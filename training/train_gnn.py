"""Train a heterogenous GNN for Person–Job link prediction using PyTorch Geometric.

This version includes detailed validation metrics, configurable logging, and
an optional decision threshold adjustment for evaluation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import negative_sampling


def _load_dataset(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_heterodata(dataset: Dict[str, Any]) -> Tuple[HeteroData, Dict[str, List[str]]]:
    data = HeteroData()
    id_lists = {}

    nodes = dataset["nodes"]
    for node_type, payload in nodes.items():
        features = torch.tensor(payload["features"], dtype=torch.float)
        data[node_type].x = features
        id_lists[node_type] = payload["ids"]

    edges = dataset["edges"]
    for edge_type, payload in edges.items():
        src = torch.tensor(payload["src"], dtype=torch.long)
        dst = torch.tensor(payload["dst"], dtype=torch.long)
        data[(payload["src_type"], edge_type, payload["dst_type"])].edge_index = torch.stack([src, dst])

    return data, id_lists


class HGTLinkPredictor(nn.Module):
    def __init__(self, in_channels: Dict[str, int], metadata, hidden_dim: int = 64):
        super().__init__()
        self.preprocess = nn.ModuleDict(
            {ntype: nn.Linear(dim, hidden_dim) for ntype, dim in in_channels.items()}
        )
        self.conv = HGTConv({ntype: hidden_dim for ntype in in_channels}, hidden_dim, metadata=metadata, heads=2)
        self.lin_src = nn.Linear(hidden_dim, hidden_dim)
        self.lin_dst = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, data: HeteroData):
        x_pre = {ntype: torch.relu(self.preprocess[ntype](feat)) for ntype, feat in data.x_dict.items()}
        x_conv = self.conv(x_pre, data.edge_index_dict)
        x = {}
        for ntype in x_pre.keys():
            if ntype in x_conv:
                x[ntype] = torch.relu(x_conv[ntype])
            else:
                x[ntype] = x_pre[ntype]
        return x

    def forward(self, data: HeteroData) -> HeteroData:
        z = self.encode(data)
        data = data.clone()
        data.x_dict = z
        return data

    def decode(self, z: dict, edge_index: torch.Tensor, src_type: str, dst_type: str) -> torch.Tensor:
        key_map = {k.lower(): k for k in z.keys()}
        src_key = key_map.get(src_type.lower(), src_type)
        dst_key = key_map.get(dst_type.lower(), dst_type)
        if src_key not in z:
            raise KeyError(f"Source node type '{src_type}' not found. Available: {list(z.keys())}")
        if dst_key not in z:
            raise KeyError(f"Destination node type '{dst_type}' not found. Available: {list(z.keys())}")
        z_src = z[src_key][edge_index[0]]
        z_dst = z[dst_key][edge_index[1]]
        src = self.lin_src(z_src)
        dst = self.lin_dst(z_dst)
        return (src * dst).sum(dim=-1)


def _prepare_logger(log_file: str | None) -> Tuple[Path, callable]:
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path("artifacts") / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"gnn_train_{timestamp}.log"

    handle = log_path.open("a", encoding="utf-8")

    def _log(msg: str) -> None:
        print(msg, flush=True)
        handle.write(msg + "\n")
        handle.flush()

    return log_path, _log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hetero GNN for Person–Job link prediction.")
    parser.add_argument("--dataset", required=True, help="Path to artifacts/gnn_dataset.json.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-model", default="artifacts/gnn_model.pt")
    parser.add_argument("--output-embeddings", default="artifacts/gnn_embeddings.pt")
    parser.add_argument("--log-file", default=None, help="Optional log path (defaults to artifacts/training_logs/…)")
    parser.add_argument("--eval-threshold", type=float, default=0.5, help="Decision threshold for precision/recall.")
    args = parser.parse_args()

    dataset = _load_dataset(Path(args.dataset))
    data, id_lists = _build_heterodata(dataset)
    log_path, logger = _prepare_logger(args.log_file)

    in_channels = {node_type: data[node_type].num_features for node_type in data.node_types}
    metadata = (list(data.node_types), list(data.edge_types))
    model = HGTLinkPredictor(in_channels, metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    node_type_map = {ntype.lower(): ntype for ntype in data.node_types}
    edge_type = next((et for et in data.edge_types if "matches" in et[1].lower()), None)
    if edge_type is None:
        raise ValueError("MATCHES edge type not found. Ensure dataset includes Person-Job edges.")
    raw_src_type, _, raw_dst_type = edge_type
    src_type = node_type_map.get(raw_src_type.lower(), raw_src_type)
    dst_type = node_type_map.get(raw_dst_type.lower(), raw_dst_type)

    edge_index = data[edge_type].edge_index
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    train_end = int(num_edges * 0.8)
    val_end = int(num_edges * 0.9)
    train_pos = edge_index[:, perm[:train_end]]
    val_pos = edge_index[:, perm[train_end:val_end]]
    test_pos = edge_index[:, perm[val_end:]]

    num_person = data["Person"].num_nodes
    num_job = data["JobPosting"].num_nodes

    def sample_negative(num_samples: int) -> torch.Tensor:
        return negative_sampling(
            edge_index=edge_index,
            num_nodes=(num_person, num_job),
            num_neg_samples=num_samples,
            method="sparse",
        )

    def train_step():
        model.train()
        optimizer.zero_grad()
        z = model.encode(data)
        pos_logits = model.decode(z, train_pos, src_type, dst_type)
        neg_edges = sample_negative(train_pos.size(1))
        neg_logits = model.decode(z, neg_edges, src_type, dst_type)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(pos_edges: torch.Tensor) -> Tuple[float, float, float]:
        model.eval()
        z = model.encode(data)
        pos_logits = model.decode(z, pos_edges, src_type, dst_type)
        neg_edges = sample_negative(pos_edges.size(1))
        neg_logits = model.decode(z, neg_edges, src_type, dst_type)
        logits = torch.cat([pos_logits, neg_logits]).sigmoid()
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        preds = (logits > args.eval_threshold).float()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        return accuracy, precision, recall

    for epoch in range(args.epochs):
        loss = train_step()
        if epoch % 5 == 0:
            val_acc, val_prec, val_recall = evaluate(val_pos)
            logger(
                f"Epoch {epoch} - Loss: {loss:.4f} - "
                f"ValAcc: {val_acc:.4f} - ValPrec: {val_prec:.4f} - ValRec: {val_recall:.4f}"
            )

    test_acc, test_prec, test_recall = evaluate(test_pos)
    logger(
        f"Test accuracy: {test_acc:.4f} - precision: {test_prec:.4f} - recall: {test_recall:.4f}"
    )

    torch.save(model.state_dict(), args.output_model)
    z = model.encode(data)
    torch.save({"node_ids": id_lists, "embeddings": z}, args.output_embeddings)
    logger(f"Saved model to {args.output_model} and embeddings to {args.output_embeddings}")
    logger(f"Logs written to {log_path}")


if __name__ == "__main__":
    main()

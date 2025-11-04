"""Export the knowledge graph to NetworkX and write GraphML/summary stats.

Usage
-----

    python scripts/export_networkx.py \
        --resume-json data/resumes_curated.json \
        --resume-limit 200 \
        --jobs data/jobs_curated.json \
        --jobs-limit 500 \
        --graphml output/networkx_hiremate.graphml \
        --summary

Outputs
-------
* GraphML file (optional) consumable by Gephi/NetworkX tooling.
* Console summary (node/edge counts per label/type) if `--summary` flag is set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any
import sys

try:
    import networkx as nx  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NetworkX is required. Install it via `pip install networkx`.") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.knowledge_graph import build_knowledge_graph, graph_to_networkx


def _load_resumes(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]
    return data


def _load_jobs(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]
    return data


def _print_summary(G: nx.Graph) -> None:
    node_labels = Counter(data.get("type") for _, data in G.nodes(data=True))
    edge_types = Counter(data.get("type") for _, _, data in G.edges(data=True))
    print("Nodes by label:")
    for label, count in node_labels.most_common():
        print(f"  {label or 'Unlabelled'}: {count}")
    print("Edges by type:")
    for edge_type, count in edge_types.most_common():
        print(f"  {edge_type or 'Unlabelled'}: {count}")


def _make_graphml_safe(G: nx.Graph) -> None:
    """Convert dict/list attributes to JSON strings so GraphML can handle them."""

    def _to_serialisable(value: Any) -> Any:
        if isinstance(value, dict):
            return json.dumps(
                {k: _to_serialisable(v) for k, v in value.items()},
                ensure_ascii=False,
            )
        if isinstance(value, (list, tuple, set)):
            serialisable = [_to_serialisable(v) for v in value]
            return json.dumps(serialisable, ensure_ascii=False)
        return value

    for _, data in G.nodes(data=True):
        for key in list(data.keys()):
            data[key] = _to_serialisable(data[key])

    for _, _, data in G.edges(data=True):
        for key in list(data.keys()):
            data[key] = _to_serialisable(data[key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NetworkX graph from HireMate data.")
    parser.add_argument("--resume-json", required=True, help="Path to resume JSON (parsed Kaggle export).")
    parser.add_argument("--resume-limit", type=int, default=None, help="Optional resume slice limit.")
    parser.add_argument("--jobs-json", required=True, help="Path to job JSON.")
    parser.add_argument("--jobs-limit", type=int, default=None, help="Optional job slice limit.")
    parser.add_argument("--graphml", help="Optional GraphML output path.")
    parser.add_argument("--summary", action="store_true", help="Print node/edge summaries.")
    parser.add_argument("--directed", action="store_true", help="Create directed graph (default undirected).")
    args = parser.parse_args()

    resumes = _load_resumes(Path(args.resume_json), args.resume_limit)
    jobs = _load_jobs(Path(args.jobs_json), args.jobs_limit)

    graph = build_knowledge_graph(resumes, jobs)
    G = graph_to_networkx(graph, directed=args.directed)

    if args.summary:
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        _print_summary(G)

    if args.graphml:
        _make_graphml_safe(G)
        out_path = Path(args.graphml)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, out_path)
        print(f"Wrote GraphML to {out_path}")


if __name__ == "__main__":
    main()

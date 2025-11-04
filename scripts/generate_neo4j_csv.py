"""Generate CSV files for Neo4j bulk import from resumes and job datasets.

Usage
-----

    python scripts/generate_neo4j_csv.py \
        --resume-json data/resumes_curated.json \
        --resume-limit 50 \
        --jobs data/jobs_curated.json \
        --jobs-limit 200 \
        --output-dir output/neo4j

The script writes ``nodes.csv`` and ``edges.csv`` (overridable via command line
options) containing rows ready for `LOAD CSV` ingestion in Neo4j. Extra
properties are stored as JSON strings so they can be expanded with APOC or
custom Cypher.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.parser import parse_cv_text
from services.knowledge_graph import build_knowledge_graph, graph_to_neo4j_rows


def _load_jobs(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Job dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Job dataset must be a list of job records.")
        if limit is not None:
            data = data[:limit]
        return data


def _load_structured_resumes(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Structured resumes JSON must be a list.")
        if limit is not None:
            data = data[:limit]
        return data


def _collect_resumes(resume_files: List[str] | None, resume_json: str | None, resume_limit: int | None) -> List[Dict[str, Any]]:
    resumes: List[Dict[str, Any]] = []
    if resume_json:
        resume_json_path = Path(resume_json)
        if not resume_json_path.exists():
            raise FileNotFoundError(resume_json_path)
        resumes.extend(_load_structured_resumes(resume_json_path, resume_limit))

    if resume_files:
        from data_loader import load_resume  # lazy import to avoid optional DOCX dependency unless needed
        for resume_path_str in resume_files:
            resume_path = Path(resume_path_str)
            if not resume_path.exists():
                raise FileNotFoundError(resume_path)
            resume_text = load_resume(resume_path)
            parsed_resume = parse_cv_text(resume_text)
            resumes.append(parsed_resume)
    return resumes


def _resolve_jobs_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        Path("data/jobs_curated.json"),
        Path("data/jobs_curated_gen.json"),
        Path("data/jobs_sample.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No job dataset provided and no default dataset found.")


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Neo4j-ready CSV exports.")
    parser.add_argument("--resume", action="append", help="Path to resume file (PDF/DOCX/TXT). Can be supplied multiple times.")
    parser.add_argument("--resume-json", help="Path to structured resume JSON (e.g., Kaggle export).")
    parser.add_argument("--resume-limit", type=int, help="Optional limit when using --resume-json.")
    parser.add_argument("--jobs", help="Path to job JSON dataset.")
    parser.add_argument("--jobs-limit", type=int, help="Optional limit on job records.")
    parser.add_argument("--output-dir", default="output/neo4j", help="Directory to write CSV files into.")
    parser.add_argument("--nodes-file", default="nodes.csv", help="Filename for nodes CSV.")
    parser.add_argument("--edges-file", default="edges.csv", help="Filename for edges CSV.")
    parser.add_argument(
        "--exclude-resume-id",
        action="append",
        default=None,
        help="Resume ID to exclude (can be provided multiple times), e.g. resume-00001.",
    )
    parser.add_argument(
        "--exclude-job-id",
        action="append",
        default=None,
        help="Job ID to exclude (can be provided multiple times), e.g. job-00006.",
    )
    args = parser.parse_args()

    resumes = _collect_resumes(args.resume, args.resume_json, args.resume_limit)
    exclude_ids = {rid for rid in (args.exclude_resume_id or []) if rid}
    if exclude_ids:
        resumes = [resume for resume in resumes if str(resume.get("id")) not in exclude_ids]

    if not resumes:
        raise ValueError("Provide at least one resume via --resume or --resume-json.")

    jobs_path = _resolve_jobs_path(args.jobs)
    jobs = _load_jobs(jobs_path, limit=args.jobs_limit)
    exclude_job_ids = {jid for jid in (args.exclude_job_id or []) if jid}
    if exclude_job_ids:
        jobs = [job for job in jobs if str(job.get("id") or job.get("job_id")) not in exclude_job_ids]

    graph = build_knowledge_graph(resumes, jobs)
    node_rows, edge_rows = graph_to_neo4j_rows(graph)

    output_dir = Path(args.output_dir)
    nodes_path = output_dir / args.nodes_file
    edges_path = output_dir / args.edges_file

    _write_csv(nodes_path, node_rows, fieldnames=["id", "type", "label", "properties"])
    _write_csv(edges_path, edge_rows, fieldnames=["type", "source", "target", "properties"])

    print(f"Wrote {len(node_rows)} node rows to {nodes_path}")
    print(f"Wrote {len(edge_rows)} edge rows to {edges_path}")


if __name__ == "__main__":
    main()

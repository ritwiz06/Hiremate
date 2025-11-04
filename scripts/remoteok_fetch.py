"""Fetch remote job listings from the RemoteOK public API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.parser import SKILL_BANK, extract_skills  # noqa: E402
from services.job_schema import build_job_record

API_URL = "https://remoteok.com/api"


def fetch_remoteok_jobs() -> List[Dict[str, object]]:
    resp = requests.get(API_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # first element is metadata
    jobs = [normalize_job(item) for item in data if isinstance(item, dict) and item.get("id")]
    return jobs


def normalize_job(item: Dict[str, object]) -> Dict[str, object]:
    title = item.get("position") or item.get("title") or ""
    company = item.get("company") or ""
    location = item.get("location") or "Remote"
    description = item.get("description") or ""
    url = item.get("url") or item.get("apply_url") or ""
    tags = item.get("tags") or []
    extracted_skills = set(extract_skills("\n".join(filter(None, [title, description])), bank=list(SKILL_BANK)))
    for tag in tags:
        if isinstance(tag, str) and tag:
            extracted_skills.add(tag.lower())

    job_type = item.get("type") or ""

    categories = []
    if isinstance(item.get("tags"), list):
        categories = [tag for tag in item["tags"] if isinstance(tag, str)]

    metadata = {
        "company_logo": item.get("company_logo"),
        "salary": item.get("salary"),
    }

    return build_job_record(
        job_id=f"remoteok-{item.get('id')}",
        source="remoteok",
        title=title,
        company=company,
        location=location,
        employment_type=job_type,
        categories=categories,
        skills=sorted(extracted_skills),
        description=description,
        apply_url=url,
        published=item.get("date", ""),
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch jobs from RemoteOK API")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of jobs")
    args = parser.parse_args()

    jobs = fetch_remoteok_jobs()
    if args.limit is not None:
        jobs = jobs[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
    print(f"Wrote {len(jobs)} jobs to {output_path}")


if __name__ == "__main__":
    main()

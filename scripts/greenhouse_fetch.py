"""Fetch jobs from Greenhouse (board JSON or Harvest API) and output JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from requests import Session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.parser import SKILL_BANK, extract_skills  # noqa: E402
from services.job_schema import build_job_record


def fetch_board_jobs(board_id: str) -> List[Dict[str, object]]:
    url = f"https://boards.greenhouse.io/{board_id}.json"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return [normalize_board_job(item) for item in data.get("jobs", [])]


def fetch_harvest_jobs(subdomain: str, token: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
    base_url = f"https://{subdomain}.greenhouse.io/v1/jobs"
    jobs: List[Dict[str, object]] = []
    page = 1
    per_page = 500
    session = Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    while True:
        params = {"per_page": per_page, "page": page, "content": "true"}
        response = session.get(base_url, params=params, timeout=30)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        for item in batch:
            jobs.append(normalize_harvest_job(item, subdomain))
            if limit is not None and len(jobs) >= limit:
                return jobs
        if len(batch) < per_page:
            break
        page += 1
        # Respect Harvest rate limit (max 120/min) -> ~0.5s between calls
        time.sleep(0.6)
    return jobs


def normalize_board_job(item: Dict[str, object]) -> Dict[str, object]:
    title = item.get("title", "")
    internal_id = item.get("id")
    location = (item.get("location") or {}).get("name", "")
    content = item.get("content", "")
    absolute_url = item.get("absolute_url", "")
    metadata = item.get("metadata") or []
    company = ""
    for entry in metadata:
        if entry.get("name", "").lower() == "company" and entry.get("value"):
            company = entry["value"].strip()
            break

    description = content
    skills = extract_skills("\n".join(filter(None, [title, description])), bank=list(SKILL_BANK))
    categories = [entry.get("value", "") for entry in metadata if entry.get("name", "").lower() == "department"]

    return build_job_record(
        job_id=f"greenhouse-{internal_id}",
        source="greenhouse",
        title=title,
        company=company,
        location=location,
        description=description,
        apply_url=absolute_url,
        skills=skills,
        categories=categories,
        metadata={"metadata": metadata},
    )


def normalize_harvest_job(item: Dict[str, object], subdomain: str) -> Dict[str, object]:
    title = item.get("name", "")
    internal_id = item.get("id")
    absolute_url = item.get("absolute_url", "")
    location_names = [office.get("name", "") for office in item.get("offices", []) if isinstance(office, dict)]
    location = ", ".join(filter(None, location_names))
    content = item.get("content", "")
    company = item.get("company") or subdomain.title()

    skills = extract_skills("\n".join(filter(None, [title, content])), bank=list(SKILL_BANK))
    categories = [dept.get("name", "") for dept in item.get("departments", []) if isinstance(dept, dict)]
    job_type = item.get("employment_type", "")

    metadata = {
        "offices": item.get("offices", []),
        "departments": item.get("departments", []),
    }

    return build_job_record(
        job_id=f"greenhouse-harvest-{internal_id}",
        source="greenhouse-harvest",
        title=title,
        company=company,
        location=location,
        description=content,
        apply_url=absolute_url,
        skills=skills,
        employment_type=job_type,
        categories=categories,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch job postings from Greenhouse")
    parser.add_argument("--board", help="Public board id, e.g. 'airbnb'")
    parser.add_argument("--subdomain", help="Harvest subdomain, e.g. 'api' for api.greenhouse.io")
    parser.add_argument("--token", help="Harvest API token")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on jobs")
    args = parser.parse_args()

    if args.token and args.subdomain:
        jobs = fetch_harvest_jobs(args.subdomain, args.token, limit=args.limit)
    elif args.board:
        jobs = fetch_board_jobs(args.board)
        if args.limit is not None:
            jobs = jobs[: args.limit]
    else:
        parser.error("Provide either --board or (--subdomain and --token) for Harvest API access")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
    print(f"Wrote {len(jobs)} jobs to {output_path}")


if __name__ == "__main__":
    main()

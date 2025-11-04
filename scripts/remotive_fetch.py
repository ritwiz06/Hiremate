"""Fetch remote job listings from Remotive public API in HireMate format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.parser import SKILL_BANK, extract_skills  # noqa: E402
from services.job_schema import build_job_record

API_URL = "https://remotive.com/api/remote-jobs"

BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
CLOSE_PARAGRAPH_RE = re.compile(r"</p>", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = BR_TAG_RE.sub("\n", text)
    text = CLOSE_PARAGRAPH_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_remotive_jobs(category: Optional[str] = None) -> List[Dict[str, object]]:
    params = {"limit": 1000}
    if category:
        params["category"] = category
    resp = requests.get(API_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return [normalize_job(item) for item in payload.get("jobs", [])]


def normalize_job(item: Dict[str, object]) -> Dict[str, object]:
    title = item.get("title", "")
    company = item.get("company_name", "")
    description_raw = item.get("description", "")
    description = _strip_html(description_raw)
    location = item.get("candidate_required_location", "Remote")
    url = item.get("url", "")
    job_type = item.get("job_type", "")
    text_for_skills = "\n".join(filter(None, [title, description]))
    skills = extract_skills(text_for_skills, bank=list(SKILL_BANK))

    categories = []
    category = item.get("category")
    if category:
        categories.append(category)

    metadata = {
        "tags": item.get("tags", []),
        "salary": item.get("salary"),
    }

    return build_job_record(
        job_id=f"remotive-{item.get('id')}",
        source="remotive",
        title=title,
        company=company,
        location=location,
        employment_type=job_type,
        categories=categories,
        skills=skills,
        description=description,
        apply_url=url,
        published=item.get("publication_date", ""),
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch jobs from Remotive API")
    parser.add_argument("--category", help="Optional category filter, e.g. 'software-dev'")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of jobs to keep")
    args = parser.parse_args()

    jobs = fetch_remotive_jobs(args.category)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
    print(f"Wrote {len(jobs)} jobs to {output_path}")


if __name__ == "__main__":
    main()

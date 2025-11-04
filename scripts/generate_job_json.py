"""Convert a job-listing dataset (e.g., from Kaggle) into HireMate job JSON.

Usage
-----

    python scripts/generate_job_json.py \
        --input data/raw_jobs.csv \
        --output data/jobs_curated.json \
        --limit 2000

Assumptions
-----------
- The CSV contains at least `title`, `company`, and either `description` or
  `job_description` columns. Optional columns: `location`, `skills`, `url`.
- When an explicit `skills` column is missing, we fall back to matching the
  HireMate skill bank against title + description text.

The generated JSON is a list of documents with the fields expected by the
job matcher and (later) the GNN pipeline:

    {
        "id": "job-0001",
        "title": "Senior Data Scientist",
        "company": "Example Corp",
        "location": "Remote",
        "skills": ["python", "machine learning", ...],
        "description": "...",
        "apply_url": "https://..."
    }

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from nlp.parser import SKILL_BANK, extract_skills
from services.job_schema import build_job_record


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def _extract_skills(row: dict) -> List[str]:
    explicit = _normalize_text(row.get("skills"))
    if explicit:
        parts: List[str] = []
        for chunk in explicit.replace("/", ",").split(","):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
        if parts:
            return sorted({skill.lower() for skill in parts})

    text_blobs: List[str] = []
    for key in ("title", "job_title", "description", "job_description", "summary", "skills_desc"):
        value = _normalize_text(row.get(key))
        if value:
            text_blobs.append(value)
    combined = "\n".join(text_blobs)
    if not combined:
        return []
    skills = extract_skills(combined, bank=list(SKILL_BANK))
    return [skill.lower() for skill in skills]


def _records_from_iter(iterable: Iterable[dict], limit: int | None) -> List[dict]:
    records: List[dict] = []
    for idx, row in enumerate(iterable, start=1):
        if limit is not None and len(records) >= limit:
            break

        title = _normalize_text(row.get("title")) or _normalize_text(row.get("job_title"))
        company = (
            _normalize_text(row.get("company"))
            or _normalize_text(row.get("company_name"))
            or _normalize_text(row.get("employer_name"))
        )
        description = (
            _normalize_text(row.get("description"))
            or _normalize_text(row.get("job_description"))
            or _normalize_text(row.get("summary"))
            or _normalize_text(row.get("skills_desc"))
        )
        if not title or not company:
            continue

        location = (
            _normalize_text(row.get("location"))
            or _normalize_text(row.get("job_location"))
            or _normalize_text(row.get("city"))
        )
        apply_url = (
            _normalize_text(row.get("url"))
            or _normalize_text(row.get("job_url"))
            or _normalize_text(row.get("job_posting_url"))
            or _normalize_text(row.get("application_url"))
        )

        skills = _extract_skills(row)

        employment_type = _normalize_text(
            row.get("job_type")
            or row.get("employment_type")
            or row.get("formatted_work_type")
            or row.get("work_type")
        )

        categories = []
        for key in ("category", "job_category", "job_function"):
            value = _normalize_text(row.get(key))
            if value:
                categories.append(value)

        published = _normalize_text(
            row.get("original_listed_time")
            or row.get("listed_time")
            or row.get("posted_at")
            or row.get("posted_on")
        )

        metadata = {}
        for key in (
            "max_salary",
            "min_salary",
            "med_salary",
            "currency",
            "compensation_type",
            "pay_period",
            "job_posting_url",
            "posting_domain",
        ):
            value = _normalize_text(row.get(key))
            if value:
                metadata[key] = value

        record = build_job_record(
            job_id=f"kaggle-linkedin-{idx:05d}",
            source="kaggle_linkedin",
            title=title,
            company=company,
            location=location,
            employment_type=employment_type,
            categories=categories,
            skills=skills,
            description=description,
            apply_url=apply_url,
            published=published,
            metadata=metadata,
        )
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate job JSON from a Kaggle/GitHub dataset")
    parser.add_argument("--input", required=True, help="Path to CSV downloaded from Kaggle or GitHub")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to convert")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    iterable = df.to_dict(orient="records")

    records = _records_from_iter(iterable, limit=args.limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} job records to {output_path}")


if __name__ == "__main__":
    main()

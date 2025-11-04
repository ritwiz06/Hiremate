"""Convert the Kaggle structured resume dataset into HireMate JSON format.

Usage
-----

    python scripts/generate_resume_json.py \
        --people data/KaggleResume/01_people.csv \
        --education data/KaggleResume/03_education.csv \
        --experience data/KaggleResume/04_experience.csv \
        --skills data/KaggleResume/05_person_skills.csv \
        --output data/resumes_curated.json \
        --limit 10000

The output mirrors the structure produced by ``nlp.parser.parse_cv_text`` so
it can feed the job matcher and future GNN pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def build_resumes(
    people_path: Path,
    education_path: Path,
    experience_path: Path,
    skills_path: Path,
    limit: int | None,
) -> List[Dict[str, object]]:
    people_rows = _read_csv(people_path)
    education_rows = _read_csv(education_path)
    experience_rows = _read_csv(experience_path)
    skill_rows = _read_csv(skills_path)

    education_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in education_rows:
        person_id = row.get("person_id")
        if not person_id:
            continue
        entry = {
            "degree": _normalize(row.get("program")),
            "institution": _normalize(row.get("institution")),
            "location": _normalize(row.get("location")),
            "years": _normalize(row.get("start_date")),
            "summary": "",
        }
        education_map[person_id].append(entry)

    experience_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in experience_rows:
        person_id = row.get("person_id")
        if not person_id:
            continue
        years = " - ".join(filter(None, [_normalize(row.get("start_date")), _normalize(row.get("end_date"))]))
        entry = {
            "role": _normalize(row.get("title")),
            "company": _normalize(row.get("firm")),
            "location": _normalize(row.get("location")),
            "years": years,
            "summary": "",
        }
        experience_map[person_id].append(entry)

    skills_map: Dict[str, List[str]] = defaultdict(list)
    for row in skill_rows:
        person_id = row.get("person_id")
        skill = _normalize(row.get("skill"))
        if person_id and skill:
            normalized = skill.lower()
            if normalized not in skills_map[person_id]:
                skills_map[person_id].append(normalized)

    resumes: List[Dict[str, object]] = []
    for idx, row in enumerate(people_rows, start=1):
        if limit is not None and len(resumes) >= limit:
            break
        person_id = row.get("person_id")
        if not person_id:
            continue

        contact = {
            "emails": [_normalize(row.get("email"))] if _normalize(row.get("email")) else [],
            "phones": [_normalize(row.get("phone"))] if _normalize(row.get("phone")) else [],
            "links": {
                "urls": [],
                "linkedin": [_normalize(row.get("linkedin"))] if _normalize(row.get("linkedin")) else [],
                "github": [],
            },
        }

        resume = {
            "id": f"resume-{idx:05d}",
            "name": _normalize(row.get("name")) or None,
            "summary": "",
            "contact": contact,
            "skills": skills_map.get(person_id, []),
            "education": education_map.get(person_id, []),
            "experience": experience_map.get(person_id, []),
            "projects": [],
            "publications": [],
        }
        resumes.append(resume)

    return resumes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resume JSON from Kaggle structured dataset")
    parser.add_argument("--people", required=True, help="Path to 01_people.csv")
    parser.add_argument("--education", required=True, help="Path to 03_education.csv")
    parser.add_argument("--experience", required=True, help="Path to 04_experience.csv")
    parser.add_argument("--skills", required=True, help="Path to 05_person_skills.csv")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Optional resume cap")
    args = parser.parse_args()

    people_path = Path(args.people)
    if not people_path.exists():
        raise FileNotFoundError(people_path)

    output_path = Path(args.output)

    resumes = build_resumes(
        people_path,
        Path(args.education),
        Path(args.experience),
        Path(args.skills),
        limit=args.limit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(resumes, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(resumes)} resumes to {output_path}")


if __name__ == "__main__":
    main()

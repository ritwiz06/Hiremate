"""Fetch job listings from We Work Remotely RSS feeds and convert to JSON.

Usage
-----

    python scripts/weworkremotely_scrape.py \
        --output data/weworkremotely_jobs.json \
        --limit 500

By default the script queries a handful of popular categories (programming,
design, data, product). You can override the feed list via the `--feed`
argument repeated multiple times.

The resulting JSON aligns with the format used by the job matcher / GNN
pipeline and enriches each entry with inferred skills by reusing the
HireMate skill bank.
"""

from __future__ import annotations

import argparse
import html
import re
import json
import sys
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

try:
    import requests
except ImportError as exc:  # pragma: no cover - ensure user awareness
    raise ImportError("The 'requests' package is required. Install it via pip install requests") from exc

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from nlp.parser import SKILL_BANK, extract_skills  # noqa: E402
from services.job_schema import build_job_record


COMPANY_OVERRIDES = {
    "remote",
    "remote (full-time, offshore)",
    "part time",
    "full time",
    "contract",
    "temporary",
}

DEFAULT_FEEDS = [
    "https://weworkremotely.com/remote-jobs.rss",
    "https://weworkremotely.com/categories/remote-customer-support-jobs.rss",
    "https://weworkremotely.com/categories/remote-product-jobs.rss",
    "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-front-end-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-sales-and-marketing-jobs.rss",
    "https://weworkremotely.com/categories/remote-management-and-finance-jobs.rss",
    "https://weworkremotely.com/categories/remote-design-jobs.rss",
    "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
    "https://weworkremotely.com/categories/all-other-remote-jobs.rss",
]


BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
CLOSE_PARAGRAPH_RE = re.compile(r"</p>", re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = BR_TAG_RE.sub("\n", text)
    text = CLOSE_PARAGRAPH_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _infer_skills(text: str) -> List[str]:
    if not text:
        return []
    return extract_skills(text, bank=list(SKILL_BANK))


def fetch_feed(url: str) -> List[Dict[str, object]]:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network
        print(f"Failed to fetch {url}: {exc}")
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        print(f"Failed to parse RSS from {url}: {exc}")
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    items: List[Dict[str, object]] = []
    for item in channel.findall("item"):
        record: Dict[str, object] = {}
        for tag in ("title", "link", "description", "pubDate", "guid"):
            element = item.find(tag)
            record[tag] = element.text.strip() if element is not None and element.text else ""
        categories = [el.text.strip() for el in item.findall("category") if el.text]
        if categories:
            record["categories"] = categories
        items.append(record)
    return items


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _split_company_title(text: str) -> tuple[Optional[str], str]:
    if not text:
        return None, ""
    for delimiter in (" – ", " — ", " - ", "—", "–", ":", " -", "- "):
        if delimiter in text:
            left, right = text.split(delimiter, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    return None, text.strip()


def normalise_item(raw: Dict[str, object]) -> Dict[str, object]:
    title = _coerce_str(raw.get("title"))
    link = _coerce_str(raw.get("link"))
    guid = _coerce_str(raw.get("guid", link))
    description_html = _coerce_str(raw.get("description"))
    description = _strip_html(description_html)
    published_raw = _coerce_str(raw.get("pubDate"))

    try:
        published = datetime.strptime(published_raw, "%a, %d %b %Y %H:%M:%S %z").isoformat()
    except ValueError:
        published = published_raw

    text_for_skills = "\n".join(filter(None, [title, description]))
    skills = _infer_skills(text_for_skills)

    company_part = _coerce_str(raw.get("company")).strip()

    company_candidate = ""
    title_part = title

    if company_part:
        company_from_field, extra_title = _split_company_title(company_part)
        if company_from_field and company_from_field.lower() not in COMPANY_OVERRIDES:
            company_candidate = company_from_field
            if extra_title and extra_title not in title_part:
                title_part = f"{extra_title} {title_part}".strip()
        elif company_part and company_part.lower() not in COMPANY_OVERRIDES:
            company_candidate = company_part

    company_from_title, title_from_title = _split_company_title(title)
    if (not company_candidate) and company_from_title and company_from_title.lower() not in COMPANY_OVERRIDES:
        company_candidate = company_from_title
        title_part = title_from_title

    if (not company_candidate) and company_from_title and company_from_title.lower() in COMPANY_OVERRIDES:
        company_candidate = ""
        title_part = title_from_title

    if company_part and not company_candidate and company_part.lower() not in COMPANY_OVERRIDES:
        company_candidate = company_part

    record = build_job_record(
        job_id=sha1(guid.encode("utf-8")).hexdigest()[:16],
        source="weworkremotely",
        title=title_part or title,
        company=company_candidate,
        location="Remote",
        categories=raw.get("categories", []),
        skills=[skill for skill in skills if len(skill) > 1],
        description=description,
        apply_url=link,
        published=published,
    )
    return record


def collect_jobs(feed_urls: Iterable[str], limit: Optional[int]) -> List[Dict[str, object]]:
    seen = set()
    collected: List[Dict[str, object]] = []
    for feed_url in feed_urls:
        for raw in fetch_feed(feed_url):
            job = normalise_item(raw)
            if job["id"] in seen:
                continue
            seen.add(job["id"])
            collected.append(job)
            if limit is not None and len(collected) >= limit:
                return collected
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape We Work Remotely RSS feeds")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    parser.add_argument(
        "--feed",
        action="append",
        default=None,
        help="RSS feed URL (can be provided multiple times)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total jobs")
    args = parser.parse_args()

    feeds = args.feed if args.feed else DEFAULT_FEEDS
    jobs = collect_jobs(feeds, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(jobs, fh, indent=2, ensure_ascii=False)

    print(f"Saved {len(jobs)} jobs to {output_path}")


if __name__ == "__main__":
    main()

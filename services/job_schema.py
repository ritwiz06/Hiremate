"""Shared helpers for constructing normalized job JSON records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _unique_list(values: Optional[Iterable[str]]) -> List[str]:
    unique: List[str] = []
    seen = set()
    if not values:
        return unique
    for value in values:
        if not value:
            continue
        text = _coerce_text(value)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(text)
    return unique


def build_job_record(
    *,
    job_id: str,
    source: str,
    title: str = "",
    company: str = "",
    location: str = "",
    employment_type: str = "",
    categories: Optional[Iterable[str]] = None,
    skills: Optional[Iterable[str]] = None,
    description: str = "",
    apply_url: str = "",
    published: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a normalized job dictionary consumed across the application."""

    record: Dict[str, Any] = {
        "id": _coerce_text(job_id),
        "source": _coerce_text(source),
        "title": _coerce_text(title),
        "company": _coerce_text(company),
        "location": _coerce_text(location),
        "employment_type": _coerce_text(employment_type),
        "categories": _unique_list(categories),
        "skills": _unique_list(skills),
        "description": _coerce_text(description),
        "apply_url": _coerce_text(apply_url),
        "published": _coerce_text(published),
        "metadata": metadata or {},
    }
    return record


__all__ = ["build_job_record"]


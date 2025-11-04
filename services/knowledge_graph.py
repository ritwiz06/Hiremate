"""Knowledge graph construction utilities for HireMate.

The helper in this module materialises the node/edge schema agreed for the
project.  It receives parsed resume dictionaries (as produced by
``nlp.parser.parse_cv_text`` or the curated Kaggle export) plus normalised job
records (via ``services.job_schema.build_job_record`` or the fetch scripts) and
emits a graph payload consumable by downstream KG/RAG components.

Design goals
------------
* Normalise identifiers with deterministic prefixes so multiple resumes/jobs can
  coexist in the same graph snapshot.
* Deduplicate frequently repeated entities such as skills, companies, and
  locations.
* Preserve lightweight provenance (e.g. ``source_resume_id`` on experience
  nodes) to simplify human review and future feature engineering.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple
import re


Node = Dict[str, Any]
Edge = Dict[str, Any]


_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _sanitize_text(text: str) -> str:
    sanitized = _CTRL_RE.sub(" ", text)
    sanitized = sanitized.replace("\n", "\\n").replace("\r", "\\n")
    return sanitized


@dataclass
class _KnowledgeGraphBuilder:
    """Stateful helper that keeps nodes deduplicated and edges append-only."""

    _nodes: Dict[Tuple[str, str], Node] = field(default_factory=dict)
    _edges: List[Edge] = field(default_factory=list)
    _skill_aliases: Dict[str, str] = field(default_factory=dict)
    _location_aliases: Dict[str, str] = field(default_factory=dict)

    _DISPLAY_KEYS: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "Person": ["name", "person_id"],
            "Skill": ["name"],
            "Company": ["name"],
            "Experience": ["role", "company"],
            "Education": ["degree", "institution"],
            "Project": ["name"],
            "Certification": ["name"],
            "Location": ["name"],
            "JobPosting": ["title", "job_id"],
            "Category": ["name"],
            "Competency": ["name"],
        }
    )

    def add_node(self, node_type: str, node_id: str, **properties: Any) -> Node:
        key = (node_type, node_id)
        node = self._nodes.get(key)
        if node is None:
            node = {"id": node_id, "type": node_type, "label": _sanitize_text(node_type)}
            self._nodes[key] = node
        for prop_key, value in properties.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            node[prop_key] = value
        self._update_display_label(node)
        return node

    def _update_display_label(self, node: Node) -> None:
        node_type = node.get("type")
        keys = self._DISPLAY_KEYS.get(node_type, [])
        for key in keys:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                node["label"] = _sanitize_text(value.strip())
                return
            if value not in (None, "", [], {}):
                node["label"] = _sanitize_text(str(value))
                return
        node["label"] = _sanitize_text(node_type or "Node")

    def add_edge(
        self,
        edge_type: str,
        source: str,
        target: str,
        **properties: Any,
    ) -> Edge:
        edge = {"type": edge_type, "source": source, "target": target}
        for prop_key, value in properties.items():
            if value in (None, "", []):
                continue
            edge[prop_key] = value
        self._edges.append(edge)
        return edge

    # -- Normalisation helpers -------------------------------------------------

    def _skill_id(self, name: str) -> str:
        canonical = name.strip().lower()
        if not canonical:
            raise ValueError("Skill name cannot be empty")
        if canonical not in self._skill_aliases:
            node_id = f"skill:{canonical}"
            self._skill_aliases[canonical] = node_id
        return self._skill_aliases[canonical]

    def _location_id(self, raw: str) -> str:
        canonical = raw.strip().lower()
        if not canonical:
            canonical = "unknown"
        if canonical not in self._location_aliases:
            sha = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
            node_id = f"location:{sha}"
            self._location_aliases[canonical] = node_id
            self.add_node("Location", node_id, name=raw)
        return self._location_aliases[canonical]


def _resume_identifier(resume: Dict[str, Any], index: int) -> str:
    """Return a deterministic identifier for the resume/person."""
    resume_id = resume.get("id")
    if resume_id:
        return str(resume_id)
    key_material = "|".join(
        part or ""
        for part in [
            resume.get("name"),
            resume.get("summary"),
            ",".join(resume.get("skills", []) or []),
        ]
    )
    digest = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:12]
    return f"resume-{index:05d}-{digest}"


def _as_list(value: Any) -> List[Any]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_knowledge_graph(
    resumes: Iterable[Dict[str, Any]],
    jobs: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Construct nodes and edges following the HireMate KG schema.

    Parameters
    ----------
    resumes:
        Iterable of parsed resume dictionaries (from the parser or curated CSV).
    jobs:
        Iterable of normalised job posting dictionaries (from build_job_record
        or the fetch/generation scripts).

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...]}`` where nodes contain ``id`` and
        ``label`` plus properties, and edges contain ``type``, ``source``,
        ``target`` plus optional metadata such as ``weight``.
    """

    builder = _KnowledgeGraphBuilder()

    for idx, resume in enumerate(resumes, start=1):
        person_slug = _resume_identifier(resume, idx)
        person_node_id = f"person:{person_slug}"
        name = resume.get("name")
        builder.add_node(
            "Person",
            person_node_id,
            person_id=person_slug,
            name=name,
            summary=resume.get("summary"),
            contact=resume.get("contact"),
        )

        # Person -> Skill
        for skill in _as_list(resume.get("skills")):
            if not skill:
                continue
            skill_id = builder._skill_id(str(skill))
            builder.add_node("Skill", skill_id, name=str(skill))
            builder.add_edge(
                "PERSON_HAS_SKILL",
                source=person_node_id,
                target=skill_id,
                weight=1.0,
                provenance="resume.skills",
            )

        # Person -> Competency
        for comp in _as_list(resume.get("competencies")):
            if not comp:
                continue
            comp_id = f"competency:{comp.strip().lower()}"
            builder.add_node("Competency", comp_id, name=comp)
            builder.add_edge(
                "PERSON_HAS_COMPETENCY",
                source=person_node_id,
                target=comp_id,
                provenance="resume.competencies",
            )

        # Person -> Education
        for edu_idx, edu in enumerate(_as_list(resume.get("education")), start=1):
            edu_node_id = f"education:{person_slug}:{edu_idx:02d}"
            builder.add_node(
                "Education",
                edu_node_id,
                degree=edu.get("degree"),
                institution=edu.get("institution"),
                location=edu.get("location"),
                years=edu.get("years"),
                summary=edu.get("summary"),
                source_resume_id=person_slug,
            )
            builder.add_edge(
                "PERSON_COMPLETED_EDUCATION",
                source=person_node_id,
                target=edu_node_id,
            )
            location = edu.get("location")
            if location:
                loc_id = builder._location_id(location)
                builder.add_edge(
                    "EDUCATION_LOCATED_IN",
                    source=edu_node_id,
                    target=loc_id,
                )

        # Person -> Experience
        for exp_idx, exp in enumerate(_as_list(resume.get("experience")), start=1):
            exp_node_id = f"experience:{person_slug}:{exp_idx:02d}"
            builder.add_node(
                "Experience",
                exp_node_id,
                role=exp.get("role"),
                summary=exp.get("summary"),
                years=exp.get("years"),
                source_resume_id=person_slug,
            )
            builder.add_edge(
                "PERSON_HELD_EXPERIENCE",
                source=person_node_id,
                target=exp_node_id,
            )

            company = exp.get("company")
            if company:
                company_id = f"company:{company.strip().lower()}"
                builder.add_node("Company", company_id, name=company)
                builder.add_edge(
                    "EXPERIENCE_AT_COMPANY",
                    source=exp_node_id,
                    target=company_id,
                )

            location = exp.get("location")
            if location:
                loc_id = builder._location_id(location)
                builder.add_edge(
                    "EXPERIENCE_LOCATED_IN",
                    source=exp_node_id,
                    target=loc_id,
                )

            for skill in _as_list(exp.get("skills")):
                if not skill:
                    continue
                skill_id = builder._skill_id(str(skill))
                builder.add_node("Skill", skill_id, name=str(skill))
                builder.add_edge(
                    "EXPERIENCE_USES_SKILL",
                    source=exp_node_id,
                    target=skill_id,
                )

        # Person -> Projects & Certifications
        for proj_idx, proj in enumerate(_as_list(resume.get("projects")), start=1):
            proj_node_id = f"project:{person_slug}:{proj_idx:02d}"
            builder.add_node(
                "Project",
                proj_node_id,
                name=proj.get("name"),
                details=proj.get("details"),
                source_resume_id=person_slug,
            )
            builder.add_edge(
                "PERSON_COMPLETED_PROJECT",
                source=person_node_id,
                target=proj_node_id,
            )
            for skill in _as_list(proj.get("skills")):
                if not skill:
                    continue
                skill_id = builder._skill_id(str(skill))
                builder.add_node("Skill", skill_id, name=str(skill))
                builder.add_edge(
                    "PROJECT_USES_SKILL",
                    source=proj_node_id,
                    target=skill_id,
                )

        for cert_idx, cert in enumerate(_as_list(resume.get("certifications")), start=1):
            cert_node_id = f"certification:{person_slug}:{cert_idx:02d}"
            builder.add_node(
                "Certification",
                cert_node_id,
                name=cert.get("name") if isinstance(cert, dict) else cert,
                provider=(cert or {}).get("provider") if isinstance(cert, dict) else None,
                issued_date=(cert or {}).get("issued_date") if isinstance(cert, dict) else None,
                source_resume_id=person_slug,
            )
            builder.add_edge(
                "PERSON_HAS_CERTIFICATION",
                source=person_node_id,
                target=cert_node_id,
            )

    # Job postings and related edges
    for job in jobs:
        job_id_raw = job.get("id") or job.get("job_id")
        if not job_id_raw:
            continue
        job_node_id = f"job:{job_id_raw}"
        builder.add_node(
            "JobPosting",
            job_node_id,
            job_id=str(job_id_raw),
            title=job.get("title"),
            description=job.get("description"),
            apply_url=job.get("apply_url") or job.get("url"),
            source=job.get("source"),
            published=job.get("published"),
            employment_type=job.get("employment_type"),
        )

        company = job.get("company")
        if company:
            company_id = f"company:{company.strip().lower()}"
            builder.add_node("Company", company_id, name=company)
            builder.add_edge(
                "JOB_POSTED_BY",
                source=job_node_id,
                target=company_id,
            )

        location = job.get("location")
        if location:
            loc_id = builder._location_id(location)
            builder.add_edge(
                "JOB_LOCATED_IN",
                source=job_node_id,
                target=loc_id,
            )

        for skill in _as_list(job.get("skills")):
            if not skill:
                continue
            skill_id = builder._skill_id(str(skill))
            builder.add_node("Skill", skill_id, name=str(skill))
            builder.add_edge(
                "JOB_REQUIRES_SKILL",
                source=job_node_id,
                target=skill_id,
            )

        for category in _as_list(job.get("categories")):
            if not category:
                continue
            cat_id = f"category:{category.strip().lower()}"
            builder.add_node("Category", cat_id, name=category)
            builder.add_edge(
                "JOB_IN_CATEGORY",
                source=job_node_id,
                target=cat_id,
            )

        for competency in _as_list(job.get("competencies")):
            if not competency:
                continue
            comp_id = f"competency:{competency.strip().lower()}"
            builder.add_node("Competency", comp_id, name=competency)
            builder.add_edge(
                "JOB_PREFERRED_COMPETENCY",
                source=job_node_id,
                target=comp_id,
            )

    return {
        "nodes": list(builder._nodes.values()),
        "edges": builder._edges,
    }

def graph_to_neo4j_rows(
    graph: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Prepare node and edge rows suitable for Neo4j CSV imports."""

    def _prepare(value: Any) -> Any:
        if isinstance(value, str):
            return _sanitize_text(value)
        if isinstance(value, dict):
            return {k: _prepare(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_prepare(v) for v in value]
        return value

    node_rows: List[Dict[str, str]] = []
    for node in graph.get("nodes", []):
        node_id = str(node.get("id"))
        node_type = node.get("type") or ""
        label = _sanitize_text(str(node.get("label") or node_type or ""))
        raw_props = {
            key: _prepare(value)
            for key, value in node.items()
            if key not in {"id", "type", "label"} and value not in (None, "", [], {})
        }
        props_json = json.dumps(raw_props, ensure_ascii=False)
        encoded_props = base64.b64encode(props_json.encode("utf-8")).decode("ascii")
        node_rows.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "properties": encoded_props,
            }
        )

    edge_rows: List[Dict[str, str]] = []
    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        edge_type = edge.get("type") or ""
        raw_props = {
            key: _prepare(value)
            for key, value in edge.items()
            if key not in {"type", "source", "target"} and value not in (None, "", [], {})
        }
        props_json = json.dumps(raw_props, ensure_ascii=False)
        encoded_props = base64.b64encode(props_json.encode("utf-8")).decode("ascii")
        edge_rows.append(
            {
                "type": edge_type,
                "source": str(source),
                "target": str(target),
                "properties": encoded_props,
            }
        )
    return node_rows, edge_rows


try:  # optional dependency
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore


def graph_to_networkx(graph: Dict[str, List[Dict[str, Any]]], directed: bool = True):
    """Return a NetworkX graph representation of the knowledge graph.

    Parameters
    ----------
    graph:
        Output from :func:`build_knowledge_graph`.
    directed:
        Whether to create a directed graph (default True).

    Returns
    -------
    networkx.Graph or networkx.DiGraph

    Raises
    ------
    ImportError
        If NetworkX is not installed.
    """

    if nx is None:
        raise ImportError("networkx is not installed. Install it via `pip install networkx`.")

    G = nx.DiGraph() if directed else nx.Graph()

    for node in graph.get("nodes", []):
        node_id = node.get("id")
        if not node_id:
            continue
        G.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})

    for edge in graph.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        attrs = {k: v for k, v in edge.items() if k not in {"source", "target"}}
        if directed or G.is_multigraph():
            G.add_edge(source, target, **attrs)
        else:
            if G.has_edge(source, target):
                G[source][target].setdefault("relations", []).append(attrs)
            else:
                G.add_edge(source, target, relations=[attrs])
    return G


__all__ = ["build_knowledge_graph", "graph_to_neo4j_rows", "graph_to_networkx"]

import base64
import json
import pytest

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from services.knowledge_graph import (
    build_knowledge_graph,
    graph_to_neo4j_rows,
    graph_to_networkx,
)


def _edge_exists(edges, edge_type, source=None, target=None):
    for edge in edges:
        if edge["type"] != edge_type:
            continue
        if source is not None and edge["source"] != source:
            continue
        if target is not None and edge["target"] != target:
            continue
        return True
    return False


def test_build_knowledge_graph_creates_expected_nodes_and_edges():
    resume = {
        "id": "resume-001",
        "name": "Jane Doe",
        "summary": "Product-minded data scientist.",
        "contact": {"emails": ["jane@example.com"]},
        "skills": ["Python", "SQL"],
        "competencies": ["Leadership"],
        "education": [
            {"degree": "BSc Computer Science", "institution": "State University", "location": "Austin, TX", "years": "2014-2018"}
        ],
        "experience": [
            {
                "role": "Data Scientist",
                "company": "Insight Labs",
                "location": "Remote",
                "years": "2019-Present",
                "summary": "Built ML models.",
                "skills": ["Python", "Machine Learning"],
            }
        ],
        "projects": [
            {"name": "Forecasting Platform", "details": "Implemented demand forecasting service.", "skills": ["Python", "Time Series"]}
        ],
        "certifications": [{"name": "AWS Practitioner", "provider": "AWS"}],
    }

    job = {
        "id": "job-123",
        "title": "Senior Data Scientist",
        "company": "Insight Labs",
        "location": "Remote",
        "skills": ["Python", "Machine Learning"],
        "categories": ["Data"],
        "employment_type": "Full Time",
        "source": "remoteok",
    }

    graph = build_knowledge_graph([resume], [job])
    nodes = graph["nodes"]
    edges = graph["edges"]

    node_map = {(node["type"], node["id"]): node for node in nodes}

    assert ("Person", "person:resume-001") in node_map
    assert ("JobPosting", "job:job-123") in node_map
    assert any(node["type"] == "Company" and node.get("name") == "Insight Labs" for node in nodes)
    assert any(node["type"] == "Skill" and node.get("name") == "Python" for node in nodes)
    assert any(node["type"] == "Person" and node.get("label") == "Jane Doe" for node in nodes)
    assert any(node["type"] == "Skill" and node.get("label") == "Python" for node in nodes)

    person_id = "person:resume-001"
    experience_id = "experience:resume-001:01"
    project_id = "project:resume-001:01"
    certification_id = "certification:resume-001:01"

    assert _edge_exists(edges, "PERSON_HAS_SKILL", source=person_id)
    assert _edge_exists(edges, "PERSON_HELD_EXPERIENCE", source=person_id, target=experience_id)
    assert _edge_exists(edges, "EXPERIENCE_AT_COMPANY", source=experience_id)
    assert _edge_exists(edges, "PROJECT_USES_SKILL", source=project_id)
    assert _edge_exists(edges, "PERSON_HAS_CERTIFICATION", source=person_id, target=certification_id)
    assert _edge_exists(edges, "JOB_POSTED_BY", source="job:job-123")
    assert _edge_exists(edges, "JOB_REQUIRES_SKILL", source="job:job-123")


def test_graph_to_neo4j_rows_returns_json_properties():
    graph = build_knowledge_graph(
        [{"id": "resume-001", "name": "Jane Doe", "contact": {"emails": ["jane@example.com"]}, "skills": ["Python"], "experience": []}],
        [{"id": "job-1", "title": "Data Scientist", "skills": ["Python"]}],
    )
    node_rows, edge_rows = graph_to_neo4j_rows(graph)
    person_row = next(row for row in node_rows if row["type"] == "Person")
    props = json.loads(base64.b64decode(person_row["properties"]).decode("utf-8"))
    assert props.get("name") == "Jane Doe"
    assert isinstance(props.get("contact"), dict)
    assert person_row["label"] == "Jane Doe"
    if edge_rows:
        props_edge = json.loads(base64.b64decode(edge_rows[0]["properties"]).decode("utf-8"))
        assert isinstance(props_edge, dict)


@pytest.mark.skipif(nx is None, reason="networkx not installed")
def test_graph_to_networkx():
    graph = build_knowledge_graph(
        [{"id": "resume-001", "name": "Jane Doe", "skills": ["Python"], "experience": []}],
        [{"id": "job-1", "title": "Data Scientist", "skills": ["Python"]}],
    )
    G = graph_to_networkx(graph)
    assert G.has_node("person:resume-001")
    assert any(data.get("type") == "Person" for _, data in G.nodes(data=True))

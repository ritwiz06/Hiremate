import json
from pathlib import Path

import nlp.parser as parser_module


parser_module.extract_name_via_llm = None
parser_module.refine_experience_via_llm = None
parser_module.compare_experience_outputs = None

parse_cv_text = parser_module.parse_cv_text


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "resume_samples.json"


def test_resume_regressions():
    samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for sample in samples:
        raw_text = sample["text"]
        expected = sample["expected"]

        result = parse_cv_text(raw_text)

        assert result["name"] == expected["name"]

        for email in expected.get("emails", []):
            assert email in result["contact"]["emails"]

        for skill in expected.get("skills", []):
            normalized = skill.strip()
            assert any(normalized.lower() == s.lower() for s in result["skills"])

        roles = [entry.get("role", "") for entry in result.get("experience", [])]
        for expected_role in expected.get("experience_roles", []):
            assert any(expected_role.lower() in role.lower() for role in roles)

        for entry in result.get("experience", []):
            assert "skills" in entry

    assert isinstance(result.get("competencies"), list)
    assert isinstance(result.get("experience_projects"), list)

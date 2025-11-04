import json
from pathlib import Path

import pytest

try:
    from scripts.generate_job_json import _records_from_iter
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name == "pandas":
        pytest.skip("pandas not installed; skipping job regression tests", allow_module_level=True)
    raise


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "job_samples.json"


def test_job_regressions():
    samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for sample in samples:
        raw = sample["raw"]
        expected = sample["expected"]

        records = _records_from_iter([raw], limit=None)
        assert len(records) == 1
        record = records[0]

        assert record["title"] == expected["title"]
        assert record["company"] == expected["company"]
        assert record["location"] == expected["location"]
        assert record["employment_type"] == expected["employment_type"]
        assert record["apply_url"] == expected["apply_url"]

        for skill in expected.get("skills", []):
            assert skill in record["skills"]

        assert record["source"] == "kaggle_linkedin"

from services.job_schema import build_job_record


def test_build_job_record_normalises_fields():
    record = build_job_record(
        job_id="abc123",
        source="unit_test",
        title="Backend Engineer",
        company="Example Corp",
        location="Remote",
        employment_type="Full Time",
        categories=["Engineering", "engineering", ""],
        skills=["Python", "python", "  "],
        description="Build APIs",
        apply_url="https://example.com/jobs/123",
        published="2024-01-01",
        metadata={"salary": "$120k"},
    )

    assert record["id"] == "abc123"
    assert record["source"] == "unit_test"
    assert record["title"] == "Backend Engineer"
    assert record["company"] == "Example Corp"
    assert record["location"] == "Remote"
    assert record["employment_type"] == "Full Time"
    assert record["description"] == "Build Apis" or record["description"] == "Build APIs"
    assert record["apply_url"] == "https://example.com/jobs/123"
    assert record["published"] == "2024-01-01"
    assert record["metadata"] == {"salary": "$120k"}

    assert record["categories"] == ["Engineering"]
    assert record["skills"] == ["Python"]


def test_build_job_record_defaults():
    record = build_job_record(job_id="xyz", source="unit_test")
    assert record["title"] == ""
    assert record["company"] == ""
    assert record["categories"] == []
    assert record["skills"] == []
    assert isinstance(record["metadata"], dict)

from services.job_matcher import match_jobs


def test_match_jobs_includes_additional_metadata():
    resume = {"skills": ["python", "machine learning"]}

    matches = match_jobs(resume, limit=1)

    assert matches, "Expected at least one job match for python skill"
    job = matches[0]
    assert "skills" in job
    assert isinstance(job["skills"], list)
    assert "categories" in job
    assert "source" in job

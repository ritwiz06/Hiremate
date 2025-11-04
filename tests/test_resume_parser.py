import nlp.parser as parser_module


# Disable optional LLM helpers for deterministic tests.
parser_module.extract_name_via_llm = None
parser_module.refine_experience_via_llm = None
parser_module.compare_experience_outputs = None

parse_cv_text = parser_module.parse_cv_text


def test_parse_cv_text_structure_and_content():
    sample_resume = """John Doe
    Email: john.doe@example.com
    Phone: +1 123-456-7890

    Summary
    Experienced software engineer with a focus on backend systems and cloud services.

    Experience
    Software Engineer, Acme Corp
    Jan 2020 - Present
    - Led migration of legacy APIs to a microservices architecture on AWS.

    Junior Developer, Example Labs
    Jun 2018 - Dec 2019
    - Implemented data ingestion pipelines using Python and PostgreSQL.

    Education
    Bachelor of Science in Computer Science
    University of Somewhere
    2014 - 2018

    Skills
    Python, Java, AWS, Docker, PostgreSQL, communication, teamwork

    Projects
    Real-time Monitoring Platform – Built event-driven alerting with Kafka and FastAPI.
    """

    result = parse_cv_text(sample_resume)

    expected_top_level_keys = {
        "name",
        "contact",
        "summary",
        "skills",
        "education",
        "experience",
        "projects",
        "publications",
        "competencies",
        "experience_projects",
        "sections_found",
    }
    assert set(result.keys()) == expected_top_level_keys

    assert result["name"] == "John Doe"
    assert "john.doe@example.com" in result["contact"]["emails"]
    assert "+1 123-456-7890" in result["contact"]["phones"]
    assert "AWS" in result["skills"]

    education_entries = result["education"]
    assert education_entries, "Education entries should not be empty"
    first_degree = education_entries[0]
    assert first_degree["institution"].lower().startswith("university of somewhere".lower())

    experience_entries = result["experience"]
    assert experience_entries, "Experience entries should not be empty"
    roles = {entry["role"] for entry in experience_entries}
    assert "Software Engineer, Acme Corp" in roles or "Software Engineer" in roles

    assert all("skills" in entry for entry in experience_entries)

    projects = result["projects"]
    assert isinstance(projects, list)
    if projects:
        combined_text = " ".join(
            f"{project.get('name', '')} {project.get('details', '')}" for project in projects if isinstance(project, dict)
        ).lower()
        assert combined_text.strip()


def test_project_skill_extraction_filters_single_character_skills():
    resume = {
        "summary": "",
        "experience": [],
        "projects": [
            {
                "name": "Smart Contactless Airport Baggage System",
                "details": "Designed and implemented a contactless baggage management system using RFID and computer vision.",
            }
        ],
    }
    parser_module._annotate_resume(resume)
    skills = resume["projects"][0]["skills"]
    assert "C" not in skills
    assert "R" not in skills
    assert "Computer Vision" in skills


def test_publication_fragments_are_merged():
    text = """Publications
Agarwal, Ritik and et. al. (2023). "Smart Distributed Contactless Airport Baggage Management and Han-
dling System". In: Scalable and Distributed Machine Learning and Deep Learning Patterns (10.4018/978-
1-6684-9804-0.ch005)."""
    pubs = parser_module.extract_publications(text)
    assert len(pubs) == 1
    publication = pubs[0]
    assert "Handling System" in publication
    assert "\n" not in publication

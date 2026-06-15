from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_package_source_has_no_ai_or_app_imports() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "optcpv").rglob("*.py"))

    assert "google.genai" not in source
    assert "fastapi" not in source.lower()


def test_core_dependencies_exclude_app_and_ai_packages() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_block = pyproject.split("[project]", 1)[1].split("[project.optional-dependencies]", 1)[0]
    dependencies_line = next(line for line in project_block.splitlines() if line.startswith("dependencies"))
    dependencies = {
        dependency.strip().strip('"').split(">=")[0].lower()
        for dependency in dependencies_line.split("[", 1)[1].split("]", 1)[0].split(",")
        if dependency.strip()
    }

    assert dependencies.isdisjoint({"fastapi", "uvicorn", "python-multipart", "google-genai"})

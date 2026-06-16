from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def test_cv_dependency_packages_available() -> None:
    for module in ["cairosvg", "cv2", "numpy", "PIL", "schemdraw"]:
        assert importlib.util.find_spec(module) is not None

    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import schemdraw  # noqa: F401

    try:
        import cairosvg  # noqa: F401
    except OSError as exc:
        # Some local macOS CI hosts have the Python package but not libcairo.
        # OptCPV's raster module must still exercise its deterministic fallback.
        assert "cairo" in str(exc).lower()


def test_core_dependencies_include_cv_and_renderer() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_block = pyproject.split("[project]", 1)[1].split("[project.optional-dependencies]", 1)[0]

    assert "opencv-python-headless" in project_block
    assert "schemdraw" in project_block
    assert "cairosvg" in project_block
    assert "fastapi" not in project_block.lower()
    assert "google-genai" not in project_block.lower()

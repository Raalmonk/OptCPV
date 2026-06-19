import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "extract_textbook_corpus.py"
SPEC = importlib.util.spec_from_file_location("extract_textbook_corpus", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def test_caption_detection_uses_textbook_caption_marker_font() -> None:
    caption = extractor.TextLine(
        text="Figure 1.12 (a) The right side shows a one-op-amp differential amplifier.",
        x0=54.0,
        top=391.0,
        x1=378.0,
        bottom=402.0,
        first_font="KFCNFC+AntiqueOliveStd-Black",
        first_size=8.0,
    )
    body_reference = extractor.TextLine(
        text="Figure 1.11(d), the input-output characteristic, shows that a circuit can have positive gain.",
        x0=72.0,
        top=198.0,
        x1=378.0,
        bottom=221.0,
        first_font="KFCGEB+TimesTenLTStd-Roman",
        first_size=10.0,
    )

    assert extractor._is_caption_line(caption.text, caption)
    assert not extractor._is_caption_line(body_reference.text, body_reference)


def test_problem_split_only_accepts_current_chapter_ids() -> None:
    section = {
        "chapter": 1,
        "chapter_title": "CHAPTER 1 BASIC CONCEPTS",
        "start_page": 109,
        "end_page": 112,
    }
    text = """
    1.37 Use Figure P1.37 and assume Vout switches at 3.3 V.
    3.3 V U1 LT1632
    Figure P1.37 One-amp amplifier with skin-electrode impedances.
    """

    records = extractor._split_problem_records(text, section)

    assert [record["problem_id"] for record in records] == ["1.37"]
    assert "3.3 V U1" in records[0]["statement"]


def test_graphic_clusters_keep_separate_page_regions_apart() -> None:
    boxes = [
        (50.0, 40.0, 380.0, 55.0),
        (60.0, 235.0, 210.0, 260.0),
        (210.0, 265.0, 380.0, 300.0),
    ]

    clusters = extractor._graphic_clusters_in_band(boxes, 0.0, 400.0)

    assert clusters == [(50.0, 40.0, 380.0, 55.0), (60.0, 235.0, 380.0, 300.0)]

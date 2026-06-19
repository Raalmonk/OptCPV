from pathlib import Path

from PIL import Image, ImageDraw

from optcpv.image_overlay import analyze_image_overlay, render_image_overlay_svg
from optcpv.textbook_surrogate import TextbookCorpusIndex, TextbookFigureInterpreter
from tools.run_textbook_image_overlay_batch import run_batch


CORPUS_DIR = Path("textbook_circuit_corpus")


def test_image_overlay_extracts_wires_nodes_and_component_buttons(tmp_path) -> None:
    image_path = tmp_path / "simple_circuit.png"
    _write_simple_circuit_image(image_path)

    plan = analyze_image_overlay(image_path)
    svg = render_image_overlay_svg(plan)

    assert len(plan.nodes) >= 2
    assert len(plan.wires) >= 1
    assert len(plan.component_buttons) >= 1
    assert "IMAGE_GRAPH_DRAFT" in plan.middle_layer_one
    assert "IMAGE_OVERLAY_PLAN" in plan.middle_layer_two
    assert 'data-optcpv-image-overlay="true"' in svg
    assert 'class="optcpv-wire-hit"' in svg
    assert 'role="button"' in svg
    assert "window.optcpvHighlightWires" in svg


def test_image_overlay_rejects_plain_plot_without_circuit(tmp_path) -> None:
    image_path = tmp_path / "plot.png"
    _write_plot_image(image_path)

    plan = analyze_image_overlay(image_path)

    assert plan.image_kind in {"plot", "non_circuit", "possible_circuit"}
    assert plan.metrics["overlay_mode"] == "skipped_non_circuit"
    assert len(plan.wires) == 0
    assert len(plan.component_buttons) == 0
    assert "reject_reasons" in plan.middle_layer_one


def test_image_overlay_recognizes_opamp_triangle_candidate(tmp_path) -> None:
    image_path = tmp_path / "opamp.png"
    _write_opamp_image(image_path)

    plan = analyze_image_overlay(image_path)

    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert any(button.kind == "op_amp" for button in plan.component_buttons)


def test_image_overlay_recognizes_rcl_symbol_candidates(tmp_path) -> None:
    image_path = tmp_path / "rcl.png"
    _write_rcl_image(image_path)

    plan = analyze_image_overlay(image_path)
    kinds = {button.kind for button in plan.component_buttons}

    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert "resistor" in kinds
    assert "capacitor" in kinds
    assert "inductor" in kinds


def test_textbook_caption_region_is_not_component_button() -> None:
    card = next(card for card in TextbookCorpusIndex(CORPUS_DIR).cards if card.figure_id == "1.9" and card.page == 65)
    plan = analyze_image_overlay(CORPUS_DIR / "figures/figure_1.9_p065/crop.png", card=card)

    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert not any(
        button.bbox.y > plan.height * 0.45 and button.bbox.width > plan.width * 0.45
        for button in plan.component_buttons
    )


def test_textbook_plot_like_figures_are_rejected_before_overlay() -> None:
    corpus = TextbookCorpusIndex(CORPUS_DIR)
    for figure_id, page in [("1.3", 45), ("6.20", 399), ("P6.29", 417)]:
        card = next(card for card in corpus.cards if card.figure_id == figure_id and card.page == page)
        plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)

        assert plan.image_kind == "plot"
        assert plan.metrics["overlay_mode"] == "skipped_non_circuit"
        assert len(plan.wires) == 0
        assert len(plan.component_buttons) == 0


def test_textbook_photo_panel_without_wires_is_rejected() -> None:
    card = next(card for card in TextbookCorpusIndex(CORPUS_DIR).cards if card.figure_id == "10.20" and card.page == 638)
    plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)

    assert plan.image_kind in {"non_circuit", "anatomy_or_context"}
    assert plan.metrics["overlay_mode"] == "skipped_non_circuit"
    assert len(plan.wires) == 0
    assert len(plan.component_buttons) == 0


def test_textbook_physical_context_figures_are_not_schematic_overlays() -> None:
    corpus = TextbookCorpusIndex(CORPUS_DIR)
    for figure_id, page in [("5.20", 339), ("6.12", 381), ("7.26", 471)]:
        card = next(card for card in corpus.cards if card.figure_id == figure_id and card.page == page)
        plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)

        assert plan.image_kind == "anatomy_or_context"
        assert plan.metrics["overlay_mode"] == "skipped_non_circuit"
        assert len(plan.wires) == 0
        assert len(plan.component_buttons) == 0


def test_textbook_instrumentation_system_is_functional_block_overlay() -> None:
    card = next(card for card in TextbookCorpusIndex(CORPUS_DIR).cards if card.figure_id == "1.1" and card.page == 27)

    plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)
    kinds = {button.kind for button in plan.component_buttons}

    assert plan.image_kind == "block_diagram"
    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert len(plan.component_buttons) >= 6
    assert kinds == {"functional_block"}


def test_textbook_emg_circuit_keeps_interactive_overlay() -> None:
    card = next(card for card in TextbookCorpusIndex(CORPUS_DIR).cards if card.figure_id == "E6.5" and card.page == 391)
    plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)

    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert len(plan.wires) >= 1
    assert len(plan.component_buttons) >= 1


def test_textbook_ecg_schematic_does_not_mask_caption_frequency_response_as_plot() -> None:
    card = next(card for card in TextbookCorpusIndex(CORPUS_DIR).cards if card.figure_id == "6.18" and card.page == 395)

    plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)
    kinds = {button.kind for button in plan.component_buttons}

    assert plan.image_kind == "schematic_circuit"
    assert plan.metrics["plot_region_count"] == 0
    assert len(plan.wires) >= 20
    assert len(plan.component_buttons) >= 9
    assert "op_amp" in kinds
    assert "component_group" in kinds


def test_block_diagram_overlay_uses_functional_blocks_not_rcl_symbols(tmp_path) -> None:
    image_path = tmp_path / "block_diagram.png"
    _write_block_diagram_image(image_path)
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "bd",
            "page": 1,
            "caption": "Figure bd Simplified block diagram of an EMG control system.",
            "categories": ["system_block_diagram"],
            "primary_category": "system_block_diagram",
            "is_likely_circuit": True,
        }
    )

    plan = analyze_image_overlay(image_path, card=card)
    kinds = {button.kind for button in plan.component_buttons}

    assert plan.image_kind == "block_diagram"
    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert not (kinds & {"resistor", "capacitor", "inductor", "op_amp"})
    assert "functional_block" in kinds


def test_textbook_gray_block_diagram_gets_functional_buttons() -> None:
    card = next(card for card in TextbookCorpusIndex(CORPUS_DIR).cards if card.figure_id == "13.17" and card.page == 812)

    plan = analyze_image_overlay(CORPUS_DIR / card.image, card=card)
    kinds = {button.kind for button in plan.component_buttons}

    assert plan.image_kind == "block_diagram"
    assert plan.metrics["overlay_mode"] == "interactive_overlay"
    assert "functional_block" in kinds
    assert not (kinds & {"resistor", "capacitor", "inductor", "op_amp"})


def test_textbook_image_overlay_batch_smoke(tmp_path) -> None:
    summary = run_batch(corpus_dir=CORPUS_DIR, out_dir=tmp_path, max_cards=5, contact_sheet_count=4)

    assert summary["total_cards"] == 5
    assert summary["overlay_attempts"] == 5
    assert summary["overlay_successes"] == 5
    assert summary["overlay_expected_cards"] >= 1
    assert summary["cards_with_wires"] >= 1
    assert summary["cards_with_component_buttons"] >= 1
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "contact_sheet_overlay_review.png").exists()
    middle_files = list((tmp_path / "middle_layers").glob("*.gemini_layer1_input.txt"))
    assert len(middle_files) == 5


def _write_simple_circuit_image(path: Path) -> None:
    image = Image.new("RGB", (260, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(20, 82), (72, 82)], fill="black", width=3)
    draw.rectangle([72, 68, 118, 96], outline="black", width=3)
    draw.text((82, 52), "R1", fill="black")
    draw.line([(118, 82), (210, 82)], fill="black", width=3)
    draw.line([(165, 82), (165, 130)], fill="black", width=3)
    draw.line([(150, 130), (180, 130)], fill="black", width=3)
    draw.line([(154, 138), (176, 138)], fill="black", width=3)
    draw.line([(158, 146), (172, 146)], fill="black", width=3)
    draw.ellipse([16, 78, 24, 86], fill="black")
    draw.ellipse([206, 78, 214, 86], outline="black", width=2)
    image.save(path)


def _write_plot_image(path: Path) -> None:
    image = Image.new("RGB", (260, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(28, 128), (228, 128)], fill="black", width=2)
    draw.line([(28, 128), (28, 22)], fill="black", width=2)
    points = [(28 + index * 4, 85 - int(42 * __import__("math").sin(index / 8))) for index in range(50)]
    draw.line(points, fill="black", width=2)
    draw.text((92, 135), "Time, s", fill="black")
    draw.text((4, 18), "V", fill="black")
    image.save(path)


def _write_opamp_image(path: Path) -> None:
    image = Image.new("RGB", (260, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(20, 70), (86, 70)], fill="black", width=3)
    draw.line([(20, 104), (86, 104)], fill="black", width=3)
    draw.line([(86, 42), (86, 132), (178, 87), (86, 42)], fill="black", width=3)
    draw.line([(178, 87), (230, 87)], fill="black", width=3)
    draw.text((94, 64), "+", fill="black")
    draw.text((96, 96), "-", fill="black")
    draw.ellipse([16, 66, 24, 74], fill="black")
    draw.ellipse([16, 100, 24, 108], fill="black")
    draw.ellipse([226, 83, 234, 91], outline="black", width=2)
    image.save(path)


def _write_rcl_image(path: Path) -> None:
    image = Image.new("RGB", (380, 180), "white")
    draw = ImageDraw.Draw(image)
    y = 88
    draw.line([(18, y), (66, y)], fill="black", width=3)
    zigzag = [(66, y), (76, 72), (88, 104), (100, 72), (112, 104), (124, 72), (136, y)]
    draw.line(zigzag, fill="black", width=3)
    draw.text((88, 42), "R1", fill="black")
    draw.line([(136, y), (170, y)], fill="black", width=3)
    draw.line([(184, 64), (184, 112)], fill="black", width=3)
    draw.line([(202, 64), (202, 112)], fill="black", width=3)
    draw.text((180, 118), "C1", fill="black")
    draw.line([(170, y), (184, y)], fill="black", width=3)
    draw.line([(202, y), (226, y)], fill="black", width=3)
    for x in (226, 248, 270):
        draw.arc([x, 70, x + 28, 106], start=180, end=0, fill="black", width=3)
    draw.text((250, 42), "L1", fill="black")
    draw.line([(298, y), (356, y)], fill="black", width=3)
    draw.ellipse([14, y - 4, 22, y + 4], fill="black")
    draw.ellipse([352, y - 4, 360, y + 4], outline="black", width=2)
    image.save(path)


def _write_block_diagram_image(path: Path) -> None:
    image = Image.new("RGB", (460, 180), "white")
    draw = ImageDraw.Draw(image)
    boxes = [(30, 56, 130, 112, "EMG"), (180, 56, 300, 112, "Filter"), (350, 56, 430, 112, "Stim")]
    for x1, y1, x2, y2, label in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="black", width=3)
        draw.text((x1 + 24, y1 + 20), label, fill="black")
    draw.line([(130, 84), (180, 84)], fill="black", width=3)
    draw.polygon([(180, 84), (168, 78), (168, 90)], fill="black")
    draw.line([(300, 84), (350, 84)], fill="black", width=3)
    draw.polygon([(350, 84), (338, 78), (338, 90)], fill="black")
    draw.line([(390, 112), (390, 146), (80, 146), (80, 112)], fill="black", width=2)
    image.save(path)

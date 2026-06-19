"""Run image-backed interactive overlay extraction over textbook crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

from PIL import Image, ImageDraw

from optcpv.image_overlay import ImageOverlayPlan, analyze_image_overlay, render_image_overlay_svg
from optcpv.textbook_surrogate import FigureStructureCard, TextbookCorpusIndex


DEFAULT_OUT_DIR = Path("generated/textbook_image_overlay_batch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="textbook_circuit_corpus")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-cards", type=int, default=None)
    parser.add_argument("--contact-sheet-count", type=int, default=64)
    args = parser.parse_args()
    summary = run_batch(
        corpus_dir=Path(args.corpus),
        out_dir=Path(args.out),
        max_cards=args.max_cards,
        contact_sheet_count=args.contact_sheet_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=False))


def run_batch(
    *,
    corpus_dir: Path,
    out_dir: Path,
    max_cards: int | None = None,
    contact_sheet_count: int = 0,
) -> dict[str, Any]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / "artifacts"
    middle_dir = out_dir / "middle_layers"
    artifact_dir.mkdir(exist_ok=True)
    middle_dir.mkdir(exist_ok=True)

    cards = TextbookCorpusIndex(corpus_dir).cards
    if max_cards is not None:
        cards = cards[:max_cards]

    results = []
    for index, card in enumerate(cards):
        result = _process_card(corpus_dir, card, index=index, artifact_dir=artifact_dir, middle_dir=middle_dir)
        results.append(result)
        if (index + 1) % 25 == 0:
            print(f"overlay checked {index + 1}/{len(cards)} cards")

    _write_jsonl(out_dir / "results.jsonl", results)
    summary = _summarize(results, elapsed_seconds=time.time() - started)
    _write_json(out_dir / "summary.json", summary)
    if contact_sheet_count > 0:
        _write_contact_sheet(results, out_dir / "contact_sheet_overlay_review.png", max_items=contact_sheet_count, mode="interactive")
        _write_contact_sheet(results, out_dir / "contact_sheet_rejections_review.png", max_items=contact_sheet_count, mode="skipped")
    return summary


def _process_card(
    corpus_dir: Path,
    card: FigureStructureCard,
    *,
    index: int,
    artifact_dir: Path,
    middle_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": index,
        "figure_id": card.figure_id,
        "page": card.page,
        "image": card.image,
        "is_likely_circuit": card.is_likely_circuit,
        "primary_category": card.primary_category,
        "categories": list(card.categories),
        "visual_families": list(card.visual_families),
        "overlay_attempted": False,
        "overlay_ok": False,
        "interactive_pass": False,
        "node_count": 0,
        "wire_count": 0,
        "component_button_count": 0,
        "button_kind_counts": {},
        "overlay_confidence": 0.0,
        "image_kind": "unknown",
        "circuit_confidence": 0.0,
        "reject_reasons": [],
        "overlay_mode": "not_run",
        "artifact_svg": None,
        "artifact_json": None,
        "middle_layer_one": None,
        "middle_layer_two": None,
        "semantic_layer_one": None,
        "semantic_layer_two": None,
        "gemini_layer_one_input": None,
        "gemini_layer_one_output": None,
        "gemini_layer_two_input": None,
        "gemini_layer_two_output": None,
        "error": None,
    }
    if not card.image:
        result["error"] = "missing_image_reference"
        return result
    image_path = corpus_dir / card.image
    if not image_path.exists():
        result["error"] = f"missing_image_file: {image_path}"
        return result
    result["overlay_attempted"] = True
    try:
        plan = analyze_image_overlay(image_path, card=card)
        svg = render_image_overlay_svg(plan)
    except Exception as exc:  # pragma: no cover - batch report should capture individual failures.
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    stem = _artifact_stem(card)
    svg_path = artifact_dir / f"{stem}.svg"
    json_path = artifact_dir / f"{stem}.json"
    layer_one_path = middle_dir / f"{stem}.graph_draft.txt"
    layer_two_path = middle_dir / f"{stem}.overlay_plan.txt"
    semantic_one_path = middle_dir / f"{stem}.semantic_draft.txt"
    semantic_two_path = middle_dir / f"{stem}.semantic_check.txt"
    gemini_one_input_path = middle_dir / f"{stem}.gemini_layer1_input.txt"
    gemini_one_output_path = middle_dir / f"{stem}.gemini_layer1_output.json"
    gemini_two_input_path = middle_dir / f"{stem}.gemini_layer2_input.txt"
    gemini_two_output_path = middle_dir / f"{stem}.gemini_layer2_output.json"
    svg_path.write_text(svg, encoding="utf-8")
    _write_json(json_path, plan.to_dict())
    layer_one_path.write_text(plan.middle_layer_one, encoding="utf-8")
    layer_two_path.write_text(plan.middle_layer_two, encoding="utf-8")
    semantic_one_path.write_text(str(plan.metrics.get("semantic_layer_one") or ""), encoding="utf-8")
    semantic_two_path.write_text(str(plan.metrics.get("semantic_layer_two") or ""), encoding="utf-8")
    gemini_one_input_path.write_text(str(plan.metrics.get("gemini_layer_one_input") or ""), encoding="utf-8")
    gemini_one_output_path.write_text(str(plan.metrics.get("gemini_layer_one_output") or "{}"), encoding="utf-8")
    gemini_two_input_path.write_text(str(plan.metrics.get("gemini_layer_two_input") or ""), encoding="utf-8")
    gemini_two_output_path.write_text(str(plan.metrics.get("gemini_layer_two_output") or "{}"), encoding="utf-8")

    result.update(
        {
            "overlay_ok": True,
            "interactive_pass": _interactive_pass(plan, card),
            "node_count": len(plan.nodes),
            "wire_count": len(plan.wires),
            "component_button_count": len(plan.component_buttons),
            "button_kind_counts": dict(plan.metrics.get("button_kind_counts") or {}),
            "overlay_confidence": plan.metrics.get("overlay_confidence", 0.0),
            "image_kind": plan.image_kind,
            "circuit_confidence": plan.circuit_confidence,
            "reject_reasons": list(plan.reject_reasons),
            "overlay_mode": plan.metrics.get("overlay_mode", "unknown"),
            "artifact_svg": str(svg_path),
            "artifact_json": str(json_path),
            "middle_layer_one": str(layer_one_path),
            "middle_layer_two": str(layer_two_path),
            "semantic_layer_one": str(semantic_one_path),
            "semantic_layer_two": str(semantic_two_path),
            "gemini_layer_one_input": str(gemini_one_input_path),
            "gemini_layer_one_output": str(gemini_one_output_path),
            "gemini_layer_two_input": str(gemini_two_input_path),
            "gemini_layer_two_output": str(gemini_two_output_path),
        }
    )
    return result


def _interactive_pass(plan: ImageOverlayPlan, card: FigureStructureCard) -> bool:
    if plan.image_kind in {"plot", "non_circuit", "anatomy_or_context"}:
        return plan.metrics.get("overlay_mode") == "skipped_non_circuit"
    if plan.image_kind == "block_diagram":
        return len(plan.wires) >= 1 and len(plan.component_buttons) >= 1
    if plan.image_kind == "schematic_circuit":
        return (
            len(plan.nodes) >= 2
            and len(plan.wires) >= _schematic_wire_floor(card)
            and len(plan.component_buttons) >= _schematic_button_floor(card)
        )
    if not _overlay_expected_for_card(card):
        return plan.metrics.get("overlay_mode") == "skipped_non_circuit" or bool(plan.wires or plan.component_buttons)
    return len(plan.nodes) >= 2 and len(plan.wires) >= 1 and len(plan.component_buttons) >= 1


def _schematic_button_floor(card: FigureStructureCard) -> int:
    cues = set(card.component_cues)
    families = set(card.visual_families)
    categories = set(card.categories)
    floor = 1
    if cues & {"op_amp", "feedback", "bridge", "diode", "transistor"}:
        floor = max(floor, 4)
    if cues & {"resistor", "capacitor", "inductor"}:
        floor = max(floor, 4)
    if "active_filter" in families:
        floor = max(floor, 6)
    if families & {"multi_opamp_stage", "instrumentation_amplifier"}:
        floor = max(floor, 8)
    if categories & {"bioelectric_recording"} and cues & {"op_amp", "ecg", "emg"}:
        floor = max(floor, 8)
    return floor


def _schematic_wire_floor(card: FigureStructureCard) -> int:
    return 8 if _schematic_button_floor(card) >= 8 else 1


def _overlay_expected_for_card(card: FigureStructureCard) -> bool:
    categories = set(card.categories)
    if "schematic_circuit" in categories:
        return True
    if "system_block_diagram" in categories and card.is_likely_circuit:
        return True
    return False


def _summarize(results: list[dict[str, Any]], *, elapsed_seconds: float) -> dict[str, Any]:
    attempted = [result for result in results if result["overlay_attempted"]]
    ok = [result for result in attempted if result["overlay_ok"]]
    card_expected = [result for result in results if _card_expected_for_result(result)]
    likely = [result for result in results if _overlay_expected_for_result(result)]
    likely_ok = [result for result in likely if result["overlay_ok"]]
    likely_pass = [result for result in likely_ok if result["interactive_pass"]]
    non_expected_ok = [result for result in ok if not _overlay_expected_for_result(result)]
    errors = [result for result in results if result.get("error") and not result.get("overlay_ok")]
    classification_counts: dict[str, int] = {}
    button_kind_counts: dict[str, int] = {}
    for result in ok:
        kind = str(result.get("image_kind") or "unknown")
        classification_counts[kind] = classification_counts.get(kind, 0) + 1
        for button_kind, count in dict(result.get("button_kind_counts") or {}).items():
            button_kind_counts[str(button_kind)] = button_kind_counts.get(str(button_kind), 0) + int(count)
    return {
        "total_cards": len(results),
        "card_expected_cards": len(card_expected),
        "overlay_expected_cards": len(likely),
        "overlay_attempts": len(attempted),
        "overlay_successes": len(ok),
        "overlay_errors": len(errors),
        "interactive_passes": sum(1 for result in ok if result["interactive_pass"]),
        "expected_interactive_passes": len(likely_pass),
        "expected_interactive_failures": len(likely_ok) - len(likely_pass),
        "non_expected_cards": len(non_expected_ok),
        "non_expected_skips": sum(1 for result in non_expected_ok if result.get("overlay_mode") == "skipped_non_circuit"),
        "non_expected_overlays": sum(1 for result in non_expected_ok if result.get("overlay_mode") == "interactive_overlay"),
        "classification_counts": dict(sorted(classification_counts.items())),
        "button_kind_counts": dict(sorted(button_kind_counts.items())),
        "card_expected_but_visual_rejected": [
            {
                "figure_id": result["figure_id"],
                "page": result["page"],
                "image_kind": result.get("image_kind"),
                "circuit_confidence": result.get("circuit_confidence"),
                "reject_reasons": result.get("reject_reasons", [])[:5],
            }
            for result in card_expected
            if result.get("overlay_mode") == "skipped_non_circuit"
        ][:30],
        "cards_with_wires": sum(1 for result in ok if result["wire_count"] > 0),
        "cards_with_component_buttons": sum(1 for result in ok if result["component_button_count"] > 0),
        "likely_with_wires": sum(1 for result in likely_ok if result["wire_count"] > 0),
        "likely_with_component_buttons": sum(1 for result in likely_ok if result["component_button_count"] > 0),
        "average_nodes": round(_average(result["node_count"] for result in ok), 3),
        "average_wires": round(_average(result["wire_count"] for result in ok), 3),
        "average_component_buttons": round(_average(result["component_button_count"] for result in ok), 3),
        "average_overlay_confidence": round(_average(float(result["overlay_confidence"]) for result in ok), 3),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "failure_examples": [
            {
                "figure_id": result["figure_id"],
                "page": result["page"],
                "image_kind": result.get("image_kind"),
                "circuit_confidence": result.get("circuit_confidence"),
                "reject_reasons": result.get("reject_reasons", [])[:5],
                "wires": result["wire_count"],
                "component_buttons": result["component_button_count"],
                "artifact_svg": result.get("artifact_svg"),
                "error": result.get("error"),
            }
            for result in [item for item in likely_ok if not item["interactive_pass"]][:30]
        ],
        "error_examples": [
            {"figure_id": result["figure_id"], "page": result["page"], "error": result["error"]}
            for result in errors[:20]
        ],
    }


def _overlay_expected_for_result(result: dict[str, Any]) -> bool:
    categories = set(result.get("categories") or [])
    if "schematic_circuit" in categories:
        if result.get("image_kind") in {"plot", "non_circuit", "anatomy_or_context"}:
            return False
        return True
    if "system_block_diagram" in categories and result.get("is_likely_circuit"):
        return True
    return False


def _card_expected_for_result(result: dict[str, Any]) -> bool:
    categories = set(result.get("categories") or [])
    if "schematic_circuit" in categories:
        return True
    if "system_block_diagram" in categories and result.get("is_likely_circuit"):
        return True
    return False


def _write_contact_sheet(results: list[dict[str, Any]], path: Path, *, max_items: int, mode: str = "interactive") -> None:
    candidates = [
        result
        for result in results
        if result.get("overlay_ok") and result.get("artifact_json")
    ]
    if mode == "interactive":
        candidates = [result for result in candidates if result.get("overlay_mode") == "interactive_overlay"]
    elif mode == "skipped":
        candidates = [result for result in candidates if result.get("overlay_mode") == "skipped_non_circuit"]
    candidates.sort(
        key=lambda result: (
            result.get("interactive_pass") is True,
            _overlay_expected_for_result(result) is False,
            float(result.get("overlay_confidence") or 0),
            int(result.get("wire_count") or 0),
            int(result.get("component_button_count") or 0),
        )
    )
    selected = candidates[:max_items]
    if not selected:
        return
    tile_w, tile_h, caption_h = 320, 230, 62
    cols = 4
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + caption_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(selected):
        col = index % cols
        row = index // cols
        x0 = col * tile_w
        y0 = row * (tile_h + caption_h)
        try:
            plan = _plan_from_json(Path(result["artifact_json"]))
            image = _overlay_preview(plan, tile_w - 14, tile_h - 14)
        except Exception:
            image = Image.new("RGB", (tile_w - 14, tile_h - 14), "white")
        ix = x0 + (tile_w - image.width) // 2
        iy = y0 + (tile_h - image.height) // 2
        sheet.paste(image, (ix, iy))
        draw.rectangle([x0, y0 + tile_h, x0 + tile_w - 1, y0 + tile_h + caption_h - 1], outline="#d0d0d0")
        draw.multiline_text((x0 + 8, y0 + tile_h + 5), _caption(result), fill="black", spacing=3)
    sheet.save(path)


def _overlay_preview(plan: dict[str, Any], max_width: int, max_height: int) -> Image.Image:
    image = Image.open(plan["image_path"]).convert("RGB")
    sx = max_width / max(1, image.width)
    sy = max_height / max(1, image.height)
    scale = min(sx, sy)
    scaled_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    image = image.resize(scaled_size)
    draw = ImageDraw.Draw(image)
    for wire in plan.get("wires", [])[:220]:
        points = [(point["x"] * scale, point["y"] * scale) for point in wire.get("points", [])]
        if len(points) >= 2:
            draw.line(points, fill=(239, 68, 68), width=2)
    for button in plan.get("component_buttons", [])[:80]:
        bbox = button["bbox"]
        box = [
            bbox["x"] * scale,
            bbox["y"] * scale,
            (bbox["x"] + bbox["width"]) * scale,
            (bbox["y"] + bbox["height"]) * scale,
        ]
        draw.rectangle(box, outline=(37, 99, 235), width=2)
    return image


def _plan_from_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _caption(result: dict[str, Any]) -> str:
    text = (
        f"{result.get('figure_id')} p{result.get('page')} "
        f"{result.get('image_kind')} pass={str(result.get('interactive_pass')).lower()}\n"
        f"n={result.get('node_count')} w={result.get('wire_count')} "
        f"c={result.get('component_button_count')} conf={result.get('overlay_confidence')}"
    )
    return text if len(text) <= 88 else text[:85] + "..."


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _artifact_stem(card: FigureStructureCard) -> str:
    return _safe_id(f"{card.figure_id or 'figure'}_p{card.page or 0}")


def _safe_id(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "figure"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

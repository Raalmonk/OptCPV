"""Run corpus-wide textbook surrogate checks.

The batch has two layers:

* every extracted figure card is validated against its crop and structured text;
* every likely-circuit card is rendered through a card-scoped textbook surrogate
  fixture so the communication layer is exercised for the full book corpus.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcpv import Circuit, Component, draw_optimized_artifact
from optcpv.models import CriticReport, LayoutPlan
from optcpv.planning_agent import SemanticPlanningClient
from optcpv.planning_hints import SchematicLayoutHints
from optcpv.raster import rasterize_svg
from optcpv.textbook_surrogate import FigureStructureCard, TextbookCorpusIndex, TextbookStructurePlanner


DEFAULT_OUT_DIR = Path("generated/textbook_corpus_batch")
BOOKLIKE_SCORE_LIMIT = 5
BOOKLIKE_BLOCKING_CODES = {
    "blank_raster",
    "component_overlap",
    "dense_blob",
    "fill_ratio_low",
    "label_component_overlap",
    "label_dark_collision",
    "label_outside_canvas",
    "label_visual_collision",
    "label_wire_overlap",
    "local_clutter",
    "spread_excessive",
    "tiny_or_faint",
    "too_full_canvas",
    "too_much_empty_canvas",
    "unbalanced_ink_mass",
    "viewbox_too_large",
    "wire_component_visual_collision",
    "wire_length_high",
    "wire_net_overlap",
    "wire_through_component",
}


class SingleCardPlanningClient(SemanticPlanningClient):
    """Planning client that forces one retrieved textbook card into the prompt."""

    def __init__(self, card: FigureStructureCard, style_guide: dict[str, Any]) -> None:
        self.card = card
        self.style_guide = style_guide
        self.planner = TextbookStructurePlanner()
        self.last_middle_layer_text = ""

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        self.last_middle_layer_text = _middle_layer_text(circuit, self.card, self.style_guide)
        return self.planner.propose(circuit, [self.card])

    def refine_hints(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        critic_report: CriticReport,
        reference_image: bytes | None = None,
    ) -> SchematicLayoutHints | None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="textbook_circuit_corpus")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-cards", type=int, default=None)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true", help="Save per-card SVG, PNG, and render metadata.")
    parser.add_argument("--contact-sheet-count", type=int, default=0, help="Write a worst/review contact sheet with this many items.")
    args = parser.parse_args()

    summary = run_batch(
        corpus_dir=Path(args.corpus),
        out_dir=Path(args.out),
        max_cards=args.max_cards,
        render=not args.skip_render,
        save_artifacts=args.save_artifacts,
        contact_sheet_count=args.contact_sheet_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_batch(
    *,
    corpus_dir: Path,
    out_dir: Path,
    max_cards: int | None = None,
    render: bool = True,
    save_artifacts: bool = False,
    contact_sheet_count: int = 0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    middle_dir = out_dir / "middle_layers"
    artifact_dir = out_dir / "artifacts" if render and (save_artifacts or contact_sheet_count > 0) else None
    if render:
        middle_dir.mkdir(exist_ok=True)
    if artifact_dir is not None:
        artifact_dir.mkdir(exist_ok=True)

    corpus = TextbookCorpusIndex(corpus_dir)
    cards = corpus.cards[:max_cards] if max_cards is not None else corpus.cards
    results = []
    start = time.perf_counter()
    for index, card in enumerate(cards):
        result = _check_card(
            corpus_dir,
            corpus,
            card,
            index=index,
            middle_dir=middle_dir,
            artifact_dir=artifact_dir,
            render=render,
        )
        results.append(result)
        if (index + 1) % 25 == 0:
            print(f"checked {index + 1}/{len(cards)} cards", file=sys.stderr)

    summary = _summarize(results, elapsed_seconds=time.perf_counter() - start)
    _write_jsonl(out_dir / "results.jsonl", results)
    _write_json(out_dir / "summary.json", summary)
    if contact_sheet_count > 0 and artifact_dir is not None:
        _write_worst_contact_sheet(results, out_dir / "contact_sheet_worst.png", max_items=contact_sheet_count)
    return summary


def _check_card(
    corpus_dir: Path,
    corpus: TextbookCorpusIndex,
    card: FigureStructureCard,
    *,
    index: int,
    middle_dir: Path,
    artifact_dir: Path | None,
    render: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": index,
        "figure_id": card.figure_id,
        "page": card.page,
        "primary_category": card.primary_category,
        "categories": list(card.categories),
        "is_likely_circuit": card.is_likely_circuit,
        "visual_families": list(card.visual_families),
        "component_cues": list(card.component_cues),
        "image": card.image,
        "image_ok": False,
        "structured_text_ok": _structured_text_ok(card),
        "render_attempted": False,
        "render_ok": None,
        "hard_fail": None,
        "score": None,
        "booklike_pass": None,
        "quality_reasons": [],
        "quality_blocking_codes": [],
        "planning_hints_source": None,
        "middle_layer_generated": False,
        "fixture": None,
        "violation_codes": [],
        "artifact_svg": None,
        "artifact_png": None,
        "artifact_meta": None,
        "error": None,
    }
    result.update(_image_check(corpus_dir, card))
    if not render or not _should_render(card):
        result["skip_reason"] = "render_disabled" if not render else "not_likely_circuit"
        return result

    circuit, fixture_name = circuit_fixture_for_card(card, index)
    client = SingleCardPlanningClient(card, corpus.style_guide([card]))
    result["render_attempted"] = True
    result["fixture"] = fixture_name
    try:
        artifact = draw_optimized_artifact(circuit, max_iterations=2, planning_client=client)
    except Exception as exc:  # pragma: no cover - batch report should capture failures.
        result["render_ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        if client.last_middle_layer_text:
            _write_middle_layer(middle_dir, card, client.last_middle_layer_text)
        return result

    report = artifact.critic_report or {}
    result["render_ok"] = True
    result["hard_fail"] = bool(report.get("hard_fail", False))
    result["score"] = report.get("score")
    result["planning_hints_source"] = (artifact.planning_hints_used or {}).get("source")
    result["middle_layer_generated"] = bool(client.last_middle_layer_text)
    result["recognized_topology"] = (artifact.planning_hints_used or {}).get("recognized_topology")
    result["violation_codes"] = [violation.get("code") for violation in report.get("violations", [])]
    result.update(_quality_assessment(report))
    if artifact_dir is not None:
        result.update(_write_artifact_files(artifact_dir, card, artifact))
    _write_middle_layer(middle_dir, card, client.last_middle_layer_text)
    return result


def circuit_fixture_for_card(card: FigureStructureCard, index: int) -> tuple[Circuit, str]:
    families = set(card.visual_families)
    cues = set(card.component_cues)
    base = _safe_id(card.figure_id or f"card_{index}")
    if "right_leg_drive" in families:
        return _right_leg_drive_fixture(base), "right_leg_drive"
    if "voltage_clamp" in families:
        return _voltage_clamp_fixture(base), "voltage_clamp"
    if ("system_block" in families or "isolation_barrier" in families) and "schematic_circuit" not in card.categories:
        return _system_block_fixture(base), "system_block"
    if "instrumentation_amplifier" in families or "multi_opamp_stage" in families:
        return _instrumentation_fixture(base), "instrumentation_amplifier"
    if "active_filter" in families:
        return _active_filter_fixture(base), "active_filter"
    if "sensor_bridge" in families or "bridge" in cues:
        return _bridge_fixture(base), "sensor_bridge"
    if "protection_clamp" in families or "diode" in cues or "protection" in cues:
        return _protection_fixture(base), "protection_clamp"
    if "system_block" in families or "isolation_barrier" in families:
        return _system_block_fixture(base), "system_block"
    if "current_source" in families:
        return _current_source_fixture(base), "current_source"
    if "electrode_context" in families or "bioelectric_recording" in families:
        return _bioelectric_fixture(base), "bioelectric_recording"
    if "single_opamp_feedback" in families or "op_amp" in cues:
        return _non_inverting_fixture(base), "single_opamp_feedback"
    if "schematic_circuit" in card.categories:
        return _rc_fixture(base), "generic_schematic"
    return _non_inverting_fixture(base), "fallback"


def _voltage_clamp_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_voltage_clamp_{base}",
        title="Two-electrode voltage clamp fixture",
        components=[
            Component(id="VC", type="input", pins={"out": "vc"}, label="Vc"),
            Component(id="BUF", type="op_amp", pins={"+": "vm", "-": "sense", "out": "sense"}, label="Buffer Amp"),
            Component(id="DIFF", type="op_amp", pins={"+": "vc", "-": "sense", "out": "drive"}, label="Diff Amp"),
            Component(id="A", type="ammeter", pins={"a": "drive", "b": "icl"}, label="A"),
            Component(id="RO", type="resistor", pins={"a": "icl", "b": "vm"}, label="Ro"),
            Component(id="RM", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rm"),
            Component(id="VM", type="output", pins={"in": "vm"}, label="Vm"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _right_leg_drive_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_rld_{base}",
        motif="ecg_right_leg_drive",
        title="ECG main channel with driven right leg auxiliary loop",
        components=[
            Component(id="E_PLUS", type="input", pins={"out": "ecg_plus"}, label="E+"),
            Component(id="VCM", type="input", pins={"out": "common_mode"}, label="Vcm"),
            Component(id="U1", type="op_amp", pins={"+": "ecg_plus", "-": "fb", "out": "ecg_out"}, label="U1"),
            Component(id="RF", type="resistor", pins={"a": "ecg_out", "b": "fb"}, label="Rf", role="feedback"),
            Component(id="A_AUX", type="op_amp", pins={"+": "gnd", "-": "common_mode", "out": "right_leg_drive"}, label="Aaux", role="right_leg_drive"),
            Component(id="R_RL", type="resistor", pins={"a": "right_leg_drive", "b": "right_leg_electrode"}, label="Rrl"),
            Component(id="RL", type="output", pins={"in": "right_leg_electrode"}, label="RL", role="body_terminal"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _instrumentation_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_inamp_{base}",
        motif="instrumentation_amplifier",
        title="Instrumentation amplifier fixture",
        components=[
            Component(id="INP", type="input", pins={"out": "vinp"}, label="IN+"),
            Component(id="INN", type="input", pins={"out": "vinn"}, label="IN-"),
            Component(id="U1", type="op_amp", pins={"+": "vinp", "-": "n1", "out": "o1"}, label="U1"),
            Component(id="U2", type="op_amp", pins={"+": "vinn", "-": "n2", "out": "o2"}, label="U2"),
            Component(id="U3", type="op_amp", pins={"+": "n3", "-": "n4", "out": "vout"}, label="U3"),
            Component(id="R1", type="resistor", pins={"a": "o1", "b": "n1"}, label="R1"),
            Component(id="R2", type="resistor", pins={"a": "o2", "b": "n2"}, label="R2"),
            Component(id="Rg", type="resistor", pins={"a": "n1", "b": "n2"}, label="Rg", role="gain"),
            Component(id="R3", type="resistor", pins={"a": "o1", "b": "n3"}, label="R3"),
            Component(id="R4", type="resistor", pins={"a": "n3", "b": "gnd"}, label="R4"),
            Component(id="R5", type="resistor", pins={"a": "o2", "b": "n4"}, label="R5"),
            Component(id="R6", type="resistor", pins={"a": "n4", "b": "vout"}, label="R6"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="VOUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _active_filter_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_filter_{base}",
        motif="non_inverting_op_amp",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="RIN", type="resistor", pins={"a": "vin", "b": "filt"}, label="Rin"),
            Component(id="CIN", type="capacitor", pins={"a": "filt", "b": "gnd"}, label="Cin"),
            Component(id="U1", type="op_amp", pins={"+": "filt", "-": "fb", "out": "vout"}, label="U1"),
            Component(id="RF", type="resistor", pins={"a": "vout", "b": "fb"}, label="Rf", role="feedback"),
            Component(id="RG", type="resistor", pins={"a": "fb", "b": "gnd"}, label="Rg"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="OUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _bridge_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_bridge_{base}",
        components=[
            Component(id="VEX", type="input", pins={"out": "vex"}, label="Vex"),
            Component(id="R1", type="resistor", pins={"a": "vex", "b": "sense_p"}, label="R1"),
            Component(id="R2", type="resistor", pins={"a": "sense_p", "b": "gnd"}, label="R2"),
            Component(id="R3", type="resistor", pins={"a": "vex", "b": "sense_n"}, label="R3"),
            Component(id="R4", type="resistor", pins={"a": "sense_n", "b": "gnd"}, label="R4"),
            Component(id="AMP", type="op_amp", pins={"+": "sense_p", "-": "sense_n", "out": "vout"}, label="Amp"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="Vout"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _protection_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_protection_{base}",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="IN"),
            Component(id="RS", type="resistor", pins={"a": "vin", "b": "protected"}, label="Rs"),
            Component(id="D1", type="diode", pins={"a": "protected", "b": "gnd"}, label="D1"),
            Component(id="D2", type="diode", pins={"a": "gnd", "b": "protected"}, label="D2"),
            Component(id="U1", type="op_amp", pins={"+": "protected", "-": "fb", "out": "vout"}, label="Amp"),
            Component(id="RF", type="resistor", pins={"a": "vout", "b": "fb"}, label="Rf", role="feedback"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="OUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _system_block_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_system_{base}",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="IN"),
            Component(id="AMP", type="filter_block", pins={"in": "vin", "out": "amp"}, label="Amplifier"),
            Component(id="ISO", type="filter_block", pins={"in": "amp", "out": "iso"}, label="Isolation"),
            Component(id="LPF", type="filter_block", pins={"in": "iso", "out": "filt"}, label="Filter"),
            Component(id="ADC", type="filter_block", pins={"in": "filt", "out": "dig"}, label="ADC"),
            Component(id="VOUT", type="output", pins={"in": "dig"}, label="OUT"),
        ],
    )


def _current_source_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_current_source_{base}",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="Vref"),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "sense", "out": "drive"}, label="U1"),
            Component(id="RS", type="resistor", pins={"a": "sense", "b": "gnd"}, label="Rs"),
            Component(id="RL", type="resistor", pins={"a": "drive", "b": "sense"}, label="Load"),
            Component(id="IOUT", type="output", pins={"in": "drive"}, label="Iout"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _bioelectric_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_bioelectric_{base}",
        components=[
            Component(id="E_PLUS", type="input", pins={"out": "skin_plus"}, label="E+"),
            Component(id="E_MINUS", type="input", pins={"out": "skin_minus"}, label="E-"),
            Component(id="R5", type="resistor", pins={"a": "skin_plus", "b": "vin_plus"}, label="R5"),
            Component(id="R6", type="resistor", pins={"a": "skin_minus", "b": "vin_minus"}, label="R6"),
            Component(id="U1", type="op_amp", pins={"+": "vin_plus", "-": "vin_minus", "out": "vout"}, label="U1"),
            Component(id="RF", type="resistor", pins={"a": "vout", "b": "vin_minus"}, label="Rf", role="feedback"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="Vout"),
        ],
    )


def _non_inverting_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_noninv_{base}",
        motif="non_inverting_op_amp",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "fb", "out": "vout"}, label="U1"),
            Component(id="RF", type="resistor", pins={"a": "vout", "b": "fb"}, label="Rf", role="feedback"),
            Component(id="RG", type="resistor", pins={"a": "fb", "b": "gnd"}, label="Rg"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="OUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _rc_fixture(base: str) -> Circuit:
    return Circuit(
        id=f"batch_rc_{base}",
        motif="rc_low_pass",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label="R"),
            Component(id="C1", type="capacitor", pins={"a": "vout", "b": "gnd"}, label="C"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="OUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _summarize(results: list[dict[str, Any]], *, elapsed_seconds: float) -> dict[str, Any]:
    render_results = [result for result in results if result["render_attempted"]]
    hard_failures = [result for result in render_results if result.get("hard_fail")]
    errors = [result for result in render_results if result.get("render_ok") is False]
    quality_failures = [result for result in render_results if result.get("booklike_pass") is False]
    violation_counts = Counter(
        code
        for result in render_results
        for code in result.get("violation_codes", [])
        if code
    )
    blocking_counts = Counter(
        code
        for result in render_results
        for code in result.get("quality_blocking_codes", [])
        if code
    )
    by_family: dict[str, dict[str, int]] = {}
    for result in results:
        families = result["visual_families"] or ["generic"]
        for family in families:
            bucket = by_family.setdefault(
                family,
                {"cards": 0, "render_attempts": 0, "hard_failures": 0, "quality_failures": 0, "errors": 0},
            )
            bucket["cards"] += 1
            if result["render_attempted"]:
                bucket["render_attempts"] += 1
            if result.get("hard_fail"):
                bucket["hard_failures"] += 1
            if result.get("booklike_pass") is False:
                bucket["quality_failures"] += 1
            if result.get("render_ok") is False:
                bucket["errors"] += 1
    worst_quality = sorted(
        quality_failures,
        key=lambda result: (float(result.get("score") or 0), len(result.get("quality_blocking_codes", []))),
        reverse=True,
    )
    return {
        "total_cards": len(results),
        "likely_circuit_cards": sum(1 for result in results if result["is_likely_circuit"]),
        "image_passes": sum(1 for result in results if result["image_ok"]),
        "structured_text_passes": sum(1 for result in results if result["structured_text_ok"]),
        "render_attempts": len(render_results),
        "render_successes": sum(1 for result in render_results if result["render_ok"]),
        "render_hard_failures": len(hard_failures),
        "render_errors": len(errors),
        "booklike_score_limit": BOOKLIKE_SCORE_LIMIT,
        "booklike_passes": sum(1 for result in render_results if result.get("booklike_pass") is True),
        "booklike_failures": len(quality_failures),
        "score_gt_0": sum(1 for result in render_results if float(result.get("score") or 0) > 0),
        "score_gt_5": sum(1 for result in render_results if float(result.get("score") or 0) > 5),
        "score_gt_10": sum(1 for result in render_results if float(result.get("score") or 0) > 10),
        "violation_counts": dict(sorted(violation_counts.items())),
        "quality_blocking_counts": dict(sorted(blocking_counts.items())),
        "planning_hints_used": sum(1 for result in render_results if result.get("planning_hints_source") == "textbook_surrogate"),
        "middle_layers_generated": sum(1 for result in render_results if result.get("middle_layer_generated")),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "hard_failure_examples": [
            {
                "figure_id": result["figure_id"],
                "page": result["page"],
                "fixture": result["fixture"],
                "score": result["score"],
                "violations": result["violation_codes"][:8],
            }
            for result in hard_failures[:20]
        ],
        "quality_failure_examples": [
            {
                "figure_id": result["figure_id"],
                "page": result["page"],
                "fixture": result["fixture"],
                "score": result["score"],
                "quality_reasons": result["quality_reasons"][:8],
                "blocking_codes": result["quality_blocking_codes"][:8],
                "artifact_png": result.get("artifact_png"),
            }
            for result in worst_quality[:30]
        ],
        "error_examples": [
            {"figure_id": result["figure_id"], "page": result["page"], "fixture": result["fixture"], "error": result["error"]}
            for result in errors[:20]
        ],
        "by_family": dict(sorted(by_family.items())),
    }


def _image_check(corpus_dir: Path, card: FigureStructureCard) -> dict[str, Any]:
    if not card.image:
        return {"image_ok": False, "image_error": "missing_image_path"}
    path = corpus_dir / card.image
    if not path.exists():
        return {"image_ok": False, "image_error": "missing_file"}
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        return {"image_ok": False, "image_error": f"{type(exc).__name__}: {exc}"}
    return {"image_ok": width > 0 and height > 0, "image_width": width, "image_height": height}


def _structured_text_ok(card: FigureStructureCard) -> bool:
    text = card.structured_text
    return (
        f"FIGURE_CARD {card.figure_id}" in text
        and "visual_families:" in text
        and "layout_principles:" in text
        and "route_principles:" in text
    )


def _should_render(card: FigureStructureCard) -> bool:
    return card.is_likely_circuit or "schematic_circuit" in card.categories


def _middle_layer_text(circuit: Circuit, card: FigureStructureCard, style_guide: dict[str, Any]) -> str:
    payload = [
        "GEMINI_MIDDLE_LAYER mode=batch_single_card",
        f"circuit_id: {circuit.id}",
        "style_guide:",
        json.dumps(style_guide, indent=2, sort_keys=True),
        "retrieved_figure_cards:",
        card.structured_text,
        "planner_instruction: emit only legal OptCPV SchematicLayoutHints using existing component ids and nets.",
    ]
    return "\n\n".join(payload)


def _quality_assessment(report: dict[str, Any]) -> dict[str, Any]:
    score = float(report.get("score") or 0.0)
    hard_fail = bool(report.get("hard_fail", False))
    codes = [str(violation.get("code", "")) for violation in report.get("violations", []) if violation.get("code")]
    blocking = sorted({code for code in codes if code in BOOKLIKE_BLOCKING_CODES})
    reasons = []
    if hard_fail:
        reasons.append("hard_fail")
    if score > BOOKLIKE_SCORE_LIMIT:
        reasons.append(f"score>{BOOKLIKE_SCORE_LIMIT}")
    reasons.extend(blocking)
    booklike_pass = not hard_fail and score <= BOOKLIKE_SCORE_LIMIT and not blocking
    return {
        "booklike_pass": booklike_pass,
        "quality_reasons": reasons,
        "quality_blocking_codes": blocking,
    }


def _write_artifact_files(artifact_dir: Path, card: FigureStructureCard, artifact) -> dict[str, Any]:
    stem = _artifact_stem(card)
    svg_path = artifact_dir / f"{stem}.svg"
    png_path = artifact_dir / f"{stem}.png"
    meta_path = artifact_dir / f"{stem}.json"
    svg_path.write_text(artifact.svg, encoding="utf-8")
    raster = rasterize_svg(artifact.svg, output_width=1200, output_height=_display_raster_height(artifact))
    Image.fromarray(raster.rgba).save(png_path)
    _write_json(
        meta_path,
        {
            "figure_id": card.figure_id,
            "page": card.page,
            "fixture_source_image": card.image,
            "critic_report": artifact.critic_report,
            "vector_report": artifact.vector_report,
            "cv_report": artifact.cv_report,
            "layout_support": artifact.layout_support,
            "semantic_plan": artifact.semantic_plan,
            "optimization_log": artifact.optimization_log,
            "warnings": artifact.warnings,
        },
    )
    return {
        "artifact_svg": str(svg_path),
        "artifact_png": str(png_path),
        "artifact_meta": str(meta_path),
    }


def _write_worst_contact_sheet(results: list[dict[str, Any]], path: Path, *, max_items: int) -> None:
    candidates = [
        result
        for result in results
        if result.get("artifact_png") and result.get("render_ok") and result.get("booklike_pass") is False
    ]
    if not candidates:
        candidates = [
            result
            for result in results
            if result.get("artifact_png") and result.get("render_ok")
        ]
    candidates.sort(
        key=lambda result: (float(result.get("score") or 0), len(result.get("quality_blocking_codes", []))),
        reverse=True,
    )
    selected = candidates[:max_items]
    if not selected:
        return
    tile_w, tile_h, caption_h = 320, 214, 66
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
            with Image.open(result["artifact_png"]) as raw:
                image = raw.convert("RGB")
        except Exception:
            image = Image.new("RGB", (tile_w, tile_h), "white")
        image.thumbnail((tile_w - 14, tile_h - 14))
        ix = x0 + (tile_w - image.width) // 2
        iy = y0 + (tile_h - image.height) // 2
        sheet.paste(image, (ix, iy))
        caption = _contact_caption(result)
        draw.rectangle([x0, y0 + tile_h, x0 + tile_w - 1, y0 + tile_h + caption_h - 1], outline="#d0d0d0")
        draw.multiline_text((x0 + 8, y0 + tile_h + 6), caption, fill="black", spacing=3)
    sheet.save(path)


def _display_raster_height(artifact) -> int:
    try:
        width = float(artifact.viewbox.get("width") or 1200)
        height = float(artifact.viewbox.get("height") or 800)
    except Exception:
        return 800
    if width <= 0 or height <= 0:
        return 800
    return max(300, min(900, int(round(1200 * height / width))))


def _contact_caption(result: dict[str, Any]) -> str:
    codes = ",".join(result.get("quality_blocking_codes", [])[:3]) or ",".join(result.get("violation_codes", [])[:3])
    figure = result.get("figure_id") or "unknown"
    fixture = result.get("fixture") or "unknown"
    return _truncate(f"{figure} p{result.get('page')} {fixture}\nscore={result.get('score')} {codes}", 72)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_middle_layer(middle_dir: Path, card: FigureStructureCard, text: str) -> None:
    if not text:
        return
    (middle_dir / f"{_safe_id(card.figure_id)}_p{card.page}.txt").write_text(text, encoding="utf-8")


def _artifact_stem(card: FigureStructureCard) -> str:
    return f"{_safe_id(card.figure_id)}_p{card.page}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "unknown"


if __name__ == "__main__":
    main()

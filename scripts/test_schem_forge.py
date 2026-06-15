#!/usr/bin/env python3
"""Local smoke test for schem_forge instrumentation-amplifier planning."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.schem_forge import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.planner import plan_instrumentation_amplifier
from backend.app.schem_forge.renderer import render_layout
from backend.app.schem_forge.verifier import verify_equivalence


OUTPUT_DIR = REPO_ROOT / "backend" / "app" / "schem_forge" / "generated"


def build_instrumentation_amplifier_ir() -> dict:
    return {
        "id": "instrumentation_amp_fixture",
        "motif": "instrumentation_amplifier",
        "components": [
            {
                "id": "VINP",
                "type": "input",
                "role": "sensor",
                "display_label": "VIN+",
                "pins": {"out": "VINP"},
            },
            {
                "id": "VINN",
                "type": "input",
                "role": "sensor",
                "display_label": "VIN-",
                "pins": {"out": "VINN"},
            },
            {
                "id": "U1",
                "type": "op_amp",
                "role": "input_buffer_opamp",
                "display_label": "U1",
                "pins": {"+": "VINP", "-": "N_GAIN_TOP", "out": "BUF_TOP"},
            },
            {
                "id": "U2",
                "type": "op_amp",
                "role": "input_buffer_opamp",
                "display_label": "U2",
                "pins": {"+": "VINN", "-": "N_GAIN_BOTTOM", "out": "BUF_BOTTOM"},
            },
            {
                "id": "U3",
                "type": "op_amp",
                "role": "differential_stage_opamp",
                "display_label": "U3",
                "pins": {"-": "DIFF_NEG", "+": "DIFF_POS", "out": "VOUT"},
            },
            {
                "id": "RF1",
                "type": "resistor",
                "role": "feedback_resistor",
                "value_label": "RF1",
                "pins": {"a": "N_GAIN_TOP", "b": "BUF_TOP"},
            },
            {
                "id": "RF2",
                "type": "resistor",
                "role": "feedback_resistor",
                "value_label": "RF2",
                "pins": {"a": "N_GAIN_BOTTOM", "b": "BUF_BOTTOM"},
            },
            {
                "id": "RG",
                "type": "resistor",
                "role": "gain_resistor",
                "value_label": "RG",
                "pins": {"a": "N_GAIN_TOP", "b": "N_GAIN_BOTTOM"},
            },
            {
                "id": "R1",
                "type": "resistor",
                "role": "diff_input_resistor",
                "value_label": "R1",
                "pins": {"a": "BUF_TOP", "b": "DIFF_NEG"},
            },
            {
                "id": "R2",
                "type": "resistor",
                "role": "diff_input_resistor",
                "value_label": "R2",
                "pins": {"a": "BUF_BOTTOM", "b": "DIFF_POS"},
            },
            {
                "id": "RF3",
                "type": "resistor",
                "role": "feedback_resistor",
                "value_label": "RF3",
                "pins": {"a": "DIFF_NEG", "b": "VOUT"},
            },
            {
                "id": "RREF",
                "type": "resistor",
                "role": "ground_resistor",
                "value_label": "RREF",
                "pins": {"a": "DIFF_POS", "b": "GND"},
            },
            {
                "id": "VOUT",
                "type": "output",
                "role": "output",
                "display_label": "VOUT",
                "pins": {"in": "VOUT"},
            },
            {
                "id": "GND",
                "type": "ground",
                "role": "ground",
                "display_label": "GND",
                "pins": {"gnd": "GND"},
            },
        ],
    }


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    circuit_ir = build_instrumentation_amplifier_ir()

    before_plan = plan_instrumentation_amplifier(circuit_ir)
    verify_equivalence(circuit_ir, before_plan)
    before_render = render_layout(before_plan)
    before_report = critique_layout(before_plan, before_render)

    write_text(OUTPUT_DIR / "before.svg", before_render.svg)
    write_json(OUTPUT_DIR / "before_plan.json", before_plan.to_dict())

    print("Before critic score:", before_report.total_score)
    if before_report.violations:
        print("Before violations:")
        for violation in before_report.violations:
            print(f"- {violation.code}: {violation.message} (+{violation.penalty})")
    else:
        print("Before violations: none")

    result = generate_beautiful_schematic(
        circuit_ir,
        max_iterations=5,
        llm_client=MockLLMClient(),
    )
    verify_equivalence(circuit_ir, result.layout)

    write_text(OUTPUT_DIR / "after.svg", result.svg)
    write_json(OUTPUT_DIR / "after_plan.json", result.layout.to_dict())
    write_json(
        OUTPUT_DIR / "critic_report.json",
        {
            "before": before_report.to_dict(),
            "after": result.critic_report.to_dict(),
        },
    )

    print("After critic score:", result.critic_report.total_score)
    print("Agent iterations:", result.iterations)
    print("Output directory:", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

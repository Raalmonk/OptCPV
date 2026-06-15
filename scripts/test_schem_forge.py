#!/usr/bin/env python3
"""Export local schem_forge SVG fixtures and critic reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.schem_forge import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.examples import EXAMPLE_CASES, instrumentation_amp_ir
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from backend.app.schem_forge.verifier import verify_equivalence


OUTPUT_ROOT = REPO_ROOT / "backend" / "app" / "schem_forge" / "generated"


def build_instrumentation_amplifier_ir() -> dict:
    """Backward-compatible fixture hook for older local imports."""

    return instrumentation_amp_ir()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_case(case_name: str) -> dict:
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_ir = EXAMPLE_CASES[case_name]()

    before_plan = plan_circuit(circuit_ir)
    verify_equivalence(circuit_ir, before_plan)
    before_render = render_layout(before_plan)
    before_report = critique_layout(before_plan, before_render)

    result = generate_beautiful_schematic(
        circuit_ir,
        max_iterations=5,
        llm_client=MockLLMClient(),
    )
    verify_equivalence(circuit_ir, result.layout)

    write_text(output_dir / "before.svg", before_render.svg)
    write_text(output_dir / "after.svg", result.svg)
    write_json(output_dir / "before_plan.json", before_plan.to_dict())
    write_json(output_dir / "after_plan.json", result.layout.to_dict())
    write_json(
        output_dir / "critic_report.json",
        {
            "before": before_report.to_dict(),
            "after": result.critic_report.to_dict(),
            "agent_debug_log": result.debug_log,
        },
    )

    return {
        "case": case_name,
        "before": before_report.total_score,
        "after": result.critic_report.total_score,
        "fatal_after": result.critic_report.fatal_count,
        "improved": "yes" if result.critic_report.total_score < before_report.total_score else "no",
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(EXAMPLE_CASES),
        default=None,
        help="Single fixture case to export.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all fixture cases.",
    )
    return parser.parse_args()


def print_summary(rows: list[dict]) -> None:
    print(f"{'case':<24} {'before':>8} {'after':>8} {'fatal_after':>12} {'improved':>10}")
    for row in rows:
        print(
            f"{row['case']:<24} {row['before']:>8} {row['after']:>8} "
            f"{row['fatal_after']:>12} {row['improved']:>10}"
        )
    print("Output root:", OUTPUT_ROOT)


def main() -> int:
    args = parse_args()
    if args.all:
        case_names = sorted(EXAMPLE_CASES)
    else:
        case_names = [args.case or "instrumentation_amp"]

    rows = [run_case(case_name) for case_name in case_names]
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

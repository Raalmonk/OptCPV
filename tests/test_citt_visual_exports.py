from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from backend.app.schem_forge.adapters import circuit_problem_to_schem_forge_ir
from backend.app.schem_forge.agent import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.citt_examples import CITT_EXAMPLE_CASES
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from backend.app.schem_forge.verifier import verify_equivalence
from scripts.test_schem_forge import CITT_QA_THRESHOLDS, artifact_contract_errors, run_case


EXPECTED_MOTIFS = {
    "citt_bme_instrumentation_amplifier": "instrumentation_amplifier",
    "citt_non_inverting_op_amp": "non_inverting_op_amp",
    "citt_rc_low_pass": "rc_low_pass",
    "citt_voltage_divider": "voltage_divider",
}

FORBIDDEN_VISUAL_CODES = {
    "component_overlap",
    "wire_crosses_component_body",
    "wire_crossing",
}


@pytest.fixture(params=sorted(CITT_EXAMPLE_CASES))
def citt_case(request) -> tuple[str, dict, dict]:
    case_name = request.param
    payload = CITT_EXAMPLE_CASES[case_name]()
    original = copy.deepcopy(payload)
    ir = circuit_problem_to_schem_forge_ir(payload)

    assert payload == original
    return case_name, payload, ir


def test_citt_payload_adapter_preserves_motif_pins_and_virtual_markers(citt_case) -> None:
    case_name, payload, ir = citt_case

    assert ir["id"] == payload["id"]
    assert ir["motif"] == EXPECTED_MOTIFS[case_name]
    assert any(
        component["type"] == "ground" and component.get("source") == "virtual_terminal"
        for component in ir["components"]
    )
    assert any(
        component["type"] == "output" and component.get("source") == "virtual_terminal"
        for component in ir["components"]
    )
    assert any(
        component["type"] == "input" and component.get("source") == "virtual_terminal"
        for component in ir["components"]
    )

    opamps = [component for component in ir["components"] if "op_amp" in component["type"]]
    for opamp in opamps:
        assert {"+", "-", "out"} <= set(opamp["pins"])
    if case_name == "citt_non_inverting_op_amp":
        opamp = next(component for component in ir["components"] if component["id"] == "U1")
        assert opamp["pins"] == {"+": "VIN", "-": "NFB", "out": "VOUT", "ref": "GND"}


def test_citt_cases_plan_render_and_pass_visual_gates(citt_case) -> None:
    case_name, _payload, ir = citt_case
    plan = plan_circuit(ir)
    before_report = critique_layout(plan, render_layout(plan))
    result = generate_beautiful_schematic(ir, llm_client=MockLLMClient())
    after_report = result.critic_report
    codes = {violation.code for violation in after_report.violations}

    assert verify_equivalence(ir, plan) is True
    assert verify_equivalence(ir, result.layout) is True
    assert f"motif: {EXPECTED_MOTIFS[case_name]}" in plan.warnings
    assert not any("grid_fallback" in warning for warning in plan.warnings)
    assert ET.fromstring(result.svg).tag.endswith("svg")
    assert after_report.fatal_count == 0
    assert after_report.total_score <= before_report.total_score
    assert after_report.total_score <= CITT_QA_THRESHOLDS[case_name]
    assert not (codes & FORBIDDEN_VISUAL_CODES)
    assert result.artifact.components
    assert result.artifact.nets
    assert result.artifact.focus_regions
    assert result.artifact.zoom_presets
    assert result.artifact.hit_targets
    assert artifact_contract_errors(result.artifact) == []


@pytest.mark.parametrize("case_name", sorted(CITT_EXAMPLE_CASES))
def test_citt_export_case_writes_review_artifacts(case_name: str) -> None:
    row = run_case(case_name, suite="citt")

    assert row["suite"] == "citt"
    assert row["artifact_ok"] == "yes"
    assert row["fatal_after"] == 0
    assert row["threshold_ok"] is True
    assert row["violations_after"] == []
    assert Path(row["source_payload"]).exists()
    assert Path(row["adapted_ir"]).exists()
    assert Path(row["after_artifact"]).exists()
    assert Path(row["debug_html"]).exists()

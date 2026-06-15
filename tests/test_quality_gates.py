from __future__ import annotations

import pytest

from backend.app.schem_forge.agent import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.examples import EXAMPLE_CASES
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout


THRESHOLDS = {
    "voltage_divider": 100,
    "rc_low_pass": 150,
    "non_inverting_op_amp": 200,
    "instrumentation_amp": 300,
}


@pytest.mark.parametrize("case_name", sorted(EXAMPLE_CASES))
def test_builtin_visual_quality_gates(case_name: str) -> None:
    circuit_ir = EXAMPLE_CASES[case_name]()
    before_plan = plan_circuit(circuit_ir)
    before_report = critique_layout(before_plan, render_layout(before_plan))
    result = generate_beautiful_schematic(circuit_ir, llm_client=MockLLMClient())
    after_report = result.critic_report
    codes = [violation.code for violation in after_report.violations]

    assert after_report.fatal_count == 0
    assert after_report.total_score <= THRESHOLDS[case_name]
    assert after_report.total_score <= before_report.total_score
    assert "component_overlap" not in codes
    assert "wire_crosses_component_body" not in codes

    if case_name == "instrumentation_amp":
        # TODO: replace the remaining input/gain-node crossing with a jumper or cleaner bus route.
        assert codes == ["wire_crossing"]
    else:
        assert "wire_crossing" not in codes

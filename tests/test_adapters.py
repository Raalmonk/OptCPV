from __future__ import annotations

from backend.app.schem_forge.adapters import circuit_problem_to_schem_forge_ir
from backend.app.schem_forge.agent import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.verifier import verify_equivalence


def _assert_adapter_payload_plans(payload: dict) -> dict:
    ir = circuit_problem_to_schem_forge_ir(payload)
    plan = plan_circuit(ir)

    assert verify_equivalence(ir, plan) is True
    result = generate_beautiful_schematic(ir, llm_client=MockLLMClient())
    assert result.svg.startswith("<svg")
    assert verify_equivalence(ir, result.layout) is True
    return ir


def test_adapter_voltage_divider_dict() -> None:
    ir = _assert_adapter_payload_plans(
        {
            "id": "citt_divider",
            "components": [
                {"id": "VS", "type": "voltage_source", "nodes": ["VIN", "GND"], "value": "5 V"},
                {"id": "R1", "type": "resistor", "nodes": ["VIN", "VOUT"], "value": "10k"},
                {"id": "R2", "type": "resistor", "nodes": ["VOUT", "GND"], "value": "10k"},
            ],
            "goals": [{"type": "node_voltage", "node": "VOUT"}],
        }
    )

    assert any(component["type"] == "ground" for component in ir["components"])
    assert any(component["type"] == "output" for component in ir["components"])


def test_adapter_rc_low_pass_dict() -> None:
    _assert_adapter_payload_plans(
        {
            "id": "citt_rc",
            "components": [
                {"id": "VS", "type": "voltage_source", "nodes": ["VIN", "GND"]},
                {"id": "R1", "type": "resistor", "nodes": ["VIN", "VOUT"], "role": "series_resistor"},
                {"id": "C1", "type": "capacitor", "nodes": ["VOUT", "GND"], "role": "shunt_capacitor"},
            ],
            "goals": [{"type": "node_voltage", "node": "VOUT"}],
        }
    )


def test_adapter_non_inverting_opamp_dict() -> None:
    _assert_adapter_payload_plans(
        {
            "id": "citt_noninv",
            "motif": "non_inverting_op_amp",
            "components": [
                {"id": "VS", "type": "voltage_source", "nodes": ["VIN", "GND"]},
                {"id": "U1", "type": "ideal_op_amp", "nodes": ["VIN", "NFB", "VOUT", "GND"]},
                {"id": "RF", "type": "resistor", "role": "feedback_resistor", "nodes": ["NFB", "VOUT"]},
                {"id": "RG", "type": "resistor", "role": "gain_resistor", "nodes": ["NFB", "GND"]},
            ],
            "goals": [{"type": "node_voltage", "node": "VOUT"}],
        }
    )


def test_adapter_instrumentation_amplifier_like_payload() -> None:
    _assert_adapter_payload_plans(
        {
            "id": "citt_inamp",
            "motif": "instrumentation_amplifier",
            "components": [
                {"id": "VP", "type": "voltage_source", "nodes": ["VINP", "GND"]},
                {"id": "VN", "type": "voltage_source", "nodes": ["VINN", "GND"]},
                {"id": "U1", "type": "ideal_op_amp", "role": "input_buffer_opamp", "nodes": ["VINP", "N_GAIN_TOP", "BUF_TOP", "GND"]},
                {"id": "U2", "type": "ideal_op_amp", "role": "input_buffer_opamp", "nodes": ["VINN", "N_GAIN_BOTTOM", "BUF_BOTTOM", "GND"]},
                {"id": "U3", "type": "ideal_op_amp", "role": "differential_stage_opamp", "nodes": ["DIFF_POS", "DIFF_NEG", "VOUT", "GND"]},
                {"id": "RF1", "type": "resistor", "role": "feedback_resistor", "nodes": ["N_GAIN_TOP", "BUF_TOP"]},
                {"id": "RF2", "type": "resistor", "role": "feedback_resistor", "nodes": ["N_GAIN_BOTTOM", "BUF_BOTTOM"]},
                {"id": "RG", "type": "resistor", "role": "gain_resistor", "nodes": ["N_GAIN_TOP", "N_GAIN_BOTTOM"]},
                {"id": "R1", "type": "resistor", "role": "diff_input_resistor", "nodes": ["BUF_TOP", "DIFF_NEG"]},
                {"id": "R2", "type": "resistor", "role": "diff_input_resistor", "nodes": ["BUF_BOTTOM", "DIFF_POS"]},
                {"id": "RF3", "type": "resistor", "role": "feedback_resistor", "nodes": ["DIFF_NEG", "VOUT"]},
                {"id": "RREF", "type": "resistor", "role": "ground_resistor", "nodes": ["DIFF_POS", "GND"]},
            ],
            "goals": [{"type": "node_voltage", "node": "VOUT"}],
        }
    )

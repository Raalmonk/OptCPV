from __future__ import annotations

import copy

from backend.app.schem_forge.adapters import circuit_problem_to_schem_forge_ir
from backend.app.schem_forge.agent import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.verifier import verify_equivalence


def _assert_adapter_payload_plans(payload: dict) -> dict:
    original = copy.deepcopy(payload)
    ir = circuit_problem_to_schem_forge_ir(payload)
    plan = plan_circuit(ir)

    assert payload == original
    assert verify_equivalence(ir, plan) is True
    result = generate_beautiful_schematic(ir, llm_client=MockLLMClient())
    assert result.svg.startswith("<svg")
    assert verify_equivalence(ir, result.layout) is True
    assert result.artifact.focus_regions
    assert result.artifact.zoom_presets
    assert result.artifact.hit_targets
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


def _assert_realistic_payload(payload: dict, expected_motif: str, expected_output_net: str) -> dict:
    ir = _assert_adapter_payload_plans(payload)

    assert ir["motif"] == expected_motif
    assert any(component["type"] == "ground" and component["pins"]["gnd"] == payload["ground_node"] for component in ir["components"])
    assert any(component["type"] == "input" for component in ir["components"])
    assert any(
        component["type"] == "output" and expected_output_net in component["pins"].values()
        for component in ir["components"]
    )
    assert all(component["id"] for component in ir["components"])
    return ir


def test_adapter_realistic_citt_voltage_divider_payload() -> None:
    _assert_realistic_payload(
        {
            "id": "real_divider",
            "topology_id": "voltage_divider",
            "analysis_type": "dc",
            "ground_node": "0",
            "nodes": ["VIN", "VOUT", "0"],
            "components": [
                {"id": "V1", "type": "voltage_source", "nodes": ["VIN", "0"], "value": 5, "unit": "V", "label": "VIN"},
                {"id": "Rtop", "type": "resistor", "nodes": ["VIN", "VOUT"], "value": 10, "unit": "kOhm", "label": "Rtop"},
                {"id": "Rbot", "type": "resistor", "nodes": ["VOUT", "0"], "value": 10, "unit": "kOhm", "label": "Rbot"},
            ],
            "goals": [
                {"id": "goal_vout", "quantity": "node_voltage", "target": "VOUT", "reference": "0"},
                {
                    "id": "goal_vrbot",
                    "quantity": "component_voltage",
                    "target": "Rbot",
                    "reference": {"positive_node": "VOUT", "negative_node": "0"},
                },
            ],
        },
        "voltage_divider",
        "VOUT",
    )


def test_adapter_realistic_citt_rc_low_pass_payload() -> None:
    _assert_realistic_payload(
        {
            "id": "real_rc_low_pass",
            "topology_id": "rc_low_pass",
            "analysis_type": "ac",
            "ground_node": "GND",
            "nodes": ["VIN", "VOUT", "GND"],
            "components": [
                {"id": "V1", "type": "voltage_source", "nodes": ["VIN", "GND"], "value": 1, "unit": "V", "label": "VIN"},
                {"id": "R1", "type": "resistor", "nodes": ["VIN", "VOUT"], "value": 1, "unit": "kOhm", "label": "R"},
                {"id": "C1", "type": "capacitor", "nodes": ["VOUT", "GND"], "value": 100, "unit": "nF", "label": "C"},
            ],
            "goals": [{"id": "goal_vout", "quantity": "node_voltage", "target": "VOUT", "reference": "GND"}],
        },
        "rc_low_pass",
        "VOUT",
    )


def test_adapter_realistic_citt_non_inverting_opamp_payload() -> None:
    ir = _assert_realistic_payload(
        {
            "id": "real_noninv",
            "topology_id": "non_inverting_op_amp",
            "analysis_type": "dc",
            "ground_node": "GND",
            "nodes": ["VIN", "NFB", "VOUT", "GND"],
            "components": [
                {"id": "V1", "type": "voltage_source", "nodes": ["VIN", "GND"], "value": 1, "unit": "V", "label": "VIN"},
                {"id": "U1", "type": "ideal_op_amp", "nodes": ["VIN", "NFB", "VOUT", "GND"], "label": "U1"},
                {"id": "RF", "type": "resistor", "role": "feedback_resistor", "nodes": ["NFB", "VOUT"], "value": 10, "unit": "kOhm"},
                {"id": "RG", "type": "resistor", "role": "gain_resistor", "nodes": ["NFB", "GND"], "value": 1, "unit": "kOhm"},
            ],
            "goals": [{"id": "goal_vout", "quantity": "node_voltage", "target": "VOUT", "reference": "GND"}],
        },
        "non_inverting_op_amp",
        "VOUT",
    )

    opamp = next(component for component in ir["components"] if component["id"] == "U1")
    assert opamp["pins"] == {"+": "VIN", "-": "NFB", "out": "VOUT", "ref": "GND"}


def test_adapter_realistic_bme_instrumentation_amplifier_payload() -> None:
    _assert_realistic_payload(
        {
            "id": "bme_inamp",
            "topology_id": "instrumentation_amplifier",
            "analysis_type": "dc",
            "ground_node": "GND",
            "nodes": [
                "VINP",
                "VINN",
                "N_GAIN_TOP",
                "N_GAIN_BOTTOM",
                "BUF_TOP",
                "BUF_BOTTOM",
                "DIFF_NEG",
                "DIFF_POS",
                "VOUT",
                "GND",
            ],
            "components": [
                {"id": "VP", "type": "voltage_source", "nodes": ["VINP", "GND"], "value": 1.65, "unit": "V", "label": "VIN+"},
                {"id": "VN", "type": "voltage_source", "nodes": ["VINN", "GND"], "value": 1.60, "unit": "V", "label": "VIN-"},
                {"id": "U1", "type": "ideal_op_amp", "role": "input_buffer_opamp", "nodes": ["VINP", "N_GAIN_TOP", "BUF_TOP", "GND"]},
                {"id": "U2", "type": "ideal_op_amp", "role": "input_buffer_opamp", "nodes": ["VINN", "N_GAIN_BOTTOM", "BUF_BOTTOM", "GND"]},
                {"id": "U3", "type": "ideal_op_amp", "role": "differential_stage_opamp", "nodes": ["DIFF_POS", "DIFF_NEG", "VOUT", "GND"]},
                {"id": "RF1", "type": "resistor", "role": "feedback_resistor", "nodes": ["N_GAIN_TOP", "BUF_TOP"], "value": 10, "unit": "kOhm"},
                {"id": "RF2", "type": "resistor", "role": "feedback_resistor", "nodes": ["N_GAIN_BOTTOM", "BUF_BOTTOM"], "value": 10, "unit": "kOhm"},
                {"id": "RG", "type": "resistor", "role": "gain_resistor", "nodes": ["N_GAIN_TOP", "N_GAIN_BOTTOM"], "value": 499, "unit": "Ohm"},
                {"id": "R1", "type": "resistor", "role": "diff_input_resistor", "nodes": ["BUF_TOP", "DIFF_NEG"], "value": 10, "unit": "kOhm"},
                {"id": "R2", "type": "resistor", "role": "diff_input_resistor", "nodes": ["BUF_BOTTOM", "DIFF_POS"], "value": 10, "unit": "kOhm"},
                {"id": "RF3", "type": "resistor", "role": "feedback_resistor", "nodes": ["DIFF_NEG", "VOUT"], "value": 10, "unit": "kOhm"},
                {"id": "RREF", "type": "resistor", "role": "ground_resistor", "nodes": ["DIFF_POS", "GND"], "value": 10, "unit": "kOhm"},
            ],
            "goals": [{"id": "goal_vout", "quantity": "node_voltage", "target": "VOUT", "reference": "GND"}],
        },
        "instrumentation_amplifier",
        "VOUT",
    )

"""Representative CiTT-style circuit payload fixtures for schem_forge QA."""

from __future__ import annotations

from typing import Any, Callable


def citt_voltage_divider_payload() -> dict[str, Any]:
    return {
        "id": "citt_voltage_divider",
        "title": "CiTT voltage divider",
        "topology_id": "voltage_divider",
        "analysis_type": "dc",
        "ground_node": "0",
        "nodes": ["VIN", "VOUT", "0"],
        "components": [
            {
                "id": "V1",
                "type": "voltage_source",
                "nodes": ["VIN", "0"],
                "value": 5,
                "unit": "V",
                "label": "VIN",
            },
            {
                "id": "Rtop",
                "type": "resistor",
                "nodes": ["VIN", "VOUT"],
                "value": 10,
                "unit": "kOhm",
                "label": "Rtop",
            },
            {
                "id": "Rbot",
                "type": "resistor",
                "nodes": ["VOUT", "0"],
                "value": 10,
                "unit": "kOhm",
                "label": "Rbot",
            },
        ],
        "goals": [
            {
                "id": "goal_vout",
                "quantity": "node_voltage",
                "target": "VOUT",
                "reference": {"positive_node": "VOUT", "negative_node": "0"},
            },
            {
                "id": "goal_vrbot",
                "quantity": "component_voltage",
                "target": "Rbot",
                "reference": {"positive_node": "VOUT", "negative_node": "0"},
            },
        ],
    }


def citt_rc_low_pass_payload() -> dict[str, Any]:
    return {
        "id": "citt_rc_low_pass",
        "title": "CiTT RC low-pass filter",
        "topology_id": "rc_low_pass",
        "analysis_type": "ac",
        "ground_node": "GND",
        "nodes": ["VIN", "VOUT", "GND"],
        "components": [
            {
                "id": "V1",
                "type": "voltage_source",
                "nodes": ["VIN", "GND"],
                "value": 1,
                "unit": "V",
                "label": "VIN",
            },
            {
                "id": "R1",
                "type": "resistor",
                "nodes": ["VIN", "VOUT"],
                "value": 1,
                "unit": "kOhm",
                "label": "R",
            },
            {
                "id": "C1",
                "type": "capacitor",
                "nodes": ["VOUT", "GND"],
                "value": 100,
                "unit": "nF",
                "label": "C",
            },
        ],
        "goals": [
            {
                "id": "goal_vout",
                "quantity": "node_voltage",
                "target": "VOUT",
                "reference": {"positive_node": "VOUT", "negative_node": "GND"},
            }
        ],
    }


def citt_non_inverting_op_amp_payload() -> dict[str, Any]:
    return {
        "id": "citt_non_inverting_op_amp",
        "title": "CiTT non-inverting op amp",
        "topology_id": "non_inverting_op_amp",
        "analysis_type": "dc",
        "ground_node": "GND",
        "nodes": ["VIN", "NFB", "VOUT", "GND"],
        "components": [
            {
                "id": "V1",
                "type": "voltage_source",
                "nodes": ["VIN", "GND"],
                "value": 1,
                "unit": "V",
                "label": "VIN",
            },
            {
                "id": "U1",
                "type": "ideal_op_amp",
                "nodes": ["VIN", "NFB", "VOUT", "GND"],
                "label": "U1",
            },
            {
                "id": "RF",
                "type": "resistor",
                "role": "feedback_resistor",
                "nodes": ["NFB", "VOUT"],
                "value": 10,
                "unit": "kOhm",
            },
            {
                "id": "RG",
                "type": "resistor",
                "role": "gain_resistor",
                "nodes": ["NFB", "GND"],
                "value": 1,
                "unit": "kOhm",
            },
        ],
        "goals": [
            {
                "id": "goal_vout",
                "quantity": "node_voltage",
                "target": "VOUT",
                "reference": {"positive_node": "VOUT", "negative_node": "GND"},
            }
        ],
    }


def citt_bme_instrumentation_amplifier_payload() -> dict[str, Any]:
    return {
        "id": "citt_bme_instrumentation_amplifier",
        "title": "CiTT BME instrumentation amplifier",
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
            {
                "id": "VP",
                "type": "voltage_source",
                "nodes": ["VINP", "GND"],
                "value": 1.65,
                "unit": "V",
                "label": "VIN+",
            },
            {
                "id": "VN",
                "type": "voltage_source",
                "nodes": ["VINN", "GND"],
                "value": 1.60,
                "unit": "V",
                "label": "VIN-",
            },
            {
                "id": "U1",
                "type": "ideal_op_amp",
                "role": "input_buffer_opamp",
                "nodes": ["VINP", "N_GAIN_TOP", "BUF_TOP", "GND"],
            },
            {
                "id": "U2",
                "type": "ideal_op_amp",
                "role": "input_buffer_opamp",
                "nodes": ["VINN", "N_GAIN_BOTTOM", "BUF_BOTTOM", "GND"],
            },
            {
                "id": "U3",
                "type": "ideal_op_amp",
                "role": "differential_stage_opamp",
                "nodes": ["DIFF_POS", "DIFF_NEG", "VOUT", "GND"],
            },
            {
                "id": "RF1",
                "type": "resistor",
                "role": "feedback_resistor",
                "nodes": ["N_GAIN_TOP", "BUF_TOP"],
                "value": 10,
                "unit": "kOhm",
            },
            {
                "id": "RF2",
                "type": "resistor",
                "role": "feedback_resistor",
                "nodes": ["N_GAIN_BOTTOM", "BUF_BOTTOM"],
                "value": 10,
                "unit": "kOhm",
            },
            {
                "id": "RG",
                "type": "resistor",
                "role": "gain_resistor",
                "nodes": ["N_GAIN_TOP", "N_GAIN_BOTTOM"],
                "value": 499,
                "unit": "Ohm",
            },
            {
                "id": "R1",
                "type": "resistor",
                "role": "diff_input_resistor",
                "nodes": ["BUF_TOP", "DIFF_NEG"],
                "value": 10,
                "unit": "kOhm",
            },
            {
                "id": "R2",
                "type": "resistor",
                "role": "diff_input_resistor",
                "nodes": ["BUF_BOTTOM", "DIFF_POS"],
                "value": 10,
                "unit": "kOhm",
            },
            {
                "id": "RF3",
                "type": "resistor",
                "role": "feedback_resistor",
                "nodes": ["DIFF_NEG", "VOUT"],
                "value": 10,
                "unit": "kOhm",
            },
            {
                "id": "RREF",
                "type": "resistor",
                "role": "ground_resistor",
                "nodes": ["DIFF_POS", "GND"],
                "value": 10,
                "unit": "kOhm",
            },
        ],
        "goals": [
            {
                "id": "goal_vout",
                "quantity": "node_voltage",
                "target": "VOUT",
                "reference": {"positive_node": "VOUT", "negative_node": "GND"},
            }
        ],
    }


CITT_EXAMPLE_CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "citt_bme_instrumentation_amplifier": citt_bme_instrumentation_amplifier_payload,
    "citt_non_inverting_op_amp": citt_non_inverting_op_amp_payload,
    "citt_rc_low_pass": citt_rc_low_pass_payload,
    "citt_voltage_divider": citt_voltage_divider_payload,
}

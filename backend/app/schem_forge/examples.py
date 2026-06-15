"""Small built-in circuit examples for tests and local SVG export."""

from __future__ import annotations


def instrumentation_amp_ir() -> dict:
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


def voltage_divider_ir() -> dict:
    return {
        "id": "voltage_divider_fixture",
        "motif": "voltage_divider",
        "components": [
            {
                "id": "VIN",
                "type": "input",
                "role": "input_source",
                "display_label": "VIN",
                "pins": {"out": "VIN"},
            },
            {
                "id": "R1",
                "type": "resistor",
                "role": "top_resistor",
                "value_label": "R1",
                "pins": {"a": "VIN", "b": "VOUT"},
            },
            {
                "id": "R2",
                "type": "resistor",
                "role": "bottom_resistor",
                "value_label": "R2",
                "pins": {"a": "VOUT", "b": "GND"},
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
                "pins": {"gnd": "GND"},
            },
        ],
    }


def rc_low_pass_ir() -> dict:
    return {
        "id": "rc_low_pass_fixture",
        "motif": "rc_low_pass",
        "components": [
            {
                "id": "VIN",
                "type": "input",
                "role": "input_source",
                "display_label": "VIN",
                "pins": {"out": "VIN"},
            },
            {
                "id": "R1",
                "type": "resistor",
                "role": "series_resistor",
                "value_label": "R",
                "pins": {"a": "VIN", "b": "VOUT"},
            },
            {
                "id": "C1",
                "type": "capacitor",
                "role": "shunt_capacitor",
                "value_label": "C",
                "pins": {"a": "VOUT", "b": "GND"},
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
                "pins": {"gnd": "GND"},
            },
        ],
    }


def non_inverting_op_amp_ir() -> dict:
    return {
        "id": "non_inverting_op_amp_fixture",
        "motif": "non_inverting_op_amp",
        "components": [
            {
                "id": "VIN",
                "type": "input",
                "role": "input_source",
                "display_label": "VIN",
                "pins": {"out": "VIN"},
            },
            {
                "id": "U1",
                "type": "op_amp",
                "role": "opamp",
                "display_label": "U1",
                "pins": {"+": "VIN", "-": "NFB", "out": "VOUT"},
            },
            {
                "id": "RF",
                "type": "resistor",
                "role": "feedback_resistor",
                "value_label": "RF",
                "pins": {"a": "NFB", "b": "VOUT"},
            },
            {
                "id": "RG",
                "type": "resistor",
                "role": "gain_resistor",
                "value_label": "RG",
                "pins": {"a": "NFB", "b": "GND"},
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
                "pins": {"gnd": "GND"},
            },
        ],
    }


EXAMPLE_CASES = {
    "instrumentation_amp": instrumentation_amp_ir,
    "voltage_divider": voltage_divider_ir,
    "rc_low_pass": rc_low_pass_ir,
    "non_inverting_op_amp": non_inverting_op_amp_ir,
}

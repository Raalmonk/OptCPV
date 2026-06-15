"""Small native OptCPV example circuits."""

from __future__ import annotations

from .models import Circuit, Component


def voltage_divider() -> Circuit:
    return Circuit(
        id="voltage_divider",
        motif="voltage_divider",
        title="Voltage Divider",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label="R1"),
            Component(id="R2", type="resistor", pins={"a": "vout", "b": "gnd"}, label="R2"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="VOUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def rc_low_pass() -> Circuit:
    return Circuit(
        id="rc_low_pass",
        motif="rc_low_pass",
        title="RC Low-Pass Filter",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label="R"),
            Component(id="C1", type="capacitor", pins={"a": "vout", "b": "gnd"}, label="C"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="VOUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def non_inverting_op_amp() -> Circuit:
    return Circuit(
        id="non_inverting_op_amp",
        motif="non_inverting_op_amp",
        title="Non-Inverting Op Amp",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}, label="U1"),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, label="Rf", role="feedback"),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rg", role="gain"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="VOUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def instrumentation_amplifier() -> Circuit:
    return Circuit(
        id="instrumentation_amplifier",
        motif="instrumentation_amplifier",
        title="Instrumentation Amplifier",
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
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def bridge_or_wheatstone() -> Circuit:
    return Circuit(
        id="bridge_or_wheatstone",
        motif="bridge_or_wheatstone",
        title="Bridge",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vl"}, label="R1"),
            Component(id="R2", type="resistor", pins={"a": "vl", "b": "gnd"}, label="R2"),
            Component(id="R3", type="resistor", pins={"a": "vin", "b": "vr"}, label="R3"),
            Component(id="R4", type="resistor", pins={"a": "vr", "b": "gnd"}, label="R4"),
            Component(id="VOUT", type="output", pins={"in": "vl"}, label="VOUT"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


EXAMPLES = {
    "voltage_divider": voltage_divider,
    "rc_low_pass": rc_low_pass,
    "non_inverting_op_amp": non_inverting_op_amp,
    "instrumentation_amplifier": instrumentation_amplifier,
    "bridge_or_wheatstone": bridge_or_wheatstone,
}

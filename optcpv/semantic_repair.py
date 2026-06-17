"""Semantic circuit normalization for drawable known motifs.

Schemdraw is good at drawing canonical schematics. CiTT-style payloads are
often semantically equivalent but use arbitrary ids, extra source components,
or non-canonical labels. This module builds a drawable canonical circuit for
known motifs while preserving user-facing labels where possible.
"""

from __future__ import annotations

from .models import Circuit, Component


def repair_circuit(circuit: Circuit) -> Circuit:
    """Return a drawable canonical circuit for known motifs when possible."""

    motif = _validated_motif(circuit) or _motif_key(circuit.id) or _infer_motif(circuit)
    builders = {
        "voltage_divider": _repair_voltage_divider,
        "rc_low_pass": _repair_rc_low_pass,
        "non_inverting_op_amp": _repair_non_inverting_op_amp,
        "instrumentation_amplifier": _repair_instrumentation_amplifier,
        "bridge_or_wheatstone": _repair_bridge,
    }
    builder = builders.get(motif)
    if builder is None:
        return circuit
    repaired = builder(circuit)
    return repaired or circuit


def _repair_voltage_divider(circuit: Circuit) -> Circuit | None:
    resistors = _components(circuit, "resistor")
    if len(resistors) < 2:
        return None
    top, bottom = resistors[:2]
    return _with_components(
        circuit,
        "voltage_divider",
        [
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_terminal_label(circuit, "VIN", "input")),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label=_label(top, "R1")),
            Component(id="R2", type="resistor", pins={"a": "vout", "b": "gnd"}, label=_label(bottom, "R2")),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label=_terminal_label(circuit, "VOUT", "output")),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label=_terminal_label(circuit, "GND", "ground")),
        ],
    )


def _repair_rc_low_pass(circuit: Circuit) -> Circuit | None:
    resistor = _first_component(circuit, "resistor")
    capacitor = _first_component(circuit, "capacitor")
    if resistor is None or capacitor is None:
        return None
    return _with_components(
        circuit,
        "rc_low_pass",
        [
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_terminal_label(circuit, "VIN", "input")),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label=_label(resistor, "R")),
            Component(id="C1", type="capacitor", pins={"a": "vout", "b": "gnd"}, label=_label(capacitor, "C")),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label=_terminal_label(circuit, "VOUT", "output")),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label=_terminal_label(circuit, "GND", "ground")),
        ],
    )


def _repair_non_inverting_op_amp(circuit: Circuit) -> Circuit | None:
    opamp = _first_opamp(circuit)
    resistors = _components(circuit, "resistor")
    if opamp is None or len(resistors) < 2:
        return None
    feedback = _first_role(resistors, "feedback") or resistors[0]
    gain = _first_role(resistors, "gain") or next((item for item in resistors if item != feedback), resistors[1])
    return _with_components(
        circuit,
        "non_inverting_op_amp",
        [
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_terminal_label(circuit, "VIN", "input")),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}, label=_label(opamp, "U1")),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, label=_label(feedback, "Rf"), role="feedback"),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, label=_label(gain, "Rg"), role="gain"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label=_terminal_label(circuit, "VOUT", "output")),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label=_terminal_label(circuit, "GND", "ground")),
        ],
    )


def _repair_instrumentation_amplifier(circuit: Circuit) -> Circuit | None:
    opamps = [component for component in circuit.components if _is_opamp(component)]
    resistors = _components(circuit, "resistor")
    inputs = [component for component in circuit.components if _key(component.type) == "input"]
    if len(opamps) != 3 or len(resistors) < 7:
        return None
    labels = [_label(item, fallback) for item, fallback in zip(resistors[:7], ["R1", "R2", "Rg", "R3", "R4", "R5", "R6"])]
    return _with_components(
        circuit,
        "instrumentation_amplifier",
        [
            Component(id="INP", type="input", pins={"out": "vinp"}, label=_label(inputs[0], "IN+") if inputs else "IN+"),
            Component(id="INN", type="input", pins={"out": "vinn"}, label=_label(inputs[1], "IN-") if len(inputs) > 1 else "IN-"),
            Component(id="U1", type="op_amp", pins={"+": "vinp", "-": "n1", "out": "o1"}, label=_label(opamps[0], "U1")),
            Component(id="U2", type="op_amp", pins={"+": "vinn", "-": "n2", "out": "o2"}, label=_label(opamps[1], "U2")),
            Component(id="U3", type="op_amp", pins={"+": "n3", "-": "n4", "out": "vout"}, label=_label(opamps[2], "U3")),
            Component(id="R1", type="resistor", pins={"a": "o1", "b": "n1"}, label=labels[0]),
            Component(id="R2", type="resistor", pins={"a": "o2", "b": "n2"}, label=labels[1]),
            Component(id="Rg", type="resistor", pins={"a": "n1", "b": "n2"}, label=labels[2], role="gain"),
            Component(id="R3", type="resistor", pins={"a": "o1", "b": "n3"}, label=labels[3]),
            Component(id="R4", type="resistor", pins={"a": "n3", "b": "gnd"}, label=labels[4]),
            Component(id="R5", type="resistor", pins={"a": "o2", "b": "n4"}, label=labels[5]),
            Component(id="R6", type="resistor", pins={"a": "n4", "b": "vout"}, label=labels[6]),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label=_terminal_label(circuit, "VOUT", "output")),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label=_terminal_label(circuit, "GND", "ground")),
        ],
    )


def _repair_bridge(circuit: Circuit) -> Circuit | None:
    resistors = _components(circuit, "resistor")
    if len(resistors) < 4:
        return None
    return _with_components(
        circuit,
        "bridge_or_wheatstone",
        [
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_terminal_label(circuit, "VIN", "input")),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vl"}, label=_label(resistors[0], "R1")),
            Component(id="R2", type="resistor", pins={"a": "vl", "b": "gnd"}, label=_label(resistors[1], "R2")),
            Component(id="R3", type="resistor", pins={"a": "vin", "b": "vr"}, label=_label(resistors[2], "R3")),
            Component(id="R4", type="resistor", pins={"a": "vr", "b": "gnd"}, label=_label(resistors[3], "R4")),
            Component(id="VOUT", type="output", pins={"in": "vl"}, label=_terminal_label(circuit, "VOUT", "output")),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label=_terminal_label(circuit, "GND", "ground")),
        ],
    )


def _with_components(circuit: Circuit, motif: str, components: list[Component]) -> Circuit:
    return Circuit(id=circuit.id, motif=motif, title=circuit.title, components=components)


def _components(circuit: Circuit, needle: str) -> list[Component]:
    return [component for component in circuit.components if needle in _key(component.type)]


def _first_component(circuit: Circuit, needle: str) -> Component | None:
    return next(iter(_components(circuit, needle)), None)


def _first_opamp(circuit: Circuit) -> Component | None:
    return next((component for component in circuit.components if _is_opamp(component)), None)


def _first_role(components: list[Component], role: str) -> Component | None:
    return next((component for component in components if role in _key(component.role)), None)


def _terminal_label(circuit: Circuit, fallback: str, component_type: str) -> str:
    component = next((item for item in circuit.components if _key(item.type) == component_type), None)
    return _label(component, fallback) if component else fallback


def _label(component: Component | None, fallback: str) -> str:
    if component is None:
        return fallback
    return component.label or component.value or component.id or fallback


def _is_opamp(component: Component) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _infer_motif(circuit: Circuit) -> str | None:
    opamps = sum(1 for component in circuit.components if _is_opamp(component))
    resistors = len(_components(circuit, "resistor"))
    capacitors = len(_components(circuit, "capacitor"))
    if opamps >= 4:
        return "op_amp_network"
    if opamps == 3 and resistors >= 7:
        return "instrumentation_amplifier"
    if opamps >= 2:
        return "op_amp_network"
    if opamps == 1 and resistors >= 2:
        return "non_inverting_op_amp"
    if resistors >= 1 and capacitors >= 1:
        return "rc_low_pass"
    if resistors >= 4:
        return "bridge_or_wheatstone"
    if resistors == 2:
        return "voltage_divider"
    return None


def _validated_motif(circuit: Circuit) -> str | None:
    motif = _motif_key(circuit.motif)
    if motif is None:
        return None
    opamps = sum(1 for component in circuit.components if _is_opamp(component))
    resistors = len(_components(circuit, "resistor"))
    capacitors = len(_components(circuit, "capacitor"))
    if motif in {"voltage_divider", "rc_low_pass", "bridge_or_wheatstone"} and opamps:
        return None
    if motif == "voltage_divider" and (resistors != 2 or capacitors):
        return None
    if motif == "rc_low_pass" and (resistors < 1 or capacitors < 1):
        return None
    if motif == "bridge_or_wheatstone" and resistors < 4:
        return None
    if motif == "instrumentation_amplifier" and (opamps != 3 or resistors < 7):
        return None
    if motif == "non_inverting_op_amp" and (opamps != 1 or resistors < 2):
        return None
    if motif == "op_amp_network" and opamps < 2:
        return None
    return motif


def _motif_key(value: str | None) -> str | None:
    key = _key(value)
    aliases = {
        "divider": "voltage_divider",
        "potential_divider": "voltage_divider",
        "resistive_divider": "voltage_divider",
        "voltage_divider": "voltage_divider",
        "rc_filter": "rc_low_pass",
        "rc_low_pass": "rc_low_pass",
        "rc_low_pass_filter": "rc_low_pass",
        "low_pass": "rc_low_pass",
        "low_pass_filter": "rc_low_pass",
        "lowpass": "rc_low_pass",
        "non_inverting_amplifier": "non_inverting_op_amp",
        "non_inverting_op_amp": "non_inverting_op_amp",
        "non_inverting_opamp": "non_inverting_op_amp",
        "noninverting_amplifier": "non_inverting_op_amp",
        "noninverting_op_amp": "non_inverting_op_amp",
        "noninverting_opamp": "non_inverting_op_amp",
        "noninv": "non_inverting_op_amp",
        "instrumentation_amplifier": "instrumentation_amplifier",
        "instrumentation_amp": "instrumentation_amplifier",
        "in_amp": "instrumentation_amplifier",
        "ina": "instrumentation_amplifier",
        "bridge": "bridge_or_wheatstone",
        "bridge_or_wheatstone": "bridge_or_wheatstone",
        "resistor_bridge": "bridge_or_wheatstone",
        "wheatstone": "bridge_or_wheatstone",
        "wheatstone_bridge": "bridge_or_wheatstone",
        "analog_front_end": "op_amp_network",
        "multi_op_amp": "op_amp_network",
        "multi_opamp": "op_amp_network",
        "op_amp_chain": "op_amp_network",
        "op_amp_network": "op_amp_network",
        "opamp_chain": "op_amp_network",
        "opamp_network": "op_amp_network",
    }
    return aliases.get(key)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

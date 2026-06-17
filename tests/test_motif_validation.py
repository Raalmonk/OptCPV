from optcpv import Circuit, Component, repair_circuit
from optcpv.planner import plan_layout


def test_divider_mislabeled_as_bridge_uses_divider_layout() -> None:
    circuit = Circuit(
        id="mislabeled_divider",
        motif="bridge",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="Rtop", type="resistor", pins={"a": "vin", "b": "sense"}),
            Component(id="Rbot", type="resistor", pins={"a": "sense", "b": "gnd"}),
            Component(id="VOUT", type="output", pins={"in": "sense"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )

    repaired = repair_circuit(circuit)
    layout = plan_layout(circuit)

    assert repaired.motif == "voltage_divider"
    assert "motif: voltage_divider" in layout.warnings


def test_bridge_mislabeled_as_divider_uses_bridge_layout() -> None:
    circuit = Circuit(
        id="mislabeled_bridge",
        motif="voltage_divider",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vl"}),
            Component(id="R2", type="resistor", pins={"a": "vl", "b": "gnd"}),
            Component(id="R3", type="resistor", pins={"a": "vin", "b": "vr"}),
            Component(id="R4", type="resistor", pins={"a": "vr", "b": "gnd"}),
            Component(id="VOUT", type="output", pins={"in": "vl"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )

    repaired = repair_circuit(circuit)
    layout = plan_layout(circuit)

    assert repaired.motif == "bridge_or_wheatstone"
    assert "motif: bridge_or_wheatstone" in layout.warnings


def test_single_opamp_mislabeled_as_network_uses_single_opamp_layout() -> None:
    circuit = Circuit(
        id="mislabeled_single_opamp",
        motif="op_amp_network",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, role="feedback"),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, role="gain"),
            Component(id="VOUT", type="output", pins={"in": "vout"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )

    repaired = repair_circuit(circuit)
    layout = plan_layout(circuit)

    assert repaired.motif == "non_inverting_op_amp"
    assert "motif: non_inverting_op_amp" in layout.warnings

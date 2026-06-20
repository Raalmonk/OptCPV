from optcpv.adapters.citt import from_citt_payload
from optcpv.models import Point
from optcpv.planner import plan_layout


def test_citt_adapter_stays_simple_boundary_converter() -> None:
    circuit = from_citt_payload(
        {
            "id": "divider_problem",
            "motif": "voltage_divider",
            "ground_node": "0",
            "goals": [{"output_node": "vout"}],
            "components": [
                {"id": "V1", "type": "voltage_source", "nodes": ["vin", "0"]},
                {"id": "R1", "type": "resistor", "nodes": ["vin", "vout"]},
                {"id": "R2", "type": "resistor", "nodes": ["vout", "0"]},
            ],
        }
    )

    assert circuit.id == "divider_problem"
    assert any(component.type == "input" for component in circuit.components)
    assert any(component.type == "output" for component in circuit.components)
    assert any(component.type == "ground" for component in circuit.components)


def test_citt_adapter_canonicalizes_voltage_clamp_net_names_for_known_routes() -> None:
    circuit = from_citt_payload(
        {
            "id": "tevc_lowercase",
            "motif": "two_electrode_voltage_clamp",
            "ground_node": "gnd",
            "output_node": "vm",
            "components": [
                {"id": "DiffAmp", "type": "op_amp_nonideal", "nodes": ["vc", "vm", "vo", "gnd"]},
                {"id": "R_m", "type": "resistor", "nodes": ["vm", "gnd"]},
                {"id": "R_o", "type": "resistor", "nodes": ["vo", "vm"]},
            ],
        }
    )

    assert [(component.id, component.pins) for component in circuit.components] == [
        ("VC", {"out": "Vc"}),
        ("DiffAmp", {"+": "Vc", "-": "Vm", "out": "Vo"}),
        ("R_m", {"a": "Vm", "b": "0"}),
        ("R_o", {"a": "Vo", "b": "Vm"}),
        ("VM", {"in": "Vm"}),
        ("VO", {"in": "Vo"}),
        ("GND", {"gnd": "0"}),
    ]

    layout = plan_layout(circuit)
    assert {wire.net for wire in layout.wires} == {"Vc", "Vm", "Vo"}
    vm_wire = next(wire for wire in layout.wires if wire.net == "Vm")
    minus = layout.pin_map[("DiffAmp", "-")]
    ro_bottom = layout.pin_map[("R_o", "b")]
    rm_top = layout.pin_map[("R_m", "a")]

    assert Point(ro_bottom.x, minus.y) in vm_wire.points
    assert Point(ro_bottom.x, rm_top.y) in vm_wire.points

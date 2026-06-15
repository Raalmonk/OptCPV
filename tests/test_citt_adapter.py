from optcpv.adapters.citt import from_citt_payload


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

from optcpv import draw_artifact, draw_optimized_artifact, repair_circuit
from optcpv.adapters.citt import from_citt_payload


def test_semantic_repair_normalizes_citt_voltage_divider_and_preserves_labels() -> None:
    circuit = from_citt_payload(
        {
            "id": "citt_divider_01",
            "topology": "potential_divider",
            "ground_node": "0",
            "goals": {"output_node": "sense"},
            "components": [
                {"id": "Vs", "type": "voltage_source", "nodes": ["vin", "0"], "label": "VIN"},
                {"id": "upper", "type": "resistor", "nodes": ["vin", "sense"], "label": "Rtop"},
                {"id": "lower", "type": "resistor", "nodes": ["sense", "0"], "label": "Rbot"},
            ],
        }
    )

    repaired = repair_circuit(circuit)

    assert repaired.motif == "voltage_divider"
    assert [component.id for component in repaired.components] == ["VIN", "R1", "R2", "VOUT", "GND"]
    assert repaired.components[1].label == "Rtop"
    assert repaired.components[2].label == "Rbot"


def test_optimized_artifact_accepts_semantic_repair_when_it_improves_citt_input() -> None:
    circuit = from_citt_payload(
        {
            "id": "citt_rc_01",
            "topology": "rc_filter",
            "ground_node": "gnd",
            "output_node": "vo",
            "components": [
                {"id": "source", "type": "voltage_source", "nodes": ["vi", "gnd"], "label": "VIN"},
                {"id": "series", "type": "resistor", "nodes": ["vi", "vo"], "label": "Rseries"},
                {"id": "shunt", "type": "capacitor", "nodes": ["vo", "gnd"], "label": "Cshunt"},
            ],
        }
    )

    raw = draw_artifact(circuit)
    optimized = draw_optimized_artifact(circuit, max_iterations=2)

    assert optimized.critic_report["score"] <= raw.critic_report["score"] - 0.5
    assert any(item["source"] == "semantic_repair" and item["accepted"] for item in optimized.optimization_log)
    assert optimized.components["R1"]["label"] == "Rseries"
    assert optimized.components["C1"]["label"] == "Cshunt"

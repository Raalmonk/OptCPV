from dataclasses import replace

from optcpv import Component, Circuit, draw_optimized_artifact, repair_circuit
from optcpv.planner import plan_layout


def eight_opamp_chain() -> Circuit:
    return opamp_chain(8, "eight_opamp_chain", "ina")


def opamp_chain(stages: int, circuit_id: str, motif: str) -> Circuit:
    components = [
        Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
        Component(id="VOUT", type="output", pins={"in": f"o{stages}"}, label="VOUT"),
        Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
    ]
    previous = "vin"
    for index in range(1, stages + 1):
        output = f"o{index}"
        summing = f"fb{index}"
        components.extend(
            [
                Component(
                    id=f"U{index}",
                    type="op_amp",
                    pins={"+": previous, "-": summing, "out": output},
                    label=f"U{index}",
                ),
                Component(
                    id=f"Rf{index}",
                    type="resistor",
                    pins={"a": output, "b": summing},
                    label=f"Rf{index}",
                    role="feedback",
                ),
                Component(
                    id=f"Rg{index}",
                    type="resistor",
                    pins={"a": summing, "b": "gnd"},
                    label=f"Rg{index}",
                    role="gain",
                ),
            ]
        )
        previous = output
    return Circuit(id=circuit_id, motif=motif, title=f"{stages} Op Amp Chain", components=components)


def test_multi_opamp_network_is_not_reduced_to_instrumentation_amp() -> None:
    circuit = eight_opamp_chain()
    repaired = repair_circuit(circuit)
    layout = plan_layout(repaired)

    assert len([component for component in repaired.components if component.type == "op_amp"]) == 8
    assert len(repaired.components) == len(circuit.components)
    assert "motif: op_amp_network" in layout.warnings
    assert all(component.y < layout.height / layout.grid for component in layout.components)
    assert not any(component.y == 15.0 and component.type != "ground" for component in layout.components)


def test_multi_opamp_network_optimizes_without_hard_failure() -> None:
    artifact = draw_optimized_artifact(eight_opamp_chain(), max_iterations=3)

    assert artifact.critic_report["hard_fail"] is False
    assert artifact.critic_report["score"] <= 100
    assert len([component for component in artifact.components.values() if component["type"] == "op_amp"]) == 8
    assert artifact.viewbox["width"] <= 1400
    assert artifact.viewbox["height"] <= 850


def test_mislabeled_multi_opamp_network_keeps_topology() -> None:
    for motif in ["bridge", "voltage_divider", "rc_low_pass", "noninv", "ina"]:
        circuit = replace(eight_opamp_chain(), id=f"eight_as_{motif}", motif=motif)
        repaired = repair_circuit(circuit)
        layout = plan_layout(circuit)
        artifact = draw_optimized_artifact(circuit, max_iterations=3)

        assert len(repaired.components) == len(circuit.components)
        assert len(artifact.components) == len(circuit.components)
        assert len([component for component in artifact.components.values() if component["type"] == "op_amp"]) == 8
        assert "motif: op_amp_network" in layout.warnings
        assert artifact.critic_report["hard_fail"] is False


def test_three_stage_chain_mislabeled_as_ina_uses_network_layout() -> None:
    circuit = opamp_chain(3, "three_stage_chain_as_ina", "ina")
    repaired = repair_circuit(circuit)
    layout = plan_layout(circuit)

    assert len(repaired.components) == len(circuit.components)
    assert repaired.motif == "ina"
    assert "motif: op_amp_network" in layout.warnings


def test_cross_row_monitor_output_uses_clear_label_and_route() -> None:
    circuit = opamp_chain(8, "eight_stage_with_monitor", "op_amp_network")
    circuit = replace(
        circuit,
        components=[
            *circuit.components,
            Component(id="VMON4", type="output", pins={"in": "o4"}, label="MONITOR_STAGE_4"),
        ],
    )

    artifact = draw_optimized_artifact(circuit, max_iterations=3)

    assert artifact.critic_report["hard_fail"] is False
    assert artifact.critic_report["score"] <= 10
    assert artifact.labels["label:VMON4"]["text"] == "VMON4"
    assert "wire_crossings" not in {violation["code"] for violation in artifact.critic_report["violations"]}


def test_verbose_component_labels_are_compacted_for_display() -> None:
    circuit = opamp_chain(8, "verbose_eight_stage", "bridge")
    verbose = []
    for component in circuit.components:
        label = component.label
        if component.type in {"op_amp", "resistor"}:
            label = f"{component.id}_stage_13_biomed_front_end"
        verbose.append(replace(component, label=label))
    circuit = replace(circuit, components=verbose)

    artifact = draw_optimized_artifact(circuit, max_iterations=3)

    assert artifact.critic_report["hard_fail"] is False
    assert artifact.critic_report["score"] <= 10
    assert artifact.labels["label:U1"]["text"] == "U1"
    assert artifact.labels["label:Rf1"]["text"] == "Rf1"


def test_multiple_ground_labels_have_clear_spacing() -> None:
    circuit = opamp_chain(4, "multi_ground_opamp_chain", "op_amp_network")
    circuit = replace(
        circuit,
        components=[
            *circuit.components,
            Component(id="AGND", type="ground", pins={"gnd": "gnd"}, label="ANALOG_GND"),
        ],
    )
    layout = plan_layout(circuit)
    labels = {label.owner_id: label for label in layout.labels}

    assert not labels["GND"].bbox.intersects(labels["AGND"].bbox, padding=0.12)

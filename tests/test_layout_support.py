from optcpv import Circuit, Component, draw_artifact
from optcpv.examples import instrumentation_amplifier
from optcpv.planner import plan_layout


def test_known_motif_exposes_high_confidence_layout_support() -> None:
    circuit = instrumentation_amplifier()
    layout = plan_layout(circuit)
    artifact = draw_artifact(circuit)

    assert layout.support.layout_mode == "native_motif"
    assert layout.support.layout_confidence == 0.95
    assert layout.support.matched_motifs == ("instrumentation_amplifier",)
    assert layout.support.fallback_used is False
    assert artifact.layout_support["matched_motifs"] == ["instrumentation_amplifier"]
    assert artifact.layout_support["fallback_used"] is False
    assert 'data-optcpv-layout-mode="native_motif"' in artifact.svg
    assert 'data-optcpv-fallback-used="false"' in artifact.svg


def test_unknown_topology_uses_explicit_diagnostic_fallback_support() -> None:
    circuit = Circuit(
        id="unseen_lc_switch",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="L1", type="inductor", pins={"a": "vin", "b": "mid"}),
            Component(id="SW1", type="switch", pins={"a": "mid", "b": "out"}),
            Component(id="VOUT", type="output", pins={"in": "out"}),
        ],
    )

    artifact = draw_artifact(circuit)

    assert artifact.layout_support["layout_mode"] == "diagnostic_fallback"
    assert artifact.layout_support["layout_confidence"] < 0.5
    assert artifact.layout_support["matched_motifs"] == []
    assert artifact.layout_support["fallback_used"] is True
    assert artifact.layout_support["unsupported_regions"] == ["circuit:unknown_topology"]
    assert 'data-optcpv-layout-mode="diagnostic_fallback"' in artifact.svg
    assert 'data-optcpv-fallback-used="true"' in artifact.svg


def test_known_motif_with_unplaced_extra_component_is_partial_fallback() -> None:
    circuit = Circuit(
        id="divider_with_extra_probe",
        motif="voltage_divider",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "sense"}),
            Component(id="R2", type="resistor", pins={"a": "sense", "b": "gnd"}),
            Component(id="VOUT", type="output", pins={"in": "sense"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
            Component(id="TP1", type="test_point", pins={"in": "sense"}),
        ],
    )

    artifact = draw_artifact(circuit)

    assert artifact.layout_support["layout_mode"] == "partial_motif"
    assert artifact.layout_support["matched_motifs"] == ["voltage_divider"]
    assert artifact.layout_support["fallback_used"] is True
    assert artifact.layout_support["unsupported_regions"] == ["component:TP1"]
    assert artifact.layout_support["layout_confidence"] == 0.55


def test_multi_opamp_network_exposes_motif_network_support() -> None:
    circuit = Circuit(
        id="two_stage_opamp",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "fb1", "out": "stage1"}),
            Component(id="U2", type="op_amp", pins={"+": "stage1", "-": "fb2", "out": "vout"}),
            Component(id="VOUT", type="output", pins={"in": "vout"}),
        ],
    )

    layout = plan_layout(circuit)

    assert layout.support.layout_mode == "motif_network"
    assert layout.support.layout_confidence == 0.72
    assert layout.support.matched_motifs == ("op_amp_network",)
    assert layout.support.fallback_used is False

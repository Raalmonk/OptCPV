from optcpv import SchematicArtifact, draw_artifact
from optcpv.examples import voltage_divider


def test_draw_artifact_returns_small_metadata() -> None:
    artifact = draw_artifact(voltage_divider())

    assert isinstance(artifact, SchematicArtifact)
    assert artifact.svg.startswith("<svg")
    assert artifact.viewbox == {"x": 0, "y": 0, "width": 1100, "height": 800}
    assert set(artifact.components) == {"VIN", "R1", "R2", "VOUT", "GND"}
    assert {"vin", "vout", "gnd"}.issubset(artifact.nets)

from optcpv import plan_layout
from optcpv.examples import (
    instrumentation_amplifier,
    non_inverting_op_amp,
    rc_low_pass,
    voltage_divider,
)


def test_supported_motifs_have_non_diagnostic_layouts() -> None:
    for factory in [voltage_divider, rc_low_pass, non_inverting_op_amp, instrumentation_amplifier]:
        layout = plan_layout(factory())
        assert layout.components
        assert layout.wires
        assert all("diagnostic" not in warning for warning in layout.warnings)


def test_unknown_motif_uses_honest_diagnostic_fallback() -> None:
    layout = plan_layout(
        {
            "id": "unknown",
            "components": [
                {"id": "X1", "type": "component", "pins": {"a": "n1", "b": "n2"}},
                {"id": "X2", "type": "component", "pins": {"a": "n2", "b": "n3"}},
            ],
        }
    )

    assert layout.warnings == ["diagnostic: generic layout"]

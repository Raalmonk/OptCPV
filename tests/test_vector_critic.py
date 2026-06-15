from optcpv.examples import voltage_divider
from optcpv.patch import LayoutPatch, MoveComponent, apply_patch
from optcpv.planner import plan_layout
from optcpv.vector_critic import critique_layout


def test_vector_critic_catches_component_overlap() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    overlapped = apply_patch(circuit, layout, LayoutPatch(move_component=[MoveComponent("R1", 2.0, 4.0)]))
    report = critique_layout(overlapped)

    assert any(violation.code == "component_overlap" for violation in report.violations)
    assert report.hard_fail

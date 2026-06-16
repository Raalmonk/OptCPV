from dataclasses import replace

from optcpv.critic import critique
from optcpv.examples import voltage_divider
from optcpv.patch import LayoutPatch, MoveComponent, apply_patch
from optcpv.planner import plan_layout
from optcpv.renderer import render_svg
from optcpv.vector_critic import critique_layout


def test_enlarging_canvas_alone_does_not_reduce_score() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    base = critique(circuit, layout, render_svg(layout))
    huge = replace(layout, width=2600, height=1800)
    huge_score = critique(circuit, huge, render_svg(huge))

    assert huge_score.score >= base.score
    assert any(violation.code == "viewbox_too_large" for violation in huge_score.violations)


def test_sparse_huge_layout_receives_penalty() -> None:
    layout = plan_layout(voltage_divider())
    huge = replace(layout, width=2600, height=1800)
    report = critique_layout(huge)

    assert any(violation.code in {"viewbox_too_large", "fill_ratio_low"} for violation in report.violations)


def test_group_transform_scale_hack_is_penalized() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800"><g transform="scale(0.05)"><rect x="20" y="20" width="200" height="100" fill="black"/></g></svg>'
    report = critique(circuit, layout, svg)

    assert any(violation.code == "group_scale_too_small" for violation in report.violations)
    assert report.hard_fail


def test_viewbox_rewrite_hack_is_penalized() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100"><line x1="10" y1="10" x2="190" y2="90" stroke="black" stroke-width="2"/></svg>'
    report = critique(circuit, layout, svg)

    assert any(violation.code == "viewbox_not_fixed_frame" for violation in report.violations)


def test_abnormally_small_stroke_is_penalized() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800"><line x1="100" y1="100" x2="900" y2="100" stroke="black" stroke-width="0.3"/></svg>'
    report = critique(circuit, layout, svg)

    assert any(violation.code == "stroke_width_too_small" for violation in report.violations)
    assert report.hard_fail


def test_excessive_spread_patch_is_rejected() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)

    try:
        apply_patch(circuit, layout, LayoutPatch(move_component=[MoveComponent("VOUT", 40.0, 20.0)]))
    except ValueError as exc:
        assert "spreads components excessively" in str(exc)
    else:
        raise AssertionError("Expected excessive spread patch to be rejected.")

from dataclasses import replace

from optcpv.critic import critique
from optcpv.examples import voltage_divider
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

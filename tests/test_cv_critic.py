from optcpv.critic import critique
from optcpv.examples import voltage_divider
from optcpv.planner import plan_layout
from optcpv.renderer import render_svg


def test_cv_critic_detects_dense_black_blob() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800"><rect width="1100" height="800" fill="black"/></svg>'
    report = critique(circuit, layout, svg)

    assert any(violation.code == "dense_blob" for violation in report.violations)
    assert report.hard_fail


def test_cv_critic_detects_tiny_scaled_down_drawing() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 800"><g transform="scale(0.05)"><rect x="20" y="20" width="200" height="100" fill="black"/></g></svg>'
    report = critique(circuit, layout, svg)

    assert any(violation.code in {"tiny_or_faint", "too_much_empty_canvas"} for violation in report.violations)


def test_known_good_voltage_divider_has_low_cv_score() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    report = critique(circuit, layout, render_svg(layout))

    assert report.score < 120

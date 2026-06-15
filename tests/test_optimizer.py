import xml.etree.ElementTree as ET

from optcpv import draw_optimized_artifact, draw_optimized_svg
from optcpv.critic import critique
from optcpv.examples import instrumentation_amplifier
from optcpv.planner import plan_layout
from optcpv.renderer import render_svg
from optcpv.vector_critic import critique_layout


def test_optimizer_improves_instrumentation_amplifier_without_canvas_growth() -> None:
    circuit = instrumentation_amplifier()
    raw_layout = plan_layout(circuit)
    raw_report = critique(circuit, raw_layout, render_svg(raw_layout))
    optimized = draw_optimized_artifact(circuit)

    assert optimized.viewbox == {"x": 0, "y": 0, "width": raw_layout.width, "height": raw_layout.height}
    assert optimized.critic_report["score"] <= raw_report.score
    assert optimized.optimization_log


def test_optimized_output_has_no_component_overlaps_and_healthy_fill() -> None:
    optimized = draw_optimized_artifact(instrumentation_amplifier())

    assert not any(v["code"] == "component_overlap" for v in optimized.critic_report["violations"])
    assert optimized.critic_report["metrics"]["vector.component_fill_ratio"] >= 0.025


def test_draw_optimized_svg_returns_valid_svg_for_instrumentation_amp() -> None:
    svg = draw_optimized_svg(instrumentation_amplifier())
    root = ET.fromstring(svg)

    assert root.tag.endswith("svg")
    assert 'data-renderer="optcpv.schemdraw"' in svg
    assert "data-component-id" in svg

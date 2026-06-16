from optcpv.critic import critique
from optcpv.cv_critic import critique_raster
from optcpv.examples import voltage_divider
from optcpv.patch import LayoutPatch, MoveLabel, apply_patch
from optcpv.planner import plan_layout
from optcpv.renderer import render_svg
from optcpv.renderers.svg_postprocess import render_layer_svg
from optcpv.raster import rasterize_svg


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


def test_layered_cv_does_not_count_label_text_as_collision() -> None:
    layout = plan_layout(voltage_divider())
    labels = rasterize_svg(render_layer_svg(layout, "labels"))
    blank = rasterize_svg(_empty_svg(layout))
    report = critique_raster(layout, labels, wire_raster=blank, component_raster=blank, label_raster=labels)

    assert report.metrics["layered_masks"] is True
    assert report.metrics["label_layer_collision_count"] == 0
    assert not any(violation.code == "label_visual_collision" for violation in report.violations)


def test_layered_cv_detects_label_on_wire_fixture() -> None:
    circuit = voltage_divider()
    layout = plan_layout(circuit)
    wire = layout.wires[0]
    start, end = wire.points[0], wire.points[1]
    moved = apply_patch(
        circuit,
        layout,
        LayoutPatch(move_label=[MoveLabel(layout.labels[0].id, (start.x + end.x) / 2.0, (start.y + end.y) / 2.0)]),
    )
    labels = rasterize_svg(render_layer_svg(moved, "labels"))
    wires = rasterize_svg(render_layer_svg(moved, "wires"))
    components = rasterize_svg(render_layer_svg(moved, "components"))
    report = critique_raster(moved, rasterize_svg(render_svg(moved)), wire_raster=wires, component_raster=components, label_raster=labels)

    assert report.metrics["label_layer_collision_count"] > 0
    assert any(violation.code == "label_visual_collision" for violation in report.violations)


def _empty_svg(layout) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {layout.width} {layout.height}" '
        f'width="{layout.width}" height="{layout.height}"></svg>'
    )

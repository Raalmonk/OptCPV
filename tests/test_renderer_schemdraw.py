from dataclasses import replace
import xml.etree.ElementTree as ET

from optcpv import draw_svg
from optcpv.examples import EXAMPLES, instrumentation_amplifier
from optcpv.models import LayoutWire, Point
from optcpv.planner import plan_layout
from optcpv.renderers.schemdraw_backend import FALLBACK_RENDERER_ID, SchemdrawRenderer
from optcpv.renderers.svg_postprocess import render_debug_svg, render_layer_svg


def test_default_renderer_is_schemdraw_and_preserves_metadata() -> None:
    circuit = instrumentation_amplifier()
    svg = draw_svg(circuit)
    root = ET.fromstring(svg)

    assert root.tag.endswith("svg")
    assert 'data-renderer="optcpv.schemdraw"' in svg
    assert "data-component-id" in svg
    assert "data-net-name" in svg
    assert "data-pin-name" in svg
    for component in circuit.components:
        assert f'data-component-id="{component.id}"' in svg


def test_default_examples_use_real_schemdraw_renderer() -> None:
    for factory in EXAMPLES.values():
        svg = draw_svg(factory())

        assert 'data-renderer="optcpv.schemdraw"' in svg
        assert FALLBACK_RENDERER_ID not in svg
        assert 'viewBox="0 0 1100 800"' in svg
        assert 'id="schemdraw-canvas"' in svg


def test_schemdraw_fallback_metadata_does_not_lie(monkeypatch) -> None:
    circuit = instrumentation_amplifier()
    layout = plan_layout(circuit)

    def fail_render(self, layout):
        raise ValueError("boom")

    monkeypatch.setattr(SchemdrawRenderer, "_render_schemdraw_svg", fail_render)
    svg = SchemdrawRenderer().render(layout)

    assert f'data-renderer="{FALLBACK_RENDERER_ID}"' in svg
    assert 'data-renderer="optcpv.schemdraw"' not in svg
    assert 'data-schemdraw-error="ValueError: boom"' in svg


def test_label_halo_preserves_label_metadata() -> None:
    layout = plan_layout(instrumentation_amplifier())
    labels_svg = render_layer_svg(layout, "labels")

    assert 'data-label-id="label:U1"' in labels_svg
    assert 'class="label-halo"' in labels_svg


def test_debug_renderer_draws_wire_crossing_as_jump_bridge() -> None:
    layout = replace(
        plan_layout(instrumentation_amplifier()),
        wires=[
            LayoutWire(net="horizontal", points=[Point(2.0, 2.0), Point(8.0, 2.0)], connected_pins=[]),
            LayoutWire(net="vertical", points=[Point(5.0, 0.8), Point(5.0, 4.2)], connected_pins=[]),
        ],
    )
    svg = render_debug_svg(layout)

    assert '<path class="wire"' in svg
    assert " Q " in svg
    assert 'data-net-name="horizontal"' in svg

import xml.etree.ElementTree as ET

from optcpv import draw_svg
from optcpv.examples import instrumentation_amplifier


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

from __future__ import annotations

import xml.etree.ElementTree as ET

from backend.app.schem_forge.models import LayoutPlan
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout


def test_layout_plan_serialization_roundtrip(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    roundtrip = LayoutPlan.from_dict(plan.to_dict())

    assert roundtrip.to_dict() == plan.to_dict()
    assert roundtrip.topology_signature == plan.topology_signature


def test_renderer_returns_valid_svg_xml(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    rendered = render_layout(plan)

    root = ET.fromstring(rendered.svg)

    assert root.tag.endswith("svg")
    assert root.attrib["data-schem-forge-renderer"] == plan.renderer
    assert root.attrib["data-circuit-id"] == plan.circuit_id


def test_renderer_preserves_component_and_net_metadata(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    rendered = render_layout(plan)

    assert 'data-component-id="R1"' in rendered.svg
    assert 'data-component-id="R2"' in rendered.svg
    assert 'data-net-name="VOUT"' in rendered.svg
    assert 'data-pin-ref="R1.a"' in rendered.svg or 'data-pin-ref="R1.b"' in rendered.svg


def test_instrumentation_svg_renders_without_xml_errors(instrumentation_ir_fixture: dict) -> None:
    plan = plan_circuit(instrumentation_ir_fixture)
    rendered = render_layout(plan)

    ET.fromstring(rendered.svg)
    assert 'data-component-id="U1"' in rendered.svg
    assert 'data-component-id="U3"' in rendered.svg

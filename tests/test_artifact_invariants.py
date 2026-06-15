from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import pytest

from backend.app.schem_forge.agent import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.artifact import (
    ArtifactBBox,
    ComponentArtifact,
    FocusRegion,
    HitTarget,
    LabelArtifact,
    NetArtifact,
    OverlayArtifact,
    PinArtifact,
    SchematicArtifact,
    ViewBox,
    ZoomPreset,
    build_schematic_artifact,
)
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.examples import EXAMPLE_CASES
from backend.app.schem_forge.models import Point
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from scripts.test_schem_forge import debug_html


def _svg_root(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def _root_viewbox(svg: str) -> ViewBox:
    root = _svg_root(svg)
    x, y, width, height = [float(part) for part in root.attrib["viewBox"].split()]
    return ViewBox(x=x, y=y, width=width, height=height)


def _all_svg_attrs(svg: str, attr_name: str) -> set[str]:
    root = _svg_root(svg)
    return {
        value
        for element in root.iter()
        if (value := element.attrib.get(attr_name)) is not None
    }


def _bbox_positive(bbox: ArtifactBBox) -> bool:
    return bbox.width > 0 and bbox.height > 0


def _inside(inner: ViewBox, outer: ViewBox) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.width <= outer.x + outer.width + 0.001
        and inner.y + inner.height <= outer.y + outer.height + 0.001
    )


@pytest.fixture(params=sorted(EXAMPLE_CASES))
def case_artifact(request):
    circuit_ir = EXAMPLE_CASES[request.param]()
    result = generate_beautiful_schematic(circuit_ir, llm_client=MockLLMClient())
    return request.param, circuit_ir, result.layout, result.artifact


def test_all_case_artifact_svg_metadata_invariants(case_artifact) -> None:
    _case, _circuit_ir, layout_plan, artifact = case_artifact
    payload = artifact.to_dict()
    root_viewbox = _root_viewbox(artifact.svg)
    svg_component_ids = _all_svg_attrs(artifact.svg, "data-component-id")
    svg_net_names = _all_svg_attrs(artifact.svg, "data-net-name")
    svg_pin_refs = _all_svg_attrs(artifact.svg, "data-pin-ref")
    svg_label_ids = _all_svg_attrs(artifact.svg, "data-label-id")

    assert json.dumps(payload)
    assert _svg_root(artifact.svg).tag.endswith("svg")
    assert artifact.svg_viewbox == root_viewbox

    for component_id, component in artifact.components.items():
        assert component_id in svg_component_ids
        assert _bbox_positive(component.bbox)
        assert component.pins
        for pin_name, pin in component.pins.items():
            assert f"{component_id}.{pin_name}" in svg_pin_refs
            assert pin.net_name in artifact.nets

    for net_name, net in artifact.nets.items():
        if net.segments:
            assert net_name in svg_net_names
            assert net.bbox is not None
            assert _bbox_positive(net.bbox)
        assert sorted(net.connected_pins) == sorted(layout_plan.net_to_pins[net_name])

    for label_id, label in artifact.labels.items():
        assert label_id in svg_label_ids
        assert _bbox_positive(label.bbox)


def test_all_case_zoom_focus_overlay_and_hit_target_invariants(case_artifact) -> None:
    _case, _circuit_ir, _layout_plan, artifact = case_artifact
    component_ids = set(artifact.components)
    net_names = set(artifact.nets)
    label_ids = set(artifact.labels)
    pin_refs = {
        f"{component_id}.{pin_name}"
        for component_id, component in artifact.components.items()
        for pin_name in component.pins
    }
    focus_ids = {region.id for region in artifact.focus_regions}
    preset_ids = [preset.id for preset in artifact.zoom_presets]

    assert preset_ids.count("fit_all") == 1
    assert all(preset.viewbox.width > 0 and preset.viewbox.height > 0 for preset in artifact.zoom_presets)
    assert all(_inside(preset.viewbox, artifact.svg_viewbox) for preset in artifact.zoom_presets)

    for region in artifact.focus_regions:
        assert _bbox_positive(region.bbox)
        assert set(region.components) <= component_ids
        assert set(region.nets) <= net_names
        assert set(region.pins) <= pin_refs
        assert set(region.labels) <= label_ids
        assert f"focus_{region.id}" in preset_ids

    for component_id in component_ids:
        assert f"component_{component_id}" in preset_ids

    for overlay in artifact.overlays:
        assert set(overlay.components) <= component_ids
        assert set(overlay.nets) <= net_names
        assert set(overlay.pins) <= pin_refs
        assert overlay.focus_region_id in focus_ids

    valid_targets = {
        "component": component_ids,
        "net": net_names,
        "pin": pin_refs,
        "label": label_ids,
        "focus_region": focus_ids,
    }
    for target in artifact.hit_targets:
        assert _bbox_positive(target.bbox)
        assert target.target_id in valid_targets[target.kind]


def test_artifact_viewbox_matches_expanded_svg_viewbox(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    plan.canvas_width = 100
    plan.canvas_height = 100
    output = next(component for component in plan.components if component.id == "VOUT")
    output.grid_x = 40
    output.grid_y = 30
    render_result = render_layout(plan)
    report = critique_layout(plan, render_result)
    artifact = build_schematic_artifact(voltage_divider_ir_fixture, plan, render_result, report)

    root_viewbox = _root_viewbox(artifact.svg)

    assert root_viewbox.width > plan.canvas_width
    assert root_viewbox.height > plan.canvas_height
    assert artifact.svg_viewbox == root_viewbox


def test_artifact_topology_svg_consistency(case_artifact) -> None:
    _case, _circuit_ir, layout_plan, artifact = case_artifact
    svg_pin_refs = _all_svg_attrs(artifact.svg, "data-pin-ref")
    svg_net_names = _all_svg_attrs(artifact.svg, "data-net-name")
    artifact_pin_refs = set()

    for component_id, pin_map in layout_plan.component_pin_nets.items():
        assert component_id in artifact.components
        for pin_name, net_name in pin_map.items():
            assert pin_name in artifact.components[component_id].pins
            pin = artifact.components[component_id].pins[pin_name]
            assert pin.net_name == net_name
            artifact_pin_refs.add(f"{component_id}.{pin_name}")
            assert pin.net_name in artifact.nets

    for net_name, pins in layout_plan.net_to_pins.items():
        assert net_name in artifact.nets
        assert sorted(artifact.nets[net_name].connected_pins) == sorted(pins)

    assert svg_pin_refs <= artifact_pin_refs
    assert svg_net_names <= set(artifact.nets)
    assert all(region_net in artifact.nets for region in artifact.focus_regions for region_net in region.nets)


def test_debug_html_uses_safe_dom_event_listeners_with_weird_ids() -> None:
    component_id = "R-1: sense \"A\""
    net_name = "VIN/REF:1"
    label_id = "label:R/1?'"
    artifact = SchematicArtifact(
        artifact_version="schem_forge.artifact.v1",
        circuit_id="weird_debug",
        renderer="schem_forge.svg.v1",
        svg=(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160">'
            f'<g data-component-id="{component_id}"></g>'
            f'<line data-net-name="{net_name}" x1="0" y1="0" x2="10" y2="0"/>'
            f'<circle data-pin-ref="{component_id}.a" data-net-name="{net_name}"/>'
            f'<text data-label-id="{label_id}">R</text>'
            "</svg>"
        ),
        svg_viewbox=ViewBox(0, 0, 200, 160),
        layout_plan={},
        components={
            component_id: ComponentArtifact(
                id=component_id,
                type="resistor",
                role=None,
                display_label=component_id,
                value_label=None,
                bbox=ArtifactBBox(20, 20, 50, 20),
                label_bbox=ArtifactBBox(20, 10, 30, 12),
                pins={
                    "a": PinArtifact(component_id, "a", net_name, Point(20, 30), "left")
                },
            )
        },
        nets={
            net_name: NetArtifact(
                name=net_name,
                connected_pins=[f"{component_id}.a"],
                segments=[],
                bbox=ArtifactBBox(0, 0, 20, 8),
                junctions=[],
            )
        },
        labels={
            label_id: LabelArtifact(label_id, "R", component_id, ArtifactBBox(20, 10, 30, 12))
        },
        focus_regions=[
            FocusRegion(
                id="focus:odd/id",
                label="Odd Focus",
                kind="qa",
                bbox=ArtifactBBox(10, 10, 80, 60),
                components=[component_id],
                nets=[net_name],
                pins=[f"{component_id}.a"],
                labels=[label_id],
                goals=[],
            )
        ],
        zoom_presets=[
            ZoomPreset(
                id="fit_all",
                label="Fit All",
                viewbox=ViewBox(0, 0, 200, 160),
                focus_region_id=None,
                target_type=None,
                target_id=None,
            ),
            ZoomPreset(
                id="focus_focus:odd/id",
                label="Odd Focus",
                viewbox=ViewBox(0, 0, 120, 100),
                focus_region_id="focus:odd/id",
                target_type="focus_region",
                target_id="focus:odd/id",
            ),
        ],
        hit_targets=[
            HitTarget("component:odd", "component", component_id, ArtifactBBox(20, 20, 50, 20))
        ],
        overlays=[
            OverlayArtifact(
                id="overlay:odd",
                kind="qa",
                label="Odd",
                components=[component_id],
                nets=[net_name],
                pins=[f"{component_id}.a"],
                focus_region_id="focus:odd/id",
            )
        ],
        critic_report={},
        warnings=[],
        provenance={},
    )
    html = debug_html(artifact)
    match = re.search(r'<script id="artifact-json" type="application/json">(.*?)</script>', html, re.S)

    assert match is not None
    assert json.loads(match.group(1))["circuit_id"] == "weird_debug"
    assert "onclick=" not in html
    assert "addEventListener" in html
    assert "dataset.presetId" in html
    assert "dataset.regionId" in html
    assert "diagram.innerHTML = artifact.svg" in html
    assert 'id="reset-view"' in html

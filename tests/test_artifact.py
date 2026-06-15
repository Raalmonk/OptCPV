from __future__ import annotations

import json

from backend.app.schem_forge.artifact import ARTIFACT_VERSION, build_schematic_artifact
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from scripts.test_schem_forge import run_case


def _artifact_for(circuit_ir: dict):
    plan = plan_circuit(circuit_ir)
    render_result = render_layout(plan)
    report = critique_layout(plan, render_result)
    return build_schematic_artifact(circuit_ir, plan, render_result, report), plan


def test_artifact_json_serializes_and_includes_svg(voltage_divider_ir_fixture: dict) -> None:
    artifact, _plan = _artifact_for(voltage_divider_ir_fixture)
    payload = artifact.to_dict()

    encoded = json.dumps(payload)

    assert artifact.artifact_version == ARTIFACT_VERSION
    assert payload["svg"].startswith("<svg")
    assert "zoom_presets" in encoded


def test_artifact_has_root_viewbox(voltage_divider_ir_fixture: dict) -> None:
    artifact, plan = _artifact_for(voltage_divider_ir_fixture)

    assert artifact.svg_viewbox.x == 0
    assert artifact.svg_viewbox.y == 0
    assert artifact.svg_viewbox.width == plan.canvas_width
    assert artifact.svg_viewbox.height == plan.canvas_height


def test_artifact_has_component_bboxes_and_pins(voltage_divider_ir_fixture: dict) -> None:
    artifact, plan = _artifact_for(voltage_divider_ir_fixture)

    assert set(artifact.components) == {component.id for component in plan.components}
    r1 = artifact.components["R1"]
    assert r1.bbox.width > 0
    assert r1.bbox.height > 0
    assert r1.pins
    assert all(pin.net_name for pin in r1.pins.values())


def test_net_artifacts_include_connected_pins_and_segments(voltage_divider_ir_fixture: dict) -> None:
    artifact, _plan = _artifact_for(voltage_divider_ir_fixture)

    assert "VOUT" in artifact.nets
    assert artifact.nets["VOUT"].connected_pins
    assert artifact.nets["VOUT"].segments
    assert artifact.nets["VOUT"].bbox is not None


def test_focus_regions_exist_for_instrumentation_amp(instrumentation_ir_fixture: dict) -> None:
    artifact, _plan = _artifact_for(instrumentation_ir_fixture)
    region_ids = {region.id for region in artifact.focus_regions}

    assert {
        "input_pair",
        "top_input_buffer",
        "bottom_input_buffer",
        "gain_resistor",
        "differential_stage",
        "output",
        "ground_reference",
        "full_signal_path",
    }.issubset(region_ids)


def test_zoom_presets_exist_for_fit_all_and_focus_regions(instrumentation_ir_fixture: dict) -> None:
    artifact, _plan = _artifact_for(instrumentation_ir_fixture)
    preset_ids = {preset.id for preset in artifact.zoom_presets}

    assert "fit_all" in preset_ids
    for region in artifact.focus_regions:
        assert f"focus_{region.id}" in preset_ids


def test_hit_targets_include_components_and_nets(voltage_divider_ir_fixture: dict) -> None:
    artifact, _plan = _artifact_for(voltage_divider_ir_fixture)
    hit_kinds = {target.kind for target in artifact.hit_targets}

    assert "component" in hit_kinds
    assert "net" in hit_kinds
    assert any(target.target_id == "R1" for target in artifact.hit_targets)
    assert any(target.target_id == "VOUT" for target in artifact.hit_targets)


def test_renderer_svg_includes_interaction_data_attributes(voltage_divider_ir_fixture: dict) -> None:
    artifact, _plan = _artifact_for(voltage_divider_ir_fixture)

    assert 'data-artifact-version="schem_forge.artifact.v1"' in artifact.svg
    assert 'data-component-role=' in artifact.svg
    assert 'data-wire-index=' in artifact.svg
    assert 'data-owner-id=' in artifact.svg
    assert 'data-pin-ref=' in artifact.svg


def test_debug_html_is_generated_by_script() -> None:
    row = run_case("voltage_divider")

    assert row["after_artifact"].endswith("after_artifact.json")
    assert row["debug_html"].endswith("debug.html")
    with open(row["debug_html"], encoding="utf-8") as handle:
        html = handle.read()
    assert "zoomToPreset" in html
    assert "highlightFocusRegion" in html

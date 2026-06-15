#!/usr/bin/env python3
"""Export local schem_forge SVG fixtures and critic reports."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.schem_forge import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.artifact import SchematicArtifact, build_schematic_artifact
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.examples import EXAMPLE_CASES, instrumentation_amp_ir
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from backend.app.schem_forge.verifier import verify_equivalence


OUTPUT_ROOT = REPO_ROOT / "backend" / "app" / "schem_forge" / "generated"
QA_THRESHOLDS = {
    "instrumentation_amp": 300,
    "non_inverting_op_amp": 200,
    "rc_low_pass": 150,
    "voltage_divider": 100,
}


def build_instrumentation_amplifier_ir() -> dict:
    """Backward-compatible fixture hook for older local imports."""

    return instrumentation_amp_ir()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def root_viewbox(svg: str) -> dict[str, float]:
    root = ET.fromstring(svg)
    x, y, width, height = [float(part) for part in root.attrib["viewBox"].split()]
    return {"x": x, "y": y, "width": width, "height": height}


def artifact_contract_errors(artifact: SchematicArtifact) -> list[str]:
    errors: list[str] = []
    payload = artifact.to_dict()
    try:
        json.dumps(payload)
        root = ET.fromstring(artifact.svg)
    except Exception as exc:  # pragma: no cover - defensive report path
        return [f"artifact serialization/svg parse failed: {exc}"]

    if root_viewbox(artifact.svg) != artifact.svg_viewbox.__dict__:
        errors.append("SVG root viewBox does not match artifact.svg_viewbox")

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

    if preset_ids.count("fit_all") != 1:
        errors.append("fit_all zoom preset must exist exactly once")

    for preset in artifact.zoom_presets:
        if preset.viewbox.width <= 0 or preset.viewbox.height <= 0:
            errors.append(f"zoom preset {preset.id} has non-positive viewbox")
    for region in artifact.focus_regions:
        if region.bbox.width <= 0 or region.bbox.height <= 0:
            errors.append(f"focus region {region.id} has non-positive bbox")
        if not set(region.components) <= component_ids:
            errors.append(f"focus region {region.id} references missing components")
        if not set(region.nets) <= net_names:
            errors.append(f"focus region {region.id} references missing nets")
        if not set(region.pins) <= pin_refs:
            errors.append(f"focus region {region.id} references missing pins")
        if not set(region.labels) <= label_ids:
            errors.append(f"focus region {region.id} references missing labels")
        if f"focus_{region.id}" not in preset_ids:
            errors.append(f"focus region {region.id} has no matching zoom preset")
    for component_id in component_ids:
        if f"component_{component_id}" not in preset_ids:
            errors.append(f"component {component_id} has no matching zoom preset")
    for overlay in artifact.overlays:
        if not set(overlay.components) <= component_ids:
            errors.append(f"overlay {overlay.id} references missing components")
        if not set(overlay.nets) <= net_names:
            errors.append(f"overlay {overlay.id} references missing nets")
        if not set(overlay.pins) <= pin_refs:
            errors.append(f"overlay {overlay.id} references missing pins")
        if overlay.focus_region_id not in focus_ids:
            errors.append(f"overlay {overlay.id} references missing focus region")
    valid_targets = {
        "component": component_ids,
        "net": net_names,
        "pin": pin_refs,
        "label": label_ids,
        "focus_region": focus_ids,
    }
    for target in artifact.hit_targets:
        if target.bbox.width <= 0 or target.bbox.height <= 0:
            errors.append(f"hit target {target.id} has non-positive bbox")
        if target.target_id not in valid_targets[target.kind]:
            errors.append(f"hit target {target.id} references missing {target.kind}")

    svg_component_ids = {
        value for element in root.iter() if (value := element.attrib.get("data-component-id"))
    }
    svg_net_names = {
        value for element in root.iter() if (value := element.attrib.get("data-net-name"))
    }
    svg_pin_refs = {
        value for element in root.iter() if (value := element.attrib.get("data-pin-ref"))
    }
    if not component_ids <= svg_component_ids:
        errors.append("one or more artifact components missing from SVG metadata")
    if not {name for name, net in artifact.nets.items() if net.segments} <= svg_net_names:
        errors.append("one or more segmented artifact nets missing from SVG metadata")
    if not pin_refs <= svg_pin_refs:
        errors.append("one or more artifact pins missing from SVG metadata")
    return errors


def debug_html(artifact: SchematicArtifact) -> str:
    artifact_json = json.dumps(artifact.to_dict(), sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SchemForge Debug Viewer - {artifact.circuit_id}</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #f4f5f7; color: #111827; }}
    main {{ display: grid; grid-template-columns: 320px 1fr; height: 100vh; }}
    aside {{ overflow: auto; padding: 16px; background: #ffffff; border-right: 1px solid #d1d5db; }}
    #diagram {{ overflow: hidden; background: #fbfaf7; }}
    button {{ display: block; width: 100%; margin: 6px 0; padding: 7px 9px; border: 1px solid #cbd5e1; background: #fff; text-align: left; cursor: pointer; }}
    button:hover {{ background: #eef2ff; }}
    h1 {{ font-size: 16px; margin: 0 0 10px; }}
    h2 {{ font-size: 13px; margin: 18px 0 8px; color: #374151; }}
    .meta {{ font-size: 12px; color: #4b5563; line-height: 1.4; }}
    .highlight {{ stroke: #dc2626 !important; stroke-width: 4 !important; filter: drop-shadow(0 0 4px rgba(220,38,38,.35)); }}
    .highlight-fill {{ outline: 3px solid #dc2626; }}
  </style>
</head>
<body>
<main>
  <aside>
    <h1>{artifact.circuit_id}</h1>
    <div class="meta">critic score: {artifact.critic_report.get("total_score")}<br>renderer: {artifact.renderer}<br>artifact: {artifact.artifact_version}</div>
    <h2>View</h2>
    <button type="button" id="reset-view">Reset viewBox</button>
    <h2>Zoom Presets</h2>
    <div id="zoom-buttons"></div>
    <h2>Focus Regions</h2>
    <div id="focus-buttons"></div>
  </aside>
  <section id="diagram"></section>
</main>
<script id="artifact-json" type="application/json">{artifact_json}</script>
<script>
const artifact = JSON.parse(document.getElementById("artifact-json").textContent);
const diagram = document.getElementById("diagram");
diagram.innerHTML = artifact.svg;
const svg = diagram.querySelector("svg");
const originalViewBox = `${{artifact.svg_viewbox.x}} ${{artifact.svg_viewbox.y}} ${{artifact.svg_viewbox.width}} ${{artifact.svg_viewbox.height}}`;
svg.setAttribute("width", "100%");
svg.setAttribute("height", "100%");
svg.setAttribute("viewBox", originalViewBox);

function cssEscape(value) {{
  if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(value);
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
}}

function clearHighlights() {{
  svg.querySelectorAll(".highlight").forEach(el => el.classList.remove("highlight"));
}}

function setViewBox(viewbox) {{
  svg.setAttribute("viewBox", `${{viewbox.x}} ${{viewbox.y}} ${{viewbox.width}} ${{viewbox.height}}`);
}}

function resetView() {{
  clearHighlights();
  svg.setAttribute("viewBox", originalViewBox);
}}

function zoomToPreset(presetId) {{
  const preset = artifact.zoom_presets.find(p => p.id === presetId);
  if (!preset) return;
  setViewBox(preset.viewbox);
  if (preset.focus_region_id) highlightFocusRegion(preset.focus_region_id);
}}

function highlightFocusRegion(regionId) {{
  clearHighlights();
  const region = artifact.focus_regions.find(r => r.id === regionId);
  if (!region) return;
  region.components.forEach(id => svg.querySelector(`[data-component-id="${{cssEscape(id)}}"]`)?.classList.add("highlight"));
  region.nets.forEach(net => svg.querySelectorAll(`[data-net-name="${{cssEscape(net)}}"]`).forEach(el => el.classList.add("highlight")));
  region.labels.forEach(id => svg.querySelector(`[data-label-id="${{cssEscape(id)}}"]`)?.classList.add("highlight"));
}}

document.getElementById("reset-view").addEventListener("click", resetView);

const zoomButtons = document.getElementById("zoom-buttons");
artifact.zoom_presets.forEach(preset => {{
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.presetId = preset.id;
  button.textContent = preset.label;
  button.addEventListener("click", () => zoomToPreset(button.dataset.presetId));
  zoomButtons.appendChild(button);
}});

const focusButtons = document.getElementById("focus-buttons");
artifact.focus_regions.forEach(region => {{
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.regionId = region.id;
  button.textContent = region.label;
  button.addEventListener("click", () => {{
    const regionId = button.dataset.regionId;
    highlightFocusRegion(regionId);
    zoomToPreset(`focus_${{regionId}}`);
  }});
  focusButtons.appendChild(button);
}});
</script>
</body>
</html>
"""


def run_case(case_name: str) -> dict:
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_ir = EXAMPLE_CASES[case_name]()

    before_plan = plan_circuit(circuit_ir)
    verify_equivalence(circuit_ir, before_plan)
    before_render = render_layout(before_plan)
    before_report = critique_layout(before_plan, before_render)
    before_artifact = build_schematic_artifact(
        circuit_ir,
        before_plan,
        before_render,
        before_report,
    )

    result = generate_beautiful_schematic(
        circuit_ir,
        max_iterations=5,
        llm_client=MockLLMClient(),
    )
    verify_equivalence(circuit_ir, result.layout)

    write_text(output_dir / "before.svg", before_render.svg)
    write_text(output_dir / "after.svg", result.svg)
    write_json(output_dir / "before_plan.json", before_plan.to_dict())
    write_json(output_dir / "after_plan.json", result.layout.to_dict())
    write_json(output_dir / "before_artifact.json", before_artifact.to_dict())
    write_json(output_dir / "after_artifact.json", result.artifact.to_dict())
    write_text(output_dir / "debug.html", debug_html(result.artifact))
    write_json(
        output_dir / "critic_report.json",
        {
            "before": before_report.to_dict(),
            "after": result.critic_report.to_dict(),
            "agent_debug_log": result.debug_log,
        },
    )
    artifact_errors = artifact_contract_errors(result.artifact)
    after_codes = [violation.code for violation in result.critic_report.violations]
    known_visual_issues = []
    if case_name == "instrumentation_amp" and after_codes == ["wire_crossing"]:
        known_visual_issues.append(
            "Known instrumentation_amp crossing between N_GAIN_TOP and VINP; fatal-free but not zero-score."
        )
    threshold = QA_THRESHOLDS[case_name]

    return {
        "case": case_name,
        "before": before_report.total_score,
        "after": result.critic_report.total_score,
        "fatal_after": result.critic_report.fatal_count,
        "focus_regions": len(result.artifact.focus_regions),
        "zoom_presets": len(result.artifact.zoom_presets),
        "hit_targets": len(result.artifact.hit_targets),
        "artifact_ok": "yes" if not artifact_errors else "no",
        "improved": "yes" if result.critic_report.total_score < before_report.total_score else "no",
        "output_dir": str(output_dir),
        "before_svg": str(output_dir / "before.svg"),
        "after_svg": str(output_dir / "after.svg"),
        "after_artifact": str(output_dir / "after_artifact.json"),
        "debug_html": str(output_dir / "debug.html"),
        "critic_report": str(output_dir / "critic_report.json"),
        "threshold": threshold,
        "threshold_ok": result.critic_report.total_score <= threshold,
        "artifact_errors": artifact_errors,
        "known_visual_issues": known_visual_issues,
        "violations_after": [violation.to_dict() if hasattr(violation, "to_dict") else violation.__dict__ for violation in result.critic_report.violations],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(EXAMPLE_CASES),
        default=None,
        help="Single fixture case to export.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all fixture cases.",
    )
    parser.add_argument(
        "--allow-known-issues",
        action="store_true",
        help="Do not return non-zero when known non-fatal visual issues are present.",
    )
    return parser.parse_args()


def print_summary(rows: list[dict]) -> None:
    print(
        f"{'case':<24} {'before':>8} {'after':>8} {'fatal_after':>12} "
        f"{'focus_regions':>14} {'zoom_presets':>13} {'hit_targets':>12} "
        f"{'artifact_ok':>11} {'improved':>10}"
    )
    for row in rows:
        print(
            f"{row['case']:<24} {row['before']:>8} {row['after']:>8} "
            f"{row['fatal_after']:>12} {row['focus_regions']:>14} "
            f"{row['zoom_presets']:>13} {row['hit_targets']:>12} "
            f"{row['artifact_ok']:>11} {row['improved']:>10}"
        )
    print("Output root:", OUTPUT_ROOT)
    for row in rows:
        print(f"{row['case']} artifact:", row["after_artifact"])
        print(f"{row['case']} debug:", row["debug_html"])


def write_qa_summary(rows: list[dict]) -> Path:
    summary_path = OUTPUT_ROOT / "qa_summary.json"
    summary = {
        "all_artifacts_ok": all(row["artifact_ok"] == "yes" for row in rows),
        "any_fatal_after": any(row["fatal_after"] > 0 for row in rows),
        "cases": {
            row["case"]: {
                "before": row["before"],
                "after": row["after"],
                "fatal_after": row["fatal_after"],
                "threshold": row["threshold"],
                "threshold_ok": row["threshold_ok"],
                "improved": row["improved"],
                "artifact_ok": row["artifact_ok"],
                "artifact_errors": row["artifact_errors"],
                "artifact_counts": {
                    "focus_regions": row["focus_regions"],
                    "zoom_presets": row["zoom_presets"],
                    "hit_targets": row["hit_targets"],
                },
                "paths": {
                    "before_svg": row["before_svg"],
                    "after_svg": row["after_svg"],
                    "after_artifact": row["after_artifact"],
                    "debug_html": row["debug_html"],
                    "critic_report": row["critic_report"],
                },
                "known_visual_issues": row["known_visual_issues"],
                "violations_after": row["violations_after"],
            }
            for row in rows
        },
        "known_visual_issues": [
            issue for row in rows for issue in row["known_visual_issues"]
        ],
    }
    write_json(summary_path, summary)
    return summary_path


def main() -> int:
    args = parse_args()
    if args.all:
        case_names = sorted(EXAMPLE_CASES)
    else:
        case_names = [args.case or "instrumentation_amp"]

    rows = [run_case(case_name) for case_name in case_names]
    print_summary(rows)
    summary_path = write_qa_summary(rows)
    print("QA summary:", summary_path)
    has_fatal = any(row["fatal_after"] > 0 for row in rows)
    has_artifact_errors = any(row["artifact_errors"] for row in rows)
    if has_artifact_errors or (has_fatal and not args.allow_known_issues):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

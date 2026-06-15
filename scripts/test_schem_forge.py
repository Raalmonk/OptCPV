#!/usr/bin/env python3
"""Export local schem_forge SVG fixtures and critic reports."""

from __future__ import annotations

import argparse
import json
import sys
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


def build_instrumentation_amplifier_ir() -> dict:
    """Backward-compatible fixture hook for older local imports."""

    return instrumentation_amp_ir()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    <button type="button" onclick="resetView()">Reset viewBox</button>
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
  region.components.forEach(id => svg.querySelector(`[data-component-id="${{CSS.escape(id)}}"]`)?.classList.add("highlight"));
  region.nets.forEach(net => svg.querySelectorAll(`[data-net-name="${{CSS.escape(net)}}"]`).forEach(el => el.classList.add("highlight")));
  region.labels.forEach(id => svg.querySelector(`[data-label-id="${{CSS.escape(id)}}"]`)?.classList.add("highlight"));
}}

document.getElementById("zoom-buttons").innerHTML = artifact.zoom_presets
  .map(preset => `<button type="button" onclick="zoomToPreset('${{preset.id}}')">${{preset.label}}</button>`)
  .join("");
document.getElementById("focus-buttons").innerHTML = artifact.focus_regions
  .map(region => `<button type="button" onclick="highlightFocusRegion('${{region.id}}'); zoomToPreset('focus_${{region.id}}')">${{region.label}}</button>`)
  .join("");
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

    return {
        "case": case_name,
        "before": before_report.total_score,
        "after": result.critic_report.total_score,
        "fatal_after": result.critic_report.fatal_count,
        "improved": "yes" if result.critic_report.total_score < before_report.total_score else "no",
        "output_dir": str(output_dir),
        "after_artifact": str(output_dir / "after_artifact.json"),
        "debug_html": str(output_dir / "debug.html"),
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
    return parser.parse_args()


def print_summary(rows: list[dict]) -> None:
    print(f"{'case':<24} {'before':>8} {'after':>8} {'fatal_after':>12} {'improved':>10}")
    for row in rows:
        print(
            f"{row['case']:<24} {row['before']:>8} {row['after']:>8} "
            f"{row['fatal_after']:>12} {row['improved']:>10}"
        )
    print("Output root:", OUTPUT_ROOT)
    for row in rows:
        print(f"{row['case']} artifact:", row["after_artifact"])
        print(f"{row['case']} debug:", row["debug_html"])


def main() -> int:
    args = parse_args()
    if args.all:
        case_names = sorted(EXAMPLE_CASES)
    else:
        case_names = [args.case or "instrumentation_amp"]

    rows = [run_case(case_name) for case_name in case_names]
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

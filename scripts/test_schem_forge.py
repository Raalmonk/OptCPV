#!/usr/bin/env python3
"""Export local schem_forge SVG fixtures and critic reports."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from html import escape as html_escape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.schem_forge import MockLLMClient, generate_beautiful_schematic
from backend.app.schem_forge.adapters import circuit_problem_to_schem_forge_ir
from backend.app.schem_forge.artifact import SchematicArtifact, build_schematic_artifact
from backend.app.schem_forge.citt_examples import CITT_EXAMPLE_CASES
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.examples import EXAMPLE_CASES, instrumentation_amp_ir
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from backend.app.schem_forge.verifier import verify_equivalence


OUTPUT_ROOT = REPO_ROOT / "backend" / "app" / "schem_forge" / "generated"
BUILTIN_QA_THRESHOLDS = {
    "instrumentation_amp": 300,
    "non_inverting_op_amp": 200,
    "rc_low_pass": 150,
    "voltage_divider": 100,
}
CITT_QA_THRESHOLDS = {
    "citt_bme_instrumentation_amplifier": 300,
    "citt_non_inverting_op_amp": 200,
    "citt_rc_low_pass": 150,
    "citt_voltage_divider": 100,
}
QA_THRESHOLDS = {**BUILTIN_QA_THRESHOLDS, **CITT_QA_THRESHOLDS}
SUITES = ("builtins", "citt")
ALL_CASES = {**EXAMPLE_CASES, **CITT_EXAMPLE_CASES}


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


def infer_suite(case_name: str, requested_suite: str | None = None) -> str:
    if requested_suite == "builtins":
        if case_name not in EXAMPLE_CASES:
            raise SystemExit(f"Case {case_name!r} is not a built-in fixture.")
        return "builtins"
    if requested_suite == "citt":
        if case_name not in CITT_EXAMPLE_CASES:
            raise SystemExit(f"Case {case_name!r} is not a CiTT fixture.")
        return "citt"
    if case_name in EXAMPLE_CASES:
        return "builtins"
    if case_name in CITT_EXAMPLE_CASES:
        return "citt"
    raise SystemExit(f"Unknown case {case_name!r}.")


def build_case_ir(case_name: str, suite: str) -> tuple[dict, dict | None]:
    if suite == "builtins":
        return EXAMPLE_CASES[case_name](), None
    payload = CITT_EXAMPLE_CASES[case_name]()
    return circuit_problem_to_schem_forge_ir(payload), payload


def selected_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all:
        return [
            *[(case_name, "builtins") for case_name in sorted(EXAMPLE_CASES)],
            *[(case_name, "citt") for case_name in sorted(CITT_EXAMPLE_CASES)],
        ]
    if args.case:
        return [(args.case, infer_suite(args.case, args.suite))]
    if args.suite == "builtins":
        return [(case_name, "builtins") for case_name in sorted(EXAMPLE_CASES)]
    if args.suite == "citt":
        return [(case_name, "citt") for case_name in sorted(CITT_EXAMPLE_CASES)]
    return [("instrumentation_amp", "builtins")]


def relative_to_output(path: str | Path) -> str:
    return Path(path).relative_to(OUTPUT_ROOT).as_posix()


def write_visual_review(rows: list[dict]) -> Path:
    review_path = OUTPUT_ROOT / "visual_review.html"
    cards: list[str] = []
    for row in rows:
        after_svg = relative_to_output(row["after_svg"])
        debug_path = relative_to_output(row["debug_html"])
        artifact_path = relative_to_output(row["after_artifact"])
        critic_path = relative_to_output(row["critic_report"])
        violations = row["violations_after"]
        if violations:
            violation_items = "\n".join(
                f"<li><code>{html_escape(item['code'])}</code> {html_escape(item['message'])}</li>"
                for item in violations
            )
        else:
            violation_items = "<li>none</li>"
        status = "pass" if row["threshold_ok"] and row["fatal_after"] == 0 and row["artifact_ok"] == "yes" else "fail"
        escaped_case = html_escape(row["case"])
        escaped_suite = html_escape(row["suite"])
        escaped_artifact_ok = html_escape(row["artifact_ok"])
        cards.append(
            f"""
    <article class="case-card {status}">
      <header>
        <div>
          <h2>{escaped_case}</h2>
          <p>{escaped_suite} - threshold {row["threshold"]}</p>
        </div>
        <strong>{row["after"]}</strong>
      </header>
      <img src="{html_escape(after_svg)}" alt="{escaped_case} schematic">
      <dl>
        <div><dt>before</dt><dd>{row["before"]}</dd></div>
        <div><dt>after</dt><dd>{row["after"]}</dd></div>
        <div><dt>fatal</dt><dd>{row["fatal_after"]}</dd></div>
        <div><dt>artifact</dt><dd>{escaped_artifact_ok}</dd></div>
      </dl>
      <nav>
        <a href="{html_escape(after_svg)}">SVG</a>
        <a href="{html_escape(debug_path)}">Debug</a>
        <a href="{html_escape(artifact_path)}">Artifact</a>
        <a href="{html_escape(critic_path)}">Critic</a>
      </nav>
      <h3>Violations</h3>
      <ul>{violation_items}</ul>
    </article>
"""
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>schem_forge visual review</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #f6f7f9; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 18px; font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .case-card {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; }}
    .case-card.pass {{ border-top: 4px solid #15803d; }}
    .case-card.fail {{ border-top: 4px solid #b91c1c; }}
    header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }}
    h2 {{ margin: 0; font-size: 16px; }}
    h3 {{ margin: 12px 0 6px; font-size: 13px; }}
    p {{ margin: 3px 0 0; color: #4b5563; font-size: 12px; }}
    strong {{ font-size: 26px; }}
    img {{ width: 100%; height: 320px; object-fit: contain; background: #fbfaf7; border: 1px solid #e5e7eb; margin: 12px 0; }}
    dl {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0; }}
    dt {{ color: #6b7280; font-size: 11px; }}
    dd {{ margin: 0; font-weight: 600; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    ul {{ margin: 0; padding-left: 18px; font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <h1>schem_forge visual review</h1>
  <section class="grid">
{''.join(cards)}
  </section>
</main>
</body>
</html>
"""
    write_text(review_path, html)
    return review_path


def run_case(case_name: str, suite: str | None = None) -> dict:
    resolved_suite = infer_suite(case_name, suite)
    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_ir, source_payload = build_case_ir(case_name, resolved_suite)
    if source_payload is not None:
        write_json(output_dir / "citt_payload.json", source_payload)
        write_json(output_dir / "adapted_ir.json", circuit_ir)

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
    threshold = QA_THRESHOLDS[case_name]
    payload_path = str(output_dir / "citt_payload.json") if source_payload is not None else None
    adapted_ir_path = str(output_dir / "adapted_ir.json") if source_payload is not None else None

    return {
        "case": case_name,
        "suite": resolved_suite,
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
        "source_payload": payload_path,
        "adapted_ir": adapted_ir_path,
        "threshold": threshold,
        "threshold_ok": result.critic_report.total_score <= threshold,
        "artifact_errors": artifact_errors,
        "violations_after": [violation.to_dict() if hasattr(violation, "to_dict") else violation.__dict__ for violation in result.critic_report.violations],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default=None,
        help="Fixture suite to export. Without --case, exports every case in the suite.",
    )
    parser.add_argument(
        "--case",
        choices=sorted(ALL_CASES),
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
    print(
        f"{'case':<40} {'suite':<9} {'before':>8} {'after':>8} {'fatal_after':>12} "
        f"{'focus_regions':>14} {'zoom_presets':>13} {'hit_targets':>12} "
        f"{'artifact_ok':>11} {'threshold':>10}"
    )
    for row in rows:
        print(
            f"{row['case']:<40} {row['suite']:<9} {row['before']:>8} {row['after']:>8} "
            f"{row['fatal_after']:>12} {row['focus_regions']:>14} "
            f"{row['zoom_presets']:>13} {row['hit_targets']:>12} "
            f"{row['artifact_ok']:>11} {row['threshold']:>10}"
        )
    print("Output root:", OUTPUT_ROOT)
    for row in rows:
        print(f"{row['case']} artifact:", row["after_artifact"])
        print(f"{row['case']} debug:", row["debug_html"])


def write_qa_summary(rows: list[dict], visual_review_path: Path) -> Path:
    summary_path = OUTPUT_ROOT / "qa_summary.json"
    summary = {
        "all_artifacts_ok": all(row["artifact_ok"] == "yes" for row in rows),
        "any_fatal_after": any(row["fatal_after"] > 0 for row in rows),
        "all_thresholds_ok": all(row["threshold_ok"] for row in rows),
        "visual_review": str(visual_review_path),
        "cases": {
            row["case"]: {
                "suite": row["suite"],
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
                    "source_payload": row["source_payload"],
                    "adapted_ir": row["adapted_ir"],
                    "visual_review": str(visual_review_path),
                },
                "violation_codes_after": [
                    violation["code"] for violation in row["violations_after"]
                ],
                "violations_after": row["violations_after"],
            }
            for row in rows
        },
        "remaining_visual_issues": [
            {"case": row["case"], "violations": row["violations_after"]}
            for row in rows
            if row["violations_after"]
        ],
    }
    write_json(summary_path, summary)
    return summary_path


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)

    rows = [run_case(case_name, suite=suite) for case_name, suite in cases]
    print_summary(rows)
    visual_review_path = write_visual_review(rows)
    summary_path = write_qa_summary(rows, visual_review_path)
    print("QA summary:", summary_path)
    print("Visual review:", visual_review_path)
    has_fatal = any(row["fatal_after"] > 0 for row in rows)
    has_artifact_errors = any(row["artifact_errors"] for row in rows)
    has_threshold_failures = any(not row["threshold_ok"] for row in rows)
    if has_artifact_errors or has_fatal or has_threshold_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

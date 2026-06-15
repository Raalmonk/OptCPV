#!/usr/bin/env python3
"""Exercise the schem_forge FastAPI product endpoints and export artifacts."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import app
from backend.app.schem_forge.citt_examples import citt_voltage_divider_payload


OUTPUT_ROOT = REPO_ROOT / "backend" / "app" / "schem_forge" / "generated" / "api_smoke"


def patch_testclient_httpx() -> None:
    original_init = httpx.Client.__init__
    if "app" in inspect.signature(original_init).parameters:
        return

    def compatible_init(self, *args, app=None, **kwargs):
        return original_init(self, *args, **kwargs)

    httpx.Client.__init__ = compatible_init


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def debug_html(artifact: dict[str, Any]) -> str:
    artifact_json = json.dumps(artifact, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>schem_forge API smoke - {artifact.get("circuit_id", "artifact")}</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #f4f6f8; color: #111827; }}
    main {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); height: 100vh; }}
    aside {{ overflow: auto; padding: 16px; background: #fff; border-right: 1px solid #d1d5db; }}
    #diagram {{ overflow: hidden; background: #fbfaf7; }}
    button {{ display: block; width: 100%; margin: 6px 0; padding: 7px 9px; border: 1px solid #cbd5e1; background: #fff; text-align: left; cursor: pointer; }}
    h1 {{ font-size: 16px; margin: 0 0 10px; }}
    h2 {{ font-size: 13px; margin: 18px 0 8px; }}
    .highlight {{ stroke: #dc2626 !important; stroke-width: 4 !important; }}
  </style>
</head>
<body>
<main>
  <aside>
    <h1>{artifact.get("circuit_id", "schematic")}</h1>
    <h2>Zoom</h2>
    <div id="zoom-buttons"></div>
    <h2>Focus</h2>
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
svg.setAttribute("width", "100%");
svg.setAttribute("height", "100%");

function cssEscape(value) {{
  if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(value);
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
}}
function setViewBox(viewbox) {{
  svg.setAttribute("viewBox", `${{viewbox.x}} ${{viewbox.y}} ${{viewbox.width}} ${{viewbox.height}}`);
}}
function clearHighlights() {{
  svg.querySelectorAll(".highlight").forEach(element => element.classList.remove("highlight"));
}}
function focusRegion(regionId) {{
  const region = artifact.focus_regions.find(item => item.id === regionId);
  const preset = artifact.zoom_presets.find(item => item.id === `focus_${{regionId}}`);
  if (preset) setViewBox(preset.viewbox);
  if (!region) return;
  clearHighlights();
  region.components.forEach(id => svg.querySelector(`[data-component-id="${{cssEscape(id)}}"]`)?.classList.add("highlight"));
  region.nets.forEach(net => svg.querySelectorAll(`[data-net-name="${{cssEscape(net)}}"]`).forEach(element => element.classList.add("highlight")));
}}
artifact.zoom_presets.forEach(preset => {{
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = preset.label || preset.id;
  button.addEventListener("click", () => setViewBox(preset.viewbox));
  document.getElementById("zoom-buttons").appendChild(button);
}});
artifact.focus_regions.forEach(region => {{
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = region.label || region.id;
  button.addEventListener("click", () => focusRegion(region.id));
  document.getElementById("focus-buttons").appendChild(button);
}});
</script>
</body>
</html>
"""


def export_response(case_name: str, response_payload: dict[str, Any]) -> dict[str, Any]:
    if response_payload.get("status") != "ok":
        raise RuntimeError(f"{case_name} did not return ok: {response_payload}")

    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "artifact.json"
    svg_path = output_dir / "schematic.svg"
    debug_path = output_dir / "debug.html"
    summary_path = output_dir / "response_summary.json"

    write_json(artifact_path, response_payload["artifact"])
    write_text(svg_path, response_payload["svg"])
    write_text(debug_path, debug_html(response_payload["artifact"]))
    summary = {
        "case": case_name,
        "status": response_payload["status"],
        "critic": response_payload["critic"],
        "warnings": response_payload["warnings"],
        "input_motif": response_payload["input_ir"].get("motif"),
        "paths": {
            "artifact": str(artifact_path),
            "svg": str(svg_path),
            "debug_html": str(debug_path),
            "response_summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    patch_testclient_httpx()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)

    cases = [
        (
            "text_voltage_divider",
            client.post("/v1/schematic/from-text", json={"prompt": "Draw a voltage divider"}),
        ),
        (
            "text_rc_low_pass",
            client.post("/v1/schematic/from-text", json={"prompt": "Draw an RC low pass filter"}),
        ),
        (
            "text_non_inverting_op_amp",
            client.post("/v1/schematic/from-text", json={"prompt": "Draw a non-inverting op amp"}),
        ),
        (
            "text_instrumentation_amplifier",
            client.post(
                "/v1/schematic/from-text",
                json={"prompt": "Draw an instrumentation amplifier"},
            ),
        ),
        (
            "ir_citt_voltage_divider",
            client.post(
                "/v1/schematic/from-ir",
                json={"input_format": "auto", "circuit": citt_voltage_divider_payload()},
            ),
        ),
    ]

    summaries = []
    for case_name, response in cases:
        response.raise_for_status()
        summaries.append(export_response(case_name, response.json()))

    index_path = OUTPUT_ROOT / "response_summary.json"
    write_json(
        index_path,
        {
            "status": "ok",
            "case_count": len(summaries),
            "cases": summaries,
        },
    )
    print(f"API smoke artifacts: {OUTPUT_ROOT}")
    print(f"API smoke summary: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

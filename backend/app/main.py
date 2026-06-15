"""FastAPI product surface for schem_forge."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from backend.app.api_models import FromIRRequest, FromTextRequest
from backend.app.services.artifact_service import (
    UnsupportedStudentFacingDiagram,
    generate_schematic_payload,
    normalize_input_circuit,
)
from backend.app.services.image_to_ir import image_parser_available, parse_image_to_ir
from backend.app.services.text_to_ir import (
    CircuitParserError,
    CircuitParserUnavailable,
    DeterministicDemoParser,
    GeminiCircuitParser,
    UnrecognizedDemoPrompt,
)


app = FastAPI(
    title="schem_forge",
    description="Beautiful, zoomable, interactive schematic artifacts from text, images, or circuit IR.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "schem_forge"}


@app.get("/", response_class=HTMLResponse)
def viewer() -> str:
    return VIEWER_HTML


@app.post("/v1/schematic/from-ir")
def schematic_from_ir(request: FromIRRequest) -> Any:
    try:
        circuit_ir = normalize_input_circuit(request.circuit, request.input_format)
        return generate_schematic_payload(
            circuit_ir,
            max_iterations=request.max_iterations,
            use_mock_agent=request.use_mock_agent,
        )
    except UnsupportedStudentFacingDiagram as exc:
        return JSONResponse({"status": "unsupported_motif", "message": str(exc)}, status_code=422)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/schematic")
async def schematic(request: Request) -> Any:
    """Compatibility product endpoint backed by schem_forge, never fallback graph."""

    payload = await request.json()
    if isinstance(payload, dict) and "circuit" in payload:
        request_model = FromIRRequest.model_validate(payload)
    elif isinstance(payload, dict):
        request_model = FromIRRequest(circuit=payload)
    else:
        raise HTTPException(status_code=400, detail="Request body must be a circuit object.")
    return schematic_from_ir(request_model)


@app.post("/v1/schematic/from-text")
def schematic_from_text(request: FromTextRequest) -> Any:
    parser = DeterministicDemoParser()
    try:
        circuit_ir = parser.parse_text(request.prompt)
    except UnrecognizedDemoPrompt:
        if not GeminiCircuitParser.is_configured():
            return JSONResponse(
                {
                    "status": "needs_parser",
                    "message": "Text parsing requires GEMINI_API_KEY or a recognized demo prompt.",
                }
            )
        try:
            circuit_ir = GeminiCircuitParser().parse_text(request.prompt)
        except CircuitParserUnavailable as exc:
            return JSONResponse({"status": "parser_unavailable", "message": str(exc)})
        except CircuitParserError as exc:
            return JSONResponse({"status": "parser_error", "message": str(exc)}, status_code=422)

    try:
        circuit_ir = normalize_input_circuit(circuit_ir, "auto")
        return generate_schematic_payload(
            circuit_ir,
            max_iterations=request.max_iterations,
            use_mock_agent=request.use_mock_agent,
        )
    except UnsupportedStudentFacingDiagram as exc:
        return JSONResponse({"status": "unsupported_motif", "message": str(exc)}, status_code=422)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/schematic/from-image")
async def schematic_from_image(request: Request) -> Any:
    if not image_parser_available():
        return JSONResponse(
            {
                "status": "vision_backend_unavailable",
                "message": "Image-to-circuit parsing requires GEMINI_API_KEY or GOOGLE_API_KEY.",
            }
        )

    try:
        form = await request.form()
    except Exception as exc:
        return JSONResponse(
            {
                "status": "image_upload_error",
                "message": f"Could not read multipart upload: {exc}",
            },
            status_code=400,
        )

    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="Multipart field 'file' is required.")
    prompt_value = form.get("prompt")
    prompt = str(prompt_value) if prompt_value is not None else None

    try:
        image_bytes = await upload.read()
        circuit_ir = parse_image_to_ir(image_bytes, prompt=prompt)
        circuit_ir = normalize_input_circuit(circuit_ir, "auto")
        return generate_schematic_payload(circuit_ir)
    except UnsupportedStudentFacingDiagram as exc:
        return JSONResponse({"status": "unsupported_motif", "message": str(exc)}, status_code=422)
    except CircuitParserUnavailable as exc:
        return JSONResponse({"status": "vision_backend_unavailable", "message": str(exc)})
    except CircuitParserError as exc:
        return JSONResponse({"status": "vision_parse_error", "message": str(exc)}, status_code=422)


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>schem_forge</title>
  <style>
    :root { color-scheme: light; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #f4f6f8; color: #111827; }
    main { display: grid; grid-template-columns: 380px minmax(0, 1fr); min-height: 100vh; }
    aside { background: #ffffff; border-right: 1px solid #d1d5db; padding: 18px; overflow: auto; }
    section { min-width: 0; padding: 18px; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 14px; }
    h1 { margin: 0 0 14px; font-size: 20px; }
    h2 { margin: 14px 0 8px; font-size: 13px; color: #374151; }
    label { display: block; margin: 10px 0 5px; font-size: 12px; color: #374151; }
    textarea, input[type="text"], input[type="file"] { width: 100%; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; padding: 9px; font: inherit; background: #fff; }
    textarea { min-height: 150px; resize: vertical; }
    button, a.download { border: 1px solid #9ca3af; border-radius: 6px; background: #fff; color: #111827; padding: 8px 10px; font: inherit; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; }
    button.primary { width: 100%; background: #111827; color: #fff; border-color: #111827; margin-top: 12px; }
    button:hover, a.download:hover { background: #eef2ff; }
    button.primary:hover { background: #1f2937; }
    .tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 10px; }
    .tabs button.active { background: #dbeafe; border-color: #60a5fa; }
    .panel { display: none; }
    .panel.active { display: block; }
    .status { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }
    .status strong { font-size: 24px; }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; }
    .stage { background: #fbfaf7; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; min-height: 520px; }
    .stage svg { width: 100%; height: 100%; min-height: 520px; display: block; }
    .error { color: #991b1b; white-space: pre-wrap; }
    .meta { color: #4b5563; font-size: 12px; line-height: 1.4; }
    .highlight { stroke: #dc2626 !important; stroke-width: 4 !important; filter: drop-shadow(0 0 4px rgba(220, 38, 38, .35)); }
    @media (max-width: 840px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #d1d5db; } }
  </style>
</head>
<body>
<main>
  <aside>
    <h1>schem_forge</h1>
    <div class="tabs">
      <button type="button" class="active" data-tab="text">Text</button>
      <button type="button" data-tab="json">JSON / IR</button>
      <button type="button" data-tab="image">Image</button>
    </div>
    <div id="panel-text" class="panel active">
      <label for="prompt">Circuit prompt</label>
      <textarea id="prompt">Draw a non-inverting op amp</textarea>
    </div>
    <div id="panel-json" class="panel">
      <label for="json-input">CiTT JSON or schem_forge IR</label>
      <textarea id="json-input">{
  "id": "demo_divider",
  "motif": "voltage_divider",
  "components": [
    {"id": "VIN", "type": "input", "role": "input_source", "display_label": "VIN", "pins": {"out": "VIN"}},
    {"id": "R1", "type": "resistor", "role": "top_resistor", "value_label": "R1", "pins": {"a": "VIN", "b": "VOUT"}},
    {"id": "R2", "type": "resistor", "role": "bottom_resistor", "value_label": "R2", "pins": {"a": "VOUT", "b": "GND"}},
    {"id": "VOUT", "type": "output", "role": "output", "display_label": "VOUT", "pins": {"in": "VOUT"}},
    {"id": "GND", "type": "ground", "role": "ground", "pins": {"gnd": "GND"}}
  ]
}</textarea>
    </div>
    <div id="panel-image" class="panel">
      <label for="image-file">Circuit image</label>
      <input id="image-file" type="file" accept="image/*">
      <label for="image-prompt">Optional hint</label>
      <input id="image-prompt" type="text" placeholder="e.g. instrumentation amplifier">
    </div>
    <button id="generate" class="primary" type="button">Generate schematic</button>
    <h2>Focus Regions</h2>
    <div id="focus-buttons" class="controls"></div>
    <h2>Zoom Presets</h2>
    <div id="zoom-buttons" class="controls"></div>
    <h2>Downloads</h2>
    <div class="controls">
      <a id="download-artifact" class="download" href="#">Artifact JSON</a>
      <a id="download-svg" class="download" href="#">SVG</a>
    </div>
  </aside>
  <section>
    <div class="status">
      <div><div class="meta">critic score</div><strong id="score">-</strong></div>
      <div><div class="meta">status</div><span id="status">idle</span></div>
      <div><div class="meta">warnings</div><span id="warnings">-</span></div>
    </div>
    <div id="stage" class="stage"></div>
  </section>
</main>
<script>
let activeTab = "text";
let currentArtifact = null;

function setTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tabs button").forEach(button => button.classList.toggle("active", button.dataset.tab === tab));
  document.querySelectorAll(".panel").forEach(panel => panel.classList.toggle("active", panel.id === `panel-${tab}`));
}

document.querySelectorAll(".tabs button").forEach(button => {
  button.addEventListener("click", () => setTab(button.dataset.tab));
});

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(value);
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "\\\\$&");
}

function renderArtifact(response) {
  if (response.status !== "ok") {
    document.getElementById("status").textContent = response.status;
    document.getElementById("stage").innerHTML = `<pre class="error">${response.message || "Request failed"}</pre>`;
    return;
  }
  currentArtifact = response.artifact;
  document.getElementById("score").textContent = response.critic.total_score;
  document.getElementById("status").textContent = response.status;
  document.getElementById("warnings").textContent = (response.warnings || []).join(", ") || "none";
  const stage = document.getElementById("stage");
  stage.innerHTML = response.svg;
  const svg = stage.querySelector("svg");
  if (svg) {
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", "100%");
  }
  buildButtons();
  setDownload("download-artifact", "artifact.json", JSON.stringify(currentArtifact, null, 2), "application/json");
  setDownload("download-svg", "schematic.svg", response.svg, "image/svg+xml");
}

function setDownload(id, filename, text, mimeType) {
  const link = document.getElementById(id);
  const blob = new Blob([text], {type: mimeType});
  link.href = URL.createObjectURL(blob);
  link.download = filename;
}

function buildButtons() {
  const zoomBox = document.getElementById("zoom-buttons");
  const focusBox = document.getElementById("focus-buttons");
  zoomBox.innerHTML = "";
  focusBox.innerHTML = "";
  (currentArtifact.zoom_presets || []).forEach(preset => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = preset.label || preset.id;
    button.addEventListener("click", () => setViewBox(preset.viewbox));
    zoomBox.appendChild(button);
  });
  (currentArtifact.focus_regions || []).forEach(region => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = region.label || region.id;
    button.addEventListener("click", () => focusRegion(region.id));
    focusBox.appendChild(button);
  });
}

function setViewBox(viewbox) {
  const svg = document.querySelector("#stage svg");
  if (!svg || !viewbox) return;
  svg.setAttribute("viewBox", `${viewbox.x} ${viewbox.y} ${viewbox.width} ${viewbox.height}`);
}

function clearHighlights() {
  document.querySelectorAll("#stage .highlight").forEach(element => element.classList.remove("highlight"));
}

function focusRegion(regionId) {
  if (!currentArtifact) return;
  const region = currentArtifact.focus_regions.find(item => item.id === regionId);
  const preset = currentArtifact.zoom_presets.find(item => item.id === `focus_${regionId}`);
  if (preset) setViewBox(preset.viewbox);
  if (!region) return;
  clearHighlights();
  const svg = document.querySelector("#stage svg");
  region.components.forEach(id => svg?.querySelector(`[data-component-id="${cssEscape(id)}"]`)?.classList.add("highlight"));
  region.nets.forEach(net => svg?.querySelectorAll(`[data-net-name="${cssEscape(net)}"]`).forEach(element => element.classList.add("highlight")));
  region.labels.forEach(id => svg?.querySelector(`[data-label-id="${cssEscape(id)}"]`)?.classList.add("highlight"));
}

async function generate() {
  document.getElementById("status").textContent = "working";
  let response;
  if (activeTab === "text") {
    response = await fetch("/v1/schematic/from-text", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt: document.getElementById("prompt").value})
    });
  } else if (activeTab === "json") {
    response = await fetch("/v1/schematic/from-ir", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({input_format: "auto", circuit: JSON.parse(document.getElementById("json-input").value)})
    });
  } else {
    const form = new FormData();
    const file = document.getElementById("image-file").files[0];
    if (file) form.append("file", file);
    form.append("prompt", document.getElementById("image-prompt").value);
    response = await fetch("/v1/schematic/from-image", {method: "POST", body: form});
  }
  const payload = await response.json();
  renderArtifact(payload);
}

document.getElementById("generate").addEventListener("click", () => {
  generate().catch(error => {
    document.getElementById("status").textContent = "error";
    document.getElementById("stage").innerHTML = `<pre class="error">${error.message}</pre>`;
  });
});
</script>
</body>
</html>
"""

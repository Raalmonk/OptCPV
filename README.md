# schem_forge

`schem_forge` turns circuit descriptions into beautiful, zoomable, interactive schematic artifacts for OptCPV/CiTT analog tutoring.

A user can provide:

- text describing a common demo circuit
- a CiTT-style `CircuitProblem` JSON payload
- schem_forge-native circuit IR
- an image, when the optional Gemini vision parser is configured

Core product contract: **SVG is the display layer; artifact JSON is the product interface.**

The API returns `SchematicArtifact` JSON with SVG plus structured metadata for zoom presets, focus regions, components, nets, hit targets, overlays, critic status, and provenance. Frontends should consume the artifact JSON instead of reverse-engineering circuit meaning from raw SVG.

## Run Locally

```bash
python3 -m pip install -e ".[dev]"
python3 -m uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## Generate From Text

No API key is required for built-in demo recognizers:

```bash
curl -s http://127.0.0.1:8000/v1/schematic/from-text \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Draw a non-inverting op amp"}'
```

Recognized local demos:

- `voltage divider`
- `RC low pass` / `low-pass`
- `non-inverting op amp`
- `instrumentation amplifier`

Unknown text prompts require `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Gemini is only used to produce structured circuit IR; schem_forge still renders the SVG/artifact deterministically.

For local debugging, you can put the key in `.env`; the API loads it automatically:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
```

## Generate From IR

schem_forge-native IR:

```bash
curl -s http://127.0.0.1:8000/v1/schematic/from-ir \
  -H "Content-Type: application/json" \
  -d '{
    "input_format": "auto",
    "circuit": {
      "id": "demo_divider",
      "motif": "voltage_divider",
      "components": [
        {"id":"VIN","type":"input","role":"input_source","display_label":"VIN","pins":{"out":"VIN"}},
        {"id":"R1","type":"resistor","role":"top_resistor","value_label":"R1","pins":{"a":"VIN","b":"VOUT"}},
        {"id":"R2","type":"resistor","role":"bottom_resistor","value_label":"R2","pins":{"a":"VOUT","b":"GND"}},
        {"id":"VOUT","type":"output","role":"output","display_label":"VOUT","pins":{"in":"VOUT"}},
        {"id":"GND","type":"ground","role":"ground","pins":{"gnd":"GND"}}
      ]
    }
  }'
```

CiTT-style payloads with `components[].nodes` are converted automatically through `circuit_problem_to_schem_forge_ir()`.

## Generate From Image

```bash
curl -s http://127.0.0.1:8000/v1/schematic/from-image \
  -F "file=@schematic.png" \
  -F "prompt=instrumentation amplifier"
```

Image parsing requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Without a key, the endpoint returns:

```json
{
  "status": "vision_backend_unavailable",
  "message": "Image-to-circuit parsing requires GEMINI_API_KEY or GOOGLE_API_KEY."
}
```

## Response Shape

Successful API responses include:

```json
{
  "status": "ok",
  "artifact": { "...": "SchematicArtifact JSON" },
  "svg": "<svg ...>",
  "critic": { "total_score": 0 },
  "warnings": [],
  "input_ir": { "...": "normalized schem_forge IR" }
}
```

Important artifact fields:

- `artifact.svg`
- `artifact.zoom_presets`
- `artifact.focus_regions`
- `artifact.components`
- `artifact.nets`
- `artifact.hit_targets`
- `artifact.critic_report`

## API Endpoints

- `GET /`
- `GET /health`
- `POST /v1/schematic/from-ir`
- `POST /v1/schematic/from-text`
- `POST /v1/schematic/from-image`

## Validation

```bash
python3 -m compileall backend/app scripts
python3 -m pytest -q
python3 scripts/test_schem_forge.py --all
python3 scripts/api_smoke_test.py
```

In the Codex sandbox, Python bytecode cache writes may need a writable prefix:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/schem_forge_pycache python3 -m compileall backend/app scripts
```

## Exported Artifacts

Visual QA exports:

```bash
python3 scripts/test_schem_forge.py --all
```

API smoke exports:

```bash
python3 scripts/api_smoke_test.py
```

Generated files are written under:

```text
backend/app/schem_forge/generated/
backend/app/schem_forge/generated/api_smoke/
```

## Architecture Notes

The core compiler still lives in `backend/app/schem_forge`:

- `generate_beautiful_schematic()`
- `SchematicArtifact`
- deterministic planners/renderers/critic/verifier
- `circuit_problem_to_schem_forge_ir()`

The API layer is deliberately thin. Text/image parsers may produce circuit IR only. They never produce SVG. Layout, SVG, zoom metadata, focus regions, and critic reports always come from schem_forge.

## Current Limitations

- Free-form text beyond the deterministic demos needs optional Gemini configuration.
- Image-to-circuit parsing needs optional Gemini configuration.
- Gemini integration is isolated behind `GeminiCircuitParser`; offline tests do not make network calls.
- The bridge planner is a conservative first pass.

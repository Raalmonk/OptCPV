# OptCPV

OptCPV is a CV-native circuit schematic drawing optimizer.

It is lightweight in product scope: it is a Python library, not a web product, tutor, arbitrary text parser, or image-to-circuit recognizer. But CV is core. OptCPV renders its own schematic, rasterizes that SVG into a fixed evaluation frame, inspects the pixels with OpenCV, and applies topology-safe layout patches until the drawing is cleaner.

Pipeline:

```text
Circuit
  -> LayoutPlan DSL
  -> Schemdraw renderer
  -> fixed-size raster image
  -> vector + OpenCV critic
  -> topology-safe patch
  -> optimized SVG + artifact + visual QA report
```

## Install

```bash
python -m pip install --upgrade pip
python -m pip install ".[dev]"
```

Core dependencies are mandatory:

```toml
dependencies = [
  "schemdraw>=0.19",
  "numpy>=1.24",
  "opencv-python-headless>=4.8",
  "pillow>=10",
  "cairosvg>=2.7",
]
```

Optional extras:

```toml
dev = ["pytest>=7"]
vision = ["google-genai>=1.0"]
```

`vision` is only for an optional patch-proposal client. The default optimizer works without it.

## Optional Gemini Feedback

The default loop uses local vector/OpenCV criticism plus `HeuristicVisionClient` for Gemini-shaped feedback without spending credits. To attach Gemini for selected cases, install the optional extra and pass an explicit client:

```python
from optcpv import GeminiVisionClient, draw_optimized_artifact

artifact = draw_optimized_artifact(
    circuit,
    vision_client=GeminiVisionClient(model="gemini-3.5-flash"),
)
```

The Gemini client sends the rendered schematic raster as PNG plus layout/topology metadata, and expects a topology-safe `LayoutPatch` JSON response. Patch validation still runs locally before any move is accepted.

## Public API

```python
from optcpv import Circuit, Component, draw_optimized_svg

circuit = Circuit(
    id="demo",
    motif="non_inverting_op_amp",
    components=[
        Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
        Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}),
        Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, label="Rf", role="feedback"),
        Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rg", role="gain"),
        Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        Component(id="VOUT", type="output", pins={"in": "vout"}, label="VOUT"),
    ],
)

svg = draw_optimized_svg(circuit)
```

Also available:

- `draw_svg(circuit)` for one-pass rendering through the same Schemdraw backend
- `draw_artifact(circuit)` for raw SVG plus critic reports
- `draw_optimized_artifact(circuit)` for optimized SVG plus QA metadata and optimization log
- `plan_layout(circuit)` for the LayoutPlan DSL

## What CV Means Here

CV means OptCPV inspects its own rendered output:

- dark-pixel density and local clutter
- visible label/wire/component collisions
- huge empty canvas and tiny scale hacks
- component density and compactness
- left-to-right balance and schematic conventions

CV does not mean OCR or recognizing arbitrary uploaded schematic images.

## Topology Safety

The optimizer may move components, labels, orientations, and wire points. It may not change component IDs, component types, pin names, nets, pin-to-net mappings, topology maps, or canvas size.

The topology verifier runs after deterministic planning, after every patch, and before artifact output.

## Fixed-Scale Evaluation

All visual scoring uses a fixed raster frame:

```python
EVAL_WIDTH = 1200
EVAL_HEIGHT = 800
```

Increasing SVG width/height, shrinking the viewBox, or spreading components apart does not reduce penalties. The critic penalizes low fill ratio, excessive whitespace, excessive spread, excessive wire length, and over-large viewboxes.

## Export Examples

```bash
python examples/export_examples.py
```

For each bundled example, this writes:

- `generated/<name>.raw.svg`
- `generated/<name>.optimized.svg`
- `generated/<name>.artifact.json`
- `generated/<name>.critic.json`

The command prints raw score to optimized score for each circuit.

## Boundaries

OptCPV core does not ship a FastAPI product surface, arbitrary text parsing, arbitrary image parsing, or CiTT tutor logic. Simple input adapters may live under `optcpv.adapters`.

## Validation

```bash
python -m compileall optcpv examples tests
python -m pytest -q
python examples/export_examples.py
```

## BME Analog Benchmark

The BME stress benchmark synthesizes source-inspired biomedical analog front-end cases, converts them into OptCPV `Circuit` IR, and evaluates raw, local-optimized, and Gemini-simulated visual-feedback paths.

The benchmark is intentionally outside the library core: cases are generated on demand, outputs stay under `generated/`, and no large dataset or downloaded image corpus is bundled with OptCPV.

```bash
python examples/bme_analog_200.py \
  --text-count 250 \
  --image-count 250 \
  --out-dir generated/bme_analog_500 \
  --fail-on-failure

python examples/bme_analog_200.py \
  --text-count 250 \
  --image-count 250 \
  --adversarial \
  --out-dir generated/bme_analog_500_adversarial \
  --fail-on-failure
```

Each run writes `cases.json`, `results.json`, and `summary.json`. The summary includes pass rates, score histograms, failure IDs, and clustered violation codes so regressions are actionable without spending Gemini credit.

Use `--start-index N` to run a shifted deterministic batch without changing the benchmark source lists. For example, `--start-index 500 --text-count 250 --image-count 250 --adversarial` creates the next 500 dirty-input variants.

Add `--contact-sheet --contact-sheet-count 12` when you want a local PNG review sheet of the worst or most representative optimized cases.

For quick CI or pre-commit checks, add `--local-only` to skip the second Gemini-sim optimization pass. Use the default mode for deeper local QA when you want the local critic and Gemini-sim feedback loops compared side by side.

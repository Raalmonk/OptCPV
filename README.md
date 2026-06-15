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
python -m pip install -e ".[dev]"
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

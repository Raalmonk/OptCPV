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

The default loop is local: deterministic semantic planning, vector/OpenCV criticism, and topology-safe patches. Gemini is opt-in and can participate at three boundaries:

- `GeminiPlanningClient` proposes semantic layout hints before the first render.
- The same planning client can receive the failed SVG, local critic report, and optional reference image after a local patch fails, then return refined stage/lane/route hints.
- `GeminiVisualReviewClient` can review the rendered raster and suggest topology-safe visual patches.
- `GeminiFigureSemanticClient` can perform the two-pass figure classification/check used before image-backed overlays. The local default uses the same prompt contract without spending quota.

To attach Gemini for selected cases, install the optional extra and pass an explicit client:

```python
from optcpv import GeminiPlanningClient, GeminiVisualReviewClient, draw_optimized_artifact

artifact = draw_optimized_artifact(
    circuit,
    planning_client=GeminiPlanningClient(model="gemini-3.5-flash"),
    visual_review_client=GeminiVisualReviewClient(model="gemini-3.5-flash"),
)
```

Or enable clients explicitly from the environment:

```bash
export OPTCPV_PLANNING_CLIENT=textbook
# or, for real Gemini:
export OPTCPV_USE_GEMINI_PLANNER=1
export OPTCPV_USE_GEMINI_VISUAL_REVIEW=1
export OPTCPV_USE_GEMINI_FIGURE_SEMANTICS=1
export GEMINI_API_KEY=...
```

`OPTCPV_PLANNING_CLIENT=gemini`, `OPTCPV_VISUAL_REVIEW_CLIENT=gemini`, and `OPTCPV_FIGURE_SEMANTIC_CLIENT=gemini` are equivalent opt-in switches. `OPTCPV_USE_HEURISTIC_VISUAL_REVIEW=1` enables the local Gemini-shaped reviewer without network calls.

`OPTCPV_PLANNING_CLIENT=textbook` enables the local two-layer Gemini surrogate:

1. `TextbookFigureInterpreter` reads the extracted textbook corpus and compresses figures into structured visual grammar cards.
2. `TextbookStructurePlanner` retrieves relevant cards for a circuit and emits legal `SchematicLayoutHints`.

This is the preferred low-quota path. It lets OptCPV use the textbook corpus without asking Gemini to inspect hundreds of images each run.

Gemini responses are never trusted blindly. Planning hints are legalized against the existing topology, visual patches are verified locally, and no client may create/delete/rename/rewire components, pins, or nets.

## Textbook Corpus Extraction

When a local textbook PDF is present in the repository root, extract figure crops, problem statements, and category indexes with:

```bash
python tools/extract_textbook_corpus.py --dpi 180 --out textbook_circuit_corpus
```

The generated corpus contains:

- `figures/figure_*/crop.png` and `metadata.json`
- `problems/chapter_*.jsonl`
- `indexes/figures.jsonl`
- `indexes/likely_circuit_figures.jsonl`
- `indexes/problems.jsonl`
- `indexes/circuit_or_design_problems.jsonl`
- `classified/` category folders with browsable image links for circuit-like figures
- `structured_text/figure_cards.jsonl`
- `structured_text/figure_cards.txt`
- `structured_text/style_guide.json`
- `structured_text/by_family/*.txt`

The extractor uses the textbook caption font, nearby vector/image graphics, and chapter-scoped problem IDs so body references such as `Figure ... shows` and numeric values such as `3.3 V` do not become false figures/problems.

Build or refresh the structured cards with:

```bash
python tools/build_textbook_structured_cards.py --corpus textbook_circuit_corpus
```

These cards are deliberately text-first. They summarize figure family, component cues, layout principles, route principles, image metrics, and source crop paths so a real Gemini call can receive a compact textbook memory instead of raw bulk images.

To run a local smoke check of that communication layer without Gemini quota:

```bash
python tools/run_textbook_surrogate_smoke.py
```

It writes SVG/PNG artifacts, summaries, and the exact `GEMINI_MIDDLE_LAYER` structured text to `generated/textbook_surrogate_smoke/`.

To run the full textbook batch over every extracted figure card:

```bash
python tools/run_textbook_corpus_batch.py --out generated/textbook_corpus_batch
```

That batch validates every crop and structured card, then renders every likely circuit figure through a card-scoped textbook surrogate fixture. It writes `summary.json`, `results.jsonl`, and per-card middle-layer prompts under `generated/textbook_corpus_batch/`.

## Image-Backed Interactive Overlays

For problem statements that already include a textbook figure, OptCPV can keep the original image as the source of truth and add transparent interactive SVG hit targets on top:

```python
from optcpv import analyze_image_overlay, render_image_overlay_svg

plan = analyze_image_overlay("textbook_circuit_corpus/figures/figure_1.9_p065/crop.png")
svg = render_image_overlay_svg(plan)
```

The local low-quota Gemini surrogate has two semantic layers before CV overlay:

1. `FIGURE_SEMANTIC_DRAFT`: generate basic figure grammar, identify the input/figure kind, describe plot axes/quantity when applicable, and apply circuit grammar only to true schematics.
2. `FIGURE_SEMANTIC_CHECK`: reject contradictions before overlay. Plots/waveforms/photos are skipped; block diagrams only create functional block buttons, never R/C/L/op-amp symbol buttons.

Only after that gate does CV run:

1. `IMAGE_GRAPH_DRAFT`: extract wire runs, node candidates, and component or block regions from the allowed image regions.
2. `IMAGE_OVERLAY_PLAN`: snap those primitives into highlightable wires and `role=button` regions while preserving the source image.

Rendered overlay SVGs expose `window.optcpvHighlightWires([...])`, `window.optcpvClearHighlight()`, wire `data-wire-id` attributes, and component `data-component-id`/`data-wire-ids` attributes. Clicking a component button highlights its connected wires.

Run the full textbook overlay audit with:

```bash
python tools/run_textbook_image_overlay_batch.py --out generated/textbook_image_overlay_batch
```

The batch writes per-figure SVG/JSON artifacts, both overlay middle-layer text files, `results.jsonl`, `summary.json`, and visual contact sheets. For every figure it also writes auditable Gemini-shaped communication files under `middle_layers/`: `*.gemini_layer1_input.txt`, `*.gemini_layer1_output.json`, `*.gemini_layer2_input.txt`, and `*.gemini_layer2_output.json`. It reports card-level expectations separately from the semantic/visual classifier, so mislabeled textbook crops such as waveforms or anatomy photos are surfaced as semantic rejections instead of being forced into circuit overlays.

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

Artifacts include explicit layout capability metadata:

```python
artifact = draw_optimized_artifact(circuit)
artifact.layout_support
# {
#   "layout_mode": "native_motif",
#   "layout_confidence": 0.95,
#   "matched_motifs": ["instrumentation_amplifier"],
#   "fallback_used": False,
#   "unsupported_regions": [],
#   "notes": [...]
# }
```

The same information is also written on the SVG root as `data-optcpv-layout-mode`,
`data-optcpv-layout-confidence`, `data-optcpv-matched-motifs`,
`data-optcpv-fallback-used`, and `data-optcpv-unsupported-regions`.

## Layout Support Contract

OptCPV is currently a known-motif canonicalizer plus a Schemdraw/native motif renderer, vector/CV critic, and topology-safe patch loop. It does not promise arbitrary circuit topology to textbook schematic conversion.

Layout modes are explicit:

- `native_motif`: a recognized motif with a native Schemdraw rendering path, currently voltage divider, RC low-pass, non-inverting op amp, instrumentation amplifier, and bridge/Wheatstone.
- `motif_network`: heuristic multi-op-amp network placement and routing. This is motif-aware and useful for composed analog front ends, but it is not a textbook-layout guarantee.
- `partial_motif`: a known motif was matched, but one or more components required generic fallback placement. Read `unsupported_regions` before treating the drawing as complete.
- `diagnostic_fallback`: no known motif matched. OptCPV still returns a topology-preserving diagnostic schematic, but `layout_confidence` is low and `fallback_used` is true.

## What CV Means Here

CV means OptCPV inspects its own rendered output:

- dark-pixel density and local clutter
- visible label/wire/component collisions
- huge empty canvas and tiny scale hacks
- component density and compactness
- left-to-right balance and schematic conventions

For existing figure images, CV can produce image-backed interaction overlays. That is not the same as recovering a complete electrical netlist; the original image remains the visual source of truth, and OptCPV adds nodes, wires, component buttons, and highlight metadata for interaction.

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

CiTT-style adapters are boundary converters, not proof that arbitrary CiTT circuits will render like textbook figures. Unknown or weakly supported topologies are surfaced through `layout_support` instead of being silently presented as fully supported motif output.

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

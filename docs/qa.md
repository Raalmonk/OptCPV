# schem_forge QA

## How To Run QA

```bash
python3 -m pytest -q
python3 scripts/test_schem_forge.py --all
python3 scripts/test_schem_forge.py --suite citt
python3 -m compileall backend/app/schem_forge scripts
```

In sandboxed macOS environments where Python cannot write bytecode under `~/Library/Caches`, use:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/schem_forge_pycache python3 -m compileall backend/app/schem_forge scripts
```

## Test Coverage Map

- `test_models_renderer.py`: LayoutPlan roundtrip, SVG XML validity, renderer metadata.
- `test_critic_verifier_agent.py`: critic defect detection, topology verifier rejection, restricted patch safety, agent best-layout retention.
- `test_adapters.py`: dict-like and realistic CiTT-style payloads, topology_id mapping, node-voltage goals, op-amp pin mapping, virtual terminal creation.
- `test_artifact.py`: artifact serialization, component/net/focus/zoom/hit basics, debug HTML generation.
- `test_artifact_invariants.py`: per-case artifact contract invariants and layout/artifact/SVG consistency.
- `test_quality_gates.py`: fatal-free visual quality thresholds for built-in motifs.
- `test_citt_visual_exports.py`: CiTT payload fixtures, adapter realism, motif planning, visual gates, and export artifacts.

## Artifact Contract Invariants

For every generated built-in and CiTT artifact:

- artifact JSON must serialize with `json.dumps`.
- artifact SVG must parse as XML.
- SVG root `viewBox` must equal `artifact.svg_viewbox`.
- every artifact component must exist in SVG `data-component-id`.
- every segmented net must exist in SVG `data-net-name`.
- every artifact pin must exist in SVG `data-pin-ref`.
- every label must exist in SVG `data-label-id`.
- all zoom preset viewboxes must be positive and inside the root viewBox.
- focus regions, overlays, and hit targets may reference only existing artifact ids.
- every focus region must have a matching `focus_<region_id>` zoom preset.
- every component must have a matching `component_<id>` zoom preset.
- layout topology, artifact pins/nets, and SVG metadata must agree.

## Current Score Thresholds

After `generate_beautiful_schematic(..., MockLLMClient())`:

- `voltage_divider`: score <= 100, fatal_count == 0
- `rc_low_pass`: score <= 150, fatal_count == 0
- `non_inverting_op_amp`: score <= 200, fatal_count == 0
- `instrumentation_amp`: score <= 300, fatal_count == 0
- `citt_voltage_divider`: score <= 100, fatal_count == 0
- `citt_rc_low_pass`: score <= 150, fatal_count == 0
- `citt_non_inverting_op_amp`: score <= 200, fatal_count == 0
- `citt_bme_instrumentation_amplifier`: score <= 300, fatal_count == 0

All built-ins and CiTT fixtures must have no component overlaps, no `wire_crosses_component_body` violations, and no `wire_crossing` violations.

## Known Visual Issues

None currently tracked. Instrumentation amplifier and the BME CiTT instrumentation payload are expected to be zero-penalty.

## Debug Viewer

Each case writes:

```text
backend/app/schem_forge/generated/<case>/debug.html
backend/app/schem_forge/generated/visual_review.html
```

The per-case viewer embeds artifact JSON inline, injects `artifact.svg`, creates zoom/focus buttons with DOM event listeners, and highlights components/nets/labels by SVG data attributes. The visual review page shows every exported case with score, fatal count, links, and remaining violations. It intentionally avoids inline `onclick` strings so unusual ids cannot break JavaScript.

## Adding New Circuit Cases

1. Add a fixture builder in `backend/app/schem_forge/examples.py` or a CiTT payload in `backend/app/schem_forge/citt_examples.py`.
2. Add or update a canonical planner if the motif is common.
3. Add a score threshold in `scripts/test_schem_forge.py` and `tests/test_quality_gates.py`.
4. Run `python3 scripts/test_schem_forge.py --all`.
5. Inspect `debug.html`, `visual_review.html`, and `qa_summary.json`.
6. Add focused adapter fixtures if the case comes from CiTT data.

## Before CiTT Integration

Must pass:

- pytest
- artifact contract invariants
- fatal-free built-in cases
- realistic CiTT adapter tests
- fatal-free, crossing-free CiTT visual gates
- `scripts/test_schem_forge.py --all`
- manual inspection of generated `debug.html` and `visual_review.html`

CiTT should consume `generate_beautiful_schematic(circuit).artifact`, not raw SVG alone. SVG is the display layer; artifact JSON is the product interface.

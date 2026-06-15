# schem_forge

`schem_forge` is a lightweight circuit-to-schematic compiler for OptCPV/CiTT analog tutoring diagrams. It is not a generic graph drawing tool: its first job is to recognize common circuit motifs and produce textbook-style schematics with deterministic, topology-safe planning.

Core product contract: **SVG 是展示层，artifact JSON 才是产品接口。**

`generate_beautiful_schematic(circuit).artifact` returns a `SchematicArtifact` containing the display SVG plus structured metadata for zooming, focus, hit-testing, overlays, critic status, and provenance. The frontend should not parse raw SVG to rediscover circuit meaning; it should consume the artifact JSON.

## Why Not Generic Graph Layout

Generic graph layout treats components as nodes and electrical nets as edges. That is the wrong abstraction for circuit tutoring diagrams. A good analog schematic needs semantic conventions: inputs on the left, outputs on the right, feedback loops above op-amps, ground at the bottom, symmetric instrumentation-amplifier stages, and clear orthogonal routing. `schem_forge` starts with canonical motif planners and only uses an LLM-style patch loop as a constrained polishing layer.

## Layout DSL

The core DSL lives in `backend/app/schem_forge/models.py` and separates:

- electrical connectivity: `net_to_pins`, `component_pin_nets`, `topology_signature`
- visual placement: component grid coordinates and orientation
- pin anchors: per-pin sides and offsets
- wire routing: route waypoints and rendered segments
- labels: owner-aware text positions
- critic geometry: rendered bboxes, pins, junctions, wire segments
- renderer metadata: deterministic SVG renderer id

## SchematicArtifact

`backend/app/schem_forge/artifact.py` builds the tutor-facing artifact from the verified `LayoutPlan`, deterministic `RenderResult`, and `CriticReport`.

The artifact includes:

- `svg`: the display layer
- `components`: bbox, labels, and pin artifacts for every component
- `nets`: connected pins, wire segments, junctions, and net bboxes
- `labels`: owner-aware label bboxes
- `focus_regions`: semantic regions such as `differential_stage` or `feedback_network`
- `zoom_presets`: `fit_all`, one preset per focus region, component presets, and major net presets
- `hit_targets`: component, net, pin, label, and focus-region bboxes for probing
- `overlays`: optional semantic highlight groups
- `critic_report` and `provenance`

Focus regions let a lesson step sync directly to schematic meaning. For example, an instrumentation-amplifier lesson can zoom to `focus_differential_stage`, highlight its op-amp and nets, then attach probes to hit targets without searching the SVG DOM for topology.

## Topology Safety

The agent may only patch visual fields: component position, orientation, label position, and wire waypoints. It may not mutate component ids, component types, pins, net names, topology maps, or signatures. After every patch, `verify_equivalence()` compares the layout against the original circuit IR and raises `ElectricalTopologyError` on drift.

This is the guardrail that prevents LLM electrical hallucination: a visual patch can move a resistor, but it cannot silently reconnect it.

## Local Setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 scripts/test_schem_forge.py --all
```

In the Codex sandbox, `compileall` may need Python's bytecode cache redirected:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/schem_forge_pycache python3 -m compileall backend/app/schem_forge scripts
```

On a normal local checkout, this should work directly:

```bash
python3 -m compileall backend/app/schem_forge scripts
```

## Export SVGs

```bash
python3 scripts/test_schem_forge.py --case instrumentation_amp
python3 scripts/test_schem_forge.py --case voltage_divider
python3 scripts/test_schem_forge.py --case rc_low_pass
python3 scripts/test_schem_forge.py --case non_inverting_op_amp
python3 scripts/test_schem_forge.py --all
```

Generated files are written under `backend/app/schem_forge/generated/<case>/` and are gitignored:

- `before.svg`
- `after.svg`
- `before_plan.json`
- `after_plan.json`
- `before_artifact.json`
- `after_artifact.json`
- `critic_report.json`
- `debug.html`

`debug.html` is a standalone local viewer with buttons for zoom presets and focus regions.

## Frontend Usage

```js
diagramPanel.innerHTML = artifact.svg;
```

```js
function zoomToPreset(artifact, presetId) {
  const preset = artifact.zoom_presets.find(p => p.id === presetId);
  svg.setAttribute(
    "viewBox",
    `${preset.viewbox.x} ${preset.viewbox.y} ${preset.viewbox.width} ${preset.viewbox.height}`
  );
}

function highlightFocusRegion(artifact, regionId) {
  const region = artifact.focus_regions.find(r => r.id === regionId);
  region.components.forEach(id =>
    svg.querySelector(`[data-component-id="${id}"]`)?.classList.add("highlight")
  );
  region.nets.forEach(net =>
    svg.querySelectorAll(`[data-net-name="${net}"]`).forEach(el => el.classList.add("highlight"))
  );
}
```

Lesson step example:

```js
zoomToPreset(artifact, "focus_differential_stage");
highlightFocusRegion(artifact, "differential_stage");
```

Hit targets support component/net probes by using artifact bboxes directly instead of reverse-engineering layout from raw SVG.

## Supported Motifs

- instrumentation amplifier
- non-inverting op-amp
- RC low-pass filter
- voltage divider
- bridge / Wheatstone-style circuits
- simple grid fallback for unknown circuits

## Current Limitations

- Instrumentation-amplifier output is topology-safe and fatal-free, but still has one scored crossing between the top input and gain-node route.
- The bridge planner is a conservative first pass.
- The Gemini client is intentionally a placeholder; local quality is being improved before network-backed visual polishing is wired in.
- The CiTT adapter handles common node-list payloads, but real integration should add fixtures from production CircuitProblem objects.

## Next CiTT Steps

1. Feed real CiTT `CircuitProblem` examples through `circuit_problem_to_schem_forge_ir()`.
2. Compare generated SVGs against the current graph-based diagrams.
3. Add motif-specific planners for any frequent CiTT patterns that fall back to grid layout.
4. Wire `compile_schematic_for_generator()` behind a feature flag in `schematic_generator.py`.

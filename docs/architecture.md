# schem_forge Architecture

## Circuit IR And Adapter

`adapters.py` converts CiTT-style payloads into a small schem_forge IR:

- components have stable `id`, `type`, optional labels/roles, and `pins`
- node-list components are mapped into named pins
- terminal and ground symbols are added only on existing nets
- the adapter does not mutate the source payload

## Planner

`planner.py` is the main quality layer. It detects known motifs and creates deterministic canonical layouts before any agent loop runs. This avoids the trap of starting from a crammed generic graph and hoping an LLM fixes it later.

Current planners include instrumentation amplifier, non-inverting op-amp, RC low-pass, voltage divider, bridge/Wheatstone, and a simple grid fallback.

## Renderer

`renderer.py` is intentionally dumb. It obeys `LayoutPlan` coordinates, draws SVG primitives, and emits metadata such as:

- `data-schem-forge-renderer`
- `data-component-id`
- `data-pin-ref`
- `data-net-name`
- rendered component and label bboxes
- pin, wire, and junction geometry

It does not invent auto-layout.

## Critic

`critic.py` scores rendered geometry rather than abstract connectivity. It checks overlap, body crossings, label conflicts, diagonal wires, signal-flow conventions, ground placement, feedback placement, and wire crossings.

Reports include total score, fatal/warning counts, score breakdown, component bbox summaries, wire segment count, suggestions, and safe visual patch suggestions where possible.

## Verifier

`verifier.py` is the electrical safety layer. It hashes a canonical topology payload and checks:

- same component ids
- same component types
- same pin-to-net mapping
- same net membership
- no missing or fake pins
- no disconnected nets
- unchanged stored topology signature

Invalid layouts raise `ElectricalTopologyError`.

## Agent Loop

`agent.py` applies restricted visual patches only. `MockLLMClient` is an offline deterministic stand-in that can move labels, fix orientations, move outputs rightward, and apply known route patches. Every candidate layout is verified before it can become the best result.

The loop keeps the best valid layout and records debug info for each iteration.

## Future Gemini Integration

`GeminiLLMClient` is a placeholder by design. The intended flow is:

1. deterministic planner creates a good canonical schematic
2. renderer produces SVG and geometry metadata
3. critic emits precise violations and suggested visual patches
4. Gemini proposes restricted patch operations only
5. verifier rejects any topology drift
6. the best verified layout is returned

Gemini should polish diagrams, not discover circuit topology.

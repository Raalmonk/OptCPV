# OptCPV

OptCPV is a lightweight Python circuit schematic drawing library.

It takes a small native circuit description, or a simple adapter-produced circuit, and returns a clean SVG schematic. When useful, it can also return lightweight artifact metadata for components, nets, the viewbox, and warnings.

OptCPV core does not parse arbitrary text, parse images, call Gemini, run a tutor workflow, or ship an HTTP API as the core product. Those concerns belong outside the drawing library.

## Install

```bash
python -m pip install -e ".[dev]"
```

Core dependencies are intentionally empty:

```toml
dependencies = []
```

Optional extras are available for development, small demos, or future renderer experiments:

```toml
dev = ["pytest>=7"]
server = ["fastapi>=0.100", "uvicorn>=0.23"]
schemdraw = ["schemdraw>=0.19"]
```

## Public API

```python
from optcpv import Circuit, Component, draw_artifact, draw_svg

circuit = Circuit(
    id="demo",
    motif="non_inverting_op_amp",
    components=[
        Component(id="VIN", type="input", pins={"out": "vin"}, label="VIN"),
        Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}),
        Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, label="Rf"),
        Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rg"),
        Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        Component(id="VOUT", type="output", pins={"in": "vout"}, label="VOUT"),
    ],
)

svg = draw_svg(circuit)
artifact = draw_artifact(circuit)
```

`draw_svg()` also accepts a plain dictionary with the same shape.

## Supported Motifs

The deterministic planner currently has direct layouts for:

- `voltage_divider`
- `rc_low_pass`
- `non_inverting_op_amp`
- `instrumentation_amplifier`
- `bridge_or_wheatstone`

Unknown circuits fall back to a conservative diagnostic left-to-right layout. The fallback is not presented as a polished student-facing diagram.

## Adapter Boundary

CiTT-style payload conversion lives outside the core API:

```python
from optcpv.adapters.citt import from_citt_payload

circuit = from_citt_payload(payload)
svg = draw_svg(circuit)
```

The adapter only converts simple `components[].nodes`, `ground_node`, and `goals` fields into native OptCPV IR. It does not implement CiTT product behavior.

## Export Examples

```bash
python examples/export_examples.py
```

This writes SVG files to `generated/`, which is gitignored.

## Validation

```bash
python -m compileall optcpv examples tests
python -m pytest -q
python examples/export_examples.py
```

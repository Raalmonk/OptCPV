"""Public drawing entry points for OptCPV."""

from __future__ import annotations

from .models import Circuit, Layout, SchematicArtifact
from .planner import plan_layout
from .renderer import render_svg


def draw_svg(circuit: Circuit | dict, *, style: str = "textbook") -> str:
    """Return an SVG schematic for a native OptCPV circuit description."""

    return render_svg(plan_layout(circuit), style=style)


def draw_artifact(circuit: Circuit | dict, *, style: str = "textbook") -> SchematicArtifact:
    """Return SVG plus lightweight component and net metadata."""

    layout = plan_layout(circuit)
    svg = render_svg(layout, style=style)
    return SchematicArtifact(
        svg=svg,
        components=_component_metadata(layout),
        nets=_net_metadata(layout),
        viewbox={"x": 0, "y": 0, "width": layout.width, "height": layout.height},
        warnings=list(layout.warnings),
    )


def _component_metadata(layout: Layout) -> dict[str, dict]:
    return {
        component.id: {
            "id": component.id,
            "type": component.type,
            "label": component.label or component.value or component.id,
            "pins": dict(component.pins),
            "position": {"x": component.x, "y": component.y},
            "orientation": component.orientation,
        }
        for component in layout.components
    }


def _net_metadata(layout: Layout) -> dict[str, dict]:
    return {
        wire.net: {
            "name": wire.net,
            "points": [{"x": x, "y": y} for x, y in wire.points],
        }
        for wire in layout.wires
    }

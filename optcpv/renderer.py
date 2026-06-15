"""Minimal custom SVG renderer for OptCPV layouts."""

from __future__ import annotations

from html import escape

from .models import Layout, LayoutComponent
from .planner import GRID, _pin_point


def render_svg(layout: Layout, *, style: str = "textbook") -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {layout.width} {layout.height}" '
        f'width="{layout.width}" height="{layout.height}" data-renderer="optcpv.svg" data-style="{escape(style)}">',
        "<defs>",
        "<style>",
        ".wire{fill:none;stroke:#111827;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}",
        ".component{fill:#fffaf0;stroke:#111827;stroke-width:2}",
        ".symbol{fill:none;stroke:#111827;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}",
        ".pin{fill:#111827}",
        ".label{font:14px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#111827}",
        ".terminal-label{font:12px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#374151}",
        "</style>",
        "</defs>",
    ]
    for wire in layout.wires:
        parts.append(_draw_wire(wire.net, wire.points))
    for component in layout.components:
        parts.append(_draw_component(component))
    parts.append("</svg>")
    return "\n".join(parts)


def _draw_wire(net: str, points: list[tuple[float, float]]) -> str:
    px_points = " ".join(f"{x * GRID:.1f},{y * GRID:.1f}" for x, y in points)
    return f'<polyline class="wire" points="{px_points}" data-net-name="{escape(net)}"/>'


def _draw_component(component: LayoutComponent) -> str:
    x, y = component.x * GRID, component.y * GRID
    key = _key(component.type)
    inner = _draw_default(component, x, y)
    if "op_amp" in key or "opamp" in key or "operational_amplifier" in key:
        inner = _draw_opamp(component, x, y)
    elif "resistor" in key or key.startswith("r"):
        inner = _draw_resistor(component, x, y)
    elif "capacitor" in key or key.startswith("c"):
        inner = _draw_capacitor(component, x, y)
    elif key in {"ground", "gnd"}:
        inner = _draw_ground(component, x, y)
    elif key in {"input", "output", "voltage_source", "source", "input_terminal"} or "source" in key:
        inner = _draw_terminal(component, x, y)

    pins = []
    for pin_name, net in component.pins.items():
        px, py = _pin_point(component, pin_name)
        pins.append(
            f'<circle class="pin" cx="{px * GRID:.1f}" cy="{py * GRID:.1f}" r="2.7" '
            f'data-component-id="{escape(component.id)}" data-pin-name="{escape(pin_name)}" '
            f'data-net-name="{escape(net)}"/>'
        )

    label = escape(component.label or component.value or component.id)
    return (
        f'<g data-component-id="{escape(component.id)}" data-component-type="{escape(component.type)}">\n'
        f"{inner}\n"
        f"{''.join(pins)}\n"
        f'<text class="label" x="{x:.1f}" y="{y - 34:.1f}">{label}</text>\n'
        "</g>"
    )


def _draw_opamp(component: LayoutComponent, x: float, y: float) -> str:
    points = [
        (x - 58, y - 44),
        (x - 58, y + 44),
        (x + 72, y),
    ]
    polygon = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    return "\n".join(
        [
            f'<polygon class="component" points="{polygon}"/>',
            f'<text class="terminal-label" x="{x - 44:.1f}" y="{y - 22:.1f}">-</text>',
            f'<text class="terminal-label" x="{x - 44:.1f}" y="{y + 30:.1f}">+</text>',
        ]
    )


def _draw_resistor(component: LayoutComponent, x: float, y: float) -> str:
    if component.orientation in {"up", "down"}:
        return (
            f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 24:.1f}"/>'
            f'<rect class="component" x="{x - 12:.1f}" y="{y - 24:.1f}" width="24" height="48" rx="3"/>'
            f'<line class="symbol" x1="{x:.1f}" y1="{y + 24:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>'
        )
    return (
        f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 24:.1f}" y2="{y:.1f}"/>'
        f'<rect class="component" x="{x - 24:.1f}" y="{y - 12:.1f}" width="48" height="24" rx="3"/>'
        f'<line class="symbol" x1="{x + 24:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>'
    )


def _draw_capacitor(component: LayoutComponent, x: float, y: float) -> str:
    if component.orientation in {"up", "down"}:
        return "\n".join(
            [
                f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 10:.1f}"/>',
                f'<line class="symbol" x1="{x - 18:.1f}" y1="{y - 10:.1f}" x2="{x + 18:.1f}" y2="{y - 10:.1f}"/>',
                f'<line class="symbol" x1="{x - 18:.1f}" y1="{y + 10:.1f}" x2="{x + 18:.1f}" y2="{y + 10:.1f}"/>',
                f'<line class="symbol" x1="{x:.1f}" y1="{y + 10:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>',
            ]
        )
    return "\n".join(
        [
            f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 10:.1f}" y2="{y:.1f}"/>',
            f'<line class="symbol" x1="{x - 10:.1f}" y1="{y - 18:.1f}" x2="{x - 10:.1f}" y2="{y + 18:.1f}"/>',
            f'<line class="symbol" x1="{x + 10:.1f}" y1="{y - 18:.1f}" x2="{x + 10:.1f}" y2="{y + 18:.1f}"/>',
            f'<line class="symbol" x1="{x + 10:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>',
        ]
    )


def _draw_ground(component: LayoutComponent, x: float, y: float) -> str:
    return "\n".join(
        [
            f'<line class="symbol" x1="{x:.1f}" y1="{y - 34:.1f}" x2="{x:.1f}" y2="{y - 12:.1f}"/>',
            f'<line class="symbol" x1="{x - 28:.1f}" y1="{y - 12:.1f}" x2="{x + 28:.1f}" y2="{y - 12:.1f}"/>',
            f'<line class="symbol" x1="{x - 18:.1f}" y1="{y:.1f}" x2="{x + 18:.1f}" y2="{y:.1f}"/>',
            f'<line class="symbol" x1="{x - 8:.1f}" y1="{y + 12:.1f}" x2="{x + 8:.1f}" y2="{y + 12:.1f}"/>',
        ]
    )


def _draw_terminal(component: LayoutComponent, x: float, y: float) -> str:
    return (
        f'<circle class="component" cx="{x:.1f}" cy="{y:.1f}" r="18"/>'
        f'<text class="terminal-label" x="{x:.1f}" y="{y + 4:.1f}">{escape(component.label or component.id)}</text>'
    )


def _draw_default(component: LayoutComponent, x: float, y: float) -> str:
    return f'<rect class="component" x="{x - 28:.1f}" y="{y - 18:.1f}" width="56" height="36" rx="3"/>'


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

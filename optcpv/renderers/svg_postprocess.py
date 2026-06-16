"""SVG metadata injection and debug fallback rendering."""

from __future__ import annotations

from html import escape
from typing import Literal

from ..models import LayoutComponent, LayoutLabel, LayoutPlan, Point


LayerName = Literal["wires", "components", "labels"]


def inject_metadata(svg: str, layout: LayoutPlan, *, renderer: str) -> str:
    patched = _set_root_attr(svg, "data-renderer", renderer)
    patched = _set_root_attr(patched, "data-optcpv-circuit-id", layout.circuit_id)
    overlay = _metadata_overlay(layout)
    if "</svg>" in patched:
        return patched.replace("</svg>", overlay + "\n</svg>")
    return patched + overlay


def render_debug_svg(layout: LayoutPlan, *, renderer: str = "optcpv.raw_svg", style: str = "textbook") -> str:
    return _render_layout_svg(
        layout,
        renderer=renderer,
        style=style,
        layers=("wires", "components", "labels"),
        mask_layer=None,
    )


def render_layer_svg(layout: LayoutPlan, layer: LayerName, *, style: str = "textbook") -> str:
    return _render_layout_svg(
        layout,
        renderer=f"optcpv.layer.{layer}",
        style=style,
        layers=(layer,),
        mask_layer=layer,
    )


def _render_layout_svg(
    layout: LayoutPlan,
    *,
    renderer: str,
    style: str,
    layers: tuple[LayerName, ...],
    mask_layer: LayerName | None,
) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {layout.width} {layout.height}" '
        f'width="{layout.width}" height="{layout.height}" data-renderer="{escape(renderer)}" data-style="{escape(style)}">',
        "<defs>",
        "<style>",
        *_style_rules(mask_layer),
        "</style>",
        "</defs>",
    ]
    if "wires" in layers:
        for wire in layout.wires:
            parts.append(_draw_wire(layout, wire.net, wire.points))
    if "components" in layers:
        for component in layout.components:
            parts.append(_draw_component_symbol(layout, component))
    if "labels" in layers:
        for label in layout.labels:
            parts.append(_draw_label(layout, label))
    parts.append("</svg>")
    return "\n".join(parts)


def _style_rules(mask_layer: LayerName | None) -> list[str]:
    if mask_layer == "components":
        return [
            ".wire{fill:none;stroke:#111827;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}",
            ".component{fill:#111827;stroke:#111827;stroke-width:2}",
            ".symbol{fill:none;stroke:#111827;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}",
            ".pin{fill:#111827}",
            ".label{font:14px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#111827}",
            ".terminal-label{font:12px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#111827}",
        ]
    return [
        ".wire{fill:none;stroke:#111827;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}",
        ".component{fill:#fffaf0;stroke:#111827;stroke-width:2}",
        ".symbol{fill:none;stroke:#111827;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}",
        ".pin{fill:#111827}",
        ".label{font:14px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#111827}",
        ".terminal-label{font:12px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#374151}",
    ]


def _metadata_overlay(layout: LayoutPlan) -> str:
    parts = ['<g id="optcpv-metadata" opacity="0" pointer-events="none">']
    for wire in layout.wires:
        points = " ".join(_px(layout, point) for point in wire.points)
        parts.append(f'<polyline points="{points}" data-net-name="{escape(wire.net)}"/>')
    for key, pin in layout.pin_map.items():
        component_id, pin_name = key
        parts.append(
            f'<circle cx="{pin.x * layout.grid:.1f}" cy="{pin.y * layout.grid:.1f}" r="3" '
            f'data-component-id="{escape(component_id)}" data-pin-name="{escape(pin_name)}" '
            f'data-net-name="{escape(pin.net)}"/>'
        )
    for component in layout.components:
        parts.append(
            f'<rect x="{component.bbox.x * layout.grid:.1f}" y="{component.bbox.y * layout.grid:.1f}" '
            f'width="{component.bbox.width * layout.grid:.1f}" height="{component.bbox.height * layout.grid:.1f}" '
            f'data-component-id="{escape(component.id)}" data-component-type="{escape(component.type)}"/>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def _draw_wire(layout: LayoutPlan, net: str, points: list[Point]) -> str:
    px_points = " ".join(_px(layout, point) for point in points)
    return f'<polyline class="wire" points="{px_points}" data-net-name="{escape(net)}"/>'


def _draw_component_symbol(layout: LayoutPlan, component: LayoutComponent) -> str:
    x, y = component.x * layout.grid, component.y * layout.grid
    key = _key(component.type)
    inner = _draw_default(x, y)
    if "op_amp" in key or "opamp" in key or "operational_amplifier" in key:
        inner = _draw_opamp(x, y)
    elif "resistor" in key or key.startswith("r"):
        inner = _draw_resistor(component, x, y)
    elif "capacitor" in key or key.startswith("c"):
        inner = _draw_capacitor(component, x, y)
    elif key in {"ground", "gnd"}:
        inner = _draw_ground(x, y)
    elif key in {"input", "output", "voltage_source", "source", "input_terminal"} or "source" in key:
        inner = _draw_terminal(component, x, y)

    pins = []
    for pin_name, net in component.pins.items():
        pin = layout.pin_map[(component.id, pin_name)]
        pins.append(
            f'<circle class="pin" cx="{pin.x * layout.grid:.1f}" cy="{pin.y * layout.grid:.1f}" r="2.7" '
            f'data-component-id="{escape(component.id)}" data-pin-name="{escape(pin_name)}" '
            f'data-net-name="{escape(net)}"/>'
        )

    return (
        f'<g data-component-id="{escape(component.id)}" data-component-type="{escape(component.type)}">\n'
        f"{inner}\n"
        f"{''.join(pins)}\n"
        "</g>"
    )


def _draw_label(layout: LayoutPlan, label: LayoutLabel) -> str:
    return (
        f'<text class="label" x="{label.x * layout.grid:.1f}" y="{label.y * layout.grid:.1f}" '
        f'data-label-id="{escape(label.id)}" data-label-owner-id="{escape(label.owner_id)}">'
        f"{escape(label.text)}</text>"
    )


def _draw_opamp(x: float, y: float) -> str:
    polygon = " ".join(f"{px:.1f},{py:.1f}" for px, py in [(x - 58, y - 44), (x - 58, y + 44), (x + 72, y)])
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


def _draw_ground(x: float, y: float) -> str:
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


def _draw_default(x: float, y: float) -> str:
    return f'<rect class="component" x="{x - 28:.1f}" y="{y - 18:.1f}" width="56" height="36" rx="3"/>'


def _px(layout: LayoutPlan, point: Point) -> str:
    return f"{point.x * layout.grid:.1f},{point.y * layout.grid:.1f}"


def _set_root_attr(svg: str, name: str, value: str) -> str:
    svg_start = svg.find("<svg")
    if svg_start == -1:
        return svg
    head_end = svg.find(">", svg_start)
    if head_end == -1:
        return svg
    before = svg[:svg_start]
    head = svg[svg_start:head_end]
    tail = svg[head_end:]
    escaped = escape(value)
    marker = f"{name}="
    if marker in head:
        # Keep this intentionally simple: generated SVG roots use quoted attrs.
        prefix, rest = head.split(marker, 1)
        quote = rest[0]
        end = rest.find(quote, 1)
        if end != -1:
            head = f'{prefix}{name}="{escaped}"{rest[end + 1:]}'
    else:
        head = f'{head} {name}="{escaped}"'
    return before + head + tail


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

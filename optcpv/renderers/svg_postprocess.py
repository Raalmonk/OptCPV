"""SVG metadata injection and debug fallback rendering."""

from __future__ import annotations

from html import escape
from typing import Literal

from ..labels import wrap_label_lines
from ..models import LayoutComponent, LayoutLabel, LayoutPlan, LocalTerminalIntent, Point
from ..segments import merged_axis_aligned_segments


LayerName = Literal["wires", "components", "labels"]


def inject_metadata(svg: str, layout: LayoutPlan, *, renderer: str) -> str:
    patched = _set_root_attr(svg, "data-renderer", renderer)
    patched = _set_root_attr(patched, "data-optcpv-circuit-id", layout.circuit_id)
    patched = _set_root_attr(patched, "data-optcpv-layout-mode", layout.support.layout_mode)
    patched = _set_root_attr(patched, "data-optcpv-layout-confidence", f"{layout.support.layout_confidence:.2f}")
    patched = _set_root_attr(patched, "data-optcpv-fallback-used", _bool_attr(layout.support.fallback_used))
    patched = _set_root_attr(patched, "data-optcpv-matched-motifs", ",".join(layout.support.matched_motifs))
    patched = _set_root_attr(patched, "data-optcpv-unsupported-regions", ",".join(layout.support.unsupported_regions))
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
            if _is_redundant_terminal_component(layout, component):
                continue
            parts.append(_draw_component_symbol(layout, component))
        for terminal in layout.semantic.local_terminals:
            parts.append(_draw_local_terminal_symbol(layout, terminal))
    if "labels" in layers:
        for label in layout.labels:
            owner = next((component for component in layout.components if component.id == label.owner_id), None)
            if owner is not None and _is_redundant_terminal_component(layout, owner):
                continue
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
            ".label-halo{font:14px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:none;stroke:#ffffff;stroke-width:4px;stroke-linejoin:round}",
            ".terminal-label{font:12px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#111827}",
        ]
    return [
        ".wire{fill:none;stroke:#111827;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}",
        ".component{fill:#fffaf0;stroke:#111827;stroke-width:2}",
        ".symbol{fill:none;stroke:#111827;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}",
        ".pin{fill:#111827}",
        ".label{font:14px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:#111827}",
        ".label-halo{font:14px ui-sans-serif,system-ui,sans-serif;text-anchor:middle;fill:none;stroke:#ffffff;stroke-width:4px;stroke-linejoin:round}",
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
    for terminal in layout.semantic.local_terminals:
        pin = layout.pin_map.get((terminal.component_id, terminal.pin_name))
        if pin is None:
            continue
        parts.append(
            f'<circle cx="{pin.x * layout.grid:.1f}" cy="{pin.y * layout.grid:.1f}" r="3" '
            f'data-local-terminal="true" data-component-id="{escape(terminal.component_id)}" '
            f'data-pin-name="{escape(terminal.pin_name)}" data-net-name="{escape(terminal.net)}" '
            f'data-terminal-type="{escape(terminal.terminal_type)}"/>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def _draw_wire(layout: LayoutPlan, net: str, points: list[Point]) -> str:
    return "\n".join(
        f'<line class="wire" x1="{start.x * layout.grid:.1f}" y1="{start.y * layout.grid:.1f}" '
        f'x2="{end.x * layout.grid:.1f}" y2="{end.y * layout.grid:.1f}" data-net-name="{escape(net)}"/>'
        for start, end in _unique_segments(points)
    )


def _draw_component_symbol(layout: LayoutPlan, component: LayoutComponent) -> str:
    x, y = component.x * layout.grid, component.y * layout.grid
    key = _key(component.type)
    inner = _draw_default(x, y)
    if "op_amp" in key or "opamp" in key or "operational_amplifier" in key:
        inner = _draw_opamp(x, y, flipped="flip" in _key(component.orientation))
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


def _is_redundant_terminal_component(layout: LayoutPlan, component: LayoutComponent) -> bool:
    key = _key(component.type)
    if key not in {"ground", "gnd", "supply", "power", "vcc", "vdd", "vee", "vss"}:
        return False
    local_nets = {terminal.net for terminal in layout.semantic.local_terminals}
    return bool(set(component.pins.values()) & local_nets)


def _draw_local_terminal_symbol(layout: LayoutPlan, terminal: LocalTerminalIntent) -> str:
    pin = layout.pin_map.get((terminal.component_id, terminal.pin_name))
    if pin is None:
        return ""
    x = pin.x * layout.grid
    y = pin.y * layout.grid
    sign = -1 if terminal.preferred_direction == "up" else 1
    y1 = y + sign * 24
    label_y = y + sign * 44
    if terminal.terminal_type == "positive_supply":
        symbol = (
            f'<line class="symbol" x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y1:.1f}"/>'
            f'<path class="symbol" d="M {x - 7:.1f} {y1 + sign * 7:.1f} L {x:.1f} {y1:.1f} '
            f'L {x + 7:.1f} {y1 + sign * 7:.1f}"/>'
        )
    elif terminal.terminal_type == "negative_supply":
        symbol = (
            f'<line class="symbol" x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y1:.1f}"/>'
            f'<path class="symbol" d="M {x - 7:.1f} {y1 - sign * 7:.1f} L {x:.1f} {y1:.1f} '
            f'L {x + 7:.1f} {y1 - sign * 7:.1f}"/>'
        )
    else:
        bar_y = y + sign * 26
        symbol = (
            f'<line class="symbol" x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{bar_y:.1f}"/>'
            f'<line class="symbol" x1="{x - 15:.1f}" y1="{bar_y:.1f}" x2="{x + 15:.1f}" y2="{bar_y:.1f}"/>'
            f'<line class="symbol" x1="{x - 9:.1f}" y1="{bar_y + sign * 7:.1f}" x2="{x + 9:.1f}" y2="{bar_y + sign * 7:.1f}"/>'
            f'<line class="symbol" x1="{x - 4:.1f}" y1="{bar_y + sign * 14:.1f}" x2="{x + 4:.1f}" y2="{bar_y + sign * 14:.1f}"/>'
        )
    return (
        f'<g data-local-terminal="true" data-component-id="{escape(terminal.component_id)}" '
        f'data-pin-name="{escape(terminal.pin_name)}" data-net-name="{escape(terminal.net)}" '
        f'data-terminal-type="{escape(terminal.terminal_type)}">'
        f"{symbol}"
        f"{'' if terminal.terminal_type == 'ground' else _local_terminal_label(x, label_y, terminal.label)}"
        "</g>"
    )


def _local_terminal_label(x: float, y: float, label: str) -> str:
    return f'<text class="terminal-label" x="{x:.1f}" y="{y:.1f}">{escape(label)}</text>'


def _draw_label(layout: LayoutPlan, label: LayoutLabel) -> str:
    x = label.x * layout.grid
    y = label.y * layout.grid
    lines = wrap_label_lines(label.text)
    if len(lines) == 1:
        inner = escape(lines[0])
    else:
        start_y = y - (len(lines) - 1) * 7.8
        inner = "".join(
            f'<tspan x="{x:.1f}" y="{start_y + index * 15.6:.1f}">{escape(line)}</tspan>'
            for index, line in enumerate(lines)
        )
    halo = f'<text class="label-halo" x="{x:.1f}" y="{y:.1f}" text-anchor="{escape(label.anchor)}">{inner}</text>'
    foreground = (
        f'<text class="label" x="{x:.1f}" y="{y:.1f}" text-anchor="{escape(label.anchor)}" '
        f'data-label-id="{escape(label.id)}" data-label-owner-id="{escape(label.owner_id)}">'
        f"{inner}</text>"
    )
    return halo + foreground


def _draw_opamp(x: float, y: float, *, flipped: bool = False) -> str:
    polygon = " ".join(f"{px:.1f},{py:.1f}" for px, py in [(x - 58, y - 44), (x - 58, y + 44), (x + 72, y)])
    top_label, bottom_label = ("+", "-") if flipped else ("-", "+")
    return "\n".join(
        [
            f'<polygon class="component" points="{polygon}"/>',
            f'<text class="terminal-label" x="{x - 44:.1f}" y="{y - 22:.1f}">{top_label}</text>',
            f'<text class="terminal-label" x="{x - 44:.1f}" y="{y + 30:.1f}">{bottom_label}</text>',
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


def _unique_segments(points: list[Point]) -> list[tuple[Point, Point]]:
    return merged_axis_aligned_segments(points)


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


def _bool_attr(value: bool) -> str:
    return "true" if value else "false"


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

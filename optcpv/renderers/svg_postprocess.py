"""SVG metadata injection and debug fallback rendering."""

from __future__ import annotations

from html import escape
from typing import Literal

from ..labels import wrap_label_lines
from ..models import LayoutComponent, LayoutLabel, LayoutPlan, LocalTerminalIntent, Point
from ..segments import EPSILON, merged_axis_aligned_segments


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
        parts.append(draw_wires_with_jumps(layout))
    if "components" in layers:
        for component in layout.components:
            if _is_redundant_terminal_component(layout, component):
                continue
            parts.append(_draw_component_symbol(layout, component))
        for terminal in layout.semantic.local_terminals:
            if _should_hide_local_terminal(layout, terminal):
                continue
            parts.append(_draw_local_terminal_symbol(layout, terminal))
    if "labels" in layers:
        for label in layout.labels:
            owner = next((component for component in layout.components if component.id == label.owner_id), None)
            if owner is not None and _is_redundant_terminal_component(layout, owner):
                continue
            if owner is not None and _is_duplicate_meter_label(owner, label):
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


def draw_wires_with_jumps(layout: LayoutPlan, *, class_name: str = "wire") -> str:
    jumps = _wire_jump_points(layout)
    parts: list[str] = []
    for wire in layout.wires:
        for start, end in _unique_segments(wire.points):
            parts.append(_draw_wire_segment(layout, wire.net, start, end, jumps.get(wire.net, ()), class_name=class_name))
    return "\n".join(part for part in parts if part)


def _draw_wire_segment(
    layout: LayoutPlan,
    net: str,
    start: Point,
    end: Point,
    jump_points: tuple[Point, ...],
    *,
    class_name: str,
) -> str:
    if abs(start.y - end.y) >= EPSILON:
        return _line_svg(layout, net, start, end, class_name=class_name)
    crossings = [
        point
        for point in jump_points
        if _point_on_open_segment(point, start, end)
    ]
    if not crossings:
        return _line_svg(layout, net, start, end, class_name=class_name)
    return _horizontal_bridge_path(layout, net, start, end, crossings, class_name=class_name)


def _line_svg(layout: LayoutPlan, net: str, start: Point, end: Point, *, class_name: str) -> str:
    class_attr = f' class="{escape(class_name)}"' if class_name else ""
    return (
        f'<line{class_attr} x1="{start.x * layout.grid:.1f}" y1="{start.y * layout.grid:.1f}" '
        f'x2="{end.x * layout.grid:.1f}" y2="{end.y * layout.grid:.1f}" data-net-name="{escape(net)}"/>'
    )


def _horizontal_bridge_path(
    layout: LayoutPlan,
    net: str,
    start: Point,
    end: Point,
    crossings: list[Point],
    *,
    class_name: str,
) -> str:
    radius = 8.0 / layout.grid
    y = start.y
    direction = 1.0 if end.x >= start.x else -1.0
    ordered = sorted(crossings, key=lambda point: point.x, reverse=direction < 0)
    class_attr = f' class="{escape(class_name)}"' if class_name else ""
    commands = [f"M {start.x * layout.grid:.1f} {y * layout.grid:.1f}"]
    cursor_x = start.x
    for point in ordered:
        left_x = point.x - direction * radius
        right_x = point.x + direction * radius
        if (direction > 0 and (left_x <= cursor_x + EPSILON or right_x >= end.x - EPSILON)) or (
            direction < 0 and (left_x >= cursor_x - EPSILON or right_x <= end.x + EPSILON)
        ):
            continue
        arc_sign = 1.0 if y - radius < 0.15 else -1.0
        commands.append(f"L {left_x * layout.grid:.1f} {y * layout.grid:.1f}")
        commands.append(
            f"Q {point.x * layout.grid:.1f} {(y + arc_sign * radius) * layout.grid:.1f} "
            f"{right_x * layout.grid:.1f} {y * layout.grid:.1f}"
        )
        cursor_x = right_x
    commands.append(f"L {end.x * layout.grid:.1f} {end.y * layout.grid:.1f}")
    return f'<path{class_attr} d="{" ".join(commands)}" data-net-name="{escape(net)}"/>'


def _wire_jump_points(layout: LayoutPlan) -> dict[str, tuple[Point, ...]]:
    segments: list[tuple[str, Point, Point]] = [
        (wire.net, start, end)
        for wire in layout.wires
        for start, end in _unique_segments(wire.points)
    ]
    pins = {(_round(pin.x), _round(pin.y)) for pin in layout.pin_map.values()}
    by_net: dict[str, list[Point]] = {}
    for index, left in enumerate(segments):
        for right in segments[index + 1 :]:
            if left[0] == right[0]:
                continue
            crossing = _orthogonal_crossing(left[1], left[2], right[1], right[2])
            if crossing is None or (_round(crossing.x), _round(crossing.y)) in pins:
                continue
            jumping_net = left[0] if _is_horizontal(left[1], left[2]) else right[0]
            by_net.setdefault(jumping_net, []).append(crossing)
    return {
        net: tuple(sorted(_dedupe_points(points), key=lambda point: (point.y, point.x)))
        for net, points in by_net.items()
    }


def _orthogonal_crossing(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    if _is_horizontal(a, b) and _is_vertical(c, d):
        point = Point(c.x, a.y)
        return point if _point_on_open_segment(point, a, b) and _point_on_open_segment(point, c, d) else None
    if _is_vertical(a, b) and _is_horizontal(c, d):
        point = Point(a.x, c.y)
        return point if _point_on_open_segment(point, a, b) and _point_on_open_segment(point, c, d) else None
    return None


def _point_on_open_segment(point: Point, start: Point, end: Point) -> bool:
    if _is_horizontal(start, end):
        return abs(point.y - start.y) < EPSILON and min(start.x, end.x) + EPSILON < point.x < max(start.x, end.x) - EPSILON
    if _is_vertical(start, end):
        return abs(point.x - start.x) < EPSILON and min(start.y, end.y) + EPSILON < point.y < max(start.y, end.y) - EPSILON
    return False


def _is_horizontal(start: Point, end: Point) -> bool:
    return abs(start.y - end.y) < EPSILON and abs(start.x - end.x) >= EPSILON


def _is_vertical(start: Point, end: Point) -> bool:
    return abs(start.x - end.x) < EPSILON and abs(start.y - end.y) >= EPSILON


def _dedupe_points(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        key = (_round(point.x), _round(point.y))
        if key in seen:
            continue
        seen.add(key)
        result.append(point)
    return result


def _round(value: float) -> float:
    return round(value, 4)


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
    elif "inductor" in key or key.startswith("l"):
        inner = _draw_inductor(component, x, y)
    elif "diode" in key:
        inner = _draw_diode(component, x, y)
    elif "switch" in key or key.startswith("sw"):
        inner = _draw_switch(component, x, y)
    elif _is_meter_component(component):
        inner = _draw_meter(component, x, y)
    elif key in {"ground", "gnd"}:
        inner = _draw_ground(x, y)
    elif _is_physical_source(component):
        inner = _draw_source(component, x, y)
    elif key in {"input", "output", "input_terminal"} or ("source" in key and len(component.pins) <= 1):
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
    if _has_signal_label_terminal(layout, component) and _is_signal_label_terminal_component(component):
        return True
    key = _key(component.type)
    if key not in {"ground", "gnd", "supply", "power", "vcc", "vdd", "vee", "vss"}:
        return False
    local_nets = {terminal.net for terminal in layout.semantic.local_terminals}
    return bool(set(component.pins.values()) & local_nets)


def _is_redundant_output_signal_label_terminal(layout: LayoutPlan, terminal: LocalTerminalIntent) -> bool:
    owner = next((component for component in layout.components if component.id == terminal.component_id), None)
    if owner is None or _key(owner.type) not in {"output", "output_terminal"}:
        return False
    return any(
        other.terminal_type == "signal_label"
        and other.net == terminal.net
        and other.component_id != terminal.component_id
        and not _terminal_owner_is_output(layout, other)
        for other in layout.semantic.local_terminals
    )


def _terminal_owner_is_output(layout: LayoutPlan, terminal: LocalTerminalIntent) -> bool:
    owner = next((component for component in layout.components if component.id == terminal.component_id), None)
    return owner is not None and _key(owner.type) in {"output", "output_terminal"}


def _has_signal_label_terminal(layout: LayoutPlan, component: LayoutComponent) -> bool:
    return any(
        terminal.component_id == component.id and terminal.terminal_type == "signal_label"
        for terminal in layout.semantic.local_terminals
    )


def _is_standalone_terminal_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"ground", "gnd", "supply", "power", "vcc", "vdd", "vee", "vss"}


def _is_signal_label_terminal_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"input", "output", "input_terminal"} or _is_standalone_terminal_component(component)


def _draw_local_terminal_symbol(layout: LayoutPlan, terminal: LocalTerminalIntent) -> str:
    pin = layout.pin_map.get((terminal.component_id, terminal.pin_name))
    if pin is None:
        return ""
    if terminal.terminal_type == "signal_label":
        if _is_redundant_output_signal_label_terminal(layout, terminal):
            return ""
        return _draw_signal_label_terminal(layout, terminal, pin)
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


def _should_hide_local_terminal(layout: LayoutPlan, terminal: LocalTerminalIntent) -> bool:
    if terminal.terminal_type not in {"positive_supply", "negative_supply"}:
        return False
    owner = next((component for component in layout.components if component.id == terminal.component_id), None)
    return owner is not None and _is_opamp_component(owner)


def _local_terminal_label(x: float, y: float, label: str) -> str:
    return f'<text class="terminal-label" x="{x:.1f}" y="{y:.1f}">{escape(label)}</text>'


def _draw_signal_label_terminal(layout: LayoutPlan, terminal: LocalTerminalIntent, pin) -> str:
    x = pin.x * layout.grid
    y = pin.y * layout.grid
    direction = -1 if terminal.preferred_direction == "left" or (terminal.preferred_direction not in {"right", "left"} and pin.side == "left") else 1
    x1 = x + direction * 30
    label_x = x1 + direction * 8
    label_y = y - 7.5
    anchor = "end" if direction < 0 else "start"
    label = terminal.label or terminal.net
    return (
        f'<g data-local-terminal="true" data-component-id="{escape(terminal.component_id)}" '
        f'data-pin-name="{escape(terminal.pin_name)}" data-net-name="{escape(terminal.net)}" '
        f'data-terminal-type="{escape(terminal.terminal_type)}">'
        f'<line class="symbol" x1="{x:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="#ffffff" stroke="#111827" stroke-width="1.8"/>'
        f'<text class="terminal-label" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" '
        f'dominant-baseline="auto">{escape(label)}</text>'
        "</g>"
    )


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


def _draw_switch(component: LayoutComponent, x: float, y: float) -> str:
    if component.orientation in {"up", "down"}:
        return "\n".join(
            [
                f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 16:.1f}"/>',
                f'<circle class="component" cx="{x:.1f}" cy="{y - 16:.1f}" r="3.2"/>',
                f'<line class="symbol" x1="{x:.1f}" y1="{y + 16:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>',
                f'<circle class="component" cx="{x:.1f}" cy="{y + 16:.1f}" r="3.2"/>',
                f'<line class="symbol" x1="{x:.1f}" y1="{y - 16:.1f}" x2="{x + 22:.1f}" y2="{y + 8:.1f}"/>',
            ]
        )
    return "\n".join(
        [
            f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 16:.1f}" y2="{y:.1f}"/>',
            f'<circle class="component" cx="{x - 16:.1f}" cy="{y:.1f}" r="3.2"/>',
            f'<line class="symbol" x1="{x + 16:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>',
            f'<circle class="component" cx="{x + 16:.1f}" cy="{y:.1f}" r="3.2"/>',
            f'<line class="symbol" x1="{x - 16:.1f}" y1="{y:.1f}" x2="{x + 8:.1f}" y2="{y - 22:.1f}"/>',
        ]
    )


def _draw_inductor(component: LayoutComponent, x: float, y: float) -> str:
    if component.orientation in {"up", "down"}:
        arcs = "".join(
            f'<path class="symbol" d="M {x:.1f} {y - 24 + index * 12:.1f} '
            f'a 9 6 0 0 1 0 12"/>'
            for index in range(4)
        )
        return (
            f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 24:.1f}"/>'
            f"{arcs}"
            f'<line class="symbol" x1="{x:.1f}" y1="{y + 24:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>'
        )
    arcs = "".join(
        f'<path class="symbol" d="M {x - 24 + index * 12:.1f} {y:.1f} '
        f'a 6 9 0 0 1 12 0"/>'
        for index in range(4)
    )
    return (
        f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 24:.1f}" y2="{y:.1f}"/>'
        f"{arcs}"
        f'<line class="symbol" x1="{x + 24:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>'
    )


def _draw_diode(component: LayoutComponent, x: float, y: float) -> str:
    if component.orientation in {"up", "down"}:
        return "\n".join(
            [
                f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 18:.1f}"/>',
                f'<polygon class="component" points="{x - 16:.1f},{y - 18:.1f} {x + 16:.1f},{y - 18:.1f} {x:.1f},{y + 10:.1f}"/>',
                f'<line class="symbol" x1="{x - 17:.1f}" y1="{y + 12:.1f}" x2="{x + 17:.1f}" y2="{y + 12:.1f}"/>',
                f'<line class="symbol" x1="{x:.1f}" y1="{y + 12:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>',
            ]
        )
    return "\n".join(
        [
            f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 18:.1f}" y2="{y:.1f}"/>',
            f'<polygon class="component" points="{x - 18:.1f},{y - 16:.1f} {x - 18:.1f},{y + 16:.1f} {x + 10:.1f},{y:.1f}"/>',
            f'<line class="symbol" x1="{x + 12:.1f}" y1="{y - 17:.1f}" x2="{x + 12:.1f}" y2="{y + 17:.1f}"/>',
            f'<line class="symbol" x1="{x + 12:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>',
        ]
    )


def _draw_source(component: LayoutComponent, x: float, y: float) -> str:
    label = "I" if "current" in _key(component.type) else "V"
    if component.orientation in {"up", "down"}:
        leads = (
            f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 18:.1f}"/>'
            f'<line class="symbol" x1="{x:.1f}" y1="{y + 18:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>'
        )
    else:
        leads = (
            f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 18:.1f}" y2="{y:.1f}"/>'
            f'<line class="symbol" x1="{x + 18:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>'
        )
    return (
        f"{leads}"
        f'<circle class="component" cx="{x:.1f}" cy="{y:.1f}" r="20"/>'
        f'<text class="terminal-label" x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{label}</text>'
    )


def _draw_meter(component: LayoutComponent, x: float, y: float) -> str:
    label = "A" if _is_current_meter(component) else "V" if _is_voltage_meter(component) else "M"
    if component.orientation in {"up", "down"}:
        leads = (
            f'<line class="symbol" x1="{x:.1f}" y1="{y - 44:.1f}" x2="{x:.1f}" y2="{y - 22:.1f}"/>'
            f'<line class="symbol" x1="{x:.1f}" y1="{y + 22:.1f}" x2="{x:.1f}" y2="{y + 44:.1f}"/>'
        )
    else:
        leads = (
            f'<line class="symbol" x1="{x - 44:.1f}" y1="{y:.1f}" x2="{x - 22:.1f}" y2="{y:.1f}"/>'
            f'<line class="symbol" x1="{x + 22:.1f}" y1="{y:.1f}" x2="{x + 44:.1f}" y2="{y:.1f}"/>'
        )
    return (
        f"{leads}"
        f'<circle class="component" cx="{x:.1f}" cy="{y:.1f}" r="22"/>'
        f'<text class="terminal-label" x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{label}</text>'
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


def _is_opamp_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _is_meter_component(component: LayoutComponent) -> bool:
    identity = _key(" ".join(filter(None, [component.type, component.role, component.label, component.value])))
    return (
        "ammeter" in identity
        or "voltmeter" in identity
        or "meter" in identity
        or "current_probe" in identity
        or "voltage_probe" in identity
    )


def _is_current_meter(component: LayoutComponent) -> bool:
    identity = _key(" ".join(filter(None, [component.type, component.role, component.label, component.value])))
    return "ammeter" in identity or "current" in identity or identity in {"a", "meter_a"}


def _is_voltage_meter(component: LayoutComponent) -> bool:
    identity = _key(" ".join(filter(None, [component.type, component.role, component.label, component.value])))
    return "voltmeter" in identity or "voltage" in identity or identity in {"v", "meter_v"}


def _is_duplicate_meter_label(component: LayoutComponent, label: LayoutLabel) -> bool:
    return _is_meter_component(component) and _key(label.text) in {"a", "v", "i", "m"}


def _is_physical_source(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return len(component.pins) >= 2 and (
        key in {"voltage_source", "current_source", "source"} or "source" in key
    )

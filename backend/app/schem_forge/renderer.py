"""Deterministic SVG renderer for schem_forge layout plans."""

from __future__ import annotations

from html import escape
from typing import Iterable

from .models import (
    BBox,
    ComponentLayout,
    LabelLayout,
    LayoutPlan,
    Point,
    RenderGeometry,
    RenderResult,
    WireRoute,
    WireSegment,
)


def _type_key(component_type: str) -> str:
    return component_type.lower().replace("-", "_").replace(" ", "_")


def _role_key(role: str | None) -> str:
    return (role or "").lower().replace("-", "_").replace(" ", "_")


def _is_opamp(component: ComponentLayout) -> bool:
    key = _type_key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _is_resistor(component: ComponentLayout) -> bool:
    return "resistor" in _type_key(component.type) or component.type.lower().startswith("r")


def _is_capacitor(component: ComponentLayout) -> bool:
    return "capacitor" in _type_key(component.type) or component.type.lower().startswith("c")


def _is_ground(component: ComponentLayout) -> bool:
    role = _role_key(component.role)
    return "ground" in _type_key(component.type) or role in {"ground", "gnd", "ground_symbol"}


def _is_source(component: ComponentLayout) -> bool:
    key = _type_key(component.type)
    return "source" in key or "voltage_source" in key or "current_source" in key


def _is_output(component: ComponentLayout) -> bool:
    return "output" in _role_key(component.role) or _type_key(component.type) == "output"


def _is_input(component: ComponentLayout) -> bool:
    role = _role_key(component.role)
    component_type = _type_key(component.type)
    return role in {"input", "input_source", "input_terminal", "sensor"} or component_type in {
        "input",
        "input_terminal",
    }


def _grid_to_px(point: Point, grid_size: int) -> Point:
    return Point(point.x * grid_size, point.y * grid_size)


def _component_center(component: ComponentLayout, grid_size: int) -> Point:
    return Point(component.grid_x * grid_size, component.grid_y * grid_size)


def _pin_points(layout_plan: LayoutPlan) -> dict[str, Point]:
    points: dict[str, Point] = {}
    for component in layout_plan.components:
        for pin in component.pins:
            points[f"{component.id}.{pin.pin_name}"] = Point(
                (component.grid_x + pin.offset_x) * layout_plan.grid_size,
                (component.grid_y + pin.offset_y) * layout_plan.grid_size,
            )
    return points


def _bbox_for_component(component: ComponentLayout, grid_size: int) -> BBox:
    center = _component_center(component, grid_size)
    if _is_opamp(component):
        return BBox(center.x - 2.0 * grid_size, center.y - 1.5 * grid_size, 4.0 * grid_size, 3.0 * grid_size)
    if _is_resistor(component):
        if component.orientation in {"up", "down"}:
            return BBox(center.x - 0.45 * grid_size, center.y - 1.0 * grid_size, 0.9 * grid_size, 2.0 * grid_size)
        return BBox(center.x - 1.0 * grid_size, center.y - 0.45 * grid_size, 2.0 * grid_size, 0.9 * grid_size)
    if _is_capacitor(component):
        if component.orientation in {"up", "down"}:
            return BBox(center.x - 0.55 * grid_size, center.y - 1.0 * grid_size, 1.1 * grid_size, 2.0 * grid_size)
        return BBox(center.x - 1.0 * grid_size, center.y - 0.55 * grid_size, 2.0 * grid_size, 1.1 * grid_size)
    if _is_ground(component):
        return BBox(center.x - 0.75 * grid_size, center.y - 0.85 * grid_size, 1.5 * grid_size, 1.35 * grid_size)
    if _is_source(component):
        return BBox(center.x - 0.85 * grid_size, center.y - 0.85 * grid_size, 1.7 * grid_size, 1.7 * grid_size)
    if _is_input(component) or _is_output(component):
        return BBox(center.x - 0.45 * grid_size, center.y - 0.45 * grid_size, 0.9 * grid_size, 0.9 * grid_size)
    return BBox(center.x - grid_size, center.y - 0.6 * grid_size, 2 * grid_size, 1.2 * grid_size)


def _points_attr(points: Iterable[Point]) -> str:
    return " ".join(f"{point.x:.2f},{point.y:.2f}" for point in points)


def _pin_kind(pin_name: str) -> str:
    compact = pin_name.lower().replace("_", "").replace("-", "")
    if pin_name == "+" or compact in {"plus", "inplus", "noninv", "noninverting", "inp", "vp"}:
        return "plus"
    if pin_name == "-" or compact in {"minus", "inminus", "inv", "inverting", "inn", "vn"}:
        return "minus"
    if compact in {"out", "output", "vo", "vout"}:
        return "out"
    return compact


def _pin_point(component: ComponentLayout, pin_name: str, grid_size: int) -> Point | None:
    for pin in component.pins:
        if pin.pin_name == pin_name:
            return Point(
                (component.grid_x + pin.offset_x) * grid_size,
                (component.grid_y + pin.offset_y) * grid_size,
            )
    return None


def _draw_opamp(component: ComponentLayout, grid_size: int) -> str:
    center = _component_center(component, grid_size)
    g = grid_size
    if component.orientation == "left":
        points = [
            Point(center.x + 2 * g, center.y - 1.5 * g),
            Point(center.x + 2 * g, center.y + 1.5 * g),
            Point(center.x - 2 * g, center.y),
        ]
    elif component.orientation == "up":
        points = [
            Point(center.x - 1.5 * g, center.y + 2 * g),
            Point(center.x + 1.5 * g, center.y + 2 * g),
            Point(center.x, center.y - 2 * g),
        ]
    elif component.orientation == "down":
        points = [
            Point(center.x - 1.5 * g, center.y - 2 * g),
            Point(center.x + 1.5 * g, center.y - 2 * g),
            Point(center.x, center.y + 2 * g),
        ]
    else:
        points = [
            Point(center.x - 2 * g, center.y - 1.5 * g),
            Point(center.x - 2 * g, center.y + 1.5 * g),
            Point(center.x + 2 * g, center.y),
        ]

    parts = [
        f'<polygon points="{_points_attr(points)}" class="component-fill"/>',
    ]
    for pin in component.pins:
        pin_point = _pin_point(component, pin.pin_name, grid_size)
        if not pin_point:
            continue
        kind = _pin_kind(pin.pin_name)
        sign_offset = 0.58 * g if component.orientation == "right" else -0.58 * g
        if kind == "plus":
            text_point = Point(pin_point.x + sign_offset, pin_point.y)
            parts.append(f'<text x="{text_point.x:.2f}" y="{text_point.y:.2f}" class="pin-mark">+</text>')
        elif kind == "minus":
            text_point = Point(pin_point.x + sign_offset, pin_point.y)
            parts.append(f'<text x="{text_point.x:.2f}" y="{text_point.y:.2f}" class="pin-mark">-</text>')
    label = escape(component.display_label or component.id)
    parts.append(f'<text x="{center.x:.2f}" y="{center.y:.2f}" class="component-label">{label}</text>')
    return "\n".join(parts)


def _draw_resistor(component: ComponentLayout, grid_size: int) -> str:
    center = _component_center(component, grid_size)
    g = grid_size
    if component.orientation in {"up", "down"}:
        y1 = center.y - g
        y2 = center.y + g
        x = center.x
        points = [Point(x, y1)]
        step = (y2 - y1) / 8.0
        for index in range(1, 8):
            offset = 0.28 * g if index % 2 else -0.28 * g
            points.append(Point(x + offset, y1 + step * index))
        points.append(Point(x, y2))
    else:
        x1 = center.x - g
        x2 = center.x + g
        y = center.y
        points = [Point(x1, y)]
        step = (x2 - x1) / 8.0
        for index in range(1, 8):
            offset = 0.28 * g if index % 2 else -0.28 * g
            points.append(Point(x1 + step * index, y + offset))
        points.append(Point(x2, y))
    return f'<polyline points="{_points_attr(points)}" class="component-stroke component-fill-none"/>'


def _draw_capacitor(component: ComponentLayout, grid_size: int) -> str:
    center = _component_center(component, grid_size)
    g = grid_size
    if component.orientation in {"up", "down"}:
        return "\n".join(
            [
                f'<line x1="{center.x:.2f}" y1="{center.y - g:.2f}" x2="{center.x:.2f}" y2="{center.y - 0.22 * g:.2f}" class="component-stroke"/>',
                f'<line x1="{center.x:.2f}" y1="{center.y + 0.22 * g:.2f}" x2="{center.x:.2f}" y2="{center.y + g:.2f}" class="component-stroke"/>',
                f'<line x1="{center.x - 0.55 * g:.2f}" y1="{center.y - 0.22 * g:.2f}" x2="{center.x + 0.55 * g:.2f}" y2="{center.y - 0.22 * g:.2f}" class="component-stroke"/>',
                f'<line x1="{center.x - 0.55 * g:.2f}" y1="{center.y + 0.22 * g:.2f}" x2="{center.x + 0.55 * g:.2f}" y2="{center.y + 0.22 * g:.2f}" class="component-stroke"/>',
            ]
        )
    return "\n".join(
        [
            f'<line x1="{center.x - g:.2f}" y1="{center.y:.2f}" x2="{center.x - 0.22 * g:.2f}" y2="{center.y:.2f}" class="component-stroke"/>',
            f'<line x1="{center.x + 0.22 * g:.2f}" y1="{center.y:.2f}" x2="{center.x + g:.2f}" y2="{center.y:.2f}" class="component-stroke"/>',
            f'<line x1="{center.x - 0.22 * g:.2f}" y1="{center.y - 0.55 * g:.2f}" x2="{center.x - 0.22 * g:.2f}" y2="{center.y + 0.55 * g:.2f}" class="component-stroke"/>',
            f'<line x1="{center.x + 0.22 * g:.2f}" y1="{center.y - 0.55 * g:.2f}" x2="{center.x + 0.22 * g:.2f}" y2="{center.y + 0.55 * g:.2f}" class="component-stroke"/>',
        ]
    )


def _draw_ground(component: ComponentLayout, grid_size: int) -> str:
    center = _component_center(component, grid_size)
    g = grid_size
    top = center.y - 0.8 * g
    return "\n".join(
        [
            f'<line x1="{center.x:.2f}" y1="{top:.2f}" x2="{center.x:.2f}" y2="{center.y - 0.15 * g:.2f}" class="component-stroke"/>',
            f'<line x1="{center.x - 0.65 * g:.2f}" y1="{center.y - 0.15 * g:.2f}" x2="{center.x + 0.65 * g:.2f}" y2="{center.y - 0.15 * g:.2f}" class="component-stroke"/>',
            f'<line x1="{center.x - 0.42 * g:.2f}" y1="{center.y + 0.12 * g:.2f}" x2="{center.x + 0.42 * g:.2f}" y2="{center.y + 0.12 * g:.2f}" class="component-stroke"/>',
            f'<line x1="{center.x - 0.20 * g:.2f}" y1="{center.y + 0.38 * g:.2f}" x2="{center.x + 0.20 * g:.2f}" y2="{center.y + 0.38 * g:.2f}" class="component-stroke"/>',
        ]
    )


def _draw_source_or_terminal(component: ComponentLayout, grid_size: int) -> str:
    center = _component_center(component, grid_size)
    g = grid_size
    radius = 0.42 * g if (_is_input(component) or _is_output(component)) else 0.75 * g
    parts = [
        f'<circle cx="{center.x:.2f}" cy="{center.y:.2f}" r="{radius:.2f}" class="component-fill" data-node-id="{escape(component.id)}.node"/>'
    ]
    if _is_input(component):
        arrow = [
            Point(center.x - 1.05 * g, center.y - 0.32 * g),
            Point(center.x - 0.35 * g, center.y),
            Point(center.x - 1.05 * g, center.y + 0.32 * g),
        ]
        parts.append(f'<polyline points="{_points_attr(arrow)}" class="terminal-arrow"/>')
    elif _is_output(component):
        arrow = [
            Point(center.x + 0.35 * g, center.y - 0.32 * g),
            Point(center.x + 1.05 * g, center.y),
            Point(center.x + 0.35 * g, center.y + 0.32 * g),
        ]
        parts.append(f'<polyline points="{_points_attr(arrow)}" class="terminal-arrow"/>')
    for pin in component.pins:
        point = _pin_point(component, pin.pin_name, grid_size)
        if point:
            parts.append(
                f'<line x1="{center.x:.2f}" y1="{center.y:.2f}" x2="{point.x:.2f}" y2="{point.y:.2f}" '
                f'class="component-stroke" data-pin-ref="{escape(component.id)}.{escape(pin.pin_name)}" '
                f'data-net-name="{escape(pin.net_name)}"/>'
            )
    label = escape(component.display_label or component.id)
    if _is_input(component):
        parts.append(f'<text x="{center.x:.2f}" y="{center.y - 0.75 * g:.2f}" class="terminal-label">{label}</text>')
    elif _is_output(component):
        parts.append(f'<text x="{center.x:.2f}" y="{center.y - 0.75 * g:.2f}" class="terminal-label">{label}</text>')
    return "\n".join(parts)


def _draw_default(component: ComponentLayout, grid_size: int) -> str:
    bbox = _bbox_for_component(component, grid_size)
    label = escape(component.display_label or component.id)
    return "\n".join(
        [
            f'<rect x="{bbox.x:.2f}" y="{bbox.y:.2f}" width="{bbox.width:.2f}" height="{bbox.height:.2f}" rx="2" class="component-fill"/>',
            f'<text x="{bbox.x + bbox.width / 2:.2f}" y="{bbox.y + bbox.height / 2:.2f}" class="component-label">{label}</text>',
        ]
    )


def _draw_component(component: ComponentLayout, grid_size: int) -> str:
    if _is_opamp(component):
        inner = _draw_opamp(component, grid_size)
    elif _is_resistor(component):
        inner = _draw_resistor(component, grid_size)
    elif _is_capacitor(component):
        inner = _draw_capacitor(component, grid_size)
    elif _is_ground(component):
        inner = _draw_ground(component, grid_size)
    elif _is_source(component) or _is_input(component) or _is_output(component):
        inner = _draw_source_or_terminal(component, grid_size)
    else:
        inner = _draw_default(component, grid_size)
    pin_anchors = []
    for pin in component.pins:
        point = _pin_point(component, pin.pin_name, grid_size)
        if point:
            pin_anchors.append(
                f'<circle cx="{point.x:.2f}" cy="{point.y:.2f}" r="2.2" class="pin-anchor" '
                f'data-pin-ref="{escape(component.id)}.{escape(pin.pin_name)}" '
                f'data-net-name="{escape(pin.net_name)}"/>'
            )
    return (
        f'<g data-component-id="{escape(component.id)}" data-component-type="{escape(component.type)}">\n'
        f"{inner}\n"
        f"{chr(10).join(pin_anchors)}\n"
        "</g>"
    )


def _segments_from_route(route: WireRoute, grid_size: int) -> list[WireSegment]:
    if route.segments:
        return [
            WireSegment(
                start=_grid_to_px(segment.start, grid_size),
                end=_grid_to_px(segment.end, grid_size),
                net_name=route.net_name,
                kind=segment.kind,
            )
            for segment in route.segments
        ]
    points = [_grid_to_px(point, grid_size) for point in route.waypoints]
    return [
        WireSegment(start=start, end=end, net_name=route.net_name, kind="wire")
        for start, end in zip(points, points[1:])
    ]


def _draw_wire_segments(segments: list[WireSegment]) -> str:
    parts: list[str] = []
    for segment in segments:
        parts.append(
            '<line '
            f'x1="{segment.start.x:.2f}" y1="{segment.start.y:.2f}" '
            f'x2="{segment.end.x:.2f}" y2="{segment.end.y:.2f}" '
            f'class="wire" data-net-name="{escape(segment.net_name)}" data-wire-kind="{escape(segment.kind)}"/>'
        )
    return "\n".join(parts)


def _junction_points(segments: list[WireSegment]) -> dict[str, Point]:
    by_net_and_point: dict[tuple[str, int, int], int] = {}
    point_values: dict[tuple[str, int, int], Point] = {}
    for segment in segments:
        for point in (segment.start, segment.end):
            key = (segment.net_name, round(point.x), round(point.y))
            by_net_and_point[key] = by_net_and_point.get(key, 0) + 1
            point_values[key] = point

    junctions: dict[str, Point] = {}
    for index, (key, count) in enumerate(sorted(by_net_and_point.items())):
        if count >= 3:
            net_name = key[0]
            junctions[f"{net_name}:{index}"] = point_values[key]
    return junctions


def _draw_junctions(junctions: dict[str, Point]) -> str:
    parts = []
    for node_id, point in junctions.items():
        net_name = node_id.split(":", 1)[0]
        parts.append(
            f'<circle cx="{point.x:.2f}" cy="{point.y:.2f}" r="3.2" '
            f'class="junction" data-node-id="{escape(node_id)}" data-net-name="{escape(net_name)}"/>'
        )
    return "\n".join(parts)


def _label_bbox(label: LabelLayout, grid_size: int) -> BBox:
    point = Point(label.grid_x * grid_size, label.grid_y * grid_size)
    width = max(18.0, len(label.text) * 7.1)
    height = 14.0
    if label.anchor == "start":
        x = point.x
    elif label.anchor == "end":
        x = point.x - width
    else:
        x = point.x - width / 2.0
    return BBox(x, point.y - height / 2.0, width, height)


def _draw_label(label: LabelLayout, bbox: BBox, grid_size: int) -> str:
    point = Point(label.grid_x * grid_size, label.grid_y * grid_size)
    text_anchor = {"start": "start", "middle": "middle", "end": "end"}[label.anchor]
    return (
        f'<text x="{point.x:.2f}" y="{point.y:.2f}" '
        f'text-anchor="{text_anchor}" class="layout-label" '
        f'data-label-id="{escape(label.id)}">{escape(label.text)}</text>'
    )


def render_layout(layout_plan: LayoutPlan) -> RenderResult:
    """Render exactly the supplied plan and return SVG plus rendered geometry."""

    component_bboxes = {
        component.id: _bbox_for_component(component, layout_plan.grid_size)
        for component in layout_plan.components
    }
    pin_points = _pin_points(layout_plan)

    wire_segments: list[WireSegment] = []
    for wire in layout_plan.wires:
        wire_segments.extend(_segments_from_route(wire, layout_plan.grid_size))
    junction_points = _junction_points(wire_segments)

    label_bboxes = {
        label.id: _label_bbox(label, layout_plan.grid_size)
        for label in layout_plan.labels
    }
    max_x = layout_plan.canvas_width
    max_y = layout_plan.canvas_height
    for bbox in list(component_bboxes.values()) + list(label_bboxes.values()):
        max_x = max(max_x, int(bbox.right + 40))
        max_y = max(max_y, int(bbox.bottom + 40))
    for segment in wire_segments:
        max_x = max(max_x, int(max(segment.start.x, segment.end.x) + 40))
        max_y = max(max_y, int(max(segment.start.y, segment.end.y) + 40))

    style = """
<style>
  svg { background: #fbfaf7; }
  .wire { stroke: #1f2933; stroke-width: 2.25; fill: none; stroke-linecap: round; }
  .component-stroke { stroke: #111827; stroke-width: 2.4; fill: none; stroke-linecap: round; stroke-linejoin: round; }
  .component-fill { stroke: #111827; stroke-width: 2.4; fill: #fbfaf7; stroke-linejoin: round; }
  .component-fill-none { fill: none; }
  .component-label { font: 600 12px ui-sans-serif, system-ui, sans-serif; fill: #111827; text-anchor: middle; dominant-baseline: middle; }
  .terminal-label { font: 600 12px ui-sans-serif, system-ui, sans-serif; fill: #111827; text-anchor: middle; dominant-baseline: middle; }
  .terminal-arrow { stroke: #111827; stroke-width: 2.1; fill: none; stroke-linecap: round; stroke-linejoin: round; }
  .pin-mark { font: 700 15px ui-sans-serif, system-ui, sans-serif; fill: #111827; text-anchor: middle; dominant-baseline: middle; }
  .pin-anchor { fill: #111827; opacity: 0.001; pointer-events: all; }
  .layout-label { font: 600 13px ui-sans-serif, system-ui, sans-serif; fill: #374151; dominant-baseline: middle; }
  .junction { fill: #111827; stroke: none; }
</style>
""".strip()

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_x}" height="{max_y}" viewBox="0 0 {max_x} {max_y}" role="img" data-schem-forge-renderer="{escape(layout_plan.renderer)}" data-circuit-id="{escape(layout_plan.circuit_id)}">',
        f"<desc>{escape(layout_plan.renderer)} rendering for circuit {escape(layout_plan.circuit_id)}</desc>",
        style,
        '<g class="wires">',
        _draw_wire_segments(wire_segments),
        _draw_junctions(junction_points),
        "</g>",
        '<g class="components">',
    ]
    parts.extend(_draw_component(component, layout_plan.grid_size) for component in layout_plan.components)
    parts.append("</g>")
    parts.append('<g class="labels">')
    for label in layout_plan.labels:
        parts.append(_draw_label(label, label_bboxes[label.id], layout_plan.grid_size))
    parts.append("</g>")
    parts.append("</svg>")

    geometry = RenderGeometry(
        component_bboxes=component_bboxes,
        label_bboxes=label_bboxes,
        wire_segments=wire_segments,
        pin_points=pin_points,
        junction_points=junction_points,
    )
    return RenderResult(svg="\n".join(parts), geometry=geometry)

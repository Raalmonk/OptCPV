"""Schemdraw-backed renderer for OptCPV layout plans."""

from __future__ import annotations

from html import escape
from io import StringIO

from ..labels import component_display_label, display_label_text, wrap_label_lines
from ..models import LayoutComponent, LayoutLabel, LayoutPlan, LayoutWire, Point
from .svg_postprocess import inject_metadata, render_debug_svg, _set_root_attr


FALLBACK_RENDERER_ID = "optcpv.debug_svg_after_schemdraw_error"
SCHEMDRAW_UNIT_PX = 36.0


class SchemdrawDependencyError(RuntimeError):
    """Raised when the required Schemdraw dependency is unavailable."""


class SchemdrawRenderer:
    """Render a LayoutPlan with Schemdraw as the required primary backend."""

    renderer_id = "optcpv.schemdraw"

    def render(self, layout: LayoutPlan, *, style: str = "textbook") -> str:
        try:
            svg = self._render_schemdraw_svg(layout)
        except SchemdrawDependencyError:
            raise
        except Exception as exc:
            fallback = render_debug_svg(layout, renderer=FALLBACK_RENDERER_ID, style=style)
            fallback = inject_metadata(fallback, layout, renderer=FALLBACK_RENDERER_ID)
            return _set_root_attr(fallback, "data-schemdraw-error", f"{type(exc).__name__}: {exc}")
        return inject_metadata(svg, layout, renderer=self.renderer_id)

    def _render_schemdraw_svg(self, layout: LayoutPlan) -> str:
        try:
            import schemdraw
            import schemdraw.elements as elm
        except ImportError as exc:
            raise SchemdrawDependencyError(
                "Schemdraw is a core OptCPV dependency. Install optcpv core dependencies before rendering."
            ) from exc

        native = _render_native_motif_svg(schemdraw, elm, layout)
        if native is not None:
            return _normalize_native_svg_canvas(native, layout)

        drawing = schemdraw.Drawing(show=False)
        if hasattr(drawing, "config"):
            drawing.config(unit=1)
        self._add_components(drawing, elm, layout)
        raw = _drawing_to_svg(drawing)
        if not raw:
            raise RuntimeError("Schemdraw did not return SVG image data.")
        return _add_visible_labels(_normalize_svg_canvas(raw, layout), layout)

    def _add_wires(self, drawing, elm, layout: LayoutPlan) -> None:
        for wire in layout.wires:
            for start, end in zip(wire.points, wire.points[1:]):
                drawing += elm.Line().at(_sd(start)).to(_sd(end))

    def _add_components(self, drawing, elm, layout: LayoutPlan) -> None:
        for component in layout.components:
            element = self._element_for(elm, component)
            if element is None:
                continue
            anchor = _element_anchor(component)
            element = element.at(_sd(anchor))
            if _is_two_terminal(component) and hasattr(element, "length"):
                element = element.length(_terminal_length(component))
            if component.orientation in {"left", "west"} and hasattr(element, "left"):
                element = element.left()
            elif component.orientation in {"up", "north"} and hasattr(element, "up"):
                element = element.up()
            elif component.orientation in {"down", "south"} and hasattr(element, "down"):
                element = element.down()
            elif hasattr(element, "right"):
                element = element.right()
            drawing += element

    def _element_for(self, elm, component: LayoutComponent):
        key = _key(component.type)
        if "op_amp" in key or "opamp" in key or "operational_amplifier" in key:
            return _opamp_element(elm)
        if "resistor" in key or key.startswith("r"):
            return _first_attr(elm, ["ResistorIEC", "Resistor"])
        if "capacitor" in key or key.startswith("c"):
            return _first_attr(elm, ["Capacitor"])
        if key in {"ground", "gnd"}:
            return _first_attr(elm, ["GroundSignal", "Ground"])
        if key in {"input", "output", "voltage_source", "source", "input_terminal"} or "source" in key:
            return _first_attr(elm, ["Dot"])
        return _first_attr(elm, ["Rect", "Dot"])


def _first_attr(module, names: list[str]):
    for name in names:
        factory = getattr(module, name, None)
        if factory is not None:
            try:
                return factory()
            except Exception:
                continue
    return None


def _opamp_element(elm):
    for name in ["Opamp", "OpAmp"]:
        factory = getattr(elm, name, None)
        if factory is None:
            continue
        try:
            return factory(leads=True)
        except TypeError:
            try:
                return factory()
            except Exception:
                continue
    return _first_attr(elm, ["Triangle"])


def _drawing_to_svg(drawing) -> str:
    for method_name in ["get_imagedata", "getimage"]:
        method = getattr(drawing, method_name, None)
        if method is None:
            continue
        try:
            data = method("svg")
            if isinstance(data, bytes):
                return data.decode("utf-8")
            if isinstance(data, str):
                return data
        except Exception:
            continue
    try:
        buffer = StringIO()
        drawing.save(buffer)
        return buffer.getvalue()
    except Exception:
        return ""


def _render_native_motif_svg(schemdraw, elm, layout: LayoutPlan) -> str | None:
    labels = {
        component.id: display_label_text(
            component_display_label(component.id, component.type, component.label, component.value)
        )
        for component in layout.components
    }
    motif = _layout_motif(layout)
    builders = {
        "voltage_divider": _native_voltage_divider,
        "rc_low_pass": _native_rc_low_pass,
        "non_inverting_op_amp": _native_non_inverting_op_amp,
        "instrumentation_amplifier": _native_instrumentation_amplifier,
        "bridge_or_wheatstone": _native_bridge,
    }
    builder = builders.get(motif)
    if builder is None:
        return None
    drawing = schemdraw.Drawing(show=False)
    if hasattr(drawing, "config"):
        drawing.config(unit=2.0, fontsize=10, lw=1.2)
    builder(drawing, elm, labels)
    svg = _drawing_to_svg(drawing)
    return svg or None


def _layout_motif(layout: LayoutPlan) -> str:
    for warning in layout.warnings:
        if warning.startswith("motif:"):
            return _key(warning.split(":", 1)[1].strip())
    return _key(layout.circuit_id)


def _native_voltage_divider(drawing, elm, labels: dict[str, str]) -> None:
    drawing += elm.Dot(open=True).label(labels.get("VIN", "VIN"), loc="left")
    drawing += elm.Line().right(2)
    drawing += elm.ResistorIEC().down().label(labels.get("R1", "R1"), loc="bottom")
    drawing += elm.Dot()
    drawing.push()
    drawing += elm.Line().right(2)
    drawing += elm.Dot(open=True).label(labels.get("VOUT", "VOUT"), loc="right")
    drawing.pop()
    drawing += elm.ResistorIEC().down().label(labels.get("R2", "R2"), loc="bottom")
    drawing += elm.Line().down(0.35)
    drawing += elm.Ground().label(labels.get("GND", "GND"), loc="bottom")


def _native_rc_low_pass(drawing, elm, labels: dict[str, str]) -> None:
    drawing += elm.Dot(open=True).label(labels.get("VIN", "VIN"), loc="left")
    drawing += elm.Line().right(1.5)
    drawing += elm.ResistorIEC().right().label(labels.get("R1", "R"), loc="top")
    drawing += elm.Dot()
    drawing.push()
    drawing += elm.Line().right(2)
    drawing += elm.Dot(open=True).label(labels.get("VOUT", "VOUT"), loc="right")
    drawing.pop()
    drawing += elm.Capacitor().down().label(labels.get("C1", "C"), loc="bottom")
    drawing += elm.Line().down(0.45)
    drawing += elm.Ground().label(labels.get("GND", "GND"), loc="bottom")


def _native_non_inverting_op_amp(drawing, elm, labels: dict[str, str]) -> None:
    opamp = drawing.add(elm.Opamp(leads=True).at((4.0, 0.0)).right().label(labels.get("U1", "U1"), loc="top"))
    minus = opamp.absanchors["in1"]
    plus = opamp.absanchors["in2"]
    output = opamp.absanchors["out"]
    feedback = (minus.x - 1.6, minus.y)
    vout = (output.x + 2.0, output.y)

    input_start = (feedback[0] + 1.15, plus.y)
    drawing += elm.Dot(open=True).at(input_start).label(labels.get("VIN", "VIN"), loc="left")
    drawing += elm.Line().at(input_start).to(plus)
    drawing += elm.Line().at(output).to(vout)
    drawing += elm.Dot(open=True).at(vout).label(labels.get("VOUT", "VOUT"), loc="right")

    drawing += elm.Line().at(minus).to(feedback)
    drawing += elm.Dot().at(feedback)
    top_left = (feedback[0], minus.y + 1.65)
    top_right = (output.x, minus.y + 1.65)
    drawing += elm.Line().at(feedback).to(top_left)
    drawing += elm.ResistorIEC().at(top_left).to(top_right).label(labels.get("Rf", "Rf"), loc="top")
    drawing += elm.Line().at(top_right).to(output)

    drawing += elm.ResistorIEC().at(feedback).down().label(labels.get("Rg", "Rg"), loc="bottom")
    drawing += elm.Line().down(0.45)
    drawing += elm.Ground().label(labels.get("GND", "GND"), loc="bottom")


def _native_instrumentation_amplifier(drawing, elm, labels: dict[str, str]) -> None:
    u1 = drawing.add(elm.Opamp(leads=True).at((4.0, 2.8)).right().label(labels.get("U1", "U1"), loc="top"))
    u2 = drawing.add(elm.Opamp(leads=True).at((4.0, -2.8)).right().label(labels.get("U2", "U2"), loc="bottom"))
    u3 = drawing.add(elm.Opamp(leads=True).at((12.4, 0.0)).right().label(labels.get("U3", "U3"), loc="top"))

    u1_minus, u1_plus, u1_out = u1.absanchors["in1"], u1.absanchors["in2"], u1.absanchors["out"]
    u2_minus, u2_plus, u2_out = u2.absanchors["in1"], u2.absanchors["in2"], u2.absanchors["out"]
    u3_minus, u3_plus, u3_out = u3.absanchors["in1"], u3.absanchors["in2"], u3.absanchors["out"]

    rg_top = (0.55, u1_minus.y)
    rg_bottom = (0.55, u2_minus.y)
    drawing += elm.Line().at(rg_top).to(u1_minus)
    drawing += elm.Line().at(rg_bottom).to(u2_minus)
    drawing += elm.ResistorIEC().at(rg_top).to(rg_bottom).label(labels.get("Rg", "Rg"), loc="top")

    drawing += elm.Dot(open=True).at((1.15, u1_plus.y)).label(labels.get("INP", "IN+"), loc="left")
    drawing += elm.Line().at((1.15, u1_plus.y)).to(u1_plus)
    drawing += elm.Dot(open=True).at((1.15, u2_plus.y)).label(labels.get("INN", "IN-"), loc="left")
    drawing += elm.Line().at((1.15, u2_plus.y)).to(u2_plus)

    fb1_left = (rg_top[0], u1_minus.y + 1.55)
    fb1_right = (u1_out.x, u1_minus.y + 1.55)
    drawing += elm.Line().at(rg_top).to(fb1_left)
    drawing += elm.ResistorIEC().at(fb1_left).to(fb1_right).label(labels.get("R1", "R1"), loc="top")
    drawing += elm.Line().at(fb1_right).to(u1_out)

    fb2_left = (rg_bottom[0], u2_minus.y - 1.55)
    fb2_right = (u2_out.x, u2_minus.y - 1.55)
    drawing += elm.Line().at(rg_bottom).to(fb2_left)
    drawing += elm.ResistorIEC().at(fb2_left).to(fb2_right).label(labels.get("R2", "R2"), loc="bottom")
    drawing += elm.Line().at(fb2_right).to(u2_out)

    r5_end = (u3_minus.x, u3_minus.y)
    r3_end = (u3_plus.x, u3_plus.y)
    drawing += elm.ResistorIEC().at((u1_out.x + 1.0, u3_minus.y)).to((u3_minus.x - 0.35, u3_minus.y)).label(labels.get("R5", "R5"), loc="top")
    drawing += elm.Line().at(u1_out).to((u1_out.x + 0.7, u1_out.y)).to((u1_out.x + 0.7, u3_minus.y)).to((u1_out.x + 1.0, u3_minus.y))
    drawing += elm.Line().at((u3_minus.x - 0.35, u3_minus.y)).to(r5_end)

    drawing += elm.ResistorIEC().at((u2_out.x + 1.0, u3_plus.y)).to((u3_plus.x - 0.35, u3_plus.y)).label(labels.get("R3", "R3"), loc="bottom")
    drawing += elm.Line().at(u2_out).to((u2_out.x + 0.7, u2_out.y)).to((u2_out.x + 0.7, u3_plus.y)).to((u2_out.x + 1.0, u3_plus.y))
    drawing += elm.Line().at((u3_plus.x - 0.35, u3_plus.y)).to(r3_end)

    drawing += elm.ResistorIEC().at((u3_plus.x - 0.75, u3_plus.y)).down().label(labels.get("R4", "R4"), loc="bottom")
    drawing += elm.Line().down(0.35)
    drawing += elm.Ground().label(labels.get("GND", "GND"), loc="bottom")

    out_node = (u3_out.x + 1.3, u3_out.y)
    drawing += elm.Line().at(u3_out).to(out_node)
    drawing += elm.Dot(open=True).at(out_node).label(labels.get("VOUT", "VOUT"), loc="right")
    fb6_left = (u3_minus.x - 0.35, u3_minus.y + 1.35)
    fb6_right = (u3_out.x, u3_minus.y + 1.35)
    drawing += elm.Line().at((u3_minus.x - 0.35, u3_minus.y)).to(fb6_left)
    drawing += elm.ResistorIEC().at(fb6_left).to(fb6_right).label(labels.get("R6", "R6"), loc="top")
    drawing += elm.Line().at(fb6_right).to(u3_out)


def _native_bridge(drawing, elm, labels: dict[str, str]) -> None:
    top_left = (2.0, 2.0)
    mid_left = (2.0, 0.0)
    bottom_left = (2.0, -2.0)
    top_right = (6.0, 2.0)
    mid_right = (6.0, 0.0)
    bottom_right = (6.0, -2.0)
    drawing += elm.Dot(open=True).at((0.0, 2.0)).label(labels.get("VIN", "VIN"), loc="left")
    drawing += elm.Line().at((0.0, 2.0)).to(top_left)
    drawing += elm.ResistorIEC().at(top_left).to(mid_left).label(labels.get("R1", "R1"), loc="bottom")
    drawing += elm.ResistorIEC().at(mid_left).to(bottom_left).label(labels.get("R2", "R2"), loc="bottom")
    drawing += elm.Line().at(top_left).to(top_right)
    drawing += elm.ResistorIEC().at(top_right).to(mid_right).label(labels.get("R3", "R3"), loc="bottom")
    drawing += elm.ResistorIEC().at(mid_right).to(bottom_right).label(labels.get("R4", "R4"), loc="bottom")
    drawing += elm.Line().at(bottom_left).to(bottom_right)
    drawing += elm.Line().at(mid_left).right(2.3)
    drawing += elm.Dot(open=True).label(labels.get("VOUT", "VOUT"), loc="right")
    drawing += elm.Line().at(((bottom_left[0] + bottom_right[0]) / 2.0, bottom_left[1])).down(0.8)
    drawing += elm.Ground().label(labels.get("GND", "GND"), loc="bottom")


def _normalize_svg_canvas(svg: str, layout: LayoutPlan) -> str:
    svg_start = svg.find("<svg")
    if svg_start == -1:
        raise ValueError("Schemdraw output did not contain an SVG root.")
    head_end = svg.find(">", svg_start)
    if head_end == -1:
        raise ValueError("Schemdraw output had an unterminated SVG root.")
    before = svg[:svg_start]
    head = svg[svg_start:head_end]
    close = svg.rfind("</svg>")
    if close == -1:
        raise ValueError("Schemdraw output did not close the SVG root.")
    inner = svg[head_end + 1 : close]
    for attr in ["width", "height", "viewBox"]:
        if f"{attr}=" in head:
            prefix, rest = head.split(f"{attr}=", 1)
            quote = rest[0]
            end = rest.find(quote, 1)
            if end != -1:
                head = prefix + rest[end + 1 :]
    scale = layout.grid / SCHEMDRAW_UNIT_PX
    head = f'{head} width="{layout.width}" height="{layout.height}" viewBox="0 0 {layout.width} {layout.height}"'
    return (
        before
        + head
        + ">\n"
        + _visible_wires_svg(layout)
        + "\n"
        + f'<g id="schemdraw-canvas" transform="scale({scale:.8g})">\n'
        + inner
        + "\n</g>\n</svg>"
    )


def _normalize_native_svg_canvas(svg: str, layout: LayoutPlan) -> str:
    svg_start = svg.find("<svg")
    if svg_start == -1:
        raise ValueError("Schemdraw output did not contain an SVG root.")
    head_end = svg.find(">", svg_start)
    if head_end == -1:
        raise ValueError("Schemdraw output had an unterminated SVG root.")
    close = svg.rfind("</svg>")
    if close == -1:
        raise ValueError("Schemdraw output did not close the SVG root.")
    inner = svg[head_end + 1 : close]
    viewbox = _root_viewbox(svg)
    if viewbox is None:
        viewbox = (0.0, 0.0, float(layout.width), float(layout.height))
    vx, vy, vw, vh = viewbox
    target_width = layout.width * 0.78
    target_height = layout.height * 0.72
    scale = min(target_width / max(vw, 1.0), target_height / max(vh, 1.0), 2.3)
    tx = (layout.width - vw * scale) / 2.0
    ty = (layout.height - vh * scale) / 2.0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{layout.width}" height="{layout.height}" '
        f'viewBox="0 0 {layout.width} {layout.height}">\n'
        f'<g id="schemdraw-canvas" transform="translate({tx:.3f} {ty:.3f}) scale({scale:.5f}) '
        f'translate({-vx:.3f} {-vy:.3f})">\n'
        f"{inner}\n"
        "</g>\n</svg>"
    )


def _root_viewbox(svg: str) -> tuple[float, float, float, float] | None:
    import re

    match = re.search(r"<svg\b[^>]*\bviewBox=(\"|')([^\"']+)\1", svg)
    if not match:
        return None
    values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(2))]
    if len(values) != 4:
        return None
    return tuple(values)  # type: ignore[return-value]


def _visible_wires_svg(layout: LayoutPlan) -> str:
    parts = [
        '<g id="optcpv-visible-wires" fill="none" stroke="#111827" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round">'
    ]
    for wire in layout.wires:
        parts.append(_wire_svg(layout, wire))
    parts.append("</g>")
    parts.append('<g id="optcpv-visible-junctions" fill="#111827" stroke="none">')
    for wire in layout.wires:
        if len(wire.connected_pins) >= 3:
            junction = _junction_point(wire)
            if junction is not None:
                parts.append(
                    f'<circle cx="{junction.x * layout.grid:.1f}" cy="{junction.y * layout.grid:.1f}" r="3.6" '
                    f'data-net-name="{escape(wire.net)}"/>'
                )
    parts.append("</g>")
    return "\n".join(parts)


def _wire_svg(layout: LayoutPlan, wire: LayoutWire) -> str:
    segments = []
    for start, end in _unique_segments(wire.points):
        segments.append(
            f'<line x1="{start.x * layout.grid:.1f}" y1="{start.y * layout.grid:.1f}" '
            f'x2="{end.x * layout.grid:.1f}" y2="{end.y * layout.grid:.1f}" '
            f'data-net-name="{escape(wire.net)}"/>'
        )
    return "\n".join(segments)


def _unique_segments(points: list[Point]) -> list[tuple[Point, Point]]:
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    segments: list[tuple[Point, Point]] = []
    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        a = (round(start.x, 4), round(start.y, 4))
        b = (round(end.x, 4), round(end.y, 4))
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        segments.append((start, end))
    return segments


def _junction_point(wire: LayoutWire) -> Point | None:
    counts: dict[Point, int] = {}
    for point in wire.points:
        counts[point] = counts.get(point, 0) + 1
    if not counts:
        return None
    point = max(counts, key=counts.get)
    return point if counts[point] > 1 else None


def _element_anchor(component: LayoutComponent) -> Point:
    key = _key(component.type)
    if "resistor" in key or key.startswith("r") or "capacitor" in key or key.startswith("c"):
        if component.orientation in {"up", "north"}:
            return Point(component.x, component.y + 0.95)
        if component.orientation in {"down", "south"}:
            return Point(component.x, component.y - 0.95)
        if component.orientation in {"left", "west"}:
            return Point(component.x + 0.95, component.y)
        return Point(component.x - 0.95, component.y)
    return Point(component.x, component.y)


def _is_two_terminal(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "resistor" in key or key.startswith("r") or "capacitor" in key or key.startswith("c")


def _terminal_length(component: LayoutComponent) -> float:
    return 1.9 if component.orientation in {"up", "down", "north", "south", "left", "right", "east", "west"} else 2.0


def _add_visible_labels(svg: str, layout: LayoutPlan) -> str:
    label_svg = "\n".join(_label_svg(layout, label) for label in layout.labels)
    if "</svg>" in svg:
        return svg.replace("</svg>", label_svg + "\n</svg>")
    return svg + label_svg


def _label_svg(layout: LayoutPlan, label: LayoutLabel) -> str:
    owner = next((component for component in layout.components if component.id == label.owner_id), None)
    anchor = "middle"
    if owner is not None:
        if label.x > owner.bbox.right:
            anchor = "start"
        elif label.x < owner.bbox.x:
            anchor = "end"
    dy = "0.32em"
    if owner is not None and label.y < owner.bbox.y:
        dy = "0"
    lines = wrap_label_lines(label.text)
    if len(lines) == 1:
        inner = escape(lines[0])
    else:
        start_y = label.y * layout.grid - (len(lines) - 1) * 8.3
        inner = "".join(
            f'<tspan x="{label.x * layout.grid:.1f}" y="{start_y + index * 16.6:.1f}">{escape(line)}</tspan>'
            for index, line in enumerate(lines)
        )
    return (
        f'<text class="optcpv-visible-label" x="{label.x * layout.grid:.1f}" y="{label.y * layout.grid:.1f}" '
        f'data-label-id="{escape(label.id)}" data-label-owner-id="{escape(label.owner_id)}" '
        f'text-anchor="{anchor}" dominant-baseline="middle" dy="{dy}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#111827">{inner}</text>'
    )


def _sd(point: Point) -> tuple[float, float]:
    return (point.x, -point.y)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

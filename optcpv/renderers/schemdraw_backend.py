"""Schemdraw-backed renderer for OptCPV layout plans."""

from __future__ import annotations

from io import StringIO

from ..models import LayoutComponent, LayoutPlan, Point
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

        drawing = schemdraw.Drawing(show=False)
        if hasattr(drawing, "config"):
            drawing.config(unit=1)
        self._add_wires(drawing, elm, layout)
        self._add_components(drawing, elm, layout)
        raw = _drawing_to_svg(drawing)
        if not raw:
            raise RuntimeError("Schemdraw did not return SVG image data.")
        return _normalize_svg_canvas(raw, layout)

    def _add_wires(self, drawing, elm, layout: LayoutPlan) -> None:
        for wire in layout.wires:
            for start, end in zip(wire.points, wire.points[1:]):
                drawing += elm.Line().at(_sd(start)).to(_sd(end))

    def _add_components(self, drawing, elm, layout: LayoutPlan) -> None:
        for component in layout.components:
            element = self._element_for(elm, component)
            if element is None:
                continue
            element = element.at((component.x, -component.y))
            if component.orientation in {"left", "west"} and hasattr(element, "left"):
                element = element.left()
            elif component.orientation in {"up", "north"} and hasattr(element, "up"):
                element = element.up()
            elif component.orientation in {"down", "south"} and hasattr(element, "down"):
                element = element.down()
            elif hasattr(element, "right"):
                element = element.right()
            label = component.label or component.value or component.id
            if hasattr(element, "label"):
                element = element.label(label)
            drawing += element

    def _element_for(self, elm, component: LayoutComponent):
        key = _key(component.type)
        if "op_amp" in key or "opamp" in key or "operational_amplifier" in key:
            return _first_attr(elm, ["Opamp", "OpAmp", "Triangle"])
        if "resistor" in key or key.startswith("r"):
            return _first_attr(elm, ["ResistorIEC", "Resistor"])
        if "capacitor" in key or key.startswith("c"):
            return _first_attr(elm, ["Capacitor"])
        if key in {"ground", "gnd"}:
            return _first_attr(elm, ["Ground"])
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
        + f'<g id="schemdraw-canvas" transform="scale({scale:.8g})">\n'
        + inner
        + "\n</g>\n</svg>"
    )


def _sd(point: Point) -> tuple[float, float]:
    return (point.x, -point.y)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

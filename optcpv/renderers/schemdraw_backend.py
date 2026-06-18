"""Schemdraw-backed renderer for OptCPV layout plans."""

from __future__ import annotations

from html import escape
from io import StringIO

from ..labels import component_display_label, display_label_text, wrap_label_lines
from ..models import LayoutComponent, LayoutLabel, LayoutPlan, LayoutWire, LocalTerminalIntent, Motif, Point
from ..segments import junction_points, merged_axis_aligned_segments
from .svg_postprocess import draw_wires_with_jumps, inject_metadata, render_debug_svg, _set_root_attr


FALLBACK_RENDERER_ID = "optcpv.debug_svg_after_schemdraw_error"
SCHEMDRAW_UNIT_PX = 36.0
NATIVE_SCHEMDRAW_FONT_SIZE = 12
NATIVE_SCHEMDRAW_LINE_WIDTH = 1.35
NATIVE_TARGET_WIDTH_RATIO = 0.86
NATIVE_TARGET_HEIGHT_RATIO = 0.78
NATIVE_MAX_SCALE = 2.65
_CASCADE_OPAMP_PIN_NAMES = {
    "+",
    "plus",
    "in+",
    "non_inverting",
    "noninverting",
    "-",
    "minus",
    "in-",
    "inverting",
    "out",
    "output",
}


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
            if _is_redundant_terminal_component(layout, component):
                continue
            if _is_filter_block(component):
                continue
            element = self._element_for(elm, component)
            if element is None:
                continue
            anchor = _element_anchor(component)
            element = element.at(_sd(anchor))
            if _is_two_terminal(component) and hasattr(element, "length"):
                element = element.length(_terminal_length(component))
            if _is_flipped_opamp(component) and hasattr(element, "flip"):
                element = element.flip()
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
        if "inductor" in key or key.startswith("l"):
            return _first_attr(elm, ["Inductor"])
        if "diode" in key:
            return _first_attr(elm, ["Photodiode", "Diode"]) if "photo" in key else _first_attr(elm, ["Diode"])
        if "switch" in key or key.startswith("sw"):
            return _first_attr(elm, ["Switch"])
        if _is_filter_block(component):
            return None
        if key in {"ground", "gnd"}:
            return _first_attr(elm, ["GroundSignal", "Ground"])
        if _is_physical_source(component):
            if "current" in key:
                return _first_attr(elm, ["SourceI", "Source"])
            if "voltage" in key or key == "source":
                return _first_attr(elm, ["SourceV", "Source"])
        if key in {"input", "output", "input_terminal"} or ("source" in key and len(component.pins) <= 1):
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
    if motif == "op_amp_network":
        drawing = schemdraw.Drawing(show=False)
        if hasattr(drawing, "config"):
            drawing.config(unit=2.0, fontsize=NATIVE_SCHEMDRAW_FONT_SIZE, lw=NATIVE_SCHEMDRAW_LINE_WIDTH)
        if not _native_op_amp_cascade(drawing, elm, labels, layout):
            return None
        svg = _drawing_to_svg(drawing)
        return svg or None
    builder = builders.get(motif)
    if builder is None:
        return None
    drawing = schemdraw.Drawing(show=False)
    if hasattr(drawing, "config"):
        drawing.config(unit=2.0, fontsize=NATIVE_SCHEMDRAW_FONT_SIZE, lw=NATIVE_SCHEMDRAW_LINE_WIDTH)
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
    drawing += elm.Ground()


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
    drawing += elm.Ground()


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
    drawing += elm.Ground()


def _native_op_amp_cascade(drawing, elm, labels: dict[str, str], layout: LayoutPlan) -> bool:
    cascade = _cascade_description(layout)
    if cascade is None:
        return False

    prev_out_anchor = None
    for stage in cascade["stages"]:
        prev_out_anchor = _draw_perfect_cascade_stage(
            drawing,
            elm,
            labels.get(stage["opamp"].id, stage["opamp"].id),
            labels.get(stage["feedback"].id, stage["feedback"].id),
            labels.get(stage["gain"].id, stage["gain"].id),
            prev_out_anchor=prev_out_anchor,
            input_label=labels.get(cascade["input"].id, cascade["input"].id),
        )

    drawing.add(elm.Line().right().length(0.95).at(prev_out_anchor))
    drawing.add(elm.Dot(open=True).label(labels.get(cascade["output"].id, cascade["output"].id), loc="right"))
    return True


def _draw_perfect_cascade_stage(
    drawing,
    elm,
    u_name: str,
    rf_name: str,
    rg_name: str,
    *,
    prev_out_anchor=None,
    input_label: str,
):
    input_y = 0.0
    if prev_out_anchor is not None:
        drawing.add(elm.Line().right().length(0.58).at(prev_out_anchor))
        drawing.add(elm.Line().toy(input_y))
        drawing.add(elm.Line().right().length(0.58))
    else:
        input_start = (0.0, input_y)
        drawing.add(elm.Dot(open=True).at(input_start).label(input_label, loc="left"))
        drawing.add(elm.Line().right().length(1.5).at(input_start))

    opamp = drawing.add(elm.Opamp(leads=True).flip().anchor("in2").label(u_name, loc="top"))
    out_wire = drawing.add(elm.Line().right().length(1.2).at(opamp.out))
    drawing.add(elm.Dot().at(out_wire.end))

    drawing.push()
    drawing.add(elm.Line().down().length(1.2).at(opamp.in1))
    drawing.add(elm.Dot())

    drawing.push()
    drawing.add(elm.ResistorIEC().down().label(rg_name, loc="right"))
    drawing.add(elm.Ground())
    drawing.pop()

    drawing.add(elm.ResistorIEC().right().tox(out_wire.end).label(rf_name, loc="bottom"))
    drawing.add(elm.Line().toy(out_wire.end))
    drawing.pop()

    return out_wire.end


def _cascade_description(layout: LayoutPlan) -> dict | None:
    components = {component.id: component for component in layout.components}
    opamps = [component for component in layout.components if _is_opamp_component(component)]
    if len(opamps) < 2:
        return None

    inputs = [component for component in layout.components if _is_input_or_source_component(component)]
    outputs = [component for component in layout.components if _is_output_component(component)]
    resistors = [component for component in layout.components if _is_resistor(component)]
    if len(inputs) != 1 or len(outputs) != 1:
        return None

    allowed_ids = {component.id for component in [*opamps, *inputs, *outputs, *resistors]}
    allowed_ids.update(component.id for component in layout.components if _is_standalone_terminal_component(component))
    if set(components) - allowed_ids:
        return None

    opamp_by_input_net: dict[str, LayoutComponent] = {}
    opamp_output_nets: set[str] = set()
    for opamp in opamps:
        if any(not _is_cascade_opamp_pin(pin_name) for pin_name in opamp.pins):
            return None
        plus_net = _component_pin_net(opamp, ["+", "plus", "in+", "non_inverting", "noninverting"])
        minus_net = _component_pin_net(opamp, ["-", "minus", "in-", "inverting"])
        output_net = _component_pin_net(opamp, ["out", "output"])
        if plus_net is None or minus_net is None or output_net is None:
            return None
        if plus_net in opamp_by_input_net:
            return None
        opamp_by_input_net[plus_net] = opamp
        opamp_output_nets.add(output_net)

    current_net = next(iter(inputs[0].pins.values()), None)
    if current_net is None or current_net in opamp_output_nets:
        return None

    ground_nets = _cascade_ground_nets(layout)
    used_resistors: set[str] = set()
    stages: list[dict[str, LayoutComponent]] = []
    while current_net in opamp_by_input_net:
        opamp = opamp_by_input_net.pop(current_net)
        minus_net = _component_pin_net(opamp, ["-", "minus", "in-", "inverting"])
        output_net = _component_pin_net(opamp, ["out", "output"])
        if minus_net is None or output_net is None:
            return None
        feedback = _resistor_between(resistors, output_net, minus_net, used_resistors)
        gain = _resistor_between_any(resistors, minus_net, ground_nets, used_resistors)
        if feedback is None or gain is None:
            return None
        used_resistors.update({feedback.id, gain.id})
        stages.append({"opamp": opamp, "feedback": feedback, "gain": gain})
        current_net = output_net

    if opamp_by_input_net or len(stages) != len(opamps):
        return None
    if current_net not in set(outputs[0].pins.values()):
        return None
    if any(resistor.id not in used_resistors for resistor in resistors):
        return None
    gain_ids = {stage["gain"].id for stage in stages}
    if any(terminal.component_id not in gain_ids or terminal.terminal_type != "ground" for terminal in layout.semantic.local_terminals):
        return None
    return {"input": inputs[0], "output": outputs[0], "stages": stages}


def _cascade_ground_nets(layout: LayoutPlan) -> set[str]:
    ground_nets = {terminal.net for terminal in layout.semantic.local_terminals if terminal.terminal_type == "ground"}
    for component in layout.components:
        if _is_standalone_terminal_component(component):
            ground_nets.update(component.pins.values())
    ground_nets.update(net for net in layout.net_to_pins if _key(net) in {"gnd", "ground", "agnd", "dgnd"})
    return ground_nets


def _resistor_between(
    resistors: list[LayoutComponent],
    net_a: str,
    net_b: str,
    used_resistors: set[str],
) -> LayoutComponent | None:
    targets = {net_a, net_b}
    matches = [
        resistor
        for resistor in resistors
        if resistor.id not in used_resistors and set(resistor.pins.values()) == targets
    ]
    return matches[0] if len(matches) == 1 else None


def _resistor_between_any(
    resistors: list[LayoutComponent],
    net_a: str,
    net_b_options: set[str],
    used_resistors: set[str],
) -> LayoutComponent | None:
    matches = []
    for resistor in resistors:
        if resistor.id in used_resistors:
            continue
        nets = set(resistor.pins.values())
        if net_a in nets and nets - {net_a} and nets - {net_a} <= net_b_options:
            matches.append(resistor)
    return matches[0] if len(matches) == 1 else None


def _component_pin_net(component: LayoutComponent, names: list[str]) -> str | None:
    normalized = {_key(name) for name in names}
    for pin_name, net in component.pins.items():
        if _key(pin_name) in normalized:
            return net
    return None


def _is_cascade_opamp_pin(pin_name: str) -> bool:
    return any(_key(pin_name) == _key(allowed) for allowed in _CASCADE_OPAMP_PIN_NAMES)


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
    drawing += elm.Ground()

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
    drawing += elm.Ground()


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
        + _semantic_overlays_svg(layout)
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
    target_width = layout.width * NATIVE_TARGET_WIDTH_RATIO
    target_height = layout.height * NATIVE_TARGET_HEIGHT_RATIO
    scale = min(target_width / max(vw, 1.0), target_height / max(vh, 1.0), NATIVE_MAX_SCALE)
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
    parts.append(draw_wires_with_jumps(layout, class_name=""))
    parts.append("</g>")
    parts.append('<g id="optcpv-visible-junctions" fill="#111827" stroke="none">')
    for wire in layout.wires:
        if len(wire.connected_pins) >= 3:
            for junction in _junction_points(wire):
                parts.append(
                    f'<circle cx="{junction.x * layout.grid:.1f}" cy="{junction.y * layout.grid:.1f}" r="3.6" '
                    f'data-net-name="{escape(wire.net)}"/>'
                )
    parts.append("</g>")
    return "\n".join(parts)


def draw_signal_route(layout: LayoutPlan, wire: LayoutWire) -> str:
    """Draw an orthogonal signal route from the topology-semantic plan."""

    segments = []
    for start, end in _unique_segments(wire.points):
        segments.append(
            f'<line x1="{start.x * layout.grid:.1f}" y1="{start.y * layout.grid:.1f}" '
            f'x2="{end.x * layout.grid:.1f}" y2="{end.y * layout.grid:.1f}" '
            f'data-net-name="{escape(wire.net)}"/>'
        )
    return "\n".join(segments)


def draw_local_supply_or_ground(layout: LayoutPlan, terminal: LocalTerminalIntent) -> str:
    """Draw one local terminal symbol near its owning pin, with no global bus."""

    pin = layout.pin_map.get((terminal.component_id, terminal.pin_name))
    if pin is None:
        return ""
    direction = terminal.preferred_direction
    label = terminal.label or terminal.net
    if terminal.terminal_type == "signal_label":
        return _draw_signal_label_terminal(layout, terminal, pin, label)
    if terminal.terminal_type == "positive_supply":
        return _draw_supply_terminal(layout, terminal, Point(pin.x, pin.y), label, up=True)
    if terminal.terminal_type == "negative_supply":
        return _draw_supply_terminal(layout, terminal, Point(pin.x, pin.y), label, up=False)
    return _draw_ground_terminal(layout, terminal, Point(pin.x, pin.y), label, down=direction != "up")


def draw_resistor_to_ground_leg(layout: LayoutPlan, terminal: LocalTerminalIntent) -> str:
    return draw_local_supply_or_ground(layout, terminal)


def _draw_signal_label_terminal(
    layout: LayoutPlan,
    terminal: LocalTerminalIntent,
    pin,
    label: str,
) -> str:
    direction = -1 if terminal.preferred_direction == "left" or (terminal.preferred_direction not in {"right", "left"} and pin.side == "left") else 1
    x0 = pin.x * layout.grid
    y = pin.y * layout.grid
    x1 = (pin.x + direction * 0.52) * layout.grid
    label_x = (pin.x + direction * 0.66) * layout.grid
    label_y = y - 7.5
    anchor = "end" if direction < 0 else "start"
    return "\n".join(
        [
            f'<g class="optcpv-local-terminal" data-local-terminal="true" '
            f'data-component-id="{escape(terminal.component_id)}" data-pin-name="{escape(terminal.pin_name)}" '
            f'data-net-name="{escape(terminal.net)}" data-terminal-type="{escape(terminal.terminal_type)}">',
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
            'stroke="#111827" stroke-width="2.2" stroke-linecap="round"/>',
            f'<circle cx="{x0:.1f}" cy="{y:.1f}" r="3.8" fill="#ffffff" stroke="#111827" stroke-width="1.8"/>',
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" dominant-baseline="auto" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f"{escape(label)}</text>",
            "</g>",
        ]
    )


def draw_filter_block(layout: LayoutPlan, component: LayoutComponent) -> str:
    x = component.bbox.x * layout.grid
    y = component.bbox.y * layout.grid
    width = component.bbox.width * layout.grid
    height = component.bbox.height * layout.grid
    label = display_label_text(component_display_label(component.id, component.type, component.label, component.value))
    return "\n".join(
        [
            f'<g class="optcpv-filter-block" data-component-id="{escape(component.id)}" '
            f'data-component-type="{escape(component.type)}" data-motif-type="filter_block">',
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="6" '
            'fill="#ffffff" stroke="#111827" stroke-width="2.2"/>',
            f'<text x="{(component.x * layout.grid):.1f}" y="{(component.y * layout.grid):.1f}" '
            'text-anchor="middle" dominant-baseline="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#111827">'
            f"{escape(label)}</text>",
            "</g>",
        ]
    )


def draw_opamp_buffer(layout: LayoutPlan, motif: Motif) -> str:
    return _draw_motif_metadata(layout, motif)


def draw_opamp_feedback_stage(layout: LayoutPlan, motif: Motif) -> str:
    return _draw_motif_metadata(layout, motif)


def draw_summing_opamp(layout: LayoutPlan, motif: Motif) -> str:
    return _draw_motif_metadata(layout, motif)


def _semantic_overlays_svg(layout: LayoutPlan) -> str:
    parts = ['<g id="optcpv-semantic-overlays">']
    for component in layout.components:
        if _is_filter_block(component):
            parts.append(draw_filter_block(layout, component))
    for motif in layout.semantic.motifs:
        if motif.motif_type == "opamp_buffer":
            parts.append(draw_opamp_buffer(layout, motif))
        elif motif.motif_type == "op_amp_feedback_stage":
            parts.append(draw_opamp_feedback_stage(layout, motif))
        elif motif.motif_type == "summing_opamp":
            parts.append(draw_summing_opamp(layout, motif))
    for terminal in layout.semantic.local_terminals:
        if _should_hide_local_terminal(layout, terminal):
            continue
        if _terminal_owner_is_filter_block(layout, terminal):
            continue
        if _terminal_owner_is_resistor(layout, terminal):
            parts.append(draw_resistor_to_ground_leg(layout, terminal))
        else:
            parts.append(draw_local_supply_or_ground(layout, terminal))
    parts.append("</g>")
    return "\n".join(part for part in parts if part)


def _should_hide_local_terminal(layout: LayoutPlan, terminal: LocalTerminalIntent) -> bool:
    if terminal.terminal_type not in {"positive_supply", "negative_supply"}:
        return False
    owner = next((component for component in layout.components if component.id == terminal.component_id), None)
    return owner is not None and _is_opamp_component(owner)


def _draw_motif_metadata(layout: LayoutPlan, motif: Motif) -> str:
    boxes = [
        component.bbox
        for component in layout.components
        if component.id in set(motif.component_ids)
    ]
    if not boxes:
        return ""
    x = min(box.x for box in boxes) * layout.grid
    y = min(box.y for box in boxes) * layout.grid
    right = max(box.right for box in boxes) * layout.grid
    bottom = max(box.bottom for box in boxes) * layout.grid
    return (
        f'<rect class="optcpv-motif-metadata" x="{x:.1f}" y="{y:.1f}" '
        f'width="{right - x:.1f}" height="{bottom - y:.1f}" fill="none" stroke="none" opacity="0" '
        f'data-motif-id="{escape(motif.motif_id)}" data-motif-type="{escape(motif.motif_type)}" '
        f'data-component-ids="{escape(",".join(motif.component_ids))}"/>'
    )


def _draw_ground_terminal(
    layout: LayoutPlan,
    terminal: LocalTerminalIntent,
    pin: Point,
    label: str,
    *,
    down: bool,
) -> str:
    sign = 1 if down else -1
    x = pin.x * layout.grid
    y0 = pin.y * layout.grid
    stem_end = (pin.y + sign * 0.38) * layout.grid
    bar_y = (pin.y + sign * 0.5) * layout.grid
    label_y = (pin.y + sign * 1.05) * layout.grid
    return "\n".join(
        [
            f'<g class="optcpv-local-terminal" data-local-terminal="true" '
            f'data-component-id="{escape(terminal.component_id)}" data-pin-name="{escape(terminal.pin_name)}" '
            f'data-net-name="{escape(terminal.net)}" data-terminal-type="{escape(terminal.terminal_type)}">',
            f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{stem_end:.1f}" '
            'stroke="#111827" stroke-width="2.2" stroke-linecap="round"/>',
            f'<line x1="{x - 18:.1f}" y1="{bar_y:.1f}" x2="{x + 18:.1f}" y2="{bar_y:.1f}" '
            'stroke="#111827" stroke-width="2.2" stroke-linecap="round"/>',
            f'<line x1="{x - 11:.1f}" y1="{bar_y + sign * 8:.1f}" x2="{x + 11:.1f}" y2="{bar_y + sign * 8:.1f}" '
            'stroke="#111827" stroke-width="2.2" stroke-linecap="round"/>',
            f'<line x1="{x - 5:.1f}" y1="{bar_y + sign * 16:.1f}" x2="{x + 5:.1f}" y2="{bar_y + sign * 16:.1f}" '
            'stroke="#111827" stroke-width="2.2" stroke-linecap="round"/>',
            "</g>",
        ]
    )


def _draw_supply_terminal(
    layout: LayoutPlan,
    terminal: LocalTerminalIntent,
    pin: Point,
    label: str,
    *,
    up: bool,
) -> str:
    sign = -1 if up else 1
    x = pin.x * layout.grid
    y0 = pin.y * layout.grid
    y1 = (pin.y + sign * 0.58) * layout.grid
    label_y = (pin.y + sign * 0.98) * layout.grid
    arrow = "M {:.1f} {:.1f} L {:.1f} {:.1f} L {:.1f} {:.1f}".format(
        x - 8,
        y1 + sign * 7,
        x,
        y1,
        x + 8,
        y1 + sign * 7,
    )
    return "\n".join(
        [
            f'<g class="optcpv-local-terminal" data-local-terminal="true" '
            f'data-component-id="{escape(terminal.component_id)}" data-pin-name="{escape(terminal.pin_name)}" '
            f'data-net-name="{escape(terminal.net)}" data-terminal-type="{escape(terminal.terminal_type)}">',
            f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
            'stroke="#111827" stroke-width="2.2" stroke-linecap="round"/>',
            f'<path d="{arrow}" fill="none" stroke="#111827" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
            f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" dominant-baseline="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#374151">'
            f"{escape(label)}</text>",
            "</g>",
        ]
    )


def _unique_segments(points: list[Point]) -> list[tuple[Point, Point]]:
    return merged_axis_aligned_segments(points)


def _junction_points(wire: LayoutWire) -> list[Point]:
    return junction_points(wire.points)


def _terminal_owner_is_filter_block(layout: LayoutPlan, terminal: LocalTerminalIntent) -> bool:
    component = next((item for item in layout.components if item.id == terminal.component_id), None)
    return component is not None and _is_filter_block(component)


def _is_redundant_terminal_component(layout: LayoutPlan, component: LayoutComponent) -> bool:
    if _has_signal_label_terminal(layout, component) and _is_signal_label_terminal_component(component):
        return True
    if not _is_standalone_terminal_component(component):
        return False
    nets = set(component.pins.values())
    if not nets:
        return False
    local_nets = {terminal.net for terminal in layout.semantic.local_terminals}
    return bool(nets & local_nets)


def _has_signal_label_terminal(layout: LayoutPlan, component: LayoutComponent) -> bool:
    return any(
        terminal.component_id == component.id and terminal.terminal_type == "signal_label"
        for terminal in layout.semantic.local_terminals
    )


def _is_physical_source(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return len(component.pins) >= 2 and (
        key in {"voltage_source", "current_source", "source"} or "source" in key
    )


def _is_standalone_terminal_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"ground", "gnd", "supply", "power", "vcc", "vdd", "vee", "vss"}


def _is_signal_label_terminal_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"input", "output", "input_terminal"} or _is_standalone_terminal_component(component)


def _terminal_owner_is_resistor(layout: LayoutPlan, terminal: LocalTerminalIntent) -> bool:
    component = next((item for item in layout.components if item.id == terminal.component_id), None)
    return component is not None and _is_resistor(component)


def _is_opamp_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _is_input_or_source_component(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return key in {"input", "input_terminal", "source", "voltage_source"} or "source" in key


def _is_output_component(component: LayoutComponent) -> bool:
    return _key(component.type) in {"output", "output_terminal"}


def _is_resistor(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "resistor" in key or key.startswith("r")


def _is_flipped_opamp(component: LayoutComponent) -> bool:
    return _is_opamp_component(component) and "flip" in _key(component.orientation)


def _is_filter_block(component: LayoutComponent) -> bool:
    key = _key(component.type)
    label = _key(component.label)
    value = _key(component.value)
    return any(_filter_token(text) for text in [key, label, value])


def _filter_token(value: str) -> bool:
    return any(token in value for token in ["filter", "lpf", "hpf", "bpf", "bessel", "butterworth", "chebyshev"])


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
    label_svg = "\n".join(_label_svg(layout, label) for label in layout.labels if _is_visible_label(layout, label))
    if "</svg>" in svg:
        return svg.replace("</svg>", label_svg + "\n</svg>")
    return svg + label_svg


def _is_visible_label(layout: LayoutPlan, label: LayoutLabel) -> bool:
    owner = next((component for component in layout.components if component.id == label.owner_id), None)
    return owner is None or not _is_redundant_terminal_component(layout, owner)


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
    common = (
        f'x="{label.x * layout.grid:.1f}" y="{label.y * layout.grid:.1f}" '
        f'text-anchor="{anchor}" dominant-baseline="middle" dy="{dy}" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14"'
    )
    halo = (
        f'<text class="optcpv-visible-label-halo" {common} fill="none" stroke="#ffffff" '
        f'stroke-width="4" stroke-linejoin="round">{inner}</text>'
    )
    foreground = (
        f'<text class="optcpv-visible-label" {common} fill="#111827" '
        f'data-label-id="{escape(label.id)}" data-label-owner-id="{escape(label.owner_id)}">{inner}</text>'
    )
    return halo + foreground


def _sd(point: Point) -> tuple[float, float]:
    return (point.x, -point.y)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")

"""Interactive schematic artifact models and builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .critic import CriticReport
from .models import BBox, LayoutPlan, Point, RenderResult, dataclass_to_dict


ARTIFACT_VERSION = "schem_forge.artifact.v1"


@dataclass
class ViewBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class ArtifactBBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class PinArtifact:
    component_id: str
    pin_name: str
    net_name: str
    point: Point
    side: str


@dataclass
class ComponentArtifact:
    id: str
    type: str
    role: str | None
    display_label: str | None
    value_label: str | None
    bbox: ArtifactBBox
    label_bbox: ArtifactBBox | None
    pins: dict[str, PinArtifact]


@dataclass
class NetSegmentArtifact:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class NetArtifact:
    name: str
    connected_pins: list[str]
    segments: list[NetSegmentArtifact]
    bbox: ArtifactBBox | None
    junctions: list[Point]


@dataclass
class LabelArtifact:
    id: str
    text: str
    owner_id: str | None
    bbox: ArtifactBBox


@dataclass
class FocusRegion:
    id: str
    label: str
    kind: str
    bbox: ArtifactBBox
    components: list[str]
    nets: list[str]
    pins: list[str]
    labels: list[str]
    goals: list[str]


@dataclass
class ZoomPreset:
    id: str
    label: str
    viewbox: ViewBox
    focus_region_id: str | None
    target_type: str | None
    target_id: str | None


@dataclass
class HitTarget:
    id: str
    kind: Literal["component", "net", "pin", "label", "focus_region"]
    target_id: str
    bbox: ArtifactBBox


@dataclass
class OverlayArtifact:
    id: str
    kind: str
    label: str
    components: list[str]
    nets: list[str]
    pins: list[str]
    focus_region_id: str | None
    default_visible: bool = False


@dataclass
class SchematicArtifact:
    artifact_version: str
    circuit_id: str
    renderer: str
    svg: str
    svg_viewbox: ViewBox
    layout_plan: dict[str, Any]
    components: dict[str, ComponentArtifact]
    nets: dict[str, NetArtifact]
    labels: dict[str, LabelArtifact]
    focus_regions: list[FocusRegion]
    zoom_presets: list[ZoomPreset]
    hit_targets: list[HitTarget]
    overlays: list[OverlayArtifact]
    critic_report: dict[str, Any]
    warnings: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def _read(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


def _bbox_from_model(bbox: BBox) -> ArtifactBBox:
    return ArtifactBBox(
        x=float(bbox.x),
        y=float(bbox.y),
        width=float(bbox.width),
        height=float(bbox.height),
    )


def _point_bbox(point: Point, radius: float = 8.0) -> ArtifactBBox:
    return ArtifactBBox(point.x - radius, point.y - radius, radius * 2, radius * 2)


def _bbox_from_segment(segment: NetSegmentArtifact) -> ArtifactBBox:
    x = min(segment.x1, segment.x2)
    y = min(segment.y1, segment.y2)
    return ArtifactBBox(x, y, abs(segment.x2 - segment.x1), abs(segment.y2 - segment.y1))


def _padded_bbox(bbox: ArtifactBBox, padding: float) -> ArtifactBBox:
    return ArtifactBBox(
        bbox.x - padding,
        bbox.y - padding,
        bbox.width + padding * 2,
        bbox.height + padding * 2,
    )


def _union_bboxes(bboxes: list[ArtifactBBox]) -> ArtifactBBox | None:
    if not bboxes:
        return None
    x1 = min(bbox.x for bbox in bboxes)
    y1 = min(bbox.y for bbox in bboxes)
    x2 = max(bbox.right for bbox in bboxes)
    y2 = max(bbox.bottom for bbox in bboxes)
    return ArtifactBBox(x1, y1, x2 - x1, y2 - y1)


def _union_with_min_extent(bboxes: list[ArtifactBBox], min_size: float = 8.0) -> ArtifactBBox | None:
    expanded = []
    for bbox in bboxes:
        width = max(bbox.width, min_size)
        height = max(bbox.height, min_size)
        expanded.append(
            ArtifactBBox(
                bbox.x - (width - bbox.width) / 2.0,
                bbox.y - (height - bbox.height) / 2.0,
                width,
                height,
            )
        )
    return _union_bboxes(expanded)


def padded_viewbox(
    bbox: ArtifactBBox,
    canvas: ViewBox,
    padding_px: float = 80,
    min_width: float = 180,
    min_height: float = 140,
) -> ViewBox:
    width = max(bbox.width + padding_px * 2, min_width)
    height = max(bbox.height + padding_px * 2, min_height)
    cx = bbox.x + bbox.width / 2.0
    cy = bbox.y + bbox.height / 2.0
    x = cx - width / 2.0
    y = cy - height / 2.0

    if width >= canvas.width:
        x = canvas.x
        width = canvas.width
    else:
        x = min(max(x, canvas.x), canvas.x + canvas.width - width)

    if height >= canvas.height:
        y = canvas.y
        height = canvas.height
    else:
        y = min(max(y, canvas.y), canvas.y + canvas.height - height)

    return ViewBox(x=x, y=y, width=width, height=height)


def _motif(circuit_ir: Any, layout_plan: LayoutPlan) -> str:
    motif = _read(circuit_ir, "motif", _read(circuit_ir, "kind"))
    if motif:
        return _key(str(motif))
    for warning in layout_plan.warnings:
        if warning.startswith("motif:"):
            return _key(warning.split(":", 1)[1].strip())
    return ""


def _component_ids_by_role(layout_plan: LayoutPlan, *needles: str) -> list[str]:
    ids = []
    for component in layout_plan.components:
        role = _key(component.role)
        component_type = _key(component.type)
        if any(needle in role or needle in component_type for needle in needles):
            ids.append(component.id)
    return ids


def _component_ids_by_type(layout_plan: LayoutPlan, *needles: str) -> list[str]:
    return [
        component.id
        for component in layout_plan.components
        if any(needle in _key(component.type) for needle in needles)
    ]


def _input_component_ids(layout_plan: LayoutPlan) -> list[str]:
    ids = []
    for component in layout_plan.components:
        role = _key(component.role)
        component_type = _key(component.type)
        if role in {"input", "input_source", "input_terminal", "sensor"} or component_type in {
            "input",
            "input_terminal",
        }:
            ids.append(component.id)
    return ids


def _output_component_ids(layout_plan: LayoutPlan) -> list[str]:
    ids = []
    for component in layout_plan.components:
        role = _key(component.role)
        component_type = _key(component.type)
        if role == "output" or component_type == "output":
            ids.append(component.id)
    return ids


def _ground_component_ids(layout_plan: LayoutPlan) -> list[str]:
    ids = []
    for component in layout_plan.components:
        role = _key(component.role)
        component_type = _key(component.type)
        if role in {"ground", "gnd", "ground_symbol"} or component_type in {"ground", "gnd"}:
            ids.append(component.id)
    return ids


def _component_net_names(layout_plan: LayoutPlan, component_ids: list[str]) -> list[str]:
    nets: set[str] = set()
    for component_id in component_ids:
        nets.update(layout_plan.component_pin_nets.get(component_id, {}).values())
    return sorted(nets)


def _pins_for_components(layout_plan: LayoutPlan, component_ids: list[str]) -> list[str]:
    pins = []
    for component_id in component_ids:
        for pin_name in layout_plan.component_pin_nets.get(component_id, {}):
            pins.append(f"{component_id}.{pin_name}")
    return pins


def _labels_for_components(labels: dict[str, LabelArtifact], component_ids: list[str]) -> list[str]:
    return sorted(
        label_id
        for label_id, label in labels.items()
        if label.owner_id in component_ids
    )


def _major_net_names(nets: dict[str, NetArtifact]) -> list[str]:
    return [
        net_name
        for net_name, net in sorted(nets.items())
        if net.bbox is not None and (len(net.connected_pins) >= 2 or len(net.segments) >= 2)
    ]


def _build_region(
    region_id: str,
    label: str,
    kind: str,
    component_ids: list[str],
    net_names: list[str],
    pin_refs: list[str],
    label_ids: list[str],
    goals: list[str],
    components: dict[str, ComponentArtifact],
    nets: dict[str, NetArtifact],
    labels: dict[str, LabelArtifact],
    canvas: ViewBox,
) -> FocusRegion:
    bboxes: list[ArtifactBBox] = []
    for component_id in component_ids:
        if component_id in components:
            bboxes.append(components[component_id].bbox)
    for net_name in net_names:
        net = nets.get(net_name)
        if net and net.bbox:
            bboxes.append(net.bbox)
    for pin_ref in pin_refs:
        component_id, _, pin_name = pin_ref.partition(".")
        pin = components.get(component_id).pins.get(pin_name) if component_id in components else None
        if pin:
            bboxes.append(_point_bbox(pin.point))
    for label_id in label_ids:
        if label_id in labels:
            bboxes.append(labels[label_id].bbox)
    bbox = _union_with_min_extent(bboxes) or ArtifactBBox(
        canvas.x,
        canvas.y,
        canvas.width,
        canvas.height,
    )
    return FocusRegion(
        id=region_id,
        label=label,
        kind=kind,
        bbox=_padded_bbox(bbox, 24),
        components=sorted(set(component_ids)),
        nets=sorted(set(net_names)),
        pins=sorted(set(pin_refs)),
        labels=sorted(set(label_ids)),
        goals=goals,
    )


def _opamps_by_y(layout_plan: LayoutPlan) -> list[str]:
    opamps = [
        component
        for component in layout_plan.components
        if "op_amp" in _key(component.type) or "opamp" in _key(component.type)
    ]
    return [component.id for component in sorted(opamps, key=lambda item: item.grid_y)]


def _resistors_by_y(layout_plan: LayoutPlan) -> list[str]:
    resistors = [
        component
        for component in layout_plan.components
        if "resistor" in _key(component.type) or component.type.lower().startswith("r")
    ]
    return [component.id for component in sorted(resistors, key=lambda item: item.grid_y)]


def _focus_specs(circuit_ir: Any, layout_plan: LayoutPlan) -> list[tuple[str, str, str, list[str], list[str], list[str]]]:
    motif = _motif(circuit_ir, layout_plan)
    inputs = _input_component_ids(layout_plan)
    outputs = _output_component_ids(layout_plan)
    grounds = _ground_component_ids(layout_plan)
    opamps = _opamps_by_y(layout_plan)
    resistors = _resistors_by_y(layout_plan)
    capacitors = _component_ids_by_type(layout_plan, "capacitor")
    specs: list[tuple[str, str, str, list[str], list[str], list[str]]] = []

    if "instrumentation" in motif:
        top_buffer = [opamps[0]] if opamps else []
        bottom_buffer = [opamps[1]] if len(opamps) > 1 else []
        diff_stage = [opamps[2]] if len(opamps) > 2 else []
        gain = _component_ids_by_role(layout_plan, "gain_resistor")
        feedback = _component_ids_by_role(layout_plan, "feedback_resistor")
        ref = _component_ids_by_role(layout_plan, "ground_resistor") + grounds
        specs = [
            ("input_pair", "Input Pair", "input", inputs, _component_net_names(layout_plan, inputs), []),
            ("top_input_buffer", "Top Input Buffer", "stage", top_buffer + feedback[:1], _component_net_names(layout_plan, top_buffer + feedback[:1]), []),
            ("bottom_input_buffer", "Bottom Input Buffer", "stage", bottom_buffer + feedback[1:2], _component_net_names(layout_plan, bottom_buffer + feedback[1:2]), []),
            ("gain_resistor", "Gain Resistor", "gain", gain, _component_net_names(layout_plan, gain), []),
            ("differential_stage", "Differential Stage", "stage", diff_stage + feedback[2:] + _component_ids_by_role(layout_plan, "diff_input_resistor"), _component_net_names(layout_plan, diff_stage), []),
            ("output", "Output", "output", outputs, _component_net_names(layout_plan, outputs), []),
            ("ground_reference", "Ground Reference", "reference", ref, _component_net_names(layout_plan, ref), []),
            ("full_signal_path", "Full Signal Path", "signal_path", inputs + opamps + outputs, _component_net_names(layout_plan, inputs + opamps + outputs), []),
        ]
    elif "non_inverting" in motif:
        feedback_network = _component_ids_by_role(layout_plan, "feedback_resistor", "gain_resistor")
        specs = [
            ("input", "Input", "input", inputs, _component_net_names(layout_plan, inputs), []),
            ("op_amp", "Op Amp", "stage", opamps[:1], _component_net_names(layout_plan, opamps[:1]), []),
            ("feedback_network", "Feedback Network", "feedback", feedback_network, _component_net_names(layout_plan, feedback_network), []),
            ("output", "Output", "output", outputs, _component_net_names(layout_plan, outputs), []),
            ("ground_reference", "Ground Reference", "reference", grounds, _component_net_names(layout_plan, grounds), []),
        ]
    elif "low_pass" in motif:
        series = _component_ids_by_role(layout_plan, "series_resistor") or resistors[:1]
        specs = [
            ("input", "Input", "input", inputs, _component_net_names(layout_plan, inputs), []),
            ("series_resistor", "Series Resistor", "resistor", series, _component_net_names(layout_plan, series), []),
            ("output_node", "Output Node", "output", outputs, _component_net_names(layout_plan, outputs), []),
            ("shunt_capacitor", "Shunt Capacitor", "capacitor", capacitors[:1], _component_net_names(layout_plan, capacitors[:1]), []),
            ("ground_reference", "Ground Reference", "reference", grounds, _component_net_names(layout_plan, grounds), []),
        ]
    elif "divider" in motif:
        upper = resistors[:1]
        lower = resistors[1:2]
        specs = [
            ("source", "Source", "input", inputs, _component_net_names(layout_plan, inputs), []),
            ("upper_resistor", "Upper Resistor", "resistor", upper, _component_net_names(layout_plan, upper), []),
            ("lower_resistor", "Lower Resistor", "resistor", lower, _component_net_names(layout_plan, lower), []),
            ("output_node", "Output Node", "output", outputs, _component_net_names(layout_plan, outputs), []),
            ("ground_reference", "Ground Reference", "reference", grounds, _component_net_names(layout_plan, grounds), []),
        ]
    elif "bridge" in motif or "wheatstone" in motif:
        specs = [
            ("left_leg", "Left Leg", "bridge_leg", resistors[:2], _component_net_names(layout_plan, resistors[:2]), []),
            ("right_leg", "Right Leg", "bridge_leg", resistors[2:4], _component_net_names(layout_plan, resistors[2:4]), []),
            ("sense_nodes", "Sense Nodes", "sense", outputs, _component_net_names(layout_plan, outputs), []),
            ("excitation", "Excitation", "input", inputs, _component_net_names(layout_plan, inputs), []),
            ("ground_reference", "Ground Reference", "reference", grounds, _component_net_names(layout_plan, grounds), []),
        ]
    else:
        specs = [
            ("full_schematic", "Full Schematic", "full", [component.id for component in layout_plan.components], list(layout_plan.net_to_pins), [])
        ]
    return specs


def _build_focus_regions(
    circuit_ir: Any,
    layout_plan: LayoutPlan,
    components: dict[str, ComponentArtifact],
    nets: dict[str, NetArtifact],
    labels: dict[str, LabelArtifact],
    canvas: ViewBox,
) -> list[FocusRegion]:
    regions: list[FocusRegion] = []
    for region_id, label, kind, component_ids, net_names, goals in _focus_specs(circuit_ir, layout_plan):
        pin_refs = _pins_for_components(layout_plan, component_ids)
        label_ids = _labels_for_components(labels, component_ids)
        regions.append(
            _build_region(
                region_id,
                label,
                kind,
                component_ids,
                net_names,
                pin_refs,
                label_ids,
                goals,
                components,
                nets,
                labels,
                canvas,
            )
        )
    return regions


def _build_zoom_presets(
    canvas: ViewBox,
    focus_regions: list[FocusRegion],
    components: dict[str, ComponentArtifact],
    nets: dict[str, NetArtifact],
) -> list[ZoomPreset]:
    presets = [
        ZoomPreset(
            id="fit_all",
            label="Fit All",
            viewbox=canvas,
            focus_region_id=None,
            target_type=None,
            target_id=None,
        )
    ]
    for region in focus_regions:
        presets.append(
            ZoomPreset(
                id=f"focus_{region.id}",
                label=region.label,
                viewbox=padded_viewbox(region.bbox, canvas),
                focus_region_id=region.id,
                target_type="focus_region",
                target_id=region.id,
            )
        )
    for component_id, component in sorted(components.items()):
        presets.append(
            ZoomPreset(
                id=f"component_{component_id}",
                label=component.display_label or component_id,
                viewbox=padded_viewbox(component.bbox, canvas),
                focus_region_id=None,
                target_type="component",
                target_id=component_id,
            )
        )
    for net_name in _major_net_names(nets):
        net = nets[net_name]
        if net.bbox:
            presets.append(
                ZoomPreset(
                    id=f"net_{net_name}",
                    label=f"Net {net_name}",
                    viewbox=padded_viewbox(net.bbox, canvas),
                    focus_region_id=None,
                    target_type="net",
                    target_id=net_name,
                )
            )
    return presets


def _build_hit_targets(
    components: dict[str, ComponentArtifact],
    nets: dict[str, NetArtifact],
    labels: dict[str, LabelArtifact],
    focus_regions: list[FocusRegion],
) -> list[HitTarget]:
    targets: list[HitTarget] = []
    for component_id, component in sorted(components.items()):
        targets.append(HitTarget(f"component:{component_id}", "component", component_id, component.bbox))
        for pin_name, pin in sorted(component.pins.items()):
            pin_ref = f"{component_id}.{pin_name}"
            targets.append(HitTarget(f"pin:{pin_ref}", "pin", pin_ref, _point_bbox(pin.point)))
    for net_name, net in sorted(nets.items()):
        if net.bbox:
            targets.append(HitTarget(f"net:{net_name}", "net", net_name, _padded_bbox(net.bbox, 6)))
    for label_id, label in sorted(labels.items()):
        targets.append(HitTarget(f"label:{label_id}", "label", label_id, label.bbox))
    for region in focus_regions:
        targets.append(HitTarget(f"focus_region:{region.id}", "focus_region", region.id, region.bbox))
    return targets


def _build_overlays(focus_regions: list[FocusRegion]) -> list[OverlayArtifact]:
    overlays = []
    for region in focus_regions:
        overlays.append(
            OverlayArtifact(
                id=f"overlay_{region.id}",
                kind=region.kind,
                label=region.label,
                components=list(region.components),
                nets=list(region.nets),
                pins=list(region.pins),
                focus_region_id=region.id,
                default_visible=False,
            )
        )
    return overlays


def _build_components(
    layout_plan: LayoutPlan,
    render_result: RenderResult,
    labels: dict[str, LabelArtifact],
) -> dict[str, ComponentArtifact]:
    components: dict[str, ComponentArtifact] = {}
    label_by_owner = {label.owner_id: label for label in labels.values() if label.owner_id}
    for component in layout_plan.components:
        pins: dict[str, PinArtifact] = {}
        for pin in component.pins:
            pin_ref = f"{component.id}.{pin.pin_name}"
            point = render_result.geometry.pin_points.get(pin_ref)
            if point is None:
                continue
            pins[pin.pin_name] = PinArtifact(
                component_id=component.id,
                pin_name=pin.pin_name,
                net_name=pin.net_name,
                point=point,
                side=pin.side,
            )
        bbox = render_result.geometry.component_bboxes[component.id]
        owner_label = label_by_owner.get(component.id)
        components[component.id] = ComponentArtifact(
            id=component.id,
            type=component.type,
            role=component.role,
            display_label=component.display_label,
            value_label=component.value_label,
            bbox=_bbox_from_model(bbox),
            label_bbox=owner_label.bbox if owner_label else None,
            pins=pins,
        )
    return components


def _build_labels(layout_plan: LayoutPlan, render_result: RenderResult) -> dict[str, LabelArtifact]:
    labels = {}
    for label in layout_plan.labels:
        bbox = render_result.geometry.label_bboxes.get(label.id)
        if not bbox:
            continue
        labels[label.id] = LabelArtifact(
            id=label.id,
            text=label.text,
            owner_id=label.owner_id,
            bbox=_bbox_from_model(bbox),
        )
    return labels


def _build_nets(layout_plan: LayoutPlan, render_result: RenderResult) -> dict[str, NetArtifact]:
    segments_by_net: dict[str, list[NetSegmentArtifact]] = {
        net_name: [] for net_name in layout_plan.net_to_pins
    }
    for segment in render_result.geometry.wire_segments:
        segments_by_net.setdefault(segment.net_name, []).append(
            NetSegmentArtifact(
                x1=segment.start.x,
                y1=segment.start.y,
                x2=segment.end.x,
                y2=segment.end.y,
            )
        )

    route_pin_map = {wire.net_name: sorted(wire.connected_pins) for wire in layout_plan.wires}
    nets = {}
    for net_name in sorted(set(layout_plan.net_to_pins) | set(segments_by_net)):
        segment_bboxes = [_bbox_from_segment(segment) for segment in segments_by_net.get(net_name, [])]
        junctions = [
            point
            for node_id, point in sorted(render_result.geometry.junction_points.items())
            if node_id.split(":", 1)[0] == net_name
        ]
        point_bboxes = [_point_bbox(point, 2.0) for point in junctions]
        bbox = _union_with_min_extent(segment_bboxes + point_bboxes)
        nets[net_name] = NetArtifact(
            name=net_name,
            connected_pins=route_pin_map.get(net_name, sorted(layout_plan.net_to_pins.get(net_name, []))),
            segments=segments_by_net.get(net_name, []),
            bbox=bbox,
            junctions=junctions,
        )
    return nets


def build_schematic_artifact(
    circuit_ir: Any,
    layout_plan: LayoutPlan,
    render_result: RenderResult,
    critic_report: CriticReport,
) -> SchematicArtifact:
    """Build the tutor-facing artifact from trusted layout and render geometry."""

    canvas = ViewBox(
        x=0.0,
        y=0.0,
        width=float(layout_plan.canvas_width),
        height=float(layout_plan.canvas_height),
    )
    labels = _build_labels(layout_plan, render_result)
    components = _build_components(layout_plan, render_result, labels)
    nets = _build_nets(layout_plan, render_result)
    focus_regions = _build_focus_regions(circuit_ir, layout_plan, components, nets, labels, canvas)
    zoom_presets = _build_zoom_presets(canvas, focus_regions, components, nets)
    hit_targets = _build_hit_targets(components, nets, labels, focus_regions)
    overlays = _build_overlays(focus_regions)

    return SchematicArtifact(
        artifact_version=ARTIFACT_VERSION,
        circuit_id=layout_plan.circuit_id,
        renderer=layout_plan.renderer,
        svg=render_result.svg,
        svg_viewbox=canvas,
        layout_plan=layout_plan.to_dict(),
        components=components,
        nets=nets,
        labels=labels,
        focus_regions=focus_regions,
        zoom_presets=zoom_presets,
        hit_targets=hit_targets,
        overlays=overlays,
        critic_report=critic_report.to_dict(),
        warnings=list(layout_plan.warnings),
        provenance={
            "artifact_builder": "backend.app.schem_forge.artifact.build_schematic_artifact",
            "artifact_version": ARTIFACT_VERSION,
            "renderer": layout_plan.renderer,
            "circuit_id": layout_plan.circuit_id,
            "topology_signature": layout_plan.topology_signature,
            "source_motif": _motif(circuit_ir, layout_plan),
        },
    )

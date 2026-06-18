"""Optional vision client boundary for patch proposals."""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
import json
import re
from typing import Any

from .models import BBox, Circuit, CriticReport, LayoutComponent, LayoutLabel, LayoutPlan, NetClass, Point
from .patch import LayoutPatch, MoveComponent, MoveLabel, SetOrientation, SetRoutePolicy
from .raster import RasterImage

DEFAULT_GEMINI_VISION_MODEL = "gemini-pro-latest"


class VisionLayoutClient:
    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        raise NotImplementedError


@dataclass
class MockVisionClient(VisionLayoutClient):
    patch: LayoutPatch

    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        return self.patch


@dataclass(frozen=True)
class VisualFeedback:
    """Gemini-compatible local visual QA result.

    This is intentionally shaped like the external CV judge response so the
    optimizer can run the same feedback loop without spending vision credits.
    """

    passed: bool
    score: int
    recognized_circuit: str
    topology_errors: list[str]
    visual_errors: list[str]
    label_errors: list[str]
    specific_fixes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "score": self.score,
            "recognized_circuit": self.recognized_circuit,
            "topology_errors": list(self.topology_errors),
            "visual_errors": list(self.visual_errors),
            "label_errors": list(self.label_errors),
            "specific_fixes": list(self.specific_fixes),
        }


class HeuristicVisionClient(VisionLayoutClient):
    """Local Gemini-like feedback and patch proposal.

    Use this when Gemini quota is unavailable or too expensive for every draw.
    It reads OptCPV's vector/CV critic report, emits a strict visual-QA-shaped
    feedback object, and proposes only topology-safe layout/label moves.
    """

    def __init__(self, *, pass_threshold: float = 10.0) -> None:
        self.pass_threshold = pass_threshold
        self.last_feedback: VisualFeedback | None = None

    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        self.last_feedback = heuristic_feedback(circuit, layout, critic_report, pass_threshold=self.pass_threshold)
        return _patch_from_feedback(layout, critic_report)


class GeminiVisionClient(VisionLayoutClient):
    """Optional Gemini-backed client; never part of the default optimizer."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_GEMINI_VISION_MODEL) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiVisionClient.") from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model
        self._types = types

    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        prompt = _patch_prompt(circuit, layout, svg, raster, critic_report)
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, self._raster_part(raster)],
            config={"response_mime_type": "application/json"},
        )
        text = (getattr(response, "text", "") or "").strip()
        return _patch_from_json(text, layout)

    def _raster_part(self, raster: RasterImage):
        from PIL import Image

        buffer = BytesIO()
        Image.fromarray(raster.rgba).save(buffer, format="PNG")
        return self._types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def heuristic_feedback(
    circuit: Circuit,
    layout: LayoutPlan,
    critic_report: CriticReport,
    *,
    pass_threshold: float = 10.0,
) -> VisualFeedback:
    topology_errors: list[str] = []
    visual_errors: list[str] = []
    label_errors: list[str] = []
    specific_fixes: list[str] = []

    for violation in critic_report.violations:
        code = violation.code
        subject = f" ({violation.subject})" if violation.subject else ""
        if code == "topology_failure":
            topology_errors.append(violation.message)
            specific_fixes.append("Reject this patch and keep the original topology-preserving layout.")
        elif code in {"label_visual_collision", "label_wire_overlap", "label_component_overlap", "label_outside_canvas"}:
            label_errors.append(f"{violation.message}{subject}")
            specific_fixes.append("Move the affected label to a nearby open side of its owner component.")
        elif code in {"component_overlap", "wire_through_component", "wire_component_visual_collision"}:
            visual_errors.append(f"{violation.message}{subject}")
            specific_fixes.append("Increase local spacing and route around component bodies while preserving nets.")
        elif code in {"fill_ratio_low", "too_much_empty_canvas", "spread_excessive"}:
            visual_errors.append(f"{violation.message}{subject}")
            specific_fixes.append("Compact the drawing within the fixed canvas without shrinking strokes or labels.")
        elif code in {"viewbox_not_fixed_frame", "viewbox_too_large", "group_scale_too_small", "stroke_width_too_small"}:
            visual_errors.append(f"{violation.message}{subject}")
            specific_fixes.append("Normalize SVG output to the fixed OptCPV canvas and standard visible stroke sizes.")
        elif code in {"ground_not_low", "output_not_right", "feedback_not_above", "wire_crossings", "wire_length_high"}:
            visual_errors.append(f"{violation.message}{subject}")
            specific_fixes.append("Apply schematic conventions: inputs left, outputs right, feedback above, ground below.")

    if not (topology_errors or visual_errors or label_errors) and critic_report.score > pass_threshold:
        visual_errors.append("The local critic score is high despite no single hard visual violation.")
        specific_fixes.append("Prefer a more compact, balanced placement and re-run layered CV checks.")

    score = max(0, min(100, int(round(100 - critic_report.score))))
    return VisualFeedback(
        passed=not critic_report.hard_fail and critic_report.score <= pass_threshold,
        score=score,
        recognized_circuit=_recognized_circuit(circuit, layout),
        topology_errors=topology_errors,
        visual_errors=_dedupe_text(visual_errors),
        label_errors=_dedupe_text(label_errors),
        specific_fixes=_dedupe_text(specific_fixes),
    )


def _patch_from_feedback(layout: LayoutPlan, critic_report: CriticReport) -> LayoutPatch:
    codes = {violation.code for violation in critic_report.violations}
    if not codes:
        return LayoutPatch()

    component_moves: list[MoveComponent] = []
    for component in layout.components:
        x, y = component.x, component.y
        key = _key(component.type)
        if key in {"ground", "gnd"} and ("ground_not_low" in codes or "component_overlap" in codes):
            y = max(y, _median_y(layout) + 2.2)
        if key == "output" and "output_not_right" in codes:
            x = max(x, _max_input_x(layout) + 5.0)
        if "feedback" in _key(component.role) and "feedback_not_above" in codes:
            y = min(y, _opamp_y(layout) - 2.8)
        component_moves.append(MoveComponent(component.id, x, y))

    if "component_overlap" in codes:
        component_moves = _separate_component_moves(layout, component_moves)
    if {"fill_ratio_low", "too_much_empty_canvas", "spread_excessive"} & codes:
        component_moves = _compact_component_moves(layout, component_moves)

    label_moves = _label_moves(layout) if _needs_label_moves(codes) else []
    if not component_moves and not label_moves:
        return LayoutPatch()
    return LayoutPatch(move_component=component_moves, move_label=label_moves)


def _needs_label_moves(codes: set[str]) -> bool:
    return bool(codes & {"label_visual_collision", "label_wire_overlap", "label_component_overlap", "label_outside_canvas"})


def _label_moves(layout: LayoutPlan) -> list[MoveLabel]:
    moves: list[MoveLabel] = []
    for label in layout.labels:
        owner = next((component for component in layout.components if component.id == label.owner_id), None)
        if owner is None:
            continue
        x, y = min(_label_candidates(owner), key=lambda item: _label_score(layout, label, owner, item[0], item[1]))
        moves.append(MoveLabel(label.id, x, y))
    return moves


def _label_candidates(owner: LayoutComponent) -> list[tuple[float, float]]:
    above = (owner.x, owner.bbox.y - 0.4)
    below = (owner.x, owner.bbox.bottom + 0.45)
    right = (owner.bbox.right + 0.5, owner.y)
    left = (owner.bbox.x - 0.5, owner.y)
    far_above = (owner.x, owner.bbox.y - 1.15)
    far_below = (owner.x, owner.bbox.bottom + 1.25)
    upper_right = (owner.bbox.right + 1.15, owner.bbox.y - 0.55)
    lower_right = (owner.bbox.right + 1.15, owner.bbox.bottom + 0.75)
    upper_left = (owner.bbox.x - 1.15, owner.bbox.y - 0.55)
    lower_left = (owner.bbox.x - 1.15, owner.bbox.bottom + 0.75)
    if _key(owner.type) in {"ground", "gnd"}:
        return [below, far_below, right, left, above]
    if owner.orientation in {"up", "down"}:
        return [right, left, upper_right, lower_right, upper_left, lower_left, above, below, far_above, far_below]
    return [above, upper_right, upper_left, right, below, lower_right, lower_left, left, far_above, far_below]


def _label_score(layout: LayoutPlan, label: LayoutLabel, owner: LayoutComponent, x: float, y: float) -> float:
    bbox = _moved_label_bbox(label, x, y)
    score = abs(x - label.x) * 0.1 + abs(y - label.y) * 0.1
    if bbox.x < 0 or bbox.y < 0 or bbox.right > layout.width / layout.grid or bbox.bottom > layout.height / layout.grid:
        score += 100.0
    if bbox.intersects(owner.bbox, padding=0.02):
        score += 10.0
    for component in layout.components:
        if component.id != owner.id and bbox.intersects(component.bbox, padding=0.04):
            score += 30.0
    for wire in layout.wires:
        if _polyline_intersects_bbox(wire.points, bbox.expanded(0.03)):
            score += 20.0
    return score


def _moved_label_bbox(label: LayoutLabel, x: float, y: float) -> BBox:
    return BBox(label.bbox.x + x - label.x, label.bbox.y + y - label.y, label.bbox.width, label.bbox.height)


def _compact_component_moves(layout: LayoutPlan, moves: list[MoveComponent]) -> list[MoveComponent]:
    by_id = {move.component_id: move for move in moves}
    min_x = min(component.x for component in layout.components)
    max_x = max(component.x for component in layout.components)
    min_y = min(component.y for component in layout.components)
    max_y = max(component.y for component in layout.components)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    target_span_x = min(span_x, 15.5)
    target_span_y = min(span_y, 10.0)
    return [
        MoveComponent(
            component.id,
            2.0 + ((by_id[component.id].x - min_x) / span_x) * target_span_x,
            2.2 + ((by_id[component.id].y - min_y) / span_y) * target_span_y,
        )
        for component in layout.components
    ]


def _separate_component_moves(layout: LayoutPlan, moves: list[MoveComponent]) -> list[MoveComponent]:
    by_id = {move.component_id: move for move in moves}
    separated: list[MoveComponent] = []
    for index, component in enumerate(sorted(layout.components, key=lambda item: (item.y, item.x))):
        move = by_id[component.id]
        separated.append(MoveComponent(component.id, move.x, move.y + index * 0.14))
    return separated


def _recognized_circuit(circuit: Circuit, layout: LayoutPlan) -> str:
    motif = circuit.motif or layout.circuit_id
    return _key(motif).replace("_", " ") or "schematic"


def _median_y(layout: LayoutPlan) -> float:
    values = sorted(component.y for component in layout.components)
    return values[len(values) // 2]


def _max_input_x(layout: LayoutPlan) -> float:
    values = [component.x for component in layout.components if _key(component.type) in {"input", "source", "voltage_source"}]
    return max(values or [0.0])


def _opamp_y(layout: LayoutPlan) -> float:
    values = [component.y for component in layout.components if "op" in _key(component.type)]
    return sum(values) / len(values) if values else 0.0


def _polyline_intersects_bbox(points: list[Point], bbox: BBox) -> bool:
    for start, end in zip(points, points[1:]):
        if bbox.contains_point(start) or bbox.contains_point(end):
            return True
        edges = [
            (Point(bbox.x, bbox.y), Point(bbox.right, bbox.y)),
            (Point(bbox.right, bbox.y), Point(bbox.right, bbox.bottom)),
            (Point(bbox.right, bbox.bottom), Point(bbox.x, bbox.bottom)),
            (Point(bbox.x, bbox.bottom), Point(bbox.x, bbox.y)),
        ]
        if any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges):
            return True
    return False


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


def _patch_prompt(
    circuit: Circuit,
    layout: LayoutPlan,
    svg: str,
    raster: RasterImage,
    critic_report: CriticReport,
) -> str:
    payload = {
        "task": "Return a topology-safe OptCPV LayoutPatch JSON only.",
        "hard_rules": [
            "Do not change canvas width, height, grid, component ids, component types, pins, nets, or topology.",
            "You may move existing components, move existing labels, set existing component orientation, or request route-policy corridors for existing non-terminal nets.",
            "You may make coordinated drawing-guidance moves when they improve teaching readability, but keep the layout inside the canvas.",
            "If uncertain, return empty arrays.",
        ],
        "authority": {
            "drawing_guidance": "High. Prefer clear schematic conventions, feedback corridors, and non-overlapping labels over preserving deterministic placement.",
            "local_gate": "OptCPV will reject topology changes, terminal-net routing changes, canvas/scale hacks, and unsafe spread.",
        },
        "schema": {
            "move_component": [{"component_id": "existing id", "x": "float", "y": "float"}],
            "move_label": [{"label_id": "existing label id", "x": "float", "y": "float"}],
            "set_orientation": [{"component_id": "existing id", "orientation": "right|left|up|down|right_flip"}],
            "set_route_policy": [
                {
                    "net": "existing non-terminal net",
                    "net_role": "feedback|right_leg_drive|signal|inter_block",
                    "policy": "top_feedback_corridor|bottom_feedback_corridor|bottom_auxiliary_corridor|left_to_right_manhattan|avoid_opamp_body",
                }
            ],
            "set_wire_points": [],
        },
        "circuit": {
            "id": circuit.id,
            "motif": circuit.motif,
            "components": [
                {"id": item.id, "type": item.type, "pins": item.pins, "label": item.label, "role": item.role}
                for item in circuit.components
            ],
        },
        "layout": {
            "width": layout.width,
            "height": layout.height,
            "grid": layout.grid,
            "components": [
                {
                    "id": item.id,
                    "type": item.type,
                    "x": item.x,
                    "y": item.y,
                    "orientation": item.orientation,
                    "bbox": {
                        "x": item.bbox.x,
                        "y": item.bbox.y,
                        "width": item.bbox.width,
                        "height": item.bbox.height,
                    },
                }
                for item in layout.components
            ],
            "labels": [
                {"id": item.id, "owner_id": item.owner_id, "text": item.text, "x": item.x, "y": item.y}
                for item in layout.labels
            ],
        },
        "raster": {"width": raster.width, "height": raster.height},
        "critic_report": critic_report.to_dict(),
        "svg_excerpt": svg[:6000],
    }
    return (
        "You are an OptCPV visual-layout patch proposer. Return only valid JSON matching the schema. "
        "No markdown, no prose.\n"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )


def _patch_from_json(text: str, layout: LayoutPlan) -> LayoutPatch:
    data = _parse_json(text)
    component_ids = {component.id for component in layout.components}
    label_ids = {label.id for label in layout.labels}
    orientations = {"right", "left", "up", "down", "right_flip"}
    route_policies = {
        "left_to_right_manhattan",
        "top_feedback_corridor",
        "bottom_feedback_corridor",
        "bottom_auxiliary_corridor",
        "avoid_opamp_body",
    }

    move_component = []
    for raw in _list(data.get("move_component")):
        component_id = str(raw.get("component_id", ""))
        if component_id in component_ids and _is_number(raw.get("x")) and _is_number(raw.get("y")):
            move_component.append(MoveComponent(component_id, float(raw["x"]), float(raw["y"])))

    move_label = []
    for raw in _list(data.get("move_label")):
        label_id = str(raw.get("label_id", ""))
        if label_id in label_ids and _is_number(raw.get("x")) and _is_number(raw.get("y")):
            move_label.append(MoveLabel(label_id, float(raw["x"]), float(raw["y"])))

    set_orientation = []
    for raw in _list(data.get("set_orientation")):
        component_id = str(raw.get("component_id", ""))
        orientation = _normalize_orientation(str(raw.get("orientation", "")))
        if component_id in component_ids and orientation in orientations:
            set_orientation.append(SetOrientation(component_id, orientation))

    set_route_policy = []
    for raw in _list(data.get("set_route_policy")):
        net = str(raw.get("net", ""))
        policy = str(raw.get("policy", ""))
        if net in layout.net_to_pins and policy in route_policies and _route_policy_safe(layout, net):
            set_route_policy.append(SetRoutePolicy(net=net, policy=policy, net_role=str(raw.get("net_role", "feedback"))))

    return LayoutPatch(
        move_component=move_component,
        move_label=move_label,
        set_orientation=set_orientation,
        set_route_policy=set_route_policy,
        set_wire_points=[],
    )


def _normalize_orientation(value: str) -> str:
    key = _key(value)
    aliases = {"east": "right", "west": "left", "north": "up", "south": "down", "rightflip": "right_flip"}
    return aliases.get(key, key)


def _route_policy_safe(layout: LayoutPlan, net: str) -> bool:
    net_class = layout.semantic.net_classes.get(net, NetClass.SIGNAL)
    return net_class not in {NetClass.GROUND, NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY, NetClass.REFERENCE}


def _parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) or (
        isinstance(value, str) and bool(re.fullmatch(r"-?\d+(\.\d+)?", value.strip()))
    )

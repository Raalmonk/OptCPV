"""Post-render visual review clients and patch translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
import os
import re
from typing import Any

from .models import Circuit, CriticReport, LayoutComponent, LayoutPin, LayoutPlan, NetClass
from .patch import LayoutPatch, MoveComponent, MoveLabel, SetRoutePolicy
from .raster import RasterImage


VISUAL_PATCH_ACTIONS = {"move_component", "move_label", "assign_route_corridor", "request_reroute", "no_op"}
DEFAULT_GEMINI_VISUAL_REVIEW_MODEL = "gemini-3.5-flash"


@dataclass(frozen=True)
class VisualError:
    code: str
    message: str
    subject: str | None = None
    severity: float = 0.0

    @classmethod
    def from_any(cls, raw: Any) -> "VisualError":
        if isinstance(raw, str):
            return cls(code="visual_error", message=raw)
        if not isinstance(raw, dict):
            return cls(code="visual_error", message=str(raw))
        return cls(
            code=str(raw.get("code", "visual_error")),
            message=str(raw.get("message", "")),
            subject=None if raw.get("subject") is None else str(raw.get("subject")),
            severity=float(raw.get("severity", 0.0) or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "subject": self.subject, "severity": self.severity}


@dataclass(frozen=True)
class VisualPatch:
    action: str
    component_id: str | None = None
    label_id: str | None = None
    net: str | None = None
    corridor: str | None = None
    x: float | None = None
    y: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action not in VISUAL_PATCH_ACTIONS:
            object.__setattr__(self, "action", "no_op")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VisualPatch":
        return cls(
            action=str(raw.get("action", "no_op")),
            component_id=_optional_str(raw.get("component_id")),
            label_id=_optional_str(raw.get("label_id")),
            net=_optional_str(raw.get("net")),
            corridor=_optional_str(raw.get("corridor")),
            x=_optional_float(raw.get("x")),
            y=_optional_float(raw.get("y")),
            reason=str(raw.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "component_id": self.component_id,
            "label_id": self.label_id,
            "net": self.net,
            "corridor": self.corridor,
            "x": self.x,
            "y": self.y,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VisualReview:
    passed: bool
    score: int
    recognized_topology: str
    visual_errors: list[VisualError] = field(default_factory=list)
    patches: list[VisualPatch] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VisualReview":
        if not isinstance(raw, dict):
            raise ValueError("VisualReview must be an object.")
        return cls(
            passed=bool(raw.get("passed", raw.get("pass", False))),
            score=max(0, min(100, int(raw.get("score", 0) or 0))),
            recognized_topology=str(raw.get("recognized_topology", raw.get("recognized_circuit", ""))),
            visual_errors=[VisualError.from_any(item) for item in raw.get("visual_errors", [])],
            patches=[VisualPatch.from_dict(item) for item in raw.get("patches", []) if isinstance(item, dict)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "recognized_topology": self.recognized_topology,
            "visual_errors": [error.to_dict() for error in self.visual_errors],
            "patches": [patch.to_dict() for patch in self.patches],
        }


class VisualReviewClient:
    def review(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> VisualReview:
        raise NotImplementedError


@dataclass
class FakeVisualReviewClient(VisualReviewClient):
    review_result: VisualReview

    def review(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> VisualReview:
        return self.review_result


class HeuristicVisualReviewClient(VisualReviewClient):
    def review(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> VisualReview:
        errors = [
            VisualError(
                code=violation.code,
                message=violation.message,
                subject=violation.subject,
                severity=violation.severity,
            )
            for violation in critic_report.violations
            if _visual_violation(violation.code)
        ]
        patches = _heuristic_patches(layout, critic_report)
        return VisualReview(
            passed=not critic_report.hard_fail and critic_report.score <= 10,
            score=max(0, min(100, int(round(100 - critic_report.score)))),
            recognized_topology=circuit.motif or layout.circuit_id,
            visual_errors=errors,
            patches=patches,
        )


class GeminiVisualReviewClient(VisualReviewClient):
    """Optional Gemini-backed post-render reviewer."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiVisualReviewClient.") from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model or os.getenv("OPTCPV_GEMINI_VISUAL_REVIEW_MODEL", DEFAULT_GEMINI_VISUAL_REVIEW_MODEL)
        self._types = types

    def review(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> VisualReview:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[_review_prompt(circuit, layout, svg, critic_report), self._raster_part(raster)],
            config={"response_mime_type": "application/json"},
        )
        text = (getattr(response, "text", "") or "").strip()
        return VisualReview.from_dict(_json_object(text))

    def _raster_part(self, raster: RasterImage):
        from PIL import Image

        buffer = BytesIO()
        Image.fromarray(raster.rgba).save(buffer, format="PNG")
        return self._types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def layout_patch_from_visual_review(review: VisualReview, layout: LayoutPlan) -> LayoutPatch:
    component_ids = {component.id for component in layout.components}
    label_ids = {label.id for label in layout.labels}
    move_components: list[MoveComponent] = []
    move_labels: list[MoveLabel] = []
    route_policies: list[SetRoutePolicy] = []

    for patch in review.patches:
        if patch.action == "move_component" and patch.component_id in component_ids and patch.x is not None and patch.y is not None:
            move_components.append(MoveComponent(patch.component_id, patch.x, patch.y))
        elif patch.action == "move_label" and patch.label_id in label_ids and patch.x is not None and patch.y is not None:
            move_labels.append(MoveLabel(patch.label_id, patch.x, patch.y))
        elif patch.action in {"assign_route_corridor", "request_reroute"} and patch.net:
            route_policy = _route_policy_from_visual_patch(patch, layout)
            if route_policy is not None:
                route_policies.append(route_policy)
        elif patch.action == "no_op":
            continue
    return LayoutPatch(move_component=move_components, move_label=move_labels, set_route_policy=route_policies)


def _route_policy_from_visual_patch(patch: VisualPatch, layout: LayoutPlan) -> SetRoutePolicy | None:
    if patch.net is None or patch.net not in layout.net_to_pins:
        return None
    net_class = layout.semantic.net_classes.get(patch.net, NetClass.SIGNAL)
    if net_class in {NetClass.GROUND, NetClass.POSITIVE_SUPPLY, NetClass.NEGATIVE_SUPPLY, NetClass.REFERENCE}:
        return None
    policy = _normalize_corridor_policy(patch.corridor, patch.net, layout)
    if policy is None:
        return None
    role = "right_leg_drive" if policy == "bottom_auxiliary_corridor" else "feedback"
    return SetRoutePolicy(net=patch.net, policy=policy, net_role=role)


def _normalize_corridor_policy(corridor: str | None, net: str, layout: LayoutPlan) -> str | None:
    key = _key(corridor or "")
    if any(token in key for token in ("bottom_auxiliary", "auxiliary", "right_leg", "rld")):
        return "bottom_auxiliary_corridor"
    if "top" in key and ("bottom" in key or "or" in key):
        return _feedback_corridor_for_net(net, layout)
    if "top" in key:
        return "top_feedback_corridor"
    if "bottom" in key:
        return "bottom_feedback_corridor"
    if not key or "reroute" in key or "feedback" in key:
        return _feedback_corridor_for_net(net, layout)
    return None


def _feedback_corridor_for_net(net: str, layout: LayoutPlan) -> str:
    component_by_id = {component.id: component for component in layout.components}
    pins = [layout.pin_map[key] for key in layout.net_to_pins.get(net, []) if key in layout.pin_map]
    driver = next((pin for pin in pins if _is_opamp_output_pin(pin, component_by_id)), None)
    receivers = [pin for pin in pins if _is_opamp_input_pin(pin, component_by_id)]
    if driver is not None and receivers:
        receiver = min(receivers, key=lambda pin: abs(pin.y - driver.y))
        return "bottom_feedback_corridor" if receiver.y > driver.y else "top_feedback_corridor"
    if pins:
        median_y = sorted(pin.y for pin in pins)[len(pins) // 2]
        return "bottom_feedback_corridor" if median_y > _layout_midline_y(layout) else "top_feedback_corridor"
    return "top_feedback_corridor"


def _is_opamp_output_pin(pin: LayoutPin, component_by_id: dict[str, LayoutComponent]) -> bool:
    component = component_by_id.get(pin.component_id)
    return component is not None and _is_opamp(component) and _pin_kind(pin.pin_name) in {"out", "output", "o"}


def _is_opamp_input_pin(pin: LayoutPin, component_by_id: dict[str, LayoutComponent]) -> bool:
    component = component_by_id.get(pin.component_id)
    return component is not None and _is_opamp(component) and _pin_kind(pin.pin_name) in {"+", "-", "plus", "minus", "noninverting", "inverting"}


def _is_opamp(component: LayoutComponent) -> bool:
    key = _key(component.type)
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key


def _pin_kind(pin_name: str) -> str:
    compact = _key(pin_name).replace("_", "")
    if pin_name in {"+", "-"}:
        return pin_name
    if compact in {"plus", "noninverting", "noninv", "inp", "vp"}:
        return "+"
    if compact in {"minus", "inverting", "inv", "inn", "vn"}:
        return "-"
    return compact


def _layout_midline_y(layout: LayoutPlan) -> float:
    return (layout.height / layout.grid) / 2.0


def _heuristic_patches(layout: LayoutPlan, critic_report: CriticReport) -> list[VisualPatch]:
    patches: list[VisualPatch] = []
    label_codes = {"label_visual_collision", "label_wire_overlap", "label_component_overlap", "label_outside_canvas"}
    for violation in critic_report.violations:
        if violation.code in label_codes and violation.subject:
            label_id = violation.subject if violation.subject.startswith("label:") else None
            if label_id:
                label = next((item for item in layout.labels if item.id == label_id), None)
                owner = next((item for item in layout.components if label and item.id == label.owner_id), None)
                if label and owner:
                    patches.append(
                        VisualPatch(
                            action="move_label",
                            label_id=label.id,
                            x=owner.x,
                            y=owner.bbox.y - 0.75,
                            reason=violation.message,
                        )
                    )
        elif violation.code in {"wire_through_component", "feedback_crosses_opamp_body"} and violation.subject:
            net = violation.subject.split(":", 1)[0]
            patches.append(
                VisualPatch(
                    action="request_reroute",
                    net=net,
                    corridor="top_or_bottom_feedback_corridor",
                    reason=violation.message,
                )
            )
    return patches


def _review_prompt(circuit: Circuit, layout: LayoutPlan, svg: str, critic_report: CriticReport) -> str:
    payload = {
        "task": "Review this OptCPV schematic rendering and return strict VisualReview JSON only.",
        "hard_rules": [
            "Do not create, delete, rename, or rewire components, pins, or nets.",
            "Do not change pin mappings.",
            "Do not output absolute pixel-only SVG rewrites.",
            "Only propose move_component, move_label, assign_route_corridor, request_reroute, or no_op patches.",
            "All patches must be safe to pass through local LayoutPatch verification.",
        ],
        "schema": {
            "passed": "boolean",
            "score": "integer 0-100",
            "recognized_topology": "string",
            "visual_errors": [{"code": "string", "message": "string", "subject": "optional string", "severity": "number"}],
            "patches": [
                {
                    "action": "move_component|move_label|assign_route_corridor|request_reroute|no_op",
                    "component_id": "optional existing id",
                    "label_id": "optional existing label id",
                    "net": "optional existing net",
                    "corridor": "optional corridor name",
                    "x": "optional layout-grid x for move actions",
                    "y": "optional layout-grid y for move actions",
                    "reason": "string",
                }
            ],
        },
        "circuit": {
            "id": circuit.id,
            "motif": circuit.motif,
            "components": [
                {"id": component.id, "type": component.type, "role": component.role, "pins": dict(component.pins)}
                for component in circuit.components
            ],
        },
        "layout": {
            "components": [
                {
                    "id": component.id,
                    "type": component.type,
                    "bbox": {
                        "x": component.bbox.x,
                        "y": component.bbox.y,
                        "width": component.bbox.width,
                        "height": component.bbox.height,
                    },
                }
                for component in layout.components
            ],
            "labels": [
                {"id": label.id, "owner_id": label.owner_id, "bbox": {"x": label.bbox.x, "y": label.bbox.y, "width": label.bbox.width, "height": label.bbox.height}}
                for label in layout.labels
            ],
            "wires": [
                {"net": wire.net, "points": [{"x": point.x, "y": point.y} for point in wire.points]}
                for wire in layout.wires
            ],
            "semantic_plan": layout.semantic.to_dict(),
        },
        "critic_report": critic_report.to_dict(),
        "svg_excerpt": svg[:6000],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _visual_violation(code: str) -> bool:
    return any(token in code for token in ("label", "wire", "component", "viewbox", "canvas", "clutter", "blob"))


def _json_object(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        raw = json.loads(match.group(0))
    if not isinstance(raw, dict):
        raise ValueError("GeminiVisualReviewClient must return a JSON object.")
    return raw


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


__all__ = [
    "FakeVisualReviewClient",
    "GeminiVisualReviewClient",
    "HeuristicVisualReviewClient",
    "VisualError",
    "VisualPatch",
    "VisualReview",
    "VisualReviewClient",
    "layout_patch_from_visual_review",
]

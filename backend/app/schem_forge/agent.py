"""Constrained LLM polish loop for schem_forge layouts."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .critic import CriticReport, critique_layout
from .models import LabelLayout, LayoutPlan, Point, RenderGeometry, RenderResult
from .planner import pin_layouts_for_component, plan_circuit
from .renderer import render_layout
from .verifier import ElectricalTopologyError, verify_equivalence


ACCEPTABLE_SCORE = 100


class LayoutPatchError(ValueError):
    """Raised when an LLM patch attempts a forbidden layout change."""


@dataclass
class LayoutPatch:
    move_component: list[dict[str, Any]] = field(default_factory=list)
    move_label: list[dict[str, Any]] = field(default_factory=list)
    set_orientation: list[dict[str, Any]] = field(default_factory=list)
    set_wire_waypoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    layout: LayoutPlan
    svg: str
    geometry: RenderGeometry
    critic_report: CriticReport
    iterations: int
    improved: bool
    debug_log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.to_dict(),
            "svg": self.svg,
            "geometry": self.geometry.to_dict(),
            "critic_report": self.critic_report.to_dict(),
            "iterations": self.iterations,
            "improved": self.improved,
            "debug_log": list(self.debug_log),
        }


def _as_patch_dict(raw_patch: LayoutPatch | dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(raw_patch, LayoutPatch):
        return raw_patch.to_dict()
    if isinstance(raw_patch, (dict, list)):
        return raw_patch
    raise LayoutPatchError(f"Unsupported patch type: {type(raw_patch).__name__}")


def _operation_list(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = raw.get(key, [])
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise LayoutPatchError(f"Patch field {key} must contain objects.")
        return value
    raise LayoutPatchError(f"Patch field {key} must be an object or list.")


def _point_from_patch(item: Any) -> Point:
    if isinstance(item, Point):
        return item
    if isinstance(item, dict) and "x" in item and "y" in item:
        return Point(float(item["x"]), float(item["y"]))
    raise LayoutPatchError(f"Invalid waypoint: {item!r}")


def _refresh_component_pin_layout(component, component_pin_nets: dict[str, dict[str, str]]) -> None:
    spec = {
        "id": component.id,
        "type": component.type,
        "role": component.role,
        "pins": component_pin_nets[component.id],
    }
    component.pins = pin_layouts_for_component(spec, component.orientation)


def _apply_restricted_patch(plan: LayoutPlan, raw_patch: dict[str, Any]) -> LayoutPlan:
    allowed_keys = {"move_component", "move_label", "set_orientation", "set_wire_waypoints"}
    unknown_keys = set(raw_patch) - allowed_keys
    if unknown_keys:
        raise LayoutPatchError(
            f"Patch contains forbidden fields: {sorted(unknown_keys)}"
        )

    candidate = copy.deepcopy(plan)
    components = {component.id: component for component in candidate.components}
    labels = {label.id: label for label in candidate.labels}
    wires = {wire.net_name: wire for wire in candidate.wires}

    for operation in _operation_list(raw_patch, "move_component"):
        component_id = str(operation.get("id", operation.get("component_id", "")))
        if component_id not in components:
            raise LayoutPatchError(f"Unknown component in move_component: {component_id}")
        component = components[component_id]
        if "grid_x" in operation:
            component.grid_x = float(operation["grid_x"])
        if "grid_y" in operation:
            component.grid_y = float(operation["grid_y"])

    for operation in _operation_list(raw_patch, "move_label"):
        label_id = str(operation.get("id", operation.get("label_id", "")))
        if label_id not in labels:
            raise LayoutPatchError(f"Unknown label in move_label: {label_id}")
        label = labels[label_id]
        if "grid_x" in operation:
            label.grid_x = float(operation["grid_x"])
        if "grid_y" in operation:
            label.grid_y = float(operation["grid_y"])
        if "anchor" in operation:
            if operation["anchor"] not in {"start", "middle", "end"}:
                raise LayoutPatchError(f"Invalid label anchor: {operation['anchor']}")
            label.anchor = operation["anchor"]

    for operation in _operation_list(raw_patch, "set_orientation"):
        component_id = str(operation.get("id", operation.get("component_id", "")))
        if component_id not in components:
            raise LayoutPatchError(f"Unknown component in set_orientation: {component_id}")
        orientation = operation.get("orientation")
        if orientation not in {"up", "down", "left", "right"}:
            raise LayoutPatchError(f"Invalid component orientation: {orientation}")
        component = components[component_id]
        component.orientation = orientation
        _refresh_component_pin_layout(component, candidate.component_pin_nets)

    for operation in _operation_list(raw_patch, "set_wire_waypoints"):
        net_name = str(operation.get("net_name", ""))
        if net_name not in wires:
            raise LayoutPatchError(f"Unknown net in set_wire_waypoints: {net_name}")
        if "waypoints" not in operation:
            raise LayoutPatchError("set_wire_waypoints requires a waypoints field.")
        wires[net_name].waypoints = [_point_from_patch(item) for item in operation["waypoints"]]
        wires[net_name].segments = None

    return candidate


def _apply_json_patch(plan: LayoutPlan, operations: list[dict[str, Any]]) -> LayoutPlan:
    candidate = copy.deepcopy(plan)
    for operation in operations:
        if operation.get("op") not in {"replace", "add"}:
            raise LayoutPatchError(f"Unsupported JSON Patch op: {operation.get('op')}")
        path = str(operation.get("path", ""))
        value = operation.get("value")
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3:
            raise LayoutPatchError(f"Forbidden JSON Patch path: {path}")

        collection, index_text, field_name = parts
        try:
            index = int(index_text)
        except ValueError as exc:
            raise LayoutPatchError(f"JSON Patch index must be numeric: {path}") from exc

        if collection == "components" and field_name in {"grid_x", "grid_y", "orientation"}:
            component = candidate.components[index]
            if field_name == "orientation":
                if value not in {"up", "down", "left", "right"}:
                    raise LayoutPatchError(f"Invalid orientation: {value}")
                component.orientation = value
                _refresh_component_pin_layout(component, candidate.component_pin_nets)
            else:
                setattr(component, field_name, float(value))
        elif collection == "labels" and field_name in {"grid_x", "grid_y", "anchor"}:
            label: LabelLayout = candidate.labels[index]
            if field_name == "anchor":
                if value not in {"start", "middle", "end"}:
                    raise LayoutPatchError(f"Invalid label anchor: {value}")
                label.anchor = value
            else:
                setattr(label, field_name, float(value))
        elif collection == "wires" and field_name == "waypoints":
            candidate.wires[index].waypoints = [_point_from_patch(item) for item in value]
            candidate.wires[index].segments = None
        else:
            raise LayoutPatchError(f"Forbidden JSON Patch path: {path}")
    return candidate


def apply_layout_patch(
    original_circuit: Any,
    plan: LayoutPlan,
    patch: LayoutPatch | dict[str, Any] | list[dict[str, Any]],
) -> LayoutPlan:
    raw_patch = _as_patch_dict(patch)
    if isinstance(raw_patch, list):
        candidate = _apply_json_patch(plan, raw_patch)
    else:
        candidate = _apply_restricted_patch(plan, raw_patch)
    verify_equivalence(original_circuit, candidate)
    return candidate


def _patch_has_operations(patch: LayoutPatch | dict[str, Any] | list[dict[str, Any]]) -> bool:
    raw_patch = _as_patch_dict(patch)
    if isinstance(raw_patch, list):
        return bool(raw_patch)
    return any(bool(raw_patch.get(key)) for key in {"move_component", "move_label", "set_orientation", "set_wire_waypoints"})


class MockLLMClient:
    """Deterministic polish client used by tests and local instrumentation."""

    def _merge_patch(
        self,
        target: dict[str, list[dict[str, Any]]],
        patch: dict[str, Any],
    ) -> None:
        for key in ("move_component", "move_label", "set_orientation", "set_wire_waypoints"):
            value = patch.get(key)
            if not value:
                continue
            if isinstance(value, dict):
                target[key].append(value)
            else:
                target[key].extend(value)

    def propose_patch(
        self,
        layout_plan: LayoutPlan,
        critic_report: CriticReport,
        rendered: RenderResult,
    ) -> LayoutPatch:
        patch: dict[str, list[dict[str, Any]]] = {
            "move_component": [],
            "move_label": [],
            "set_orientation": [],
            "set_wire_waypoints": [],
        }
        components = {component.id: component for component in layout_plan.components}
        labels = {label.id: label for label in layout_plan.labels}
        input_xs = [
            component.grid_x
            for component in layout_plan.components
            if (component.role or "").lower() in {"input", "input_source", "input_terminal", "sensor"}
            or component.type.lower() in {"input", "input_terminal"}
        ]

        for violation in critic_report.violations:
            if violation.suggested_patch:
                self._merge_patch(patch, violation.suggested_patch)
                continue

            if violation.code.startswith("label_"):
                for label_id in violation.entities:
                    label = labels.get(label_id)
                    if not label:
                        continue
                    if label.text.upper().startswith("RG"):
                        patch["move_label"].append(
                            {"id": label_id, "grid_x": label.grid_x - 1.8, "grid_y": label.grid_y}
                        )
                    else:
                        patch["move_label"].append(
                            {"id": label_id, "grid_x": label.grid_x, "grid_y": label.grid_y - 1.1}
                        )
            elif violation.code == "opamp_not_facing_right":
                for component_id in violation.entities:
                    if component_id in components:
                        patch["set_orientation"].append(
                            {"id": component_id, "orientation": "right"}
                        )
            elif violation.code == "ground_not_down":
                for component_id in violation.entities:
                    if component_id in components:
                        patch["set_orientation"].append(
                            {"id": component_id, "orientation": "down"}
                        )
            elif violation.code == "output_not_right_of_input":
                min_output_x = max(input_xs, default=0.0) + 8.0
                for component in components.values():
                    role = (component.role or "").lower()
                    if role == "output" or component.type.lower() == "output":
                        patch["move_component"].append(
                            {
                                "id": component.id,
                                "grid_x": max(component.grid_x, min_output_x),
                                "grid_y": component.grid_y,
                            }
                        )
            elif violation.code == "wire_crossing" and set(violation.entities) == {"N_GAIN_TOP", "VINP"}:
                vinp_route = next((wire for wire in layout_plan.wires if wire.net_name == "VINP"), None)
                if vinp_route and len(vinp_route.waypoints) >= 2:
                    start = vinp_route.waypoints[0]
                    end = vinp_route.waypoints[-1]
                    patch["set_wire_waypoints"].append(
                        {
                            "net_name": "VINP",
                            "waypoints": [
                                {"x": start.x, "y": start.y},
                                {"x": start.x, "y": 2.0},
                                {"x": end.x, "y": 2.0},
                                {"x": end.x, "y": end.y},
                            ],
                        }
                    )

        return LayoutPatch(
            move_component=patch["move_component"],
            move_label=patch["move_label"],
            set_orientation=patch["set_orientation"],
            set_wire_waypoints=patch["set_wire_waypoints"],
        )


class GeminiLLMClient:
    """Optional client placeholder; tests never require network access."""

    def __init__(self, model: str = "gemini-1.5-pro") -> None:
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GeminiLLMClient requires GEMINI_API_KEY or GOOGLE_API_KEY."
            )

    def propose_patch(
        self,
        layout_plan: LayoutPlan,
        critic_report: CriticReport,
        rendered: RenderResult,
    ) -> LayoutPatch:
        raise NotImplementedError(
            "GeminiLLMClient is intentionally not invoked in offline tests. "
            "Wire it to a network client only in an integration environment."
        )


def generate_beautiful_schematic(
    circuit_ir: Any,
    max_iterations: int = 5,
    llm_client: Any = None,
) -> AgentResult:
    current_plan = plan_circuit(circuit_ir)
    verify_equivalence(circuit_ir, current_plan)
    current_render = render_layout(current_plan)
    current_report = critique_layout(current_plan, current_render)
    initial_score = current_report.total_score

    best_plan = current_plan
    best_render = current_render
    best_report = current_report
    iterations = 0
    debug_log: list[dict[str, Any]] = [
        {"iteration": 0, "score": current_report.total_score, "event": "initial"}
    ]

    for iteration in range(max_iterations):
        if best_report.total_score <= ACCEPTABLE_SCORE:
            break
        if llm_client is None:
            break

        iterations = iteration + 1
        patch = llm_client.propose_patch(current_plan, current_report, current_render)
        if not _patch_has_operations(patch):
            debug_log.append(
                {"iteration": iterations, "score": current_report.total_score, "event": "no_patch"}
            )
            break
        try:
            candidate_plan = apply_layout_patch(circuit_ir, current_plan, patch)
        except (ElectricalTopologyError, LayoutPatchError) as exc:
            debug_log.append(
                {
                    "iteration": iterations,
                    "score": current_report.total_score,
                    "event": "patch_rejected",
                    "error": str(exc),
                    "patch": _as_patch_dict(patch),
                }
            )
            break

        candidate_render = render_layout(candidate_plan)
        candidate_report = critique_layout(candidate_plan, candidate_render)
        previous_score = current_report.total_score
        debug_log.append(
            {
                "iteration": iterations,
                "previous_score": previous_score,
                "candidate_score": candidate_report.total_score,
                "event": "patch_evaluated",
                "patch": _as_patch_dict(patch),
            }
        )
        if candidate_report.total_score <= best_report.total_score:
            current_plan = candidate_plan
            current_render = candidate_render
            current_report = candidate_report
        if candidate_report.total_score < best_report.total_score:
            best_plan = candidate_plan
            best_render = candidate_render
            best_report = candidate_report
        if candidate_report.total_score >= previous_score:
            break

    return AgentResult(
        layout=best_plan,
        svg=best_render.svg,
        geometry=best_render.geometry,
        critic_report=best_report,
        iterations=iterations,
        improved=best_report.total_score < initial_score,
        debug_log=debug_log,
    )

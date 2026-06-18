"""Schemas for optional semantic schematic planning hints.

These objects are intentionally local dataclasses.  OptCPV should be usable
without a Pydantic runtime dependency, while still validating JSON from an
external planner before the deterministic renderer sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


INPUT_MODES = {"image_guided", "model_guided"}
ORIENTATIONS = {"RIGHT", "LEFT", "UP", "DOWN", "RIGHT_FLIP"}
ROUTE_POLICIES = {
    "left_to_right_manhattan",
    "top_feedback_corridor",
    "bottom_feedback_corridor",
    "bottom_auxiliary_corridor",
    "local_terminal_only",
    "avoid_opamp_body",
}
HINT_SOURCES = {"deterministic", "gemini", "fake", "manual"}


@dataclass(frozen=True)
class SchematicPlanningRequest:
    circuit_id: str
    components: list[dict[str, Any]]
    reference_image: dict[str, Any] | None = None
    input_mode: str = "model_guided"

    def __post_init__(self) -> None:
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {sorted(INPUT_MODES)}.")

    @classmethod
    def from_circuit(
        cls,
        circuit,
        *,
        reference_image: dict[str, Any] | None = None,
        input_mode: str = "model_guided",
    ) -> "SchematicPlanningRequest":
        return cls(
            circuit_id=circuit.id,
            reference_image=reference_image,
            input_mode=input_mode,
            components=[
                {
                    "id": component.id,
                    "type": component.type,
                    "role": component.role,
                    "label": component.label,
                    "value": component.value,
                    "pins": dict(component.pins),
                }
                for component in circuit.components
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "components": list(self.components),
            "reference_image": self.reference_image,
            "input_mode": self.input_mode,
        }


@dataclass(frozen=True)
class GridPlacementHint:
    component_id: str
    stage_x: int
    lane_y: int
    orientation: str = "RIGHT"
    role: str | None = None
    confidence: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", str(self.component_id))
        object.__setattr__(self, "stage_x", int(self.stage_x))
        object.__setattr__(self, "lane_y", int(self.lane_y))
        object.__setattr__(self, "orientation", _normalize_orientation(self.orientation))
        object.__setattr__(self, "role", None if self.role is None else str(self.role))
        object.__setattr__(self, "confidence", _clamp_float(self.confidence, 0.0, 1.0))

    @property
    def col(self) -> int:
        return self.stage_x

    @property
    def row(self) -> int:
        return self.lane_y

    @property
    def motif_role(self) -> str | None:
        return self.role

    @classmethod
    def from_dict(cls, component_id: str | None, raw: dict[str, Any]) -> "GridPlacementHint":
        if not isinstance(raw, dict):
            raise ValueError("GridPlacementHint must be an object.")
        if ("x" in raw or "y" in raw) and not ("stage_x" in raw or "lane_y" in raw or "col" in raw or "row" in raw):
            raise ValueError("Planning hints must use integer stage_x/lane_y, not absolute x/y coordinates.")
        use_component_id = str(raw.get("component_id", component_id or ""))
        if not use_component_id:
            raise ValueError("GridPlacementHint requires component_id.")
        return cls(
            component_id=use_component_id,
            stage_x=int(raw.get("stage_x", raw.get("col"))),
            lane_y=int(raw.get("lane_y", raw.get("row"))),
            orientation=str(raw.get("orientation", "RIGHT")),
            role=_optional_str(raw.get("role", raw.get("motif_role"))),
            confidence=float(raw.get("confidence", 0.5)),
        )

    def with_updates(self, **updates: Any) -> "GridPlacementHint":
        return GridPlacementHint(
            component_id=str(updates.get("component_id", self.component_id)),
            stage_x=int(updates.get("stage_x", self.stage_x)),
            lane_y=int(updates.get("lane_y", self.lane_y)),
            orientation=str(updates.get("orientation", self.orientation)),
            role=updates.get("role", self.role),
            confidence=float(updates.get("confidence", self.confidence)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "stage_x": self.stage_x,
            "lane_y": self.lane_y,
            "orientation": self.orientation.upper(),
            "role": self.role,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class StageHint:
    stage_x: int
    stage_type: str
    members: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_x", int(self.stage_x))
        object.__setattr__(self, "stage_type", str(self.stage_type))
        object.__setattr__(self, "members", tuple(str(item) for item in self.members))

    @property
    def stage(self) -> int:
        return self.stage_x

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StageHint":
        if not isinstance(raw, dict):
            raise ValueError("StageHint must be an object.")
        return cls(
            stage_x=int(raw.get("stage_x", raw.get("stage"))),
            stage_type=str(raw.get("stage_type", raw.get("type", ""))),
            members=tuple(str(item) for item in raw.get("members", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"stage_x": self.stage_x, "stage_type": self.stage_type, "members": list(self.members)}


@dataclass(frozen=True)
class LaneHint:
    lane_y: int
    lane_type: str
    members: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_y", int(self.lane_y))
        object.__setattr__(self, "lane_type", str(self.lane_type))
        object.__setattr__(self, "members", tuple(str(item) for item in self.members))

    @property
    def lane(self) -> int:
        return self.lane_y

    @property
    def source(self) -> str:
        return self.members[0] if self.members else ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LaneHint":
        if not isinstance(raw, dict):
            raise ValueError("LaneHint must be an object.")
        source = raw.get("source")
        members = list(raw.get("members", ()))
        if source and str(source) not in {str(item) for item in members}:
            members.insert(0, str(source))
        return cls(
            lane_y=int(raw.get("lane_y", raw.get("lane"))),
            lane_type=str(raw.get("lane_type", raw.get("type", raw.get("source", "")))),
            members=tuple(str(item) for item in members),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"lane_y": self.lane_y, "lane_type": self.lane_type, "members": list(self.members)}


@dataclass(frozen=True)
class RoutePolicyHint:
    net: str | None
    net_role: str
    policy: str

    def __post_init__(self) -> None:
        policy = str(self.policy)
        if policy not in ROUTE_POLICIES:
            raise ValueError(f"Unknown route policy {policy!r}.")
        object.__setattr__(self, "net", None if self.net is None else str(self.net))
        object.__setattr__(self, "net_role", str(self.net_role))
        object.__setattr__(self, "policy", policy)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RoutePolicyHint":
        if not isinstance(raw, dict):
            raise ValueError("RoutePolicyHint must be an object.")
        return cls(
            net=_optional_str(raw.get("net")),
            net_role=str(raw.get("net_role", raw.get("role", ""))),
            policy=str(raw.get("policy")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"net": self.net, "net_role": self.net_role, "policy": self.policy}


@dataclass(frozen=True)
class MotifHint:
    motif_type: str
    members: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MotifHint":
        if not isinstance(raw, dict):
            raise ValueError("MotifHint must be an object.")
        return cls(
            motif_type=str(raw.get("motif_type", raw.get("type", ""))),
            members=tuple(str(item) for item in raw.get("members", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"motif_type": self.motif_type, "members": list(self.members)}


@dataclass(frozen=True)
class BlockHint:
    block_id: str
    block_type: str
    members: tuple[str, ...] = field(default_factory=tuple)
    stage_x: int = 0
    lane_y: int = 0
    ports: dict[str, str] = field(default_factory=dict)
    route_policy: str | None = None

    def __post_init__(self) -> None:
        policy = None if self.route_policy is None else str(self.route_policy)
        if policy is not None and policy not in ROUTE_POLICIES:
            raise ValueError(f"Unknown block route policy {policy!r}.")
        object.__setattr__(self, "block_id", str(self.block_id))
        object.__setattr__(self, "block_type", str(self.block_type))
        object.__setattr__(self, "members", tuple(str(item) for item in self.members))
        object.__setattr__(self, "stage_x", int(self.stage_x))
        object.__setattr__(self, "lane_y", int(self.lane_y))
        object.__setattr__(self, "ports", {str(key): str(value) for key, value in self.ports.items()})
        object.__setattr__(self, "route_policy", policy)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BlockHint":
        if not isinstance(raw, dict):
            raise ValueError("BlockHint must be an object.")
        ports = raw.get("ports", {})
        if not isinstance(ports, dict):
            raise ValueError("BlockHint.ports must be an object.")
        return cls(
            block_id=str(raw.get("block_id", raw.get("id", ""))),
            block_type=str(raw.get("block_type", raw.get("type", ""))),
            members=tuple(str(item) for item in raw.get("members", ())),
            stage_x=int(raw.get("stage_x", raw.get("stage", 0))),
            lane_y=int(raw.get("lane_y", raw.get("lane", 0))),
            ports={str(key): str(value) for key, value in ports.items()},
            route_policy=_optional_str(raw.get("route_policy", raw.get("policy"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "members": list(self.members),
            "stage_x": self.stage_x,
            "lane_y": self.lane_y,
            "ports": dict(self.ports),
            "route_policy": self.route_policy,
        }


@dataclass(frozen=True)
class InterBlockRouteHint:
    net: str
    route_from: str
    route_to: str
    policy: str

    def __post_init__(self) -> None:
        policy = str(self.policy)
        if policy not in ROUTE_POLICIES:
            raise ValueError(f"Unknown inter-block route policy {policy!r}.")
        object.__setattr__(self, "net", str(self.net))
        object.__setattr__(self, "route_from", str(self.route_from))
        object.__setattr__(self, "route_to", str(self.route_to))
        object.__setattr__(self, "policy", policy)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InterBlockRouteHint":
        if not isinstance(raw, dict):
            raise ValueError("InterBlockRouteHint must be an object.")
        return cls(
            net=str(raw.get("net", "")),
            route_from=str(raw.get("from", raw.get("source", ""))),
            route_to=str(raw.get("to", raw.get("target", ""))),
            policy=str(raw.get("policy", "left_to_right_manhattan")),
        )

    def to_route_policy(self) -> RoutePolicyHint:
        return RoutePolicyHint(net=self.net, net_role="inter_block", policy=self.policy)

    def to_dict(self) -> dict[str, Any]:
        return {"net": self.net, "from": self.route_from, "to": self.route_to, "policy": self.policy}


@dataclass(frozen=True)
class AuxiliaryLoopHint:
    loop_id: str
    loop_type: str
    members: tuple[str, ...] = field(default_factory=tuple)
    nets: tuple[str, ...] = field(default_factory=tuple)
    route_policy: str = "bottom_auxiliary_corridor"

    def __post_init__(self) -> None:
        if self.route_policy not in ROUTE_POLICIES:
            raise ValueError(f"Unknown auxiliary loop route policy {self.route_policy!r}.")
        object.__setattr__(self, "loop_id", str(self.loop_id))
        object.__setattr__(self, "loop_type", str(self.loop_type))
        object.__setattr__(self, "members", tuple(str(item) for item in self.members))
        object.__setattr__(self, "nets", tuple(str(item) for item in self.nets))
        object.__setattr__(self, "route_policy", str(self.route_policy))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AuxiliaryLoopHint":
        if not isinstance(raw, dict):
            raise ValueError("AuxiliaryLoopHint must be an object.")
        return cls(
            loop_id=str(raw.get("loop_id", raw.get("id", ""))),
            loop_type=str(raw.get("loop_type", raw.get("type", ""))),
            members=tuple(str(item) for item in raw.get("members", ())),
            nets=tuple(str(item) for item in raw.get("nets", ())),
            route_policy=str(raw.get("route_policy", raw.get("policy", "bottom_auxiliary_corridor"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "loop_type": self.loop_type,
            "members": list(self.members),
            "nets": list(self.nets),
            "route_policy": self.route_policy,
        }


@dataclass(frozen=True)
class OrientationOverrideHint:
    component_id: str
    orientation: str
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", str(self.component_id))
        object.__setattr__(self, "orientation", _normalize_orientation(self.orientation))
        object.__setattr__(self, "reason", str(self.reason))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OrientationOverrideHint":
        if not isinstance(raw, dict):
            raise ValueError("OrientationOverrideHint must be an object.")
        return cls(
            component_id=str(raw.get("component_id", raw.get("id", ""))),
            orientation=str(raw.get("orientation", "RIGHT")),
            reason=str(raw.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "orientation": self.orientation.upper(),
            "reason": self.reason,
        }


@dataclass(frozen=True, init=False)
class SchematicLayoutHints:
    recognized_topology: str
    confidence: float
    tutor_explanation: str
    stages: tuple[StageHint, ...]
    lanes: tuple[LaneHint, ...]
    placements: tuple[GridPlacementHint, ...]
    route_policies: tuple[RoutePolicyHint, ...]
    local_terminal_policy: dict[str, str]
    source: str
    routing_rules: tuple[str, ...]
    motifs: tuple[MotifHint, ...]
    blocks: tuple[BlockHint, ...]
    block_internal_motifs: tuple[MotifHint, ...]
    inter_block_routes: tuple[InterBlockRouteHint, ...]
    auxiliary_loops: tuple[AuxiliaryLoopHint, ...]
    orientation_overrides: tuple[OrientationOverrideHint, ...]

    def __init__(
        self,
        recognized_topology: str = "",
        confidence: float = 0.0,
        tutor_explanation: str = "",
        stages: tuple[StageHint, ...] | list[StageHint] | None = None,
        lanes: tuple[LaneHint, ...] | list[LaneHint] | None = None,
        placements: tuple[GridPlacementHint, ...] | list[GridPlacementHint] | None = None,
        route_policies: tuple[RoutePolicyHint, ...] | list[RoutePolicyHint] | None = None,
        local_terminal_policy: dict[str, str] | None = None,
        source: str = "manual",
        *,
        stage_order: tuple[StageHint, ...] | list[StageHint] | None = None,
        placement_hints: dict[str, GridPlacementHint] | None = None,
        routing_rules: tuple[str, ...] | list[str] | None = None,
        motifs: tuple[MotifHint, ...] | list[MotifHint] | None = None,
        blocks: tuple[BlockHint, ...] | list[BlockHint] | None = None,
        block_internal_motifs: tuple[MotifHint, ...] | list[MotifHint] | None = None,
        inter_block_routes: tuple[InterBlockRouteHint, ...] | list[InterBlockRouteHint] | None = None,
        auxiliary_loops: tuple[AuxiliaryLoopHint, ...] | list[AuxiliaryLoopHint] | None = None,
        orientation_overrides: tuple[OrientationOverrideHint, ...] | list[OrientationOverrideHint] | None = None,
    ) -> None:
        use_stages = stages if stages is not None else stage_order
        use_placements = list(placements or ())
        if placement_hints:
            use_placements.extend(
                hint if hint.component_id == component_id else hint.with_updates(component_id=component_id)
                for component_id, hint in placement_hints.items()
            )
        use_source = source if source in HINT_SOURCES else "manual"
        object.__setattr__(self, "recognized_topology", str(recognized_topology))
        object.__setattr__(self, "confidence", _clamp_float(confidence, 0.0, 1.0))
        object.__setattr__(self, "tutor_explanation", str(tutor_explanation))
        object.__setattr__(self, "stages", tuple(use_stages or ()))
        object.__setattr__(self, "lanes", tuple(lanes or ()))
        object.__setattr__(self, "placements", tuple(use_placements))
        object.__setattr__(self, "route_policies", tuple(route_policies or ()))
        object.__setattr__(self, "local_terminal_policy", {str(k): str(v) for k, v in (local_terminal_policy or {}).items()})
        object.__setattr__(self, "source", use_source)
        object.__setattr__(self, "routing_rules", tuple(str(item) for item in (routing_rules or ())))
        object.__setattr__(self, "motifs", tuple(motifs or ()))
        object.__setattr__(self, "blocks", tuple(blocks or ()))
        object.__setattr__(self, "block_internal_motifs", tuple(block_internal_motifs or ()))
        object.__setattr__(self, "inter_block_routes", tuple(inter_block_routes or ()))
        object.__setattr__(self, "auxiliary_loops", tuple(auxiliary_loops or ()))
        object.__setattr__(self, "orientation_overrides", tuple(orientation_overrides or ()))

    @property
    def stage_order(self) -> tuple[StageHint, ...]:
        return self.stages

    @property
    def placement_hints(self) -> dict[str, GridPlacementHint]:
        return {hint.component_id: hint for hint in self.placements}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SchematicLayoutHints":
        if not isinstance(raw, dict):
            raise ValueError("SchematicLayoutHints must be a JSON object.")
        placements = _placements_from_raw(raw)
        route_policies = tuple(RoutePolicyHint.from_dict(item) for item in _list(raw.get("route_policies")))
        inter_block_routes = tuple(InterBlockRouteHint.from_dict(item) for item in _list(raw.get("inter_block_routes")))
        effective_route_policies = list(route_policies)
        for route in inter_block_routes:
            if not any(policy.net == route.net and policy.policy == route.policy for policy in effective_route_policies):
                effective_route_policies.append(route.to_route_policy())
        routing_rules = tuple(str(item) for item in raw.get("routing_rules", ()) if item is not None)
        return cls(
            recognized_topology=str(raw.get("recognized_topology", "")),
            confidence=float(raw.get("confidence", 0.0)),
            tutor_explanation=str(raw.get("tutor_explanation", "")),
            stages=tuple(StageHint.from_dict(item) for item in _list(raw.get("stages", raw.get("stage_order")))),
            lanes=tuple(LaneHint.from_dict(item) for item in _list(raw.get("lanes"))),
            placements=placements,
            route_policies=tuple(effective_route_policies),
            local_terminal_policy={str(k): str(v) for k, v in raw.get("local_terminal_policy", {}).items()},
            source=str(raw.get("source", "manual")),
            routing_rules=routing_rules,
            motifs=tuple(MotifHint.from_dict(item) for item in _list(raw.get("motifs"))),
            blocks=tuple(BlockHint.from_dict(item) for item in _list(raw.get("blocks"))),
            block_internal_motifs=tuple(MotifHint.from_dict(item) for item in _list(raw.get("block_internal_motifs"))),
            inter_block_routes=inter_block_routes,
            auxiliary_loops=tuple(AuxiliaryLoopHint.from_dict(item) for item in _list(raw.get("auxiliary_loops"))),
            orientation_overrides=tuple(OrientationOverrideHint.from_dict(item) for item in _list(raw.get("orientation_overrides"))),
        )

    @classmethod
    def from_json(cls, text: str) -> "SchematicLayoutHints":
        return cls.from_dict(_json_object(text))

    def with_updates(self, **updates: Any) -> "SchematicLayoutHints":
        return SchematicLayoutHints(
            recognized_topology=updates.get("recognized_topology", self.recognized_topology),
            confidence=updates.get("confidence", self.confidence),
            tutor_explanation=updates.get("tutor_explanation", self.tutor_explanation),
            stages=updates.get("stages", self.stages),
            lanes=updates.get("lanes", self.lanes),
            placements=updates.get("placements", self.placements),
            route_policies=updates.get("route_policies", self.route_policies),
            local_terminal_policy=updates.get("local_terminal_policy", self.local_terminal_policy),
            source=updates.get("source", self.source),
            routing_rules=updates.get("routing_rules", self.routing_rules),
            motifs=updates.get("motifs", self.motifs),
            blocks=updates.get("blocks", self.blocks),
            block_internal_motifs=updates.get("block_internal_motifs", self.block_internal_motifs),
            inter_block_routes=updates.get("inter_block_routes", self.inter_block_routes),
            auxiliary_loops=updates.get("auxiliary_loops", self.auxiliary_loops),
            orientation_overrides=updates.get("orientation_overrides", self.orientation_overrides),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognized_topology": self.recognized_topology,
            "confidence": self.confidence,
            "tutor_explanation": self.tutor_explanation,
            "stages": [stage.to_dict() for stage in self.stages],
            "lanes": [lane.to_dict() for lane in self.lanes],
            "placements": [placement.to_dict() for placement in self.placements],
            "route_policies": [policy.to_dict() for policy in self.route_policies],
            "local_terminal_policy": dict(self.local_terminal_policy),
            "source": self.source,
            "routing_rules": list(self.routing_rules),
            "motifs": [motif.to_dict() for motif in self.motifs],
            "blocks": [block.to_dict() for block in self.blocks],
            "block_internal_motifs": [motif.to_dict() for motif in self.block_internal_motifs],
            "inter_block_routes": [route.to_dict() for route in self.inter_block_routes],
            "auxiliary_loops": [loop.to_dict() for loop in self.auxiliary_loops],
            "orientation_overrides": [override.to_dict() for override in self.orientation_overrides],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


PlanningHints = SchematicLayoutHints


def validate_schematic_layout_hints(raw: dict[str, Any] | str | SchematicLayoutHints) -> SchematicLayoutHints:
    if isinstance(raw, SchematicLayoutHints):
        return raw
    if isinstance(raw, str):
        return SchematicLayoutHints.from_json(raw)
    return SchematicLayoutHints.from_dict(raw)


def _placements_from_raw(raw: dict[str, Any]) -> tuple[GridPlacementHint, ...]:
    if "placements" in raw:
        value = raw.get("placements")
        if not isinstance(value, list):
            raise ValueError("SchematicLayoutHints.placements must be a list.")
        return tuple(GridPlacementHint.from_dict(None, item) for item in value)
    legacy = raw.get("placement_hints", {})
    if not isinstance(legacy, dict):
        raise ValueError("planning_hints.placement_hints must be an object.")
    return tuple(GridPlacementHint.from_dict(str(component_id), hint) for component_id, hint in legacy.items())


def _json_object(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        raw = json.loads(match.group(0))
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object.")
    return raw


def _list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of objects.")
    return [item for item in value if isinstance(item, dict)]


def _normalize_orientation(value: str | None) -> str:
    key = (value or "RIGHT").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "east": "right",
        "west": "left",
        "north": "up",
        "south": "down",
        "rightflip": "right_flip",
    }
    key = aliases.get(key, key)
    if key.upper() not in ORIENTATIONS:
        return "right"
    return key


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)

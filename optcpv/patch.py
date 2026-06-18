"""Restricted topology-safe layout patches."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .models import Circuit, LayoutLabel, LayoutPlan, Point
from .planner import rebuild_layout_geometry
from .route_contract import assert_no_diagonal_wires, orthogonalize_route
from .verifier import verify_layout_topology
from .vector_critic import critique_layout


@dataclass(frozen=True)
class MoveComponent:
    component_id: str
    x: float
    y: float


@dataclass(frozen=True)
class MoveLabel:
    label_id: str
    x: float
    y: float


@dataclass(frozen=True)
class SetOrientation:
    component_id: str
    orientation: str


@dataclass(frozen=True)
class SetWirePoints:
    net: str
    points: list[Point]


@dataclass(frozen=True)
class SetRoutePolicy:
    net: str
    policy: str
    net_role: str = "feedback"


@dataclass(frozen=True)
class LayoutPatch:
    move_component: list[MoveComponent] = field(default_factory=list)
    move_label: list[MoveLabel] = field(default_factory=list)
    set_orientation: list[SetOrientation] = field(default_factory=list)
    set_wire_points: list[SetWirePoints] = field(default_factory=list)
    set_route_policy: list[SetRoutePolicy] = field(default_factory=list)


def apply_patch(circuit: Circuit, layout: LayoutPlan, patch: LayoutPatch) -> LayoutPlan:
    components = list(layout.components)
    for move in patch.move_component:
        components = [
            replace(component, x=move.x, y=move.y) if component.id == move.component_id else component
            for component in components
        ]
    for orientation in patch.set_orientation:
        components = [
            replace(component, orientation=orientation.orientation) if component.id == orientation.component_id else component
            for component in components
        ]

    label_moves = {move.label_id: move for move in patch.move_label}
    labels = [
        _move_label(label, label_moves[label.id]) if label.id in label_moves else label
        for label in layout.labels
    ]
    support = _support_with_route_policies(layout, patch.set_route_policy)
    candidate = replace(layout, components=components, labels=labels, support=support)
    candidate = rebuild_layout_geometry(candidate)

    if patch.set_wire_points:
        wire_points = {item.net: orthogonalize_route(item.points) for item in patch.set_wire_points}
        candidate = replace(
            candidate,
            wires=[
                replace(wire, points=wire_points[wire.net]) if wire.net in wire_points else wire
                for wire in candidate.wires
            ],
        )

    verify_layout_topology(circuit, candidate)
    assert_no_diagonal_wires(candidate)
    _reject_scale_hack(layout, candidate)
    return candidate


def _support_with_route_policies(layout: LayoutPlan, policies: list[SetRoutePolicy]):
    if not policies:
        return layout.support
    planning_hints = dict(layout.support.planning_hints or {})
    raw_policies = planning_hints.get("route_policies", [])
    route_policies = [dict(item) for item in raw_policies if isinstance(item, dict)]
    by_net = {
        str(item.get("net")): item
        for item in route_policies
        if item.get("net") is not None
    }
    ordered_nets = [str(item.get("net")) for item in route_policies if item.get("net") is not None]
    for policy in policies:
        item = {"net": policy.net, "net_role": policy.net_role, "policy": policy.policy}
        if policy.net not in by_net:
            ordered_nets.append(policy.net)
        by_net[policy.net] = item
    replaced = []
    emitted: set[str] = set()
    for item in route_policies:
        net = item.get("net")
        if net is None:
            replaced.append(item)
            continue
        net_key = str(net)
        if net_key in by_net and net_key not in emitted:
            replaced.append(by_net[net_key])
            emitted.add(net_key)
    for net in ordered_nets:
        if net not in emitted and net in by_net:
            replaced.append(by_net[net])
            emitted.add(net)
    planning_hints["route_policies"] = replaced
    return replace(layout.support, planning_hints=planning_hints)


def _move_label(label: LayoutLabel, move: MoveLabel) -> LayoutLabel:
    dx = move.x - label.x
    dy = move.y - label.y
    return replace(
        label,
        x=move.x,
        y=move.y,
        bbox=replace(label.bbox, x=label.bbox.x + dx, y=label.bbox.y + dy),
    )


def _reject_scale_hack(before: LayoutPlan, after: LayoutPlan) -> None:
    if before.width != after.width or before.height != after.height or before.grid != after.grid:
        raise ValueError("LayoutPatch may not change canvas size or grid.")
    before_metrics = critique_layout(before).metrics
    after_metrics = critique_layout(after).metrics
    if float(after_metrics["average_component_distance"]) > float(before_metrics["average_component_distance"]) * 1.35 + 0.2:
        raise ValueError("Rejected patch that spreads components excessively.")
    if (
        float(after_metrics["average_component_distance"]) > float(before_metrics["average_component_distance"]) * 1.15 + 0.2
        and float(after_metrics["total_wire_length"]) > float(before_metrics["total_wire_length"]) * 2.5 + 8.0
    ):
        raise ValueError("Rejected patch that increases wire length excessively.")

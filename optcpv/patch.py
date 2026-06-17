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
class LayoutPatch:
    move_component: list[MoveComponent] = field(default_factory=list)
    move_label: list[MoveLabel] = field(default_factory=list)
    set_orientation: list[SetOrientation] = field(default_factory=list)
    set_wire_points: list[SetWirePoints] = field(default_factory=list)


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
    candidate = replace(layout, components=components, labels=labels)
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

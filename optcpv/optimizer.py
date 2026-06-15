"""Topology-safe CV-native layout optimization."""

from __future__ import annotations

from dataclasses import replace

from .artifact import artifact_from_layout
from .critic import critique
from .models import Circuit, LayoutComponent, LayoutPlan, SchematicArtifact, circuit_from_any
from .patch import LayoutPatch, MoveComponent, MoveLabel, apply_patch
from .planner import plan_layout
from .renderer import render_svg
from .verifier import verify_layout_topology


def draw_optimized_svg(circuit: Circuit | dict, *, max_iterations: int = 5) -> str:
    return draw_optimized_artifact(circuit, max_iterations=max_iterations).svg


def draw_optimized_artifact(circuit: Circuit | dict, *, max_iterations: int = 5) -> SchematicArtifact:
    native = circuit_from_any(circuit)
    layout = plan_layout(native)
    verify_layout_topology(native, layout)
    svg = render_svg(layout)
    report = critique(native, layout, svg)
    best_layout, best_svg, best_report = layout, svg, report
    log = [{"iteration": 0, "score": report.score, "accepted": True}]

    current_layout = layout
    current_report = report
    for iteration in range(1, max_iterations + 1):
        patch = propose_local_patch(current_layout, current_report)
        if _empty_patch(patch):
            break
        try:
            candidate_layout = apply_patch(native, current_layout, patch)
        except Exception as exc:
            log.append({"iteration": iteration, "score": current_report.score, "accepted": False, "reason": str(exc)})
            break
        candidate_svg = render_svg(candidate_layout)
        candidate_report = critique(native, candidate_layout, candidate_svg)
        accepted = candidate_report.score <= current_report.score - 0.5
        log.append({"iteration": iteration, "score": candidate_report.score, "accepted": accepted})
        if accepted:
            current_layout, current_report = candidate_layout, candidate_report
            if candidate_report.score < best_report.score:
                best_layout, best_svg, best_report = candidate_layout, candidate_svg, candidate_report
        else:
            break

    return artifact_from_layout(best_layout, best_svg, critic_report=best_report, cv_report=best_report, optimization_log=log)


def propose_local_patch(layout: LayoutPlan, report) -> LayoutPatch:
    moves: list[MoveComponent] = []
    label_moves: list[MoveLabel] = []
    codes = {violation.code for violation in report.violations}

    for component in layout.components:
        x, y = component.x, component.y
        if component.type.lower() in {"ground", "gnd"}:
            y = max(y, _median_y(layout) + 2.2)
        if component.type.lower() == "output":
            x = max(x, _max_input_x(layout) + 5.0)
        if _is_feedback(component):
            y = min(y, _opamp_y(layout) - 2.8)
        moves.append(MoveComponent(component.id, x, y))

    raster_fill = float(report.metrics.get("cv.ink_bbox_fill_ratio", 0.0))
    component_fill = float(report.metrics.get("vector.component_fill_ratio", 0.0))
    should_compact = "fill_ratio_low" in codes or "too_much_empty_canvas" in codes or (
        "spread_excessive" in codes and raster_fill < 0.28 and component_fill < 0.45
    )
    if should_compact:
        moves = _compact_moves(layout, moves)

    if "component_overlap" in codes:
        moves = _separate_overlaps(layout, moves)

    for label in layout.labels:
        owner = next((component for component in layout.components if component.id == label.owner_id), None)
        if owner:
            label_moves.append(MoveLabel(label.id, owner.x, owner.bbox.y - 0.35))

    return LayoutPatch(move_component=moves, move_label=label_moves)


def _compact_moves(layout: LayoutPlan, moves: list[MoveComponent]) -> list[MoveComponent]:
    move_by_id = {move.component_id: move for move in moves}
    components = layout.components
    min_x = min(component.x for component in components)
    max_x = max(component.x for component in components)
    min_y = min(component.y for component in components)
    max_y = max(component.y for component in components)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    target_span_x = min(span_x, 15.5)
    target_span_y = min(span_y, 10.0)
    compacted: list[MoveComponent] = []
    for component in components:
        base = move_by_id[component.id]
        nx = 2.0 + ((base.x - min_x) / span_x) * target_span_x
        ny = 2.2 + ((base.y - min_y) / span_y) * target_span_y
        compacted.append(MoveComponent(component.id, nx, ny))
    return compacted


def _separate_overlaps(layout: LayoutPlan, moves: list[MoveComponent]) -> list[MoveComponent]:
    move_by_id = {move.component_id: move for move in moves}
    adjusted = dict(move_by_id)
    ordered = sorted(layout.components, key=lambda component: (component.y, component.x))
    for index, component in enumerate(ordered):
        move = adjusted[component.id]
        adjusted[component.id] = replace(move, y=move.y + index * 0.12)
    return list(adjusted.values())


def _empty_patch(patch: LayoutPatch) -> bool:
    return not (patch.move_component or patch.move_label or patch.set_orientation or patch.set_wire_points)


def _median_y(layout: LayoutPlan) -> float:
    values = sorted(component.y for component in layout.components)
    return values[len(values) // 2]


def _max_input_x(layout: LayoutPlan) -> float:
    inputs = [component.x for component in layout.components if component.type.lower() in {"input", "source", "voltage_source"}]
    return max(inputs or [0.0])


def _opamp_y(layout: LayoutPlan) -> float:
    opamps = [component.y for component in layout.components if "op" in component.type.lower()]
    return sum(opamps) / len(opamps) if opamps else 0.0


def _is_feedback(component: LayoutComponent) -> bool:
    return "feedback" in (component.role or "").lower()

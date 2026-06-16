"""Topology-safe CV-native layout optimization."""

from __future__ import annotations

from dataclasses import replace

from .artifact import artifact_from_layout
from .critic import CriticBreakdown, critique_parts
from .models import BBox, Circuit, LayoutComponent, LayoutLabel, LayoutPlan, Point, SchematicArtifact, circuit_from_any
from .patch import LayoutPatch, MoveComponent, MoveLabel, apply_patch
from .planner import plan_layout
from .raster import rasterize_svg
from .renderer import render_svg_layers
from .verifier import verify_layout_topology
from .vision_agent import VisionLayoutClient


def draw_optimized_svg(
    circuit: Circuit | dict,
    *,
    max_iterations: int = 5,
    vision_client: VisionLayoutClient | None = None,
) -> str:
    return draw_optimized_artifact(circuit, max_iterations=max_iterations, vision_client=vision_client).svg


def draw_optimized_artifact(
    circuit: Circuit | dict,
    *,
    max_iterations: int = 5,
    vision_client: VisionLayoutClient | None = None,
) -> SchematicArtifact:
    native = circuit_from_any(circuit)
    layout = plan_layout(native)
    verify_layout_topology(native, layout)
    layers = render_svg_layers(layout)
    reports = critique_parts(native, layout, layers.final_svg, layers=layers)
    best_layout, best_svg, best_reports = layout, layers.final_svg, reports
    log = [{"iteration": 0, "source": "initial", "score": reports.combined_report.score, "accepted": True}]

    current_layout = layout
    current_svg = layers.final_svg
    current_reports = reports
    for iteration in range(1, max_iterations + 1):
        local_patch = propose_local_patch(current_layout, current_reports.combined_report)
        local_result = _evaluate_patch(
            native,
            current_layout,
            current_reports,
            local_patch,
            iteration=iteration,
            source="local",
            log=log,
        )
        if local_result is not None:
            current_layout, current_svg, current_reports = local_result
            if current_reports.combined_report.score < best_reports.combined_report.score:
                best_layout, best_svg, best_reports = current_layout, current_svg, current_reports
            continue

        if vision_client is None:
            break

        try:
            raster = rasterize_svg(current_svg)
            vision_patch = vision_client.propose_patch(
                native,
                current_layout,
                current_svg,
                raster,
                current_reports.combined_report,
            )
        except Exception as exc:
            log.append(
                {
                    "iteration": iteration,
                    "source": "vision",
                    "score": current_reports.combined_report.score,
                    "accepted": False,
                    "reason": f"vision_client_error: {exc}",
                }
            )
            break
        vision_result = _evaluate_patch(
            native,
            current_layout,
            current_reports,
            vision_patch,
            iteration=iteration,
            source="vision",
            log=log,
        )
        if vision_result is None:
            break
        current_layout, current_svg, current_reports = vision_result
        if current_reports.combined_report.score < best_reports.combined_report.score:
            best_layout, best_svg, best_reports = current_layout, current_svg, current_reports

    return artifact_from_layout(
        best_layout,
        best_svg,
        vector_report=best_reports.vector_report,
        cv_report=best_reports.cv_report,
        combined_report=best_reports.combined_report,
        optimization_log=log,
    )


def _evaluate_patch(
    circuit: Circuit,
    current_layout: LayoutPlan,
    current_reports: CriticBreakdown,
    patch: LayoutPatch,
    *,
    iteration: int,
    source: str,
    log: list[dict],
) -> tuple[LayoutPlan, str, CriticBreakdown] | None:
    if _empty_patch(patch):
        log.append(
            {
                "iteration": iteration,
                "source": source,
                "score": current_reports.combined_report.score,
                "accepted": False,
                "reason": "empty_patch",
            }
        )
        return None
    try:
        candidate_layout = apply_patch(circuit, current_layout, patch)
    except Exception as exc:
        log.append(
            {
                "iteration": iteration,
                "source": source,
                "score": current_reports.combined_report.score,
                "accepted": False,
                "reason": str(exc),
            }
        )
        return None
    candidate_layers = render_svg_layers(candidate_layout)
    candidate_reports = critique_parts(circuit, candidate_layout, candidate_layers.final_svg, layers=candidate_layers)
    accepted = candidate_reports.combined_report.score <= current_reports.combined_report.score - 0.5
    log.append(
        {
            "iteration": iteration,
            "source": source,
            "score": candidate_reports.combined_report.score,
            "accepted": accepted,
        }
    )
    if not accepted:
        return None
    return candidate_layout, candidate_layers.final_svg, candidate_reports


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

    label_moves = _propose_label_moves(layout)

    return LayoutPatch(move_component=moves, move_label=label_moves)


def _propose_label_moves(layout: LayoutPlan) -> list[MoveLabel]:
    moves: list[MoveLabel] = []
    for label in layout.labels:
        owner = next((component for component in layout.components if component.id == label.owner_id), None)
        if owner is None:
            continue
        candidates = _label_candidates(owner)
        best_x, best_y = min(
            candidates,
            key=lambda candidate: _label_candidate_score(layout, label, owner, candidate[0], candidate[1]),
        )
        moves.append(MoveLabel(label.id, best_x, best_y))
    return moves


def _label_candidates(owner: LayoutComponent) -> list[tuple[float, float]]:
    above = (owner.x, owner.bbox.y - 0.35)
    below = (owner.x, owner.bbox.bottom + 0.45)
    right = (owner.bbox.right + 0.45, owner.y)
    left = (owner.bbox.x - 0.45, owner.y)
    if owner.orientation in {"up", "down"} and owner.type.lower() not in {"ground", "gnd"}:
        return [right, left, above, below]
    if owner.type.lower() in {"ground", "gnd"}:
        return [below, above, right, left]
    return [above, below, right, left]


def _label_candidate_score(
    layout: LayoutPlan,
    label: LayoutLabel,
    owner: LayoutComponent,
    x: float,
    y: float,
) -> float:
    bbox = _moved_label_bbox(label, x, y)
    score = abs(x - label.x) * 0.08 + abs(y - label.y) * 0.08
    if bbox.x < 0 or bbox.y < 0 or bbox.right > layout.width / layout.grid or bbox.bottom > layout.height / layout.grid:
        score += 100.0
    for component in layout.components:
        if component.id == owner.id:
            continue
        if bbox.intersects(component.bbox, padding=0.05):
            score += 35.0
    for wire in layout.wires:
        if _polyline_intersects_bbox(wire.points, bbox.expanded(0.03)):
            score += 20.0
    if bbox.intersects(owner.bbox, padding=0.02):
        score += 10.0
    return score


def _moved_label_bbox(label: LayoutLabel, x: float, y: float) -> BBox:
    dx = x - label.x
    dy = y - label.y
    return BBox(label.bbox.x + dx, label.bbox.y + dy, label.bbox.width, label.bbox.height)


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

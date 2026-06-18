"""Public drawing and artifact entry points for OptCPV."""

from __future__ import annotations

from .critic import critique_parts
from .models import Circuit, CriticReport, LayoutPlan, SchematicArtifact, circuit_from_any
from .planning_agent import SemanticPlanningClient, planning_client_from_env
from .planner import plan_layout
from .renderer import render_svg
from .verifier import verify_layout_topology


def draw_svg(
    circuit: Circuit | dict,
    *,
    style: str = "textbook",
    planning_client: SemanticPlanningClient | None = None,
    reference_image: bytes | None = None,
) -> str:
    """Return a one-pass Schemdraw SVG for a native OptCPV circuit description."""

    native = circuit_from_any(circuit)
    planning_client = planning_client or planning_client_from_env()
    layout = plan_layout(native, planning_client=planning_client, reference_image=reference_image)
    verify_layout_topology(native, layout)
    return render_svg(layout, style=style)


def draw_artifact(
    circuit: Circuit | dict,
    *,
    style: str = "textbook",
    planning_client: SemanticPlanningClient | None = None,
    reference_image: bytes | None = None,
) -> SchematicArtifact:
    """Return a deterministic artifact with topology, vector, and CV criticism."""

    native = circuit_from_any(circuit)
    planning_client = planning_client or planning_client_from_env()
    layout = plan_layout(native, planning_client=planning_client, reference_image=reference_image)
    verify_layout_topology(native, layout)
    svg = render_svg(layout, style=style)
    reports = critique_parts(native, layout, svg)
    return artifact_from_layout(
        layout,
        svg,
        vector_report=reports.vector_report,
        cv_report=reports.cv_report,
        combined_report=reports.combined_report,
        optimization_log=[],
    )


def artifact_from_layout(
    layout: LayoutPlan,
    svg: str,
    *,
    vector_report: CriticReport | None,
    cv_report: CriticReport | None,
    combined_report: CriticReport | None,
    optimization_log: list[dict],
) -> SchematicArtifact:
    combined_dict = combined_report.to_dict() if combined_report else None
    return SchematicArtifact(
        svg=svg,
        components=_component_metadata(layout),
        nets=_net_metadata(layout),
        labels=_label_metadata(layout),
        viewbox={"x": 0, "y": 0, "width": layout.width, "height": layout.height},
        layout_support=layout.support.to_dict(),
        semantic_plan=layout.semantic.to_dict(),
        planning_hints_used=layout.support.planning_hints,
        visual_review_result=layout.support.visual_review,
        tutor_explanation=_tutor_explanation(layout),
        fallback_used=layout.support.fallback_used,
        layout_confidence=layout.support.layout_confidence,
        critic_report=combined_dict,
        vector_report=vector_report.to_dict() if vector_report else None,
        cv_report=cv_report.to_dict() if cv_report else None,
        combined_report=combined_dict,
        optimization_log=optimization_log,
        warnings=list(layout.warnings),
    )


def _component_metadata(layout: LayoutPlan) -> dict[str, dict]:
    return {
        component.id: {
            "id": component.id,
            "type": component.type,
            "label": component.label or component.value or component.id,
            "pins": dict(component.pins),
            "position": {"x": component.x, "y": component.y},
            "orientation": component.orientation,
            "bbox": {
                "x": component.bbox.x,
                "y": component.bbox.y,
                "width": component.bbox.width,
                "height": component.bbox.height,
            },
        }
        for component in layout.components
    }


def _net_metadata(layout: LayoutPlan) -> dict[str, dict]:
    wires_by_net = {wire.net: wire for wire in layout.wires}
    terminals_by_net: dict[str, list[dict]] = {}
    for terminal in layout.semantic.local_terminals:
        terminals_by_net.setdefault(terminal.net, []).append(terminal.to_dict())
    result: dict[str, dict] = {}
    for net, pins in layout.net_to_pins.items():
        wire = wires_by_net.get(net)
        net_class = layout.semantic.net_classes.get(net)
        result[net] = {
            "name": net,
            "class": net_class.value if net_class else "signal",
            "connected_pins": list(pins),
            "points": [{"x": point.x, "y": point.y} for point in wire.points] if wire else [],
            "local_terminals": terminals_by_net.get(net, []),
            "routed": wire is not None,
        }
    return result


def _label_metadata(layout: LayoutPlan) -> dict[str, dict]:
    return {
        label.id: {
            "id": label.id,
            "text": label.text,
            "owner_id": label.owner_id,
            "position": {"x": label.x, "y": label.y},
            "anchor": label.anchor,
            "bbox": {
                "x": label.bbox.x,
                "y": label.bbox.y,
                "width": label.bbox.width,
                "height": label.bbox.height,
            },
        }
        for label in layout.labels
    }


def _tutor_explanation(layout: LayoutPlan) -> str:
    if layout.support.tutor_explanation:
        return layout.support.tutor_explanation
    if layout.support.planning_hints:
        text = str(layout.support.planning_hints.get("tutor_explanation") or "")
        if text:
            return text
    motif = ", ".join(layout.support.matched_motifs)
    if motif:
        return f"OptCPV arranged this schematic as a {motif} teaching layout."
    return "OptCPV arranged this schematic with topology-preserving deterministic placement."

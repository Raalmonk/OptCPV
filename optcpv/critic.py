"""Combined topology, vector, and CV critic."""

from __future__ import annotations

from .cv_critic import critique_raster
from .models import Circuit, CriticReport, CriticViolation, LayoutPlan
from .raster import EVAL_HEIGHT, EVAL_WIDTH, rasterize_svg
from .vector_critic import critique_layout
from .verifier import TopologyVerificationError, verify_layout_topology


def critique(circuit: Circuit, layout: LayoutPlan, svg: str) -> CriticReport:
    violations: list[CriticViolation] = []
    metrics: dict[str, float | int | str | bool] = {
        "eval_width": EVAL_WIDTH,
        "eval_height": EVAL_HEIGHT,
    }

    topology_failed = False
    try:
        verify_layout_topology(circuit, layout)
    except TopologyVerificationError as exc:
        topology_failed = True
        violations.append(CriticViolation("topology_failure", str(exc), 100, True))

    vector = critique_layout(layout)
    raster = rasterize_svg(svg, output_width=EVAL_WIDTH, output_height=EVAL_HEIGHT)
    cv = critique_raster(layout, raster)

    violations.extend(vector.violations)
    violations.extend(cv.violations)
    metrics.update({f"vector.{key}": value for key, value in vector.metrics.items()})
    metrics.update({f"cv.{key}": value for key, value in cv.metrics.items()})
    metrics["topology_failed"] = topology_failed

    score = sum(violation.severity for violation in violations)
    return CriticReport(score=score, violations=violations, metrics=metrics, hard_fail=any(v.hard for v in violations))

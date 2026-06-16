"""Combined topology, vector, and CV critic."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .cv_critic import critique_raster
from .models import Circuit, CriticReport, CriticViolation, LayoutPlan, RenderedLayers
from .raster import EVAL_HEIGHT, EVAL_WIDTH, rasterize_svg
from .renderer import render_svg_layers
from .vector_critic import critique_layout
from .verifier import TopologyVerificationError, verify_layout_topology


@dataclass(frozen=True)
class CriticBreakdown:
    vector_report: CriticReport
    cv_report: CriticReport
    combined_report: CriticReport


def critique(circuit: Circuit, layout: LayoutPlan, svg: str) -> CriticReport:
    return critique_parts(circuit, layout, svg).combined_report


def critique_parts(
    circuit: Circuit,
    layout: LayoutPlan,
    svg: str,
    *,
    layers: RenderedLayers | None = None,
) -> CriticBreakdown:
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
    use_layers = layers is not None or "data-optcpv-circuit-id=" in svg or "data-renderer=\"optcpv." in svg
    layers = layers or (render_svg_layers(layout, final_svg=svg) if use_layers else None)
    raster = rasterize_svg((layers.final_svg if layers else svg), output_width=EVAL_WIDTH, output_height=EVAL_HEIGHT)
    if layers:
        wire_raster = rasterize_svg(layers.wires_svg, output_width=EVAL_WIDTH, output_height=EVAL_HEIGHT)
        component_raster = rasterize_svg(layers.components_svg, output_width=EVAL_WIDTH, output_height=EVAL_HEIGHT)
        label_raster = rasterize_svg(layers.labels_svg, output_width=EVAL_WIDTH, output_height=EVAL_HEIGHT)
        cv = critique_raster(
            layout,
            raster,
            wire_raster=wire_raster,
            component_raster=component_raster,
            label_raster=label_raster,
        )
    else:
        cv = critique_raster(layout, raster)
    svg_violations, svg_metrics = _svg_visual_contract_violations(svg, layout)

    violations.extend(vector.violations)
    violations.extend(svg_violations)
    violations.extend(cv.violations)
    metrics.update({f"vector.{key}": value for key, value in vector.metrics.items()})
    metrics.update({f"svg.{key}": value for key, value in svg_metrics.items()})
    metrics.update({f"cv.{key}": value for key, value in cv.metrics.items()})
    metrics["topology_failed"] = topology_failed

    score = sum(violation.severity for violation in violations)
    combined = CriticReport(score=score, violations=violations, metrics=metrics, hard_fail=any(v.hard for v in violations))
    return CriticBreakdown(vector_report=vector, cv_report=cv, combined_report=combined)


def _svg_visual_contract_violations(svg: str, layout: LayoutPlan) -> tuple[list[CriticViolation], dict[str, float | int | str | bool]]:
    violations: list[CriticViolation] = []
    metrics: dict[str, float | int | str | bool] = {}

    viewbox = _root_viewbox(svg)
    expected_area = float(layout.width * layout.height)
    metrics["expected_viewbox_area"] = expected_area
    if viewbox:
        x, y, width, height = viewbox
        viewbox_area = width * height
        fixed = abs(x) < 1e-6 and abs(y) < 1e-6 and abs(width - layout.width) < 1e-6 and abs(height - layout.height) < 1e-6
        metrics["root_viewbox_area"] = viewbox_area
        metrics["viewbox_matches_fixed_frame"] = fixed
        if not fixed:
            violations.append(
                CriticViolation(
                    "viewbox_not_fixed_frame",
                    "SVG viewBox does not match the fixed OptCPV evaluation frame.",
                    28,
                    viewbox_area < expected_area * 0.65 or viewbox_area > expected_area * 1.45,
                )
            )
    else:
        metrics["viewbox_matches_fixed_frame"] = False

    scales = _transform_scales(svg)
    if scales:
        min_scale = min(scales)
        metrics["min_transform_scale"] = min_scale
        if min_scale < 0.35:
            violations.append(CriticViolation("group_scale_too_small", "SVG uses a tiny group scale transform.", 35, True))
    else:
        metrics["min_transform_scale"] = 1.0

    stroke_widths = _stroke_widths(svg)
    if stroke_widths:
        min_stroke = min(stroke_widths)
        metrics["min_stroke_width"] = min_stroke
        if min_stroke < 0.7:
            violations.append(CriticViolation("stroke_width_too_small", "SVG stroke width is too small for visual evaluation.", 22, True))
    else:
        metrics["min_stroke_width"] = 0.0

    font_sizes = _font_sizes(svg)
    if font_sizes:
        min_font = min(font_sizes)
        metrics["min_font_size"] = min_font
        if min_font < 7.0:
            violations.append(CriticViolation("label_text_too_small", "SVG label text is scaled too small.", 18, True))
    else:
        metrics["min_font_size"] = 0.0
    return violations, metrics


def _root_viewbox(svg: str) -> tuple[float, float, float, float] | None:
    match = re.search(r"<svg\b[^>]*\bviewBox=(\"|')([^\"']+)\1", svg)
    if not match:
        return None
    values = _numbers(match.group(2))
    if len(values) != 4:
        return None
    return tuple(values)  # type: ignore[return-value]


def _transform_scales(svg: str) -> list[float]:
    values = [float(match.group(1)) for match in re.finditer(r"scale\(\s*([0-9.]+)", svg)]
    for match in re.finditer(r"matrix\(\s*([0-9.]+)[,\s]+[0-9.-]+[,\s]+[0-9.-]+[,\s]+([0-9.]+)", svg):
        values.append(min(float(match.group(1)), float(match.group(2))))
    return values


def _stroke_widths(svg: str) -> list[float]:
    values = [float(item) for item in re.findall(r"\bstroke-width=(?:\"|')([0-9.]+)", svg)]
    values.extend(float(item) for item in re.findall(r"stroke-width\s*:\s*([0-9.]+)", svg))
    return values


def _font_sizes(svg: str) -> list[float]:
    values = [float(item) for item in re.findall(r"\bfont-size=(?:\"|')([0-9.]+)", svg)]
    values.extend(float(item) for item in re.findall(r"font-size\s*:\s*([0-9.]+)", svg))
    values.extend(float(item) for item in re.findall(r"font\s*:\s*([0-9.]+)px", svg))
    return values


def _numbers(text: str) -> list[float]:
    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", text)]

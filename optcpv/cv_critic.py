"""OpenCV-based visual criticism for rendered schematics."""

from __future__ import annotations

from .models import CriticReport, CriticViolation, LayoutPlan
from .raster import RasterImage


def critique_raster(
    layout: LayoutPlan,
    raster: RasterImage,
    *,
    wire_raster: RasterImage | None = None,
    component_raster: RasterImage | None = None,
    label_raster: RasterImage | None = None,
) -> CriticReport:
    import cv2
    import numpy as np

    gray = raster.gray
    dark = _dark_mask(gray)
    ink_density = float(np.mean(dark))
    ys, xs = np.where(dark)
    violations: list[CriticViolation] = []

    if len(xs) == 0:
        violations.append(CriticViolation("blank_raster", "Rendered raster has no visible ink.", 80, True))
        return CriticReport(score=80, violations=violations, metrics={"ink_density": 0.0}, hard_fail=True)

    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    fill_ratio = (bbox_width * bbox_height) / float(raster.width * raster.height)
    center_x = float(xs.mean() / raster.width)
    center_y = float(ys.mean() / raster.height)

    if ink_density > 0.20:
        violations.append(CriticViolation("dense_blob", "Raster has an unreadably dense ink mass.", 55, True))
    if ink_density < 0.0012:
        violations.append(CriticViolation("tiny_or_faint", "Drawing is too tiny or visually faint in the fixed frame.", 30, True))
    if fill_ratio < 0.10:
        violations.append(CriticViolation("too_much_empty_canvas", "Drawing occupies too little of the fixed frame.", 22))
    if fill_ratio > 0.86:
        violations.append(CriticViolation("too_full_canvas", "Drawing nearly fills the frame and loses margins.", 15))
    if abs(center_x - 0.5) > 0.22 or abs(center_y - 0.5) > 0.24:
        violations.append(CriticViolation("unbalanced_ink_mass", "Ink mass is off center.", 8))

    clutter = _grid_clutter(dark, rows=8, cols=12)
    if clutter > 0.42:
        violations.append(CriticViolation("local_clutter", "Local raster density indicates visual clutter.", 24))

    has_layers = wire_raster is not None and component_raster is not None and label_raster is not None
    label_collisions = 0
    label_collision_pixels = 0
    component_hits = 0
    component_hit_pixels = 0
    if has_layers:
        wire_dark = _dark_mask(wire_raster.gray)
        component_dark = _dark_mask(component_raster.gray)
        label_dark = _dark_mask(label_raster.gray)
        label_collisions, label_collision_pixels = _label_layer_collisions(
            layout,
            wire_dark,
            component_dark,
            label_dark,
            raster,
            cv2,
            np,
        )
        component_hits, component_hit_pixels = _wire_component_mask_hits(layout, wire_dark, raster)
        if label_collisions > 0:
            violations.append(
                CriticViolation(
                    "label_visual_collision",
                    "Label text overlaps wire or component masks.",
                    min(34, 10 + label_collisions * 4),
                )
            )
        if component_hits > 0:
            violations.append(
                CriticViolation(
                    "wire_component_visual_collision",
                    "Wire pixels pass through component interiors.",
                    30,
                    True,
                )
            )
    else:
        label_collisions = _label_line_overlap(layout, dark, raster)
        if label_collisions > 0:
            violations.append(CriticViolation("label_dark_collision", "Label area overlaps dark wires or symbols.", min(30, 8 + label_collisions * 2)))
        component_hits = _wire_component_mask_hits_legacy(layout, dark, raster)
        if component_hits > 0:
            violations.append(CriticViolation("wire_component_visual_collision", "Wire-colored pixels pass through component interiors.", 30, True))

    # A light OpenCV morphology pass makes the test environment exercise cv2.
    dilated = cv2.dilate(dark.astype("uint8"), np.ones((3, 3), dtype="uint8"), iterations=1)
    dilated_density = float(np.mean(dilated > 0))

    metrics = {
        "ink_density": ink_density,
        "ink_bbox_fill_ratio": fill_ratio,
        "ink_center_x": center_x,
        "ink_center_y": center_y,
        "max_cell_density": clutter,
        "layered_masks": has_layers,
        "label_layer_collision_count": label_collisions,
        "label_layer_collision_pixels": label_collision_pixels,
        "label_dark_collision_count": label_collisions,
        "wire_component_mask_hits": component_hits,
        "wire_component_mask_pixels": component_hit_pixels,
        "dilated_density": dilated_density,
    }
    score = sum(violation.severity for violation in violations)
    return CriticReport(score=score, violations=violations, metrics=metrics, hard_fail=any(v.hard for v in violations))


def _grid_clutter(mask, *, rows: int, cols: int) -> float:
    height, width = mask.shape
    max_density = 0.0
    for row in range(rows):
        for col in range(cols):
            y0, y1 = int(row * height / rows), int((row + 1) * height / rows)
            x0, x1 = int(col * width / cols), int((col + 1) * width / cols)
            max_density = max(max_density, float(mask[y0:y1, x0:x1].mean()))
    return max_density


def _dark_mask(gray):
    return gray < 210


def _label_layer_collisions(layout: LayoutPlan, wires, components, labels, raster: RasterImage, cv2, np) -> tuple[int, int]:
    obstacle = cv2.dilate((wires | components).astype("uint8"), np.ones((3, 3), dtype="uint8"), iterations=1) > 0
    count = 0
    pixels = 0
    for label in layout.labels:
        x0, y0, x1, y1 = _bbox_px(label.bbox, layout, raster, pad_px=3)
        if x1 <= x0 or y1 <= y0:
            continue
        label_region = labels[y0:y1, x0:x1]
        obstacle_region = obstacle[y0:y1, x0:x1]
        if label_region.any():
            collision_pixels = int(np.logical_and(label_region, obstacle_region).sum())
        else:
            collision_pixels = int(obstacle_region.sum()) if float(obstacle_region.mean()) > 0.012 else 0
        if collision_pixels > 3:
            count += 1
            pixels += collision_pixels
    return count, pixels


def _label_line_overlap(layout: LayoutPlan, dark, raster: RasterImage) -> int:
    count = 0
    for label in layout.labels:
        x0, y0, x1, y1 = _bbox_px(label.bbox, layout, raster, pad_px=2)
        if x1 <= x0 or y1 <= y0:
            continue
        density = float(dark[y0:y1, x0:x1].mean())
        if density > 0.18:
            count += 1
    return count


def _wire_component_mask_hits(layout: LayoutPlan, wire_mask, raster: RasterImage) -> tuple[int, int]:
    hits = 0
    pixels = 0
    for component in layout.components:
        key = component.type.lower().replace("-", "_")
        if key in {"input", "output", "ground", "gnd"}:
            continue
        if not _requires_interior_wire_mask(key):
            continue
        x0, y0, x1, y1 = _bbox_px(component.bbox, layout, raster, pad_px=-7)
        if x1 <= x0 or y1 <= y0:
            continue
        region = wire_mask[y0:y1, x0:x1]
        hit_pixels = int(region.sum())
        threshold = max(5, int(region.size * 0.008))
        if hit_pixels > threshold:
            hits += 1
            pixels += hit_pixels
    return hits, pixels


def _requires_interior_wire_mask(key: str) -> bool:
    return "op_amp" in key or "opamp" in key or "operational_amplifier" in key or key in {"ic", "block", "subcircuit"}


def _wire_component_mask_hits_legacy(layout: LayoutPlan, dark, raster: RasterImage) -> int:
    hits = 0
    for component in layout.components:
        key = component.type.lower().replace("-", "_")
        if key in {"input", "output", "ground", "gnd"}:
            continue
        x0, y0, x1, y1 = _bbox_px(component.bbox, layout, raster, pad_px=-3)
        if x1 <= x0 or y1 <= y0:
            continue
        interior_density = float(dark[y0:y1, x0:x1].mean())
        if interior_density > 0.34 and ("op" in key or interior_density > 0.55):
            hits += 1
    return hits


def _bbox_px(bbox, layout: LayoutPlan, raster: RasterImage, *, pad_px: int) -> tuple[int, int, int, int]:
    sx = raster.width / layout.width
    sy = raster.height / layout.height
    x0 = int(max(0, bbox.x * layout.grid * sx - pad_px))
    y0 = int(max(0, bbox.y * layout.grid * sy - pad_px))
    x1 = int(min(raster.width, bbox.right * layout.grid * sx + pad_px))
    y1 = int(min(raster.height, bbox.bottom * layout.grid * sy + pad_px))
    return x0, y0, x1, y1

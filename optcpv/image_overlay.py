"""Image-backed interactive circuit overlays.

This module is the local low-quota stand-in for a two-step Gemini workflow:

1. ``ImageNodeWireInterpreter`` reads an existing problem/figure image and
   emits an ``IMAGE_GRAPH_DRAFT``: dark-pixel metrics, wire-like runs, raw
   node candidates, and residual component regions.
2. ``ImageOverlayPlanner`` turns that draft into an ``IMAGE_OVERLAY_PLAN``:
   snapped nodes, highlightable wire edges, and component button regions.

The output keeps the original image as the visual source of truth. OptCPV only
adds SVG hit targets, metadata, and highlight affordances on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
import base64
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .figure_semantics import FigureSemanticClient, FigureVisualEvidence, analyze_figure_semantics
from .models import BBox, Point
from .textbook_surrogate import FigureStructureCard


@dataclass(frozen=True)
class ImageOverlayNode:
    id: str
    x: float
    y: float
    kind: str = "junction"
    degree: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "x": round(self.x, 3), "y": round(self.y, 3), "kind": self.kind, "degree": self.degree}


@dataclass(frozen=True)
class ImageOverlayWire:
    id: str
    points: tuple[Point, ...]
    node_ids: tuple[str, ...]
    orientation: str
    confidence: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "points": [{"x": round(point.x, 3), "y": round(point.y, 3)} for point in self.points],
            "node_ids": list(self.node_ids),
            "orientation": self.orientation,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class ImageComponentButton:
    id: str
    bbox: BBox
    kind: str = "component_region"
    label: str | None = None
    connected_wire_ids: tuple[str, ...] = ()
    confidence: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": _bbox_dict(self.bbox),
            "kind": self.kind,
            "label": self.label,
            "connected_wire_ids": list(self.connected_wire_ids),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class ImageComponentCandidate:
    id: str
    bbox: BBox
    kind: str
    confidence: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": _bbox_dict(self.bbox),
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class RawLineSegment:
    orientation: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.7

    @property
    def length(self) -> float:
        return abs(self.x2 - self.x1) + abs(self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "orientation": self.orientation,
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "x2": round(self.x2, 3),
            "y2": round(self.y2, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class ImageGraphDraft:
    image_path: str
    width: int
    height: int
    image_kind: str
    circuit_confidence: float
    reject_reasons: tuple[str, ...]
    line_segments: tuple[RawLineSegment, ...]
    node_candidates: tuple[Point, ...]
    component_regions: tuple[BBox, ...]
    component_candidates: tuple[ImageComponentCandidate, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    structured_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "image_kind": self.image_kind,
            "circuit_confidence": round(self.circuit_confidence, 3),
            "reject_reasons": list(self.reject_reasons),
            "line_segments": [segment.to_dict() for segment in self.line_segments],
            "node_candidates": [{"x": round(point.x, 3), "y": round(point.y, 3)} for point in self.node_candidates],
            "component_regions": [_bbox_dict(region) for region in self.component_regions],
            "component_candidates": [candidate.to_dict() for candidate in self.component_candidates],
            "metrics": dict(self.metrics),
            "structured_text": self.structured_text,
        }


@dataclass(frozen=True)
class ImageOverlayPlan:
    image_path: str
    width: int
    height: int
    nodes: tuple[ImageOverlayNode, ...]
    wires: tuple[ImageOverlayWire, ...]
    component_buttons: tuple[ImageComponentButton, ...]
    image_kind: str = "unknown"
    circuit_confidence: float = 0.0
    reject_reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    middle_layer_one: str = ""
    middle_layer_two: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "nodes": [node.to_dict() for node in self.nodes],
            "wires": [wire.to_dict() for wire in self.wires],
            "component_buttons": [button.to_dict() for button in self.component_buttons],
            "image_kind": self.image_kind,
            "circuit_confidence": round(self.circuit_confidence, 3),
            "reject_reasons": list(self.reject_reasons),
            "metrics": dict(self.metrics),
            "middle_layer_one": self.middle_layer_one,
            "middle_layer_two": self.middle_layer_two,
        }


class ImageNodeWireInterpreter:
    """First local Gemini layer: extract image graph primitives."""

    def interpret(
        self,
        image_path: str | Path,
        *,
        card: FigureStructureCard | None = None,
        figure_semantic_client: FigureSemanticClient | None = None,
    ) -> ImageGraphDraft:
        import cv2
        import numpy as np

        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        binary = _dark_pixel_mask(gray)
        analysis_binary, caption_regions = _analysis_binary_without_caption(binary, path)
        plot_regions = _plot_like_regions(analysis_binary, card)
        analysis_binary = _mask_regions(analysis_binary, plot_regions)
        ignored_regions = (*caption_regions, *plot_regions)

        min_h = max(18, int(round(width * 0.04)))
        min_v = max(18, int(round(height * 0.035)))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v))
        horizontal_mask = cv2.morphologyEx(analysis_binary, cv2.MORPH_OPEN, h_kernel)
        vertical_mask = cv2.morphologyEx(analysis_binary, cv2.MORPH_OPEN, v_kernel)
        wire_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)

        segments = [
            *_segments_from_mask(horizontal_mask, "h", min_len=min_h),
            *_segments_from_mask(vertical_mask, "v", min_len=min_v),
        ]
        segments = _merge_collinear_segments(segments)
        segments = [segment for segment in segments if not _text_like_segment(analysis_binary, segment)]
        node_candidates = _node_candidates_from_segments(segments)
        regions = _component_regions(analysis_binary, wire_mask, segments, width, height)
        if _card_looks_like_block_diagram(card):
            regions.extend(_gray_block_regions(gray, ignored_regions, width, height))
        geometry_regions = _component_regions_from_wire_geometry(segments, width, height, analysis_binary)
        triangle_regions = _triangle_component_regions(analysis_binary, width, height)
        large_group_candidates = _large_schematic_group_candidates(analysis_binary, ignored_regions, width, height, card)
        regions = _merge_boxes(regions, width, height)
        component_candidates = _dedupe_component_candidates(
            [
                *_component_candidates(regions, triangle_regions, analysis_binary, allow_inductor=False),
                *_component_candidates(geometry_regions, (), analysis_binary, allow_inductor=True),
                *large_group_candidates,
            ]
        )
        regions = _merge_boxes([*regions, *geometry_regions], width, height)

        dark_density = float(np.count_nonzero(analysis_binary)) / float(max(1, width * height))
        wire_density = float(np.count_nonzero(wire_mask)) / float(max(1, width * height))
        plot_region_area_fraction = _area_fraction(plot_regions, width, height)
        component_kind_counts = _kind_counts(candidate.kind for candidate in component_candidates)
        semantic_check = analyze_figure_semantics(
            card=card,
            image_path=path,
            client=figure_semantic_client,
            evidence=FigureVisualEvidence(
                width=width,
                height=height,
                dark_density=dark_density,
                wire_density=wire_density,
                plot_region_area_fraction=plot_region_area_fraction,
                plot_region_count=len(plot_regions),
                line_segment_count=len(segments),
                component_kind_counts=component_kind_counts,
            ),
        )
        image_kind = semantic_check.final_image_kind
        circuit_confidence = semantic_check.circuit_confidence
        reject_reasons = semantic_check.reject_reasons
        if semantic_check.overlay_allowed:
            component_candidates = _semantic_component_candidates(component_candidates, image_kind, segments, width, height)
            component_kind_counts = _kind_counts(candidate.kind for candidate in component_candidates)
        else:
            circuit_confidence = min(circuit_confidence, 0.39)
            segments = []
            node_candidates = []
            regions = []
            component_candidates = []
            component_kind_counts = {}
        metrics = {
            "dark_density": round(dark_density, 5),
            "wire_density": round(wire_density, 5),
            "image_kind": image_kind,
            "circuit_confidence": round(circuit_confidence, 3),
            "reject_reasons": list(reject_reasons),
            "ignored_regions": [_bbox_dict(region) for region in ignored_regions],
            "plot_region_count": len(plot_regions),
            "plot_region_area_fraction": round(plot_region_area_fraction, 5),
            "horizontal_segment_count": sum(1 for segment in segments if segment.orientation == "h"),
            "vertical_segment_count": sum(1 for segment in segments if segment.orientation == "v"),
            "raw_node_candidate_count": len(node_candidates),
            "raw_component_region_count": len(regions),
            "raw_component_candidate_count": len(component_candidates),
            "component_kind_counts": component_kind_counts,
            "semantic_layer_one": semantic_check.draft.structured_text,
            "semantic_layer_two": semantic_check.structured_text,
            "gemini_layer_one_input": semantic_check.draft.layer_one_input,
            "gemini_layer_one_output": json.dumps(semantic_check.draft.raw_layer_one_output, indent=2, sort_keys=True),
            "gemini_layer_two_input": semantic_check.layer_two_input,
            "gemini_layer_two_output": json.dumps(semantic_check.raw_layer_two_output, indent=2, sort_keys=True),
            "semantic": semantic_check.to_dict(),
            "figure_id": card.figure_id if card else None,
            "page": card.page if card else None,
            "likely_circuit": card.is_likely_circuit if card else None,
        }
        structured_text = _draft_structured_text(
            path,
            width,
            height,
            image_kind,
            circuit_confidence,
            reject_reasons,
            segments,
            node_candidates,
            component_candidates,
            metrics,
        )
        return ImageGraphDraft(
            image_path=str(path),
            width=width,
            height=height,
            image_kind=image_kind,
            circuit_confidence=circuit_confidence,
            reject_reasons=tuple(reject_reasons),
            line_segments=tuple(segments),
            node_candidates=tuple(node_candidates),
            component_regions=tuple(regions),
            component_candidates=tuple(component_candidates),
            metrics=metrics,
            structured_text=structured_text,
        )


class ImageOverlayPlanner:
    """Second local Gemini layer: snap primitives into an interactive overlay."""

    def plan(self, draft: ImageGraphDraft) -> ImageOverlayPlan:
        if not _kind_allows_overlay(draft.image_kind, draft.circuit_confidence):
            metrics = {
                **draft.metrics,
                "node_count": 0,
                "wire_count": 0,
                "component_button_count": 0,
                "button_kind_counts": {},
                "overlay_confidence": 0.0,
                "overlay_mode": "skipped_non_circuit",
            }
            structured_text = _plan_structured_text(draft, (), [], [], metrics)
            return ImageOverlayPlan(
                image_path=draft.image_path,
                width=draft.width,
                height=draft.height,
                nodes=(),
                wires=(),
                component_buttons=(),
                image_kind=draft.image_kind,
                circuit_confidence=draft.circuit_confidence,
                reject_reasons=draft.reject_reasons,
                metrics=metrics,
                middle_layer_one=draft.structured_text,
                middle_layer_two=structured_text,
            )
        nodes = _cluster_nodes(draft.node_candidates)
        wires, degree_by_node = _wire_edges(draft.line_segments, nodes)
        nodes = tuple(
            ImageOverlayNode(
                id=node.id,
                x=node.x,
                y=node.y,
                kind=_node_kind(degree_by_node.get(node.id, 0)),
                degree=degree_by_node.get(node.id, 0),
            )
            for node in nodes
        )
        buttons = _component_buttons(draft.component_candidates, wires, image_kind=draft.image_kind)
        metrics = {
            **draft.metrics,
            "node_count": len(nodes),
            "wire_count": len(wires),
            "component_button_count": len(buttons),
            "button_kind_counts": _kind_counts(button.kind for button in buttons),
            "overlay_confidence": _overlay_confidence(len(nodes), len(wires), len(buttons), draft.metrics),
            "overlay_mode": "interactive_overlay",
        }
        structured_text = _plan_structured_text(draft, nodes, wires, buttons, metrics)
        return ImageOverlayPlan(
            image_path=draft.image_path,
            width=draft.width,
            height=draft.height,
            nodes=nodes,
            wires=tuple(wires),
            component_buttons=tuple(buttons),
            image_kind=draft.image_kind,
            circuit_confidence=draft.circuit_confidence,
            reject_reasons=draft.reject_reasons,
            metrics=metrics,
            middle_layer_one=draft.structured_text,
            middle_layer_two=structured_text,
        )


def analyze_image_overlay(
    image_path: str | Path,
    *,
    card: FigureStructureCard | None = None,
    figure_semantic_client: FigureSemanticClient | None = None,
) -> ImageOverlayPlan:
    """Return an interactive overlay plan for an existing circuit/problem image."""

    draft = ImageNodeWireInterpreter().interpret(image_path, card=card, figure_semantic_client=figure_semantic_client)
    return ImageOverlayPlanner().plan(draft)


def render_image_overlay_svg(
    plan: ImageOverlayPlan,
    *,
    embed_image: bool = True,
    image_href: str | None = None,
    show_regions: bool = True,
    show_nodes: bool = False,
) -> str:
    """Render a standalone SVG with the source image and interactive overlays."""

    href = image_href or (_embedded_image_href(plan.image_path) if embed_image else Path(plan.image_path).as_posix())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{plan.width}" height="{plan.height}" '
        f'viewBox="0 0 {plan.width} {plan.height}" data-optcpv-image-overlay="true">',
        "<style>",
        ".optcpv-source-image{pointer-events:none;user-select:none}",
        ".optcpv-wire-hit{fill:none;stroke:#2563eb;stroke-width:12;stroke-linecap:round;stroke-linejoin:round;opacity:0;cursor:pointer}",
        ".optcpv-wire-hit:hover,.optcpv-wire-hit:focus{opacity:.18}",
        ".optcpv-wire-hit.is-highlighted{opacity:.95;stroke:#ef4444;stroke-width:7}",
        ".optcpv-component-button rect{fill:#2563eb;opacity:0;stroke:#2563eb;stroke-width:2;cursor:pointer}",
        ".optcpv-component-button:hover rect,.optcpv-component-button:focus rect,.optcpv-component-button.is-selected rect{opacity:.14}",
        ".optcpv-node{fill:#f59e0b;stroke:#7c2d12;stroke-width:1.5;opacity:.75}",
        "</style>",
        f'<image class="optcpv-source-image" href="{escape(href)}" x="0" y="0" width="{plan.width}" height="{plan.height}"/>',
        '<g id="optcpv-overlay-wires">',
    ]
    for wire in plan.wires:
        path = _wire_path(wire.points)
        parts.append(
            f'<path class="optcpv-wire-hit" tabindex="0" data-wire-id="{escape(wire.id)}" '
            f'data-node-ids="{escape(",".join(wire.node_ids))}" d="{path}"/>'
        )
    parts.append("</g>")
    parts.append('<g id="optcpv-component-buttons">')
    for button in plan.component_buttons:
        wire_ids = ",".join(button.connected_wire_ids)
        label = button.label or button.id
        bbox = _expanded_bbox(button.bbox, 4, plan.width, plan.height)
        parts.append(
            f'<g class="optcpv-component-button" role="button" tabindex="0" '
            f'aria-label="{escape(label)}" data-component-id="{escape(button.id)}" '
            f'data-wire-ids="{escape(wire_ids)}">'
            f'<title>{escape(label)}</title>'
            f'<rect x="{bbox.x:.1f}" y="{bbox.y:.1f}" width="{bbox.width:.1f}" height="{bbox.height:.1f}" rx="3" ry="3"/>'
            "</g>"
        )
    parts.append("</g>")
    if show_nodes:
        parts.append('<g id="optcpv-overlay-nodes">')
        for node in plan.nodes:
            parts.append(
                f'<circle class="optcpv-node" cx="{node.x:.1f}" cy="{node.y:.1f}" r="3.2" '
                f'data-node-id="{escape(node.id)}" data-degree="{node.degree}"/>'
            )
        parts.append("</g>")
    if show_regions:
        parts.append('<metadata id="optcpv-image-overlay-json">')
        parts.append(escape(json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"))))
        parts.append("</metadata>")
    parts.append(_overlay_script())
    parts.append("</svg>")
    return "\n".join(parts)


def _dark_pixel_mask(gray):
    import cv2
    import numpy as np

    threshold_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = int(min(235, max(150, threshold_value + 36)))
    binary = np.where(gray < cutoff, 255, 0).astype("uint8")
    binary = cv2.medianBlur(binary, 3)
    return binary


def _analysis_binary_without_caption(binary, image_path: Path):
    import numpy as np

    ignored: list[BBox] = []
    metadata_path = image_path.with_name("metadata.json")
    if not metadata_path.exists():
        return binary, tuple(ignored)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        caption = metadata.get("caption_bbox") or {}
        crop = metadata.get("crop_bbox") or {}
        crop_x0 = float(crop["x0"])
        crop_y0 = float(crop["top"])
        crop_w = max(1e-6, float(crop["x1"]) - crop_x0)
        crop_h = max(1e-6, float(crop["bottom"]) - crop_y0)
        height, width = binary.shape[:2]
        x = (float(caption["x0"]) - crop_x0) * width / crop_w
        y = (float(caption["top"]) - crop_y0) * height / crop_h
        right = (float(caption["x1"]) - crop_x0) * width / crop_w
        bottom = (float(caption["bottom"]) - crop_y0) * height / crop_h
    except Exception:
        return binary, tuple(ignored)
    margin = max(4.0, min(binary.shape[:2]) * 0.01)
    box = _expanded_bbox(BBox(x, y, right - x, bottom - y), margin, binary.shape[1], binary.shape[0])
    if box.width <= 0 or box.height <= 0:
        return binary, tuple(ignored)
    result = np.array(binary, copy=True)
    result[int(box.y) : int(box.bottom) + 1, int(box.x) : int(box.right) + 1] = 0
    ignored.append(box)
    return result, tuple(ignored)


def _plot_like_regions(binary, card: FigureStructureCard | None) -> tuple[BBox, ...]:
    import cv2

    caption = (card.caption if card else "").lower()
    categories = set(card.categories if card else ())
    cues = set(card.component_cues if card else ())
    primary = card.primary_category if card else ""
    if card is not None and _card_looks_like_block_diagram(card):
        return ()
    if _card_is_schematic_with_component_cues(card, caption, categories, cues, primary):
        return ()
    if not (
        _caption_looks_plot_like(caption)
        or _caption_looks_waveform_like(caption)
        or primary == "waveform_or_plot"
        or "waveform_or_plot" in categories
    ):
        return ()
    height, width = binary.shape[:2]
    h_min = max(28, int(round(width * 0.08)))
    v_min = max(28, int(round(height * 0.08)))
    horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_min, 1)))
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_min)))
    horizontal = [
        segment
        for segment in _merge_collinear_segments(_segments_from_mask(horizontal_mask, "h", min_len=h_min))
        if segment.length >= max(36.0, width * 0.12)
    ]
    vertical = [
        segment
        for segment in _merge_collinear_segments(_segments_from_mask(vertical_mask, "v", min_len=v_min))
        if segment.length >= max(36.0, height * 0.12)
    ]
    boxes: list[BBox] = []
    for h_segment in horizontal:
        for v_segment in vertical:
            crosses = h_segment.x1 - 12 <= v_segment.x1 <= h_segment.x2 + 12 and v_segment.y1 - 12 <= h_segment.y1 <= v_segment.y2 + 12
            near_corner = abs(v_segment.x1 - h_segment.x1) <= 18 and v_segment.y1 - 12 <= h_segment.y1 <= v_segment.y2 + 12
            if not (crosses or near_corner):
                continue
            x = min(h_segment.x1, v_segment.x1)
            y = min(h_segment.y1, v_segment.y1)
            right = max(h_segment.x2, v_segment.x1)
            bottom = max(h_segment.y1, v_segment.y2)
            box = _expanded_bbox(BBox(x, y, right - x, bottom - y), max(10.0, min(width, height) * 0.025), width, height)
            if _box_is_plausible_plot_region(box, width, height):
                boxes.append(box)
    if not boxes and primary == "waveform_or_plot":
        dense_box = _dominant_dark_region(binary, width, height)
        if dense_box is not None and _box_is_plausible_plot_region(dense_box, width, height):
            boxes.append(dense_box)
    return tuple(_merge_boxes(boxes, width, height)[:16])


def _card_is_schematic_with_component_cues(
    card: FigureStructureCard | None,
    caption: str,
    categories: set[str],
    cues: set[str],
    primary: str,
) -> bool:
    if card is None:
        return False
    schematic = primary == "schematic_circuit" or "schematic_circuit" in categories or _caption_claims_circuit(caption)
    component_cues = cues & {"op_amp", "feedback", "resistor", "capacitor", "inductor", "diode", "transistor", "bridge"}
    return bool(schematic and component_cues)


def _box_is_plausible_plot_region(box: BBox, width: int, height: int) -> bool:
    area_fraction = box.area() / float(max(1, width * height))
    aspect = box.width / max(1.0, box.height)
    return 0.015 <= area_fraction <= 0.78 and 0.22 <= aspect <= 8.5 and box.width >= width * 0.12 and box.height >= height * 0.08


def _dominant_dark_region(binary, width: int, height: int) -> BBox | None:
    import cv2

    dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: BBox | None = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = _expanded_bbox(BBox(float(x), float(y), float(w), float(h)), 8.0, width, height)
        if best is None or box.area() > best.area():
            best = box
    return best


def _mask_regions(binary, regions: Iterable[BBox]):
    import numpy as np

    result = np.array(binary, copy=True)
    height, width = result.shape[:2]
    for region in regions:
        x1 = max(0, int(region.x))
        y1 = max(0, int(region.y))
        x2 = min(width, int(region.right) + 1)
        y2 = min(height, int(region.bottom) + 1)
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 0
    return result


def _area_fraction(regions: Iterable[BBox], width: int, height: int) -> float:
    return sum(region.area() for region in regions) / float(max(1, width * height))


def _segments_from_mask(mask, orientation: str, *, min_len: int) -> list[RawLineSegment]:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: list[RawLineSegment] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if orientation == "h":
            if width < min_len or width < max(3, height) * 3:
                continue
            y_mid = y + height / 2.0
            segments.append(RawLineSegment("h", float(x), y_mid, float(x + width), y_mid, _segment_confidence(width, height)))
        else:
            if height < min_len or height < max(3, width) * 3:
                continue
            x_mid = x + width / 2.0
            segments.append(RawLineSegment("v", x_mid, float(y), x_mid, float(y + height), _segment_confidence(height, width)))
    return segments


def _segment_confidence(long_side: int, short_side: int) -> float:
    ratio = long_side / max(1, short_side)
    return min(0.96, 0.55 + ratio / 30.0)


def _merge_collinear_segments(segments: Iterable[RawLineSegment]) -> list[RawLineSegment]:
    horizontal = _merge_same_orientation([segment for segment in segments if segment.orientation == "h"], axis_tolerance=4.0, gap=9.0)
    vertical = _merge_same_orientation([segment for segment in segments if segment.orientation == "v"], axis_tolerance=4.0, gap=9.0)
    return [*horizontal, *vertical]


def _merge_same_orientation(segments: list[RawLineSegment], *, axis_tolerance: float, gap: float) -> list[RawLineSegment]:
    if not segments:
        return []
    if segments[0].orientation == "h":
        key = lambda item: (round(item.y1 / axis_tolerance), item.x1)
    else:
        key = lambda item: (round(item.x1 / axis_tolerance), item.y1)
    ordered = sorted(segments, key=key)
    merged: list[RawLineSegment] = []
    for segment in ordered:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        if segment.orientation != previous.orientation:
            merged.append(segment)
            continue
        if segment.orientation == "h":
            same_axis = abs(segment.y1 - previous.y1) <= axis_tolerance
            touches = segment.x1 <= previous.x2 + gap
            if same_axis and touches:
                y = (segment.y1 + previous.y1) / 2.0
                merged[-1] = RawLineSegment("h", min(previous.x1, segment.x1), y, max(previous.x2, segment.x2), y, max(previous.confidence, segment.confidence))
                continue
        else:
            same_axis = abs(segment.x1 - previous.x1) <= axis_tolerance
            touches = segment.y1 <= previous.y2 + gap
            if same_axis and touches:
                x = (segment.x1 + previous.x1) / 2.0
                merged[-1] = RawLineSegment("v", x, min(previous.y1, segment.y1), x, max(previous.y2, segment.y2), max(previous.confidence, segment.confidence))
                continue
        merged.append(segment)
    return merged


def _text_like_segment(binary, segment: RawLineSegment) -> bool:
    import cv2

    if segment.length < 36:
        return False
    if segment.orientation == "h":
        x1 = max(0, int(segment.x1))
        x2 = min(binary.shape[1], int(segment.x2))
        y1 = max(0, int(segment.y1 - 10))
        y2 = min(binary.shape[0], int(segment.y1 + 11))
    else:
        x1 = max(0, int(segment.x1 - 10))
        x2 = min(binary.shape[1], int(segment.x1 + 11))
        y1 = max(0, int(segment.y1))
        y2 = min(binary.shape[0], int(segment.y2))
    if x2 <= x1 or y2 <= y1:
        return False
    roi = binary[y1:y2, x1:x2]
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
    small_components = 0
    dark_pixels = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        dark_pixels += area
        if 2 <= area <= 90:
            small_components += 1
    length = max(1.0, segment.length)
    small_limit = max(5, int(length / 42))
    density = dark_pixels / float(max(1, roi.shape[0] * roi.shape[1]))
    return small_components > small_limit and density > 0.06


def _node_candidates_from_segments(segments: list[RawLineSegment]) -> list[Point]:
    points: list[Point] = []
    for segment in segments:
        points.append(Point(segment.x1, segment.y1))
        points.append(Point(segment.x2, segment.y2))
    horizontal = [segment for segment in segments if segment.orientation == "h"]
    vertical = [segment for segment in segments if segment.orientation == "v"]
    for h_segment in horizontal:
        for v_segment in vertical:
            if h_segment.x1 - 5 <= v_segment.x1 <= h_segment.x2 + 5 and v_segment.y1 - 5 <= h_segment.y1 <= v_segment.y2 + 5:
                points.append(Point(v_segment.x1, h_segment.y1))
    return points


def _component_regions(binary, wire_mask, segments: list[RawLineSegment], width: int, height: int) -> list[BBox]:
    import cv2

    wire_dilated = cv2.dilate(wire_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    residual = cv2.bitwise_and(binary, cv2.bitwise_not(wire_dilated))
    residual = cv2.dilate(residual, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    contours, _ = cv2.findContours(residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(24.0, width * height * 0.000035)
    max_area = width * height * 0.08
    boxes: list[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < min_area or area > max_area:
            continue
        if w < 8 or h < 8 or w > width * 0.72 or h > height * 0.72:
            continue
        aspect = w / max(1.0, h)
        if aspect > 14 or aspect < 1 / 14:
            continue
        box = _expanded_bbox(BBox(float(x), float(y), float(w), float(h)), 8.0, width, height)
        if segments and _distance_to_segments(box, segments) > max(28.0, min(width, height) * 0.035):
            continue
        boxes.append(box)
    boxes = _merge_boxes(boxes, width, height)
    boxes.sort(key=lambda item: (item.y, item.x, -item.area()))
    return boxes[:96]


def _gray_block_regions(gray, ignored_regions: tuple[BBox, ...], width: int, height: int) -> list[BBox]:
    import cv2
    import numpy as np

    mask = np.where((gray < 238) & (gray > 72), 255, 0).astype("uint8")
    for region in ignored_regions:
        mask[int(region.y) : int(region.bottom) + 1, int(region.x) : int(region.right) + 1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    min_area = max(180.0, width * height * 0.0006)
    max_area = width * height * 0.16
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < min_area or area > max_area:
            continue
        if w < 16 or h < 12 or w > width * 0.72 or h > height * 0.72:
            continue
        aspect = w / max(1.0, h)
        if aspect > 10 or aspect < 0.12:
            continue
        roi = gray[y : y + h, x : x + w]
        fill = float(np.count_nonzero((roi < 238) & (roi > 72))) / max(1.0, area)
        dark = float(np.count_nonzero(roi < 95)) / max(1.0, area)
        if fill < 0.42 or dark > 0.42:
            continue
        boxes.append(_expanded_bbox(BBox(float(x), float(y), float(w), float(h)), 5.0, width, height))
    return _merge_boxes(boxes, width, height)[:80]


def _large_schematic_group_candidates(
    binary,
    ignored_regions: tuple[BBox, ...],
    width: int,
    height: int,
    card: FigureStructureCard | None,
) -> list[ImageComponentCandidate]:
    import cv2
    import numpy as np

    if not _card_is_schematic_with_component_cues(
        card,
        (card.caption if card else "").lower(),
        set(card.categories if card else ()),
        set(card.component_cues if card else ()),
        card.primary_category if card else "",
    ):
        return []

    mask = np.array(binary, copy=True)
    for region in ignored_regions:
        mask[int(region.y) : int(region.bottom) + 1, int(region.x) : int(region.right) + 1] = 0
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = _expanded_bbox(BBox(float(x), float(y), float(w), float(h)), 6.0, width, height)
        if _large_schematic_group_plausible(box, width, height):
            boxes.append(box)
    return [
        ImageComponentCandidate(f"raw_g{index:03d}", box, "functional_block_or_plot_region", 0.58)
        for index, box in enumerate(_merge_boxes(boxes, width, height)[:24], start=1)
    ]


def _large_schematic_group_plausible(box: BBox, width: int, height: int) -> bool:
    area_fraction = box.area() / float(max(1, width * height))
    aspect = box.width / max(1.0, box.height)
    if not (0.025 <= area_fraction <= 0.42):
        return False
    if not (0.45 <= aspect <= 7.5):
        return False
    if box.y > height * 0.72:
        return False
    if box.width > width * 0.82 or box.height > height * 0.68:
        return False
    return True


def _card_looks_like_block_diagram(card: FigureStructureCard | None) -> bool:
    if card is None:
        return False
    caption = (card.caption or "").lower()
    categories = set(card.categories)
    primary = card.primary_category
    anatomy = "anatomy_or_physiology" in categories
    plot_like = _caption_looks_plot_like(caption) or primary == "waveform_or_plot" or "waveform_or_plot" in categories
    waveform_like = _caption_looks_waveform_like(caption)
    photo_like_text = any(token in caption for token in ("photo", "photograph", "patient monitor", "sensor in open position"))
    physical_context = anatomy or photo_like_text or _caption_looks_physical_context(caption)
    category_block = (
        (primary == "system_block_diagram" or "system_block_diagram" in categories)
        and not plot_like
        and not waveform_like
        and not physical_context
    )
    return _caption_claims_block(caption) or category_block


def _component_regions_from_wire_geometry(segments: list[RawLineSegment], width: int, height: int, binary) -> list[BBox]:
    boxes: list[BBox] = []
    boxes.extend(_series_gap_component_regions([segment for segment in segments if segment.orientation == "h"], "h", width, height, binary))
    boxes.extend(_series_gap_component_regions([segment for segment in segments if segment.orientation == "v"], "v", width, height, binary))
    boxes.extend(_parallel_plate_regions([segment for segment in segments if segment.orientation == "v"], "v", width, height, binary))
    boxes.extend(_parallel_plate_regions([segment for segment in segments if segment.orientation == "h"], "h", width, height, binary))
    return _merge_boxes(boxes, width, height)


def _series_gap_component_regions(segments: list[RawLineSegment], orientation: str, width: int, height: int, binary) -> list[BBox]:
    boxes: list[BBox] = []
    if not segments:
        return boxes
    axis_tolerance = 6.0
    max_gap = max(72.0, min(width, height) * 0.22)
    min_gap = 9.0
    if orientation == "h":
        ordered = sorted(segments, key=lambda item: (round(item.y1 / axis_tolerance), item.x1))
    else:
        ordered = sorted(segments, key=lambda item: (round(item.x1 / axis_tolerance), item.y1))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.orientation != current.orientation:
            continue
        if orientation == "h":
            if abs(previous.y1 - current.y1) > axis_tolerance:
                continue
            gap = current.x1 - previous.x2
            if not (min_gap <= gap <= max_gap):
                continue
            pad_y = max(24.0, min(height * 0.11, gap * 0.7))
            box = _expanded_bbox(BBox(previous.x2, previous.y1 - pad_y, gap, pad_y * 2), 5.0, width, height)
        else:
            if abs(previous.x1 - current.x1) > axis_tolerance:
                continue
            gap = current.y1 - previous.y2
            if not (min_gap <= gap <= max_gap):
                continue
            pad_x = max(24.0, min(width * 0.11, gap * 0.7))
            box = _expanded_bbox(BBox(previous.x1 - pad_x, previous.y2, pad_x * 2, gap), 5.0, width, height)
        if _region_has_symbol_shape(box, binary):
            boxes.append(box)
    return boxes


def _parallel_plate_regions(segments: list[RawLineSegment], orientation: str, width: int, height: int, binary) -> list[BBox]:
    boxes: list[BBox] = []
    ordered = sorted(segments, key=lambda item: (item.x1, item.y1) if orientation == "v" else (item.y1, item.x1))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 : index + 8]:
            if first.orientation != second.orientation:
                continue
            if orientation == "v":
                gap = abs(second.x1 - first.x1)
                overlap = min(first.y2, second.y2) - max(first.y1, second.y1)
                length = min(first.length, second.length)
                if not (6.0 <= gap <= 42.0 and overlap >= max(12.0, length * 0.45)):
                    continue
                x = min(first.x1, second.x1)
                y = min(first.y1, second.y1)
                right = max(first.x2, second.x2)
                bottom = max(first.y2, second.y2)
            else:
                gap = abs(second.y1 - first.y1)
                overlap = min(first.x2, second.x2) - max(first.x1, second.x1)
                length = min(first.length, second.length)
                if not (6.0 <= gap <= 42.0 and overlap >= max(12.0, length * 0.45)):
                    continue
                x = min(first.x1, second.x1)
                y = min(first.y1, second.y1)
                right = max(first.x2, second.x2)
                bottom = max(first.y2, second.y2)
            box = _expanded_bbox(BBox(x, y, right - x, bottom - y), 8.0, width, height)
            if _region_has_symbol_shape(box, binary, required_kind="capacitor"):
                boxes.append(box)
    return boxes


def _region_has_symbol_shape(box: BBox, binary, *, required_kind: str | None = None) -> bool:
    kind, _ = _classify_symbol_shape(box, binary)
    if required_kind is not None:
        return kind == required_kind
    return kind in {"resistor", "capacitor", "inductor"}


def _triangle_component_regions(binary, width: int, height: int) -> list[BBox]:
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[BBox] = []
    min_area = max(60.0, width * height * 0.00008)
    max_area = width * height * 0.18
    max_bbox_area = width * height * 0.24
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.045 * perimeter, True)
        if len(approx) != 3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18:
            continue
        bbox_area = float(w * h)
        if bbox_area > max_bbox_area:
            continue
        fill_ratio = area / max(1.0, bbox_area)
        if fill_ratio < 0.22 or fill_ratio > 0.68:
            continue
        aspect = w / max(1.0, h)
        if 0.65 <= aspect <= 2.8:
            boxes.append(_expanded_bbox(BBox(float(x), float(y), float(w), float(h)), 8.0, width, height))
    return boxes


def _component_candidates(
    regions: list[BBox],
    triangle_regions: list[BBox],
    binary=None,
    *,
    allow_inductor: bool = True,
) -> list[ImageComponentCandidate]:
    candidates: list[ImageComponentCandidate] = []
    for triangle in triangle_regions:
        candidates.append(ImageComponentCandidate(f"raw_c{len(candidates) + 1:03d}", triangle, "op_amp", 0.82))
    for index, region in enumerate(regions, start=1):
        if _duplicates_triangle_region(region, triangle_regions):
            continue
        kind, confidence = _classify_component_region(region, binary, allow_inductor=allow_inductor)
        candidates.append(ImageComponentCandidate(f"raw_c{len(candidates) + 1:03d}", region, kind, confidence))
    return candidates


def _semantic_component_candidates(
    candidates: list[ImageComponentCandidate],
    image_kind: str,
    segments: list[RawLineSegment],
    width: int,
    height: int,
) -> list[ImageComponentCandidate]:
    if image_kind == "block_diagram":
        block_candidates = [
            ImageComponentCandidate(candidate.id, candidate.bbox, "functional_block", min(0.86, max(0.48, candidate.confidence)))
            for candidate in candidates
            if _block_candidate_plausible(candidate)
        ]
        block_candidates.extend(_block_candidates_from_segments(segments, width, height, start_index=len(block_candidates) + 1))
        return _dedupe_component_candidates(block_candidates)
    if image_kind == "schematic_circuit":
        return [
            _schematic_candidate(candidate)
            for candidate in candidates
            if candidate.kind not in {"functional_block"}
        ]
    return []


def _schematic_candidate(candidate: ImageComponentCandidate) -> ImageComponentCandidate:
    if candidate.kind == "functional_block_or_plot_region":
        confidence = min(0.78, max(0.5, candidate.confidence + 0.12))
        return ImageComponentCandidate(candidate.id, candidate.bbox, "component_group", confidence)
    return candidate


def _block_candidate_plausible(candidate: ImageComponentCandidate) -> bool:
    area = candidate.bbox.area()
    aspect = candidate.bbox.width / max(1.0, candidate.bbox.height)
    if area < 180.0 or area > 80_000.0:
        return False
    if aspect > 9.0 or aspect < 0.12:
        return False
    return True


def _block_candidates_from_segments(
    segments: list[RawLineSegment],
    width: int,
    height: int,
    *,
    start_index: int,
) -> list[ImageComponentCandidate]:
    horizontal = [segment for segment in segments if segment.orientation == "h" and segment.length >= max(28.0, width * 0.045)]
    vertical = [segment for segment in segments if segment.orientation == "v" and segment.length >= max(18.0, height * 0.08)]
    boxes: list[BBox] = []
    for top in horizontal:
        for bottom in horizontal:
            if bottom.y1 <= top.y1:
                continue
            box_height = bottom.y1 - top.y1
            if not (18.0 <= box_height <= max(180.0, height * 0.72)):
                continue
            left_x = max(top.x1, bottom.x1)
            right_x = min(top.x2, bottom.x2)
            box_width = right_x - left_x
            if box_width < max(24.0, width * 0.045):
                continue
            left_edge = _matching_vertical_edge(vertical, left_x, top.y1, bottom.y1)
            right_edge = _matching_vertical_edge(vertical, right_x, top.y1, bottom.y1)
            if left_edge is None or right_edge is None:
                continue
            box = _expanded_bbox(BBox(left_x, top.y1, box_width, box_height), 3.0, width, height)
            if _block_box_plausible(box, width, height):
                boxes.append(box)
    merged = _merge_boxes(boxes, width, height)
    return [
        ImageComponentCandidate(f"raw_b{index:03d}", box, "functional_block", 0.72)
        for index, box in enumerate(merged, start=start_index)
    ]


def _matching_vertical_edge(
    vertical: list[RawLineSegment],
    x: float,
    top_y: float,
    bottom_y: float,
) -> RawLineSegment | None:
    for segment in vertical:
        if abs(segment.x1 - x) > 10.0:
            continue
        overlap = min(segment.y2, bottom_y) - max(segment.y1, top_y)
        if overlap >= max(12.0, (bottom_y - top_y) * 0.62):
            return segment
    return None


def _block_box_plausible(box: BBox, width: int, height: int) -> bool:
    area_fraction = box.area() / float(max(1, width * height))
    aspect = box.width / max(1.0, box.height)
    return 0.001 <= area_fraction <= 0.36 and 0.35 <= aspect <= 8.0


def _duplicates_triangle_region(region: BBox, triangle_regions: list[BBox]) -> bool:
    for triangle in triangle_regions:
        if not region.intersects(triangle, padding=8.0):
            continue
        if region.area() <= triangle.area() * 3.2:
            return True
    return False


def _classify_component_region(region: BBox, binary=None, *, allow_inductor: bool = True) -> tuple[str, float]:
    if binary is not None:
        kind, confidence = _classify_symbol_shape(region, binary, allow_inductor=allow_inductor)
        if kind is not None:
            return kind, confidence
    aspect = region.width / max(1.0, region.height)
    area = region.area()
    if aspect >= 2.2 and area < 12_000:
        return "resistor", 0.58
    if aspect <= 0.45 and area < 12_000:
        return "capacitor_or_vertical_resistor", 0.56
    if 0.75 <= aspect <= 1.35 and area < 10_000:
        return "source_meter_or_node_symbol", 0.54
    if area > 16_000:
        return "functional_block_or_plot_region", 0.5
    return "component_region", 0.55


def _dedupe_component_candidates(candidates: list[ImageComponentCandidate]) -> list[ImageComponentCandidate]:
    priority = {
        "op_amp": 8,
        "resistor": 7,
        "capacitor": 7,
        "inductor": 7,
        "functional_block": 6,
        "component_group": 6,
        "source_meter_or_node_symbol": 5,
        "capacitor_or_vertical_resistor": 4,
        "component_region": 2,
        "functional_block_or_plot_region": 1,
    }
    ordered = sorted(candidates, key=lambda item: (-priority.get(item.kind, 3), -item.confidence, item.bbox.area()))
    selected: list[ImageComponentCandidate] = []
    for candidate in ordered:
        duplicate = False
        for existing in selected:
            overlap = _bbox_overlap_area(candidate.bbox, existing.bbox)
            if overlap <= 0:
                continue
            if _allow_nested_component_candidate(candidate, existing):
                continue
            if overlap / max(1.0, min(candidate.bbox.area(), existing.bbox.area())) >= 0.58:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    selected.sort(key=lambda item: (item.bbox.y, item.bbox.x, item.kind))
    return [
        ImageComponentCandidate(f"raw_c{index:03d}", item.bbox, item.kind, item.confidence)
        for index, item in enumerate(selected, start=1)
    ]


def _allow_nested_component_candidate(candidate: ImageComponentCandidate, existing: ImageComponentCandidate) -> bool:
    container_kinds = {"functional_block_or_plot_region", "component_group"}
    if candidate.kind in container_kinds or existing.kind in container_kinds:
        larger = max(candidate.bbox.area(), existing.bbox.area())
        smaller = max(1.0, min(candidate.bbox.area(), existing.bbox.area()))
        return larger / smaller >= 3.0
    return False


def _classify_symbol_shape(region: BBox, binary, *, allow_inductor: bool = True) -> tuple[str | None, float]:
    import cv2
    import numpy as np

    x1 = max(0, int(region.x))
    y1 = max(0, int(region.y))
    x2 = min(binary.shape[1], int(region.right))
    y2 = min(binary.shape[0], int(region.bottom))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    roi = binary[y1:y2, x1:x2]
    height, width = roi.shape[:2]
    area = max(1, width * height)
    dark_pixels = int(np.count_nonzero(roi))
    density = dark_pixels / float(area)
    if dark_pixels < max(12, area * 0.015):
        return None, 0.0

    line_stats = _hough_line_stats(roi)
    if allow_inductor and _looks_like_inductor(region, roi, line_stats):
        return "inductor", 0.7
    if _looks_like_zigzag_resistor(region, density, line_stats):
        return "resistor", 0.78
    if _looks_like_capacitor(region, roi, line_stats):
        return "capacitor", 0.74
    return None, 0.0


def _hough_line_stats(roi) -> dict[str, Any]:
    import cv2
    import math

    min_dim = max(1, min(roi.shape[:2]))
    lines = cv2.HoughLinesP(
        roi,
        1,
        math.pi / 180,
        threshold=max(6, int(min_dim * 0.18)),
        minLineLength=max(5, int(min_dim * 0.22)),
        maxLineGap=max(2, int(min_dim * 0.08)),
    )
    stats: dict[str, Any] = {"horizontal": 0, "vertical": 0, "diagonal": 0, "total": 0, "length": 0.0}
    if lines is None:
        return stats
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in item)
        dx = x2 - x1
        dy = y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length < 4:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        stats["total"] += 1
        stats["length"] += length
        if angle <= 12 or angle >= 168:
            stats["horizontal"] += 1
        elif 78 <= angle <= 102:
            stats["vertical"] += 1
        elif 22 <= angle <= 158:
            stats["diagonal"] += 1
    return stats


def _looks_like_zigzag_resistor(region: BBox, density: float, line_stats: dict[str, Any]) -> bool:
    aspect = region.width / max(1.0, region.height)
    if region.area() > 12_000 or not (0.35 <= aspect <= 7.5):
        return False
    diagonal = int(line_stats.get("diagonal") or 0)
    total = int(line_stats.get("total") or 0)
    if diagonal >= 3 and total >= diagonal and density < 0.42:
        return True
    return False


def _looks_like_capacitor(region: BBox, roi, line_stats: dict[str, Any]) -> bool:
    import numpy as np

    aspect = region.width / max(1.0, region.height)
    if region.area() > 16_000 or not (0.22 <= aspect <= 4.8):
        return False
    vertical = int(line_stats.get("vertical") or 0)
    horizontal = int(line_stats.get("horizontal") or 0)
    diagonal = int(line_stats.get("diagonal") or 0)
    column_peaks = _projection_peaks(np.count_nonzero(roi, axis=0), threshold_ratio=0.42)
    row_peaks = _projection_peaks(np.count_nonzero(roi, axis=1), threshold_ratio=0.42)
    vertical_plate_pair = 2 <= len(column_peaks) <= 4 and vertical >= 2 and diagonal <= 1
    horizontal_plate_pair = 2 <= len(row_peaks) <= 4 and horizontal >= 2 and diagonal <= 1
    if vertical_plate_pair or horizontal_plate_pair:
        return True
    return False


def _looks_like_inductor(region: BBox, roi, line_stats: dict[str, Any]) -> bool:
    import cv2
    import numpy as np

    aspect = region.width / max(1.0, region.height)
    if region.area() > 18_000 or not (0.65 <= aspect <= 8.0):
        return False
    diagonal = int(line_stats.get("diagonal") or 0)
    total = int(line_stats.get("total") or 0)
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    rounded = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 5:
            continue
        contour_area = cv2.contourArea(contour)
        box_area = max(1.0, float(w * h))
        fill = contour_area / box_area
        if 0.12 <= fill <= 0.62 and 0.45 <= w / max(1.0, h) <= 2.4:
            rounded += 1
    column_peaks = _projection_peaks(np.count_nonzero(roi, axis=0), threshold_ratio=0.42)
    coil_like_projection = len(column_peaks) >= 3 and aspect >= 1.05 and total <= 12
    if diagonal >= 3 and not coil_like_projection:
        return False
    return (rounded >= 2 and aspect >= 1.05) or coil_like_projection


def _projection_peaks(values, *, threshold_ratio: float) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    maximum = max(float(values.max()), 1.0)
    threshold = maximum * threshold_ratio
    peaks: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if float(value) >= threshold:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= 1:
                peaks.append((start, index - 1))
            start = None
    if start is not None:
        peaks.append((start, len(values) - 1))
    return peaks


def _classify_image_kind(
    *,
    card: FigureStructureCard | None,
    segments: list[RawLineSegment],
    component_candidates: list[ImageComponentCandidate],
    metrics: dict[str, float],
) -> tuple[str, float, tuple[str, ...]]:
    categories = set(card.categories if card else ())
    primary = card.primary_category if card else ""
    cues = set(card.component_cues if card else ())
    families = set(card.visual_families if card else ())
    caption_key = (card.caption if card else "").lower()
    rejects: list[str] = []
    real_component_count = sum(1 for candidate in component_candidates if candidate.kind in {"op_amp", "resistor", "capacitor", "inductor"})
    plot_region_area = float(metrics.get("plot_region_area_fraction") or 0.0)
    caption_is_plot_like = _caption_looks_plot_like(caption_key) or primary == "waveform_or_plot" or "waveform_or_plot" in categories
    caption_has_circuit_claim = _caption_claims_circuit(caption_key)

    circuit_score = 0.0
    if "schematic_circuit" in categories:
        circuit_score += 0.55
    if "system_block_diagram" in categories:
        circuit_score += 0.22
    if card is not None and card.is_likely_circuit:
        circuit_score += 0.2
    if cues & {"op_amp", "resistor", "capacitor", "inductor", "diode", "transistor", "electrode", "bridge"}:
        circuit_score += 0.18
    if families & {"active_filter", "instrumentation_amplifier", "single_opamp_feedback", "sensor_bridge", "protection_clamp"}:
        circuit_score += 0.12
    if any(candidate.kind == "op_amp" for candidate in component_candidates):
        circuit_score += 0.22
    if len(segments) >= 5 and len(component_candidates) >= 2:
        circuit_score += 0.34
    elif len(segments) >= 2 and len(component_candidates) >= 1:
        circuit_score += 0.08
    if card is None and len(component_candidates) >= 2 and float(metrics.get("wire_density") or 0) >= 0.012:
        circuit_score += 0.12

    if primary == "waveform_or_plot" and "schematic_circuit" not in categories:
        circuit_score -= 0.45
        rejects.append("primary_category_is_plot")
    waveform_trace_without_components = len(component_candidates) == 0 and _caption_looks_waveform_like(caption_key)
    if waveform_trace_without_components:
        circuit_score -= 0.75
        rejects.append("waveform_trace_without_components")
    plot_dominates_without_circuit = caption_is_plot_like and plot_region_area >= 0.08 and real_component_count == 0
    if plot_dominates_without_circuit:
        circuit_score -= 0.9
        rejects.append("plot_region_without_circuit_components")
    plot_caption_not_circuit = caption_is_plot_like and not caption_has_circuit_claim
    if plot_caption_not_circuit:
        circuit_score -= 0.7
        rejects.append("plot_caption_without_circuit_claim")
    if "schematic_circuit" in categories and not cues and _caption_looks_plot_like(caption_key) and not any(
        candidate.kind == "op_amp" for candidate in component_candidates
    ):
        circuit_score -= 0.7
        rejects.append("caption_and_visual_evidence_look_like_plot_not_circuit")
    if "anatomy_or_physiology" in categories and "schematic_circuit" not in categories and "bioelectric_recording" not in categories:
        circuit_score -= 0.32
        rejects.append("anatomy_context_without_schematic")
    if primary == "uncategorized_figure" and not cues:
        circuit_score -= 0.25
        rejects.append("uncategorized_without_component_cues")
    if float(metrics.get("wire_density") or 0) < 0.0006 and "schematic_circuit" not in categories:
        circuit_score -= 0.18
        rejects.append("low_wire_density")
    insufficient_visual_primitives = len(segments) < 2 and len(component_candidates) == 0
    if insufficient_visual_primitives and "schematic_circuit" in categories:
        circuit_score -= 0.95
        rejects.append("schematic_label_without_visual_circuit_primitives")
    photo_or_scan_like = float(metrics.get("dark_density") or 0) > 0.3 and len(segments) < 2 and len(component_candidates) <= 1
    if photo_or_scan_like:
        circuit_score -= 0.95
        rejects.append("photo_or_scan_without_circuit_primitives")
    photo_panel_without_wires = len(segments) == 0 and float(metrics.get("dark_density") or 0) > 0.16 and not caption_has_circuit_claim
    if photo_panel_without_wires:
        circuit_score -= 0.95
        rejects.append("photo_or_panel_without_wires")
    plot_mask_removed_all_wires = caption_is_plot_like and plot_region_area >= 0.08 and len(segments) == 0
    if plot_mask_removed_all_wires:
        circuit_score -= 0.9
        rejects.append("plot_mask_removed_all_wires")

    confidence = max(0.0, min(0.98, circuit_score))
    if waveform_trace_without_components or plot_dominates_without_circuit or plot_caption_not_circuit or plot_mask_removed_all_wires:
        kind = "plot"
    elif photo_or_scan_like or photo_panel_without_wires:
        kind = "anatomy_or_context" if "electrode_context" in families or "anatomy_or_physiology" in categories else "non_circuit"
    elif insufficient_visual_primitives and confidence < 0.42:
        kind = "non_circuit"
    elif "schematic_circuit" in categories and confidence >= 0.42:
        kind = "schematic_circuit"
    elif "system_block_diagram" in categories and confidence >= 0.34:
        kind = "block_diagram"
    elif primary == "waveform_or_plot" or _caption_looks_plot_like(caption_key):
        kind = "plot"
    elif "anatomy_or_physiology" in categories:
        kind = "anatomy_or_context"
    elif confidence >= 0.48:
        kind = "schematic_circuit"
    elif confidence >= 0.34:
        kind = "possible_circuit"
    else:
        kind = "non_circuit"
        if not rejects:
            rejects.append("insufficient_circuit_evidence")
    if kind in {"plot", "anatomy_or_context", "non_circuit"} and not rejects:
        rejects.append(f"classified_as_{kind}")
    return kind, confidence, tuple(rejects)


def _caption_looks_plot_like(caption: str) -> bool:
    return any(
        token in caption
        for token in (
            "plot",
            "simulation",
            "response",
            "frequency",
            "bode",
            "characteristic",
            "threshold",
            "curve",
            "sensitivity",
            "linearity",
            "drift",
            "versus",
            "function of",
            "spectrum",
            "waveforms",
            "transient",
            "gain versus",
            "dependence",
            "relationship",
            "as a function",
        )
    )


def _caption_looks_waveform_like(caption: str) -> bool:
    return any(token in caption for token in ("ecg", "emg", "electrocardiogram", "electromyogram", "waveform", "trace", "distorted", "recording strip"))


def _caption_claims_circuit(caption: str) -> bool:
    return any(
        token in caption
        for token in (
            "circuit design",
            "circuit diagram",
            "circuit with",
            "ecg circuit",
            "emg circuit",
            "amplifier circuit",
            "filter circuit",
            "bridge circuit",
            "schematic",
            "block diagram",
            "equivalent",
            "electrical model",
            "circuit model",
            "op amp",
            "op-amp",
            "rectifier",
            "wheatstone",
            "protection scheme",
        )
    )


def _caption_claims_block(caption: str) -> bool:
    return any(token in caption for token in ("block diagram", "flow chart", "flowchart", "signal chain", "logic diagram", "system diagram", "controller"))


def _caption_looks_physical_context(caption: str) -> bool:
    return any(
        token in caption
        for token in (
            "patient",
            "skin",
            "artery",
            "arterial",
            "tonometer",
            "probe",
            "microelectrode",
            "electrode fabricated",
            "silicon",
            "magnetic-field pickup",
            "lead wires",
            "mechanical",
            "surface of skin",
            "body",
        )
    )


def _kind_allows_overlay(image_kind: str, circuit_confidence: float) -> bool:
    if image_kind in {"schematic_circuit", "block_diagram"}:
        return circuit_confidence >= 0.42
    if image_kind == "possible_circuit":
        return circuit_confidence >= 0.42
    return False


def _cluster_nodes(points: Iterable[Point], *, tolerance: float = 8.0) -> tuple[ImageOverlayNode, ...]:
    clusters: list[list[Point]] = []
    for point in points:
        best_index = None
        best_distance = tolerance + 1
        for index, cluster in enumerate(clusters):
            center = _point_center(cluster)
            distance = abs(center.x - point.x) + abs(center.y - point.y)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None or best_distance > tolerance:
            clusters.append([point])
        else:
            clusters[best_index].append(point)
    nodes = []
    for index, cluster in enumerate(clusters, start=1):
        center = _point_center(cluster)
        nodes.append(ImageOverlayNode(f"n{index:03d}", center.x, center.y))
    return tuple(nodes)


def _wire_edges(segments: tuple[RawLineSegment, ...], nodes: tuple[ImageOverlayNode, ...]) -> tuple[list[ImageOverlayWire], dict[str, int]]:
    wires: list[ImageOverlayWire] = []
    degree: dict[str, int] = {}
    for segment in segments:
        on_segment = [
            node
            for node in nodes
            if _node_on_segment(node, segment, tolerance=8.0)
        ]
        if len(on_segment) < 2:
            continue
        if segment.orientation == "h":
            on_segment.sort(key=lambda node: node.x)
        else:
            on_segment.sort(key=lambda node: node.y)
        for left, right in zip(on_segment, on_segment[1:]):
            if abs(left.x - right.x) + abs(left.y - right.y) < 7:
                continue
            wire_id = f"w{len(wires) + 1:04d}"
            points = (Point(left.x, left.y), Point(right.x, right.y))
            wires.append(ImageOverlayWire(wire_id, points, (left.id, right.id), segment.orientation, segment.confidence))
            degree[left.id] = degree.get(left.id, 0) + 1
            degree[right.id] = degree.get(right.id, 0) + 1
    return wires, degree


def _component_buttons(
    candidates: tuple[ImageComponentCandidate, ...],
    wires: list[ImageOverlayWire],
    *,
    image_kind: str,
) -> list[ImageComponentButton]:
    buttons: list[ImageComponentButton] = []
    for index, candidate in enumerate(candidates, start=1):
        region = candidate.bbox
        if _candidate_is_text_block(candidate, image_kind):
            continue
        connected = tuple(wire.id for wire in wires if _wire_touches_box(wire, region.expanded(10.0)))
        if not connected and candidate.kind in {"functional_block_or_plot_region", "component_region"} and image_kind != "block_diagram":
            continue
        confidence = min(0.94, candidate.confidence + (0.12 if connected else -0.04))
        buttons.append(
            ImageComponentButton(
                id=f"c{index:03d}",
                bbox=region,
                kind=candidate.kind,
                label=f"{candidate.kind} {index}",
                connected_wire_ids=connected,
                confidence=confidence,
            )
        )
    return buttons


def _candidate_is_text_block(candidate: ImageComponentCandidate, image_kind: str) -> bool:
    aspect = candidate.bbox.width / max(1.0, candidate.bbox.height)
    if image_kind == "schematic_circuit" and candidate.kind == "functional_block_or_plot_region":
        return candidate.bbox.area() > 18_000 or aspect > 3.2
    if aspect > 7.0 and candidate.bbox.height > 22:
        return True
    return False


def _node_kind(degree: int) -> str:
    if degree >= 3:
        return "junction"
    if degree == 2:
        return "pass_through"
    return "terminal"


def _overlay_confidence(node_count: int, wire_count: int, button_count: int, metrics: dict[str, Any]) -> float:
    if wire_count == 0 and button_count == 0:
        return 0.0
    density_bonus = min(0.18, float(metrics.get("wire_density") or 0) * 12)
    count_bonus = min(0.45, wire_count / 80.0 + button_count / 120.0 + node_count / 160.0)
    return round(min(0.98, 0.28 + density_bonus + count_bonus), 3)


def _draft_structured_text(
    image_path: Path,
    width: int,
    height: int,
    image_kind: str,
    circuit_confidence: float,
    reject_reasons: tuple[str, ...],
    segments: list[RawLineSegment],
    nodes: list[Point],
    component_candidates: list[ImageComponentCandidate],
    metrics: dict[str, Any],
) -> str:
    lines = [
        f"IMAGE_GRAPH_DRAFT image={image_path.as_posix()} width={width} height={height}",
        f"classification: image_kind={image_kind} circuit_confidence={circuit_confidence:.3f}",
        f"reject_reasons: {', '.join(reject_reasons) or 'none'}",
        metrics.get("semantic_layer_one", ""),
        metrics.get("semantic_layer_two", ""),
        f"metrics: {json.dumps(metrics, sort_keys=True, separators=(',', ':'))}",
        f"wire_runs: horizontal={sum(1 for item in segments if item.orientation == 'h')} vertical={sum(1 for item in segments if item.orientation == 'v')}",
        f"node_candidates: count={len(nodes)}",
        f"component_candidates: count={len(component_candidates)}",
        f"component_kind_counts: {json.dumps(_kind_counts(candidate.kind for candidate in component_candidates), sort_keys=True, separators=(',', ':'))}",
        "instructions: if image_kind is not schematic_circuit/block_diagram, preserve classification and do not invent overlay hit targets",
    ]
    for index, segment in enumerate(segments[:24], start=1):
        lines.append(f"run[{index}]: {segment.orientation} ({segment.x1:.1f},{segment.y1:.1f}) -> ({segment.x2:.1f},{segment.y2:.1f})")
    for candidate in component_candidates[:24]:
        box = candidate.bbox
        lines.append(
            f"component_candidate[{candidate.id}]: kind={candidate.kind} "
            f"bbox=({box.x:.1f},{box.y:.1f},{box.width:.1f},{box.height:.1f}) confidence={candidate.confidence:.2f}"
        )
    return "\n".join(lines)


def _plan_structured_text(
    draft: ImageGraphDraft,
    nodes: tuple[ImageOverlayNode, ...],
    wires: list[ImageOverlayWire],
    buttons: list[ImageComponentButton],
    metrics: dict[str, Any],
) -> str:
    lines = [
        f"IMAGE_OVERLAY_PLAN image={draft.image_path} width={draft.width} height={draft.height}",
        f"classification: image_kind={draft.image_kind} circuit_confidence={draft.circuit_confidence:.3f}",
        f"reject_reasons: {', '.join(draft.reject_reasons) or 'none'}",
        f"nodes={len(nodes)} wires={len(wires)} component_buttons={len(buttons)} confidence={metrics['overlay_confidence']}",
        f"button_kind_counts: {json.dumps(_kind_counts(button.kind for button in buttons), sort_keys=True, separators=(',', ':'))}",
        "wire_highlight_api: call window.optcpvHighlightWires(['w0001', ...]) or click component buttons",
        "component_button_contract: transparent SVG groups with role=button, tabindex=0, data-component-id, data-wire-ids",
    ]
    for wire in wires[:24]:
        lines.append(
            f"wire[{wire.id}]: nodes={','.join(wire.node_ids)} points="
            + " -> ".join(f"({point.x:.1f},{point.y:.1f})" for point in wire.points)
        )
    return "\n".join(lines)


def _wire_path(points: tuple[Point, ...]) -> str:
    if not points:
        return ""
    first, *rest = points
    return " ".join([f"M {first.x:.1f} {first.y:.1f}", *[f"L {point.x:.1f} {point.y:.1f}" for point in rest]])


def _overlay_script() -> str:
    return """
<script><![CDATA[
(function(){
  function asSet(ids){ return new Set((ids || []).map(String)); }
  window.optcpvHighlightWires = function(ids){
    const selected = asSet(ids);
    document.querySelectorAll('.optcpv-wire-hit').forEach(function(path){
      path.classList.toggle('is-highlighted', selected.has(path.dataset.wireId));
    });
  };
  window.optcpvClearHighlight = function(){
    document.querySelectorAll('.optcpv-wire-hit').forEach(function(path){ path.classList.remove('is-highlighted'); });
    document.querySelectorAll('.optcpv-component-button').forEach(function(item){ item.classList.remove('is-selected'); });
  };
  document.querySelectorAll('.optcpv-component-button').forEach(function(button){
    function toggle(){
      const ids = (button.dataset.wireIds || '').split(',').filter(Boolean);
      const selected = !button.classList.contains('is-selected');
      window.optcpvClearHighlight();
      if(selected){
        button.classList.add('is-selected');
        window.optcpvHighlightWires(ids);
      }
    }
    button.addEventListener('click', toggle);
    button.addEventListener('keydown', function(event){
      if(event.key === 'Enter' || event.key === ' '){ event.preventDefault(); toggle(); }
    });
  });
})();
]]></script>""".strip()


def _embedded_image_href(image_path: str) -> str:
    path = Path(image_path)
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _bbox_dict(bbox: BBox) -> dict[str, float]:
    return {"x": round(bbox.x, 3), "y": round(bbox.y, 3), "width": round(bbox.width, 3), "height": round(bbox.height, 3)}


def _bbox_overlap_area(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.right, right.right)
    y2 = min(left.bottom, right.bottom)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _kind_counts(kinds: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _expanded_bbox(bbox: BBox, amount: float, width: int, height: int) -> BBox:
    x = max(0.0, bbox.x - amount)
    y = max(0.0, bbox.y - amount)
    right = min(float(width), bbox.right + amount)
    bottom = min(float(height), bbox.bottom + amount)
    return BBox(x, y, max(0.0, right - x), max(0.0, bottom - y))


def _merge_boxes(boxes: list[BBox], width: int, height: int) -> list[BBox]:
    result: list[BBox] = []
    for box in sorted(boxes, key=lambda item: item.area(), reverse=True):
        merged = False
        for index, existing in enumerate(result):
            if existing.intersects(box, padding=7.0):
                x = min(existing.x, box.x)
                y = min(existing.y, box.y)
                right = max(existing.right, box.right)
                bottom = max(existing.bottom, box.bottom)
                result[index] = _expanded_bbox(BBox(x, y, right - x, bottom - y), 0.0, width, height)
                merged = True
                break
        if not merged:
            result.append(box)
    return result


def _point_center(points: list[Point]) -> Point:
    return Point(sum(point.x for point in points) / len(points), sum(point.y for point in points) / len(points))


def _node_on_segment(node: ImageOverlayNode, segment: RawLineSegment, *, tolerance: float) -> bool:
    if segment.orientation == "h":
        return abs(node.y - segment.y1) <= tolerance and segment.x1 - tolerance <= node.x <= segment.x2 + tolerance
    return abs(node.x - segment.x1) <= tolerance and segment.y1 - tolerance <= node.y <= segment.y2 + tolerance


def _wire_touches_box(wire: ImageOverlayWire, bbox: BBox) -> bool:
    return any(bbox.contains_point(point) for point in wire.points) or _segment_intersects_bbox(wire.points[0], wire.points[-1], bbox)


def _segment_intersects_bbox(start: Point, end: Point, bbox: BBox) -> bool:
    if abs(start.y - end.y) < 1e-6:
        y = start.y
        return bbox.y <= y <= bbox.bottom and max(min(start.x, end.x), bbox.x) <= min(max(start.x, end.x), bbox.right)
    if abs(start.x - end.x) < 1e-6:
        x = start.x
        return bbox.x <= x <= bbox.right and max(min(start.y, end.y), bbox.y) <= min(max(start.y, end.y), bbox.bottom)
    return False


def _distance_to_segments(box: BBox, segments: list[RawLineSegment]) -> float:
    center = box.center
    best = float("inf")
    for segment in segments:
        if segment.orientation == "h":
            closest_x = min(max(center.x, segment.x1), segment.x2)
            closest_y = segment.y1
        else:
            closest_x = segment.x1
            closest_y = min(max(center.y, segment.y1), segment.y2)
        best = min(best, abs(center.x - closest_x) + abs(center.y - closest_y))
    return best


def _region_kind(region: BBox) -> str:
    aspect = region.width / max(1.0, region.height)
    if aspect > 3.0:
        return "wide_component_or_label"
    if aspect < 0.4:
        return "vertical_component"
    return "component_region"

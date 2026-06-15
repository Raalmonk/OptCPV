"""Optional vision client boundary for patch proposals."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .models import Circuit, CriticReport, LayoutPlan
from .patch import LayoutPatch, MoveComponent, MoveLabel, SetOrientation
from .raster import RasterImage


class VisionLayoutClient:
    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        raise NotImplementedError


@dataclass
class MockVisionClient(VisionLayoutClient):
    patch: LayoutPatch

    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        return self.patch


class GeminiVisionClient(VisionLayoutClient):
    """Optional Gemini-backed client; never part of the default optimizer."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash") -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiVisionClient.") from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model

    def propose_patch(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        raster: RasterImage,
        critic_report: CriticReport,
    ) -> LayoutPatch:
        prompt = _patch_prompt(circuit, layout, svg, raster, critic_report)
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        text = (getattr(response, "text", "") or "").strip()
        return _patch_from_json(text, layout)


def _patch_prompt(
    circuit: Circuit,
    layout: LayoutPlan,
    svg: str,
    raster: RasterImage,
    critic_report: CriticReport,
) -> str:
    payload = {
        "task": "Return a topology-safe OptCPV LayoutPatch JSON only.",
        "hard_rules": [
            "Do not change canvas width, height, grid, component ids, component types, pins, nets, or topology.",
            "Only move existing components, move existing labels, or set existing component orientation.",
            "Do not spread components far apart. Prefer small local moves.",
            "If uncertain, return empty arrays.",
        ],
        "schema": {
            "move_component": [{"component_id": "existing id", "x": "float", "y": "float"}],
            "move_label": [{"label_id": "existing label id", "x": "float", "y": "float"}],
            "set_orientation": [{"component_id": "existing id", "orientation": "right|left|up|down"}],
            "set_wire_points": [],
        },
        "circuit": {
            "id": circuit.id,
            "motif": circuit.motif,
            "components": [
                {"id": item.id, "type": item.type, "pins": item.pins, "label": item.label, "role": item.role}
                for item in circuit.components
            ],
        },
        "layout": {
            "width": layout.width,
            "height": layout.height,
            "grid": layout.grid,
            "components": [
                {
                    "id": item.id,
                    "type": item.type,
                    "x": item.x,
                    "y": item.y,
                    "orientation": item.orientation,
                    "bbox": {
                        "x": item.bbox.x,
                        "y": item.bbox.y,
                        "width": item.bbox.width,
                        "height": item.bbox.height,
                    },
                }
                for item in layout.components
            ],
            "labels": [
                {"id": item.id, "owner_id": item.owner_id, "text": item.text, "x": item.x, "y": item.y}
                for item in layout.labels
            ],
        },
        "raster": {"width": raster.width, "height": raster.height},
        "critic_report": critic_report.to_dict(),
        "svg_excerpt": svg[:6000],
    }
    return (
        "You are an OptCPV visual-layout patch proposer. Return only valid JSON matching the schema. "
        "No markdown, no prose.\n"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    )


def _patch_from_json(text: str, layout: LayoutPlan) -> LayoutPatch:
    data = _parse_json(text)
    component_ids = {component.id for component in layout.components}
    label_ids = {label.id for label in layout.labels}
    orientations = {"right", "left", "up", "down"}

    move_component = []
    for raw in _list(data.get("move_component")):
        component_id = str(raw.get("component_id", ""))
        if component_id in component_ids and _is_number(raw.get("x")) and _is_number(raw.get("y")):
            move_component.append(MoveComponent(component_id, float(raw["x"]), float(raw["y"])))

    move_label = []
    for raw in _list(data.get("move_label")):
        label_id = str(raw.get("label_id", ""))
        if label_id in label_ids and _is_number(raw.get("x")) and _is_number(raw.get("y")):
            move_label.append(MoveLabel(label_id, float(raw["x"]), float(raw["y"])))

    set_orientation = []
    for raw in _list(data.get("set_orientation")):
        component_id = str(raw.get("component_id", ""))
        orientation = str(raw.get("orientation", ""))
        if component_id in component_ids and orientation in orientations:
            set_orientation.append(SetOrientation(component_id, orientation))

    return LayoutPatch(
        move_component=move_component,
        move_label=move_label,
        set_orientation=set_orientation,
        set_wire_points=[],
    )


def _parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) or (
        isinstance(value, str) and bool(re.fullmatch(r"-?\d+(\.\d+)?", value.strip()))
    )

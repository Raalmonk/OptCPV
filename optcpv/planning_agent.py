"""Optional pre-render semantic planning client boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any

from .models import Circuit
from .planning_hints import (
    GridPlacementHint,
    LaneHint,
    MotifHint,
    PlanningHints,
    RoutePolicyHint,
    SchematicLayoutHints,
    SchematicPlanningRequest,
    StageHint,
)


DEFAULT_GEMINI_PLANNER_MODEL = "gemini-3.5-flash"


class SemanticPlanningClient:
    """Boundary for pre-layout semantic planning hints."""

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        raise NotImplementedError


@dataclass
class FakePlanningClient(SemanticPlanningClient):
    """Test double that returns a pre-baked hint object."""

    hints: SchematicLayoutHints

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        return self.hints.with_updates(source="fake")


class GeminiPlanningClient(SemanticPlanningClient):
    """Optional Gemini-backed semantic warm-start client."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiPlanningClient.") from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model or os.getenv("OPTCPV_GEMINI_PLANNER_MODEL", DEFAULT_GEMINI_PLANNER_MODEL)
        self._types = types

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        input_mode = "image_guided" if reference_image else "model_guided"
        prompt = _planning_prompt(circuit, input_mode=input_mode, has_reference_image=reference_image is not None)
        contents: list[Any] = [prompt]
        if reference_image is not None:
            contents.append(self._types.Part.from_bytes(data=reference_image, mime_type="image/png"))
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config={"response_mime_type": "application/json"},
        )
        text = (getattr(response, "text", "") or "").strip()
        return SchematicLayoutHints.from_dict(_json_object(text)).with_updates(source="gemini")


def _planning_prompt(circuit: Circuit, *, input_mode: str, has_reference_image: bool) -> str:
    request = SchematicPlanningRequest.from_circuit(
        circuit,
        input_mode=input_mode,
        reference_image={"provided": True, "role": "relative teaching schematic reference"} if has_reference_image else None,
    )
    payload = {
        "task": "Return OptCPV pre-render semantic schematic layout hints as strict JSON only.",
        "input_mode": input_mode,
        "hard_rules": [
            "Return JSON only. No markdown, comments, or prose outside the object.",
            "Output only discrete stage_x, lane_y, and orientation hints. Never output absolute pixel coordinates.",
            "Never create, delete, rename, or rewire components, pins, or nets.",
            "Never create new components for anatomy, electrode art, annotations, or labels unless they already exist in the netlist.",
            "Never create new nets.",
            "Never route GND, VCC, VEE, VDD, VSS, or REF/reference nets as global physical wires.",
            "Identify auxiliary feedback loops, especially right-leg-drive, RLD, driven-right-leg, and common-mode feedback.",
            "Put auxiliary feedback loops in a bottom auxiliary lane using bottom_auxiliary_corridor route policies.",
            "Keep the main signal flow left-to-right.",
            "Keep parallel differential inputs aligned by stage and separated by lane.",
            "Treat reference-image anatomy or electrode art as annotation, not core electrical layout, unless represented in the netlist.",
        ],
        "mode_guidance": {
            "image_guided": "Use the reference image only to infer relative teaching-schematic stages, lanes, and motifs.",
            "model_guided": "Infer a teaching-schematic layout from netlist semantics only.",
        },
        "schema": {
            "recognized_topology": "string",
            "confidence": "number from 0 to 1",
            "tutor_explanation": "short teaching explanation",
            "stages": [{"stage_x": "integer", "stage_type": "string", "members": ["component_id"]}],
            "lanes": [{"lane_y": "integer", "lane_type": "string", "members": ["component_id"]}],
            "placements": [
                {
                    "component_id": "existing component id",
                    "stage_x": "integer",
                    "lane_y": "integer",
                    "orientation": "RIGHT|LEFT|UP|DOWN",
                    "role": "optional semantic role",
                    "confidence": "number from 0 to 1",
                }
            ],
            "route_policies": [
                {
                    "net": "existing net name or null",
                    "net_role": "signal|feedback|right_leg_drive|ground|supply|reference",
                    "policy": "left_to_right_manhattan|top_feedback_corridor|bottom_feedback_corridor|bottom_auxiliary_corridor|local_terminal_only|avoid_opamp_body",
                }
            ],
            "local_terminal_policy": {"net_name": "local_symbol_only"},
            "source": "gemini",
        },
        "request": request.to_dict(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _json_object(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        raw = json.loads(match.group(0))
    if not isinstance(raw, dict):
        raise ValueError("GeminiPlanningClient must return a JSON object.")
    return raw


__all__ = [
    "DEFAULT_GEMINI_PLANNER_MODEL",
    "FakePlanningClient",
    "GeminiPlanningClient",
    "GridPlacementHint",
    "LaneHint",
    "MotifHint",
    "PlanningHints",
    "RoutePolicyHint",
    "SchematicLayoutHints",
    "SchematicPlanningRequest",
    "SemanticPlanningClient",
    "StageHint",
]

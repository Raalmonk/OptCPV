"""Optional pre-render semantic planning client boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .models import Circuit
from .planning_hints import (
    AuxiliaryLoopHint,
    BlockHint,
    GridPlacementHint,
    InterBlockRouteHint,
    LaneHint,
    MotifHint,
    OrientationOverrideHint,
    PlanningHints,
    RoutePolicyHint,
    SchematicLayoutHints,
    SchematicPlanningRequest,
    StageHint,
)


DEFAULT_GEMINI_PLANNER_MODEL = "gemini-pro-latest"
DEFAULT_GEMINI_PLANNER_FALLBACK_MODELS = ("gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro")
_DOTENV_LOADED = False


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

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_models: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiPlanningClient.") from exc
        env_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        self._client = genai.Client(api_key=env_api_key) if env_api_key else genai.Client()
        self._model = model or os.getenv("OPTCPV_GEMINI_PLANNER_MODEL", DEFAULT_GEMINI_PLANNER_MODEL)
        self._fallback_models = tuple(fallback_models) if fallback_models is not None else _fallback_models_from_env()
        self._types = types

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        input_mode = "image_guided" if reference_image else "model_guided"
        prompt = _planning_prompt(circuit, input_mode=input_mode, has_reference_image=reference_image is not None)
        contents: list[Any] = [prompt]
        if reference_image is not None:
            contents.append(self._types.Part.from_bytes(data=reference_image, mime_type="image/png"))
        response = self._generate_content_with_fallbacks(contents)
        text = (getattr(response, "text", "") or "").strip()
        return SchematicLayoutHints.from_dict(_json_object(text)).with_updates(source="gemini")

    def _generate_content_with_fallbacks(self, contents: list[Any]):
        models = tuple(dict.fromkeys((self._model, *self._fallback_models)))
        last_error: Exception | None = None
        for model in models:
            try:
                return self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config={"response_mime_type": "application/json"},
                )
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("No Gemini planner models were configured.")


def planning_client_from_env() -> SemanticPlanningClient | None:
    """Return an env-configured Gemini planner when explicitly enabled."""

    load_dotenv_if_present()
    planner = _key(os.getenv("OPTCPV_PLANNING_CLIENT"))
    enabled = _truthy(os.getenv("OPTCPV_USE_GEMINI_PLANNER")) or planner == "gemini"
    if not enabled:
        return None
    return GeminiPlanningClient()


def load_dotenv_if_present(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding process env."""

    global _DOTENV_LOADED
    if _DOTENV_LOADED and path is None:
        return
    dotenv = Path(path) if path is not None else Path.cwd() / ".env"
    if not dotenv.exists():
        _DOTENV_LOADED = True
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    if path is None:
        _DOTENV_LOADED = True


def _planning_prompt(circuit: Circuit, *, input_mode: str, has_reference_image: bool) -> str:
    request = SchematicPlanningRequest.from_circuit(
        circuit,
        input_mode=input_mode,
        reference_image={"provided": True, "role": "relative teaching schematic reference"} if has_reference_image else None,
    )
    payload = {
        "task": "Return authoritative OptCPV pre-render schematic drawing guidance as strict JSON only.",
        "input_mode": input_mode,
        "hard_rules": [
            "Return JSON only. No markdown, comments, or prose outside the object.",
            "You may choose semantic blocks, route roles, discrete stage_x/lane_y placements, route corridors, orientation overrides, and local terminal policy for every existing component/net.",
            "Never output absolute pixel coordinates; use stage_x/lane_y and route-policy guidance.",
            "Never create, delete, rename, or rewire components, pins, or nets.",
            "Never create new components for anatomy, electrode art, annotations, or labels unless they already exist in the netlist.",
            "Never create new nets.",
            "Do not draw SVG paths or wire point lists; OptCPV will convert your higher-level drawing guidance into topology-safe geometry.",
            "Use blocks to identify functional subcircuits, their existing member components, and their interface nets.",
            "Never route GND, VCC, VEE, VDD, VSS, or REF/reference nets as global physical wires.",
            "Identify auxiliary feedback loops, especially right-leg-drive, RLD, driven-right-leg, and common-mode feedback.",
            "Put auxiliary feedback loops in a bottom auxiliary lane using bottom_auxiliary_corridor route policies.",
            "Keep the main signal flow left-to-right.",
            "Keep parallel differential inputs aligned by stage and separated by lane.",
            "Treat reference-image anatomy or electrode art as annotation, not core electrical layout, unless represented in the netlist.",
        ],
        "authority": {
            "drawing_guidance": "High. Gemini may override deterministic staging, lanes, orientation, block decomposition, and signal/feedback corridor choices.",
            "local_gate": "OptCPV will keep topology, terminal-net safety, canvas bounds, and scale-hack protections.",
        },
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
                    "orientation": "RIGHT|LEFT|UP|DOWN|RIGHT_FLIP",
                    "role": "optional semantic role",
                    "confidence": "number from 0 to 1",
                }
            ],
            "blocks": [
                {
                    "block_id": "stable semantic id",
                    "block_type": "differential_input_pair|parallel_opamp_buffers|opamp_feedback_stage|rc_filter_or_passive_filter|auxiliary_feedback_loop|generic_functional_block",
                    "members": ["existing component id"],
                    "stage_x": "integer",
                    "lane_y": "integer",
                    "ports": {"semantic_port_name": "existing net name"},
                    "route_policy": "optional route policy",
                }
            ],
            "block_internal_motifs": [{"motif_type": "string", "members": ["existing component id"]}],
            "inter_block_routes": [
                {
                    "net": "existing net name",
                    "from": "block_id.port_name",
                    "to": "block_id.port_name",
                    "policy": "left_to_right_manhattan|top_feedback_corridor|bottom_feedback_corridor|bottom_auxiliary_corridor|avoid_opamp_body",
                }
            ],
            "auxiliary_loops": [
                {
                    "loop_id": "stable semantic id",
                    "loop_type": "right_leg_drive|common_mode_feedback|auxiliary_feedback",
                    "members": ["existing component id"],
                    "nets": ["existing net name"],
                    "route_policy": "bottom_auxiliary_corridor",
                }
            ],
            "orientation_overrides": [
                {
                    "component_id": "existing component id",
                    "orientation": "RIGHT|RIGHT_FLIP",
                    "reason": "short semantic reason",
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


def _fallback_models_from_env() -> tuple[str, ...]:
    raw = os.getenv("OPTCPV_GEMINI_PLANNER_FALLBACK_MODELS")
    if raw:
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    return DEFAULT_GEMINI_PLANNER_FALLBACK_MODELS


def _truthy(value: str | None) -> bool:
    return _key(value) in {"1", "true", "yes", "on", "gemini"}


def _key(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


__all__ = [
    "AuxiliaryLoopHint",
    "BlockHint",
    "DEFAULT_GEMINI_PLANNER_MODEL",
    "DEFAULT_GEMINI_PLANNER_FALLBACK_MODELS",
    "FakePlanningClient",
    "GeminiPlanningClient",
    "GridPlacementHint",
    "InterBlockRouteHint",
    "LaneHint",
    "MotifHint",
    "OrientationOverrideHint",
    "PlanningHints",
    "RoutePolicyHint",
    "SchematicLayoutHints",
    "SchematicPlanningRequest",
    "SemanticPlanningClient",
    "StageHint",
    "load_dotenv_if_present",
    "planning_client_from_env",
]

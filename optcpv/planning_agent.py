"""Optional pre-render semantic planning client boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .models import Circuit, CriticReport, LayoutPlan
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


DEFAULT_GEMINI_PLANNER_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_PLANNER_FALLBACK_MODELS = ("gemini-2.5-flash",)
_DOTENV_LOADED = False


class SemanticPlanningClient:
    """Boundary for pre-layout semantic planning hints."""

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        raise NotImplementedError

    def refine_hints(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        critic_report: CriticReport,
        reference_image: bytes | None = None,
    ) -> SchematicLayoutHints | None:
        """Return second-pass semantic hints after local rendering/criticism."""

        return None


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

    def refine_hints(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        critic_report: CriticReport,
        reference_image: bytes | None = None,
    ) -> SchematicLayoutHints | None:
        prompt = _refinement_prompt(circuit, layout, svg, critic_report, has_reference_image=reference_image is not None)
        contents: list[Any] = [prompt]
        if reference_image is not None:
            contents.append(self._types.Part.from_bytes(data=reference_image, mime_type="image/png"))
        response = self._generate_content_with_fallbacks(contents)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return None
        return SchematicLayoutHints.from_dict(_json_object(text)).with_updates(source="gemini_refinement")

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
    if _truthy(os.getenv("OPTCPV_USE_TEXTBOOK_PLANNER")) or planner in {"textbook", "textbook_surrogate", "textbook_gemini_surrogate"}:
        from .textbook_surrogate import TextbookSurrogatePlanningClient

        return TextbookSurrogatePlanningClient()
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
        "task": "Return OptCPV schematic grammar intent plus optional pre-render layout hints as strict JSON only.",
        "input_mode": input_mode,
        "hard_rules": [
            "Return JSON only. No markdown, comments, or prose outside the object.",
            "First infer schematic grammar: component roles, pin roles, net roles, functional blocks, layout constraints, and route intents.",
            "Then optionally provide semantic blocks, route roles, discrete stage_x/lane_y placements, and orientation hints. Never output absolute pixel coordinates.",
            "Never create, delete, rename, or rewire components, pins, or nets.",
            "Never create new components for anatomy, electrode art, annotations, or labels unless they already exist in the netlist.",
            "Never create new nets.",
            "Do not draw SVG paths or wire point lists; route geometry is deterministic.",
            "Prefer grammar consistency over compactness.",
            "Use blocks to identify functional subcircuits, their existing member components, and their interface nets.",
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
            "intent": {
                "recognized_topology": "string",
                "confidence": "number from 0 to 1",
                "component_roles": {
                    "existing component id": [
                        "source_port|input_port|output_port|load|op_amp_gain_stage|op_amp_buffer|op_amp_comparator|op_amp_summing_stage|op_amp_difference_stage|feedback_element|series_input_element|shunt_reference_element|bias_element|filter_element|coupling_element|protection_element|sensor_element|local_reference_symbol|supply_symbol"
                    ]
                },
                "pin_roles": {
                    "component_id.pin_name": [
                        "signal_input|feedback_input|summing_input|reference_input|stage_output|drive_output|upstream|downstream|reference_side|signal_side"
                    ]
                },
                "net_roles": {
                    "existing net name": [
                        "forward_signal|feedback_signal|feedback_node|summing_node|virtual_ground_node|reference|ground|positive_supply|negative_supply|bias|common_mode|differential_plus|differential_minus|high_fanout_signal|sensor_node|output_drive|load_drive"
                    ]
                },
                "blocks": [
                    {
                        "block_id": "stable semantic id",
                        "block_type": "source_block|input_conditioning_block|single_opamp_feedback_stage|opamp_buffer_stage|inverting_stage|non_inverting_stage|summing_stage|difference_stage|comparator_stage|hysteresis_block|rc_filter_block|active_filter_block|coupling_bias_block|sensor_bridge_block|current_source_block|current_mirror_block|transistor_gain_stage|diode_clamp_block|power_supply_block|reference_generation_block|load_output_block|auxiliary_feedback_loop|generic_functional_block",
                        "members": ["existing component id"],
                        "input_nets": ["existing net name"],
                        "output_nets": ["existing net name"],
                        "feedback_nets": ["existing net name"],
                        "reference_nets": ["existing net name"],
                    }
                ],
                "constraints": [
                    {
                        "type": "left_of|above|below|align_y|align_x|same_stage|separate_lanes|feedback_outside_body|local_terminal_only|avoid_component_body|avoid_label_zone",
                        "subject": "existing component id or net",
                        "object": "optional existing component id or net",
                        "members": ["existing component id"],
                        "net": "optional existing net name",
                        "strength": "hard|soft",
                        "preferred_side": "optional top|bottom|left|right",
                        "reason": "short semantic reason",
                    }
                ],
                "route_intents": [
                    {
                        "net": "existing net name or null",
                        "net_role": "forward_signal|feedback_signal|local_reference|high_fanout_signal|differential_pair|auxiliary_loop",
                        "policy": "direct_short|left_to_right_manhattan|input_to_stage|stage_to_stage|output_to_load|top_feedback_corridor|bottom_feedback_corridor|outer_feedback_loop|local_terminal_only|named_net_label|paired_differential_route|shared_bus_spine|star_node|tee_branch|bottom_auxiliary_corridor|avoid_active_body|avoid_label_zone|avoid_crossing_existing_route",
                    }
                ],
                "unsupported_reasons": ["string"],
                "source": "gemini",
            },
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


def _refinement_prompt(
    circuit: Circuit,
    layout: LayoutPlan,
    svg: str,
    critic_report: CriticReport,
    *,
    has_reference_image: bool,
) -> str:
    request = SchematicPlanningRequest.from_circuit(
        circuit,
        input_mode="refinement",
        reference_image={"provided": True, "role": "teaching schematic visual target"} if has_reference_image else None,
    )
    payload = {
        "task": "Return refined OptCPV schematic layout hints as strict JSON only after seeing the failed local render.",
        "conversation_role": "You are the semantic planner. The local renderer/critic has already produced a layout and failure report. Reply with high-level grammar/layout hints only.",
        "hard_rules": [
            "Return JSON only. No markdown, comments, or prose outside the object.",
            "Never create, delete, rename, or rewire components, pins, or nets.",
            "Do not output SVG paths, wire point lists, or pixel coordinates.",
            "Use only existing component ids and net names from the request.",
            "Use semantic blocks, stage_x/lane_y placements, orientation overrides, and route_policies to fix the visible failure.",
            "Prefer textbook schematic conventions: left-to-right signal flow, op-amp feedback outside the body, local ground/supply symbols, compact readable labels, and no wires through active symbols.",
            "If a reference image is provided, infer relative stages, lanes, and motif grammar from it without copying non-electrical artwork into the netlist.",
        ],
        "schema": {
            "recognized_topology": "string",
            "confidence": "number from 0 to 1",
            "tutor_explanation": "short reason for the refined layout strategy",
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
                    "block_type": "opamp_feedback_stage|parallel_opamp_buffers|differential_input_pair|auxiliary_feedback_loop|generic_functional_block",
                    "members": ["existing component id"],
                    "stage_x": "integer",
                    "lane_y": "integer",
                    "ports": {"semantic_port_name": "existing net name"},
                    "route_policy": "optional route policy",
                }
            ],
            "orientation_overrides": [
                {"component_id": "existing component id", "orientation": "RIGHT|RIGHT_FLIP", "reason": "short semantic reason"}
            ],
            "route_policies": [
                {
                    "net": "existing net name",
                    "net_role": "signal|feedback|right_leg_drive|ground|supply|reference",
                    "policy": "left_to_right_manhattan|top_feedback_corridor|bottom_feedback_corridor|bottom_auxiliary_corridor|local_terminal_only|avoid_opamp_body",
                }
            ],
            "local_terminal_policy": {"net_name": "local_symbol_only"},
            "source": "gemini_refinement",
        },
        "request": request.to_dict(),
        "current_attempt": {
            "layout": _layout_summary(layout),
            "critic_report": critic_report.to_dict(),
            "svg_excerpt": svg[:8000],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _layout_summary(layout: LayoutPlan) -> dict[str, Any]:
    return {
        "canvas": {"width": layout.width, "height": layout.height, "grid": layout.grid},
        "planning_hints_used": layout.support.planning_hints,
        "components": [
            {
                "id": component.id,
                "type": component.type,
                "role": component.role,
                "x": component.x,
                "y": component.y,
                "orientation": component.orientation,
                "pins": dict(component.pins),
                "bbox": {
                    "x": component.bbox.x,
                    "y": component.bbox.y,
                    "width": component.bbox.width,
                    "height": component.bbox.height,
                },
            }
            for component in layout.components
        ],
        "wires": [
            {"net": wire.net, "points": [{"x": point.x, "y": point.y} for point in wire.points]}
            for wire in layout.wires
        ],
        "labels": [
            {"id": label.id, "owner_id": label.owner_id, "text": label.text, "x": label.x, "y": label.y}
            for label in layout.labels
        ],
        "semantic_plan": layout.semantic.to_dict(),
    }


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

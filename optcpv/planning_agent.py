"""Optional pre-layout semantic planning client boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from .models import Circuit


class SemanticPlanningClient:
    """Boundary for heuristic pre-layout planning hints."""

    def propose_hints(self, circuit: Circuit) -> "PlanningHints":
        raise NotImplementedError


@dataclass(frozen=True)
class StageHint:
    stage: int
    stage_type: str
    members: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StageHint":
        return cls(
            stage=int(raw["stage"]),
            stage_type=str(raw.get("type", raw.get("stage_type", ""))),
            members=tuple(str(item) for item in raw.get("members", ())),
        )


@dataclass(frozen=True)
class LaneHint:
    lane: int
    source: str
    members: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LaneHint":
        return cls(
            lane=int(raw["lane"]),
            source=str(raw.get("source", "")),
            members=tuple(str(item) for item in raw.get("members", ())),
        )


@dataclass(frozen=True)
class MotifHint:
    motif_type: str
    members: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MotifHint":
        return cls(
            motif_type=str(raw.get("type", raw.get("motif_type", ""))),
            members=tuple(str(item) for item in raw.get("members", ())),
        )


@dataclass(frozen=True)
class GridPlacementHint:
    component_id: str
    col: int
    row: int
    orientation: str | None = None
    motif_role: str | None = None

    @classmethod
    def from_dict(cls, component_id: str, raw: dict[str, Any]) -> "GridPlacementHint":
        if "x" in raw or "y" in raw:
            raise ValueError("Planning hints must use integer grid col/row, not absolute x/y coordinates.")
        return cls(
            component_id=component_id,
            col=int(raw["col"]),
            row=int(raw["row"]),
            orientation=str(raw["orientation"]) if raw.get("orientation") is not None else None,
            motif_role=str(raw["motif_role"]) if raw.get("motif_role") is not None else None,
        )


@dataclass(frozen=True)
class PlanningHints:
    recognized_topology: str = ""
    confidence: float = 0.0
    stage_order: tuple[StageHint, ...] = field(default_factory=tuple)
    lanes: tuple[LaneHint, ...] = field(default_factory=tuple)
    placement_hints: dict[str, GridPlacementHint] = field(default_factory=dict)
    local_terminal_policy: dict[str, str] = field(default_factory=dict)
    routing_rules: tuple[str, ...] = field(default_factory=tuple)
    motifs: tuple[MotifHint, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanningHints":
        placements_raw = raw.get("placement_hints", {})
        if not isinstance(placements_raw, dict):
            raise ValueError("planning_hints.placement_hints must be an object.")
        return cls(
            recognized_topology=str(raw.get("recognized_topology", "")),
            confidence=float(raw.get("confidence", 0.0)),
            stage_order=tuple(StageHint.from_dict(item) for item in raw.get("stage_order", ())),
            lanes=tuple(LaneHint.from_dict(item) for item in raw.get("lanes", ())),
            placement_hints={
                str(component_id): GridPlacementHint.from_dict(str(component_id), hint)
                for component_id, hint in placements_raw.items()
            },
            local_terminal_policy={str(net): str(policy) for net, policy in raw.get("local_terminal_policy", {}).items()},
            routing_rules=tuple(str(rule) for rule in raw.get("routing_rules", ())),
            motifs=tuple(MotifHint.from_dict(item) for item in raw.get("motifs", ())),
        )


class GeminiPlanningClient(SemanticPlanningClient):
    """Optional Gemini-backed semantic warm-start client."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash") -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiPlanningClient.") from exc
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._model = model

    def propose_hints(self, circuit: Circuit) -> PlanningHints:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[_planning_prompt(circuit)],
            config={"response_mime_type": "application/json"},
        )
        text = (getattr(response, "text", "") or "").strip()
        return PlanningHints.from_dict(_json_object(text))


def _planning_prompt(circuit: Circuit) -> str:
    payload = {
        "task": "Return OptCPV pre-layout semantic planning hints as strict JSON only.",
        "hard_rules": [
            "Do not create, delete, rename, or rewire components, pins, or nets.",
            "Do not output absolute pixel coordinates.",
            "Use only integer grid col and row placement hints.",
            "Ground, supply, and reference nets must use local symbols only, not routed rails.",
            "All physical wires must be Manhattan.",
        ],
        "schema": {
            "recognized_topology": "string",
            "confidence": "number from 0 to 1",
            "stage_order": [{"stage": "integer", "type": "string", "members": ["component_id"]}],
            "lanes": [{"lane": "integer", "source": "component_id", "members": ["component_id"]}],
            "placement_hints": {
                "component_id": {
                    "col": "integer",
                    "row": "integer",
                    "orientation": "right|right_flip|left|up|down",
                    "motif_role": "optional string",
                }
            },
            "local_terminal_policy": {"net_name": "local_symbol_only"},
            "routing_rules": ["string"],
        },
        "circuit": {
            "id": circuit.id,
            "motif": circuit.motif,
            "components": [
                {
                    "id": component.id,
                    "type": component.type,
                    "role": component.role,
                    "label": component.label,
                    "value": component.value,
                    "pins": dict(component.pins),
                }
                for component in circuit.components
            ],
        },
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

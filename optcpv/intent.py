"""First-class schematic grammar intent models.

The intent layer describes what circuit parts mean before any concrete grid
placement is chosen.  It is deliberately independent of renderer coordinates
so local analyzers and external planners can share the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CONSTRAINT_STRENGTHS = {"hard", "soft"}


@dataclass(frozen=True)
class FunctionalBlock:
    block_id: str
    block_type: str
    members: tuple[str, ...] = field(default_factory=tuple)
    input_nets: tuple[str, ...] = field(default_factory=tuple)
    output_nets: tuple[str, ...] = field(default_factory=tuple)
    feedback_nets: tuple[str, ...] = field(default_factory=tuple)
    reference_nets: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", str(self.block_id))
        object.__setattr__(self, "block_type", str(self.block_type))
        object.__setattr__(self, "members", _str_tuple(self.members))
        object.__setattr__(self, "input_nets", _str_tuple(self.input_nets))
        object.__setattr__(self, "output_nets", _str_tuple(self.output_nets))
        object.__setattr__(self, "feedback_nets", _str_tuple(self.feedback_nets))
        object.__setattr__(self, "reference_nets", _str_tuple(self.reference_nets))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FunctionalBlock":
        if not isinstance(raw, dict):
            raise ValueError("FunctionalBlock must be an object.")
        return cls(
            block_id=str(raw.get("block_id", raw.get("id", ""))),
            block_type=str(raw.get("block_type", raw.get("type", ""))),
            members=_str_tuple(raw.get("members", ())),
            input_nets=_str_tuple(raw.get("input_nets", raw.get("inputs", ()))),
            output_nets=_str_tuple(raw.get("output_nets", raw.get("outputs", ()))),
            feedback_nets=_str_tuple(raw.get("feedback_nets", ())),
            reference_nets=_str_tuple(raw.get("reference_nets", raw.get("local_reference_nets", ()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "members": list(self.members),
            "input_nets": list(self.input_nets),
            "output_nets": list(self.output_nets),
            "feedback_nets": list(self.feedback_nets),
            "reference_nets": list(self.reference_nets),
        }


@dataclass(frozen=True)
class LayoutConstraint:
    constraint_type: str
    subject: str | None = None
    object: str | None = None
    members: tuple[str, ...] = field(default_factory=tuple)
    net: str | None = None
    strength: str = "soft"
    preferred_side: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        strength = str(self.strength).lower()
        if strength not in CONSTRAINT_STRENGTHS:
            strength = "soft"
        object.__setattr__(self, "constraint_type", str(self.constraint_type))
        object.__setattr__(self, "subject", None if self.subject is None else str(self.subject))
        object.__setattr__(self, "object", None if self.object is None else str(self.object))
        object.__setattr__(self, "members", _str_tuple(self.members))
        object.__setattr__(self, "net", None if self.net is None else str(self.net))
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "preferred_side", None if self.preferred_side is None else str(self.preferred_side))
        object.__setattr__(self, "reason", str(self.reason))

    @property
    def type(self) -> str:
        return self.constraint_type

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LayoutConstraint":
        if not isinstance(raw, dict):
            raise ValueError("LayoutConstraint must be an object.")
        return cls(
            constraint_type=str(raw.get("type", raw.get("constraint_type", ""))),
            subject=_optional_str(raw.get("subject", raw.get("component", raw.get("a")))),
            object=_optional_str(raw.get("object", raw.get("b"))),
            members=_str_tuple(raw.get("members", ())),
            net=_optional_str(raw.get("net")),
            strength=str(raw.get("strength", "soft")),
            preferred_side=_optional_str(raw.get("preferred_side", raw.get("side"))),
            reason=str(raw.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.constraint_type,
            "subject": self.subject,
            "object": self.object,
            "members": list(self.members),
            "net": self.net,
            "strength": self.strength,
            "preferred_side": self.preferred_side,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouteIntentRule:
    net: str | None
    net_role: str
    policy: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "net", None if self.net is None else str(self.net))
        object.__setattr__(self, "net_role", str(self.net_role))
        object.__setattr__(self, "policy", str(self.policy))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RouteIntentRule":
        if not isinstance(raw, dict):
            raise ValueError("RouteIntentRule must be an object.")
        return cls(
            net=_optional_str(raw.get("net")),
            net_role=str(raw.get("net_role", raw.get("role", ""))),
            policy=str(raw.get("policy")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"net": self.net, "net_role": self.net_role, "policy": self.policy}


@dataclass(frozen=True)
class SchematicIntent:
    recognized_topology: str = ""
    component_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pin_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    net_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    blocks: tuple[FunctionalBlock, ...] = field(default_factory=tuple)
    constraints: tuple[LayoutConstraint, ...] = field(default_factory=tuple)
    route_intents: tuple[RouteIntentRule, ...] = field(default_factory=tuple)
    unsupported_reasons: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    source: str = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "recognized_topology", str(self.recognized_topology))
        object.__setattr__(self, "component_roles", _roles_dict(self.component_roles))
        object.__setattr__(self, "pin_roles", _roles_dict(self.pin_roles))
        object.__setattr__(self, "net_roles", _roles_dict(self.net_roles))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "route_intents", tuple(self.route_intents))
        object.__setattr__(self, "unsupported_reasons", _str_tuple(self.unsupported_reasons))
        object.__setattr__(self, "confidence", _clamp_float(self.confidence, 0.0, 1.0))
        object.__setattr__(self, "source", str(self.source))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SchematicIntent":
        if not isinstance(raw, dict):
            raise ValueError("SchematicIntent must be an object.")
        return cls(
            recognized_topology=str(raw.get("recognized_topology", "")),
            component_roles=_roles_dict(raw.get("component_roles", {})),
            pin_roles=_roles_dict(raw.get("pin_roles", {})),
            net_roles=_roles_dict(raw.get("net_roles", {})),
            blocks=tuple(FunctionalBlock.from_dict(item) for item in _list(raw.get("blocks"))),
            constraints=tuple(LayoutConstraint.from_dict(item) for item in _list(raw.get("constraints"))),
            route_intents=tuple(
                RouteIntentRule.from_dict(item)
                for item in _list(raw.get("route_intents", raw.get("route_policies")))
            ),
            unsupported_reasons=_str_tuple(raw.get("unsupported_reasons", ())),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            source=str(raw.get("source", "manual")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognized_topology": self.recognized_topology,
            "component_roles": {key: list(value) for key, value in self.component_roles.items()},
            "pin_roles": {key: list(value) for key, value in self.pin_roles.items()},
            "net_roles": {key: list(value) for key, value in self.net_roles.items()},
            "blocks": [block.to_dict() for block in self.blocks],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "route_intents": [route.to_dict() for route in self.route_intents],
            "unsupported_reasons": list(self.unsupported_reasons),
            "confidence": self.confidence,
            "source": self.source,
        }


def _roles_dict(raw: Any) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Role maps must be objects.")
    result: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        if value is None:
            result[str(key)] = ()
        elif isinstance(value, str):
            result[str(key)] = (value,)
        else:
            result[str(key)] = _str_tuple(value)
    return result


def _str_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    try:
        return tuple(str(item) for item in raw if item is not None)
    except TypeError:
        return (str(raw),)


def _list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of objects.")
    return [item for item in value if isinstance(item, dict)]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


__all__ = [
    "FunctionalBlock",
    "LayoutConstraint",
    "RouteIntentRule",
    "SchematicIntent",
]

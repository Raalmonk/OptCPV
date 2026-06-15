"""Lightweight public models for OptCPV circuit drawing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Component:
    id: str
    type: str
    pins: dict[str, str]
    label: str | None = None
    role: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class Circuit:
    id: str
    components: list[Component]
    motif: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class LayoutComponent:
    id: str
    x: float
    y: float
    orientation: str
    type: str
    pins: dict[str, str]
    label: str | None = None
    role: str | None = None
    value: str | None = None


@dataclass(frozen=True)
class LayoutWire:
    net: str
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class Layout:
    circuit_id: str
    width: int
    height: int
    components: list[LayoutComponent]
    wires: list[LayoutWire]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchematicArtifact:
    svg: str
    components: dict[str, dict[str, Any]]
    nets: dict[str, dict[str, Any]]
    viewbox: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def circuit_from_any(circuit: Circuit | dict[str, Any]) -> Circuit:
    if isinstance(circuit, Circuit):
        return circuit
    if not isinstance(circuit, dict):
        raise TypeError(f"Expected Circuit or dict, got {type(circuit).__name__}.")

    raw_components = circuit.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("Circuit requires a non-empty components list.")

    components: list[Component] = []
    for raw in raw_components:
        if not isinstance(raw, dict):
            raise ValueError("Each component must be a dictionary.")
        pins = raw.get("pins")
        if not isinstance(pins, dict) or not pins:
            raise ValueError(f"Component {raw.get('id', '<unknown>')} requires pins.")
        components.append(
            Component(
                id=str(raw["id"]),
                type=str(raw["type"]),
                pins={str(pin): str(net) for pin, net in pins.items()},
                label=_optional_str(raw.get("label", raw.get("display_label"))),
                role=_optional_str(raw.get("role")),
                value=_optional_str(raw.get("value", raw.get("value_label"))),
            )
        )

    return Circuit(
        id=str(circuit.get("id", "circuit")),
        motif=_optional_str(circuit.get("motif")),
        title=_optional_str(circuit.get("title")),
        components=components,
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)

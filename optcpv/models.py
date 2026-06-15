"""Core data models for OptCPV's schematic optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def intersects(self, other: "BBox", *, padding: float = 0.0) -> bool:
        return not (
            self.right + padding <= other.x
            or other.right + padding <= self.x
            or self.bottom + padding <= other.y
            or other.bottom + padding <= self.y
        )

    def contains_point(self, point: Point | tuple[float, float], *, padding: float = 0.0) -> bool:
        px, py = point.as_tuple() if isinstance(point, Point) else point
        return self.x - padding <= px <= self.right + padding and self.y - padding <= py <= self.bottom + padding

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def expanded(self, amount: float) -> "BBox":
        return BBox(self.x - amount, self.y - amount, self.width + 2 * amount, self.height + 2 * amount)


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
class LayoutPin:
    component_id: str
    pin_name: str
    net: str
    x: float
    y: float
    side: str


@dataclass(frozen=True)
class LayoutComponent:
    id: str
    type: str
    role: str | None
    label: str | None
    value: str | None
    x: float
    y: float
    orientation: str
    pins: dict[str, str]
    bbox: BBox


@dataclass(frozen=True)
class LayoutLabel:
    id: str
    text: str
    owner_id: str
    x: float
    y: float
    anchor: str
    bbox: BBox


@dataclass(frozen=True)
class LayoutWire:
    net: str
    points: list[Point]
    connected_pins: list[tuple[str, str]]


@dataclass(frozen=True)
class LayoutPlan:
    circuit_id: str
    width: int
    height: int
    grid: int
    components: list[LayoutComponent]
    wires: list[LayoutWire]
    labels: list[LayoutLabel]
    pin_map: dict[tuple[str, str], LayoutPin]
    net_to_pins: dict[str, list[tuple[str, str]]]
    topology_signature: str
    warnings: list[str] = field(default_factory=list)


Layout = LayoutPlan


@dataclass(frozen=True)
class CriticViolation:
    code: str
    message: str
    severity: float
    hard: bool = False
    subject: str | None = None


@dataclass(frozen=True)
class CriticReport:
    score: float
    violations: list[CriticViolation]
    metrics: dict[str, float | int | str | bool]
    hard_fail: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "hard_fail": self.hard_fail,
            "metrics": dict(self.metrics),
            "violations": [
                {
                    "code": violation.code,
                    "message": violation.message,
                    "severity": violation.severity,
                    "hard": violation.hard,
                    "subject": violation.subject,
                }
                for violation in self.violations
            ],
        }


@dataclass(frozen=True)
class SchematicArtifact:
    svg: str
    components: dict[str, dict[str, Any]]
    nets: dict[str, dict[str, Any]]
    labels: dict[str, dict[str, Any]]
    viewbox: dict[str, float]
    critic_report: dict[str, Any] | None = None
    cv_report: dict[str, Any] | None = None
    optimization_log: list[dict[str, Any]] = field(default_factory=list)
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

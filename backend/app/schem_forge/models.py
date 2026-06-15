"""Typed layout DSL for deterministic schematic planning and rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


Orientation = Literal["up", "down", "left", "right"]
PinSide = Literal["left", "right", "top", "bottom"]
WireKind = Literal["wire", "jumper", "hidden_net_label"]
LabelAnchor = Literal["start", "middle", "end"]


@dataclass
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass
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

    def intersects(self, other: "BBox", padding: float = 0.0) -> bool:
        return not (
            self.right + padding <= other.x
            or other.right + padding <= self.x
            or self.bottom + padding <= other.y
            or other.bottom + padding <= self.y
        )

    def contains_point(self, point: Point, strict: bool = False) -> bool:
        if strict:
            return self.x < point.x < self.right and self.y < point.y < self.bottom
        return self.x <= point.x <= self.right and self.y <= point.y <= self.bottom

    def inset(self, amount: float) -> "BBox":
        new_width = max(0.0, self.width - 2 * amount)
        new_height = max(0.0, self.height - 2 * amount)
        return BBox(self.x + amount, self.y + amount, new_width, new_height)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class PinLayout:
    component_id: str
    pin_name: str
    net_name: str
    side: PinSide
    offset_x: float
    offset_y: float


@dataclass
class ComponentLayout:
    id: str
    type: str
    role: str | None
    grid_x: float
    grid_y: float
    orientation: Orientation
    value_label: str | None
    display_label: str | None
    pins: list[PinLayout] = field(default_factory=list)
    bbox: BBox | None = None


@dataclass
class WireSegment:
    start: Point
    end: Point
    net_name: str
    kind: WireKind = "wire"


@dataclass
class WireRoute:
    net_name: str
    connected_pins: list[str]
    waypoints: list[Point] = field(default_factory=list)
    segments: list[WireSegment] | None = None


@dataclass
class LabelLayout:
    id: str
    text: str
    owner_id: str | None
    grid_x: float
    grid_y: float
    anchor: LabelAnchor = "middle"
    bbox: BBox | None = None


@dataclass
class LayoutPlan:
    circuit_id: str
    canvas_width: int
    canvas_height: int
    grid_size: int
    components: list[ComponentLayout]
    wires: list[WireRoute]
    labels: list[LabelLayout]
    net_to_pins: dict[str, list[str]]
    component_pin_nets: dict[str, dict[str, str]]
    topology_signature: str
    renderer: str = "schem_forge.svg.v1"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutPlan":
        return layout_plan_from_dict(data)


@dataclass
class RenderGeometry:
    component_bboxes: dict[str, BBox] = field(default_factory=dict)
    label_bboxes: dict[str, BBox] = field(default_factory=dict)
    wire_segments: list[WireSegment] = field(default_factory=list)
    pin_points: dict[str, Point] = field(default_factory=dict)
    junction_points: dict[str, Point] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass
class RenderResult:
    svg: str
    geometry: RenderGeometry
    svg_viewbox: BBox | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    return value


def point_from(data: Point | dict[str, Any]) -> Point:
    if isinstance(data, Point):
        return data
    return Point(x=float(data["x"]), y=float(data["y"]))


def bbox_from(data: BBox | dict[str, Any] | None) -> BBox | None:
    if data is None or isinstance(data, BBox):
        return data
    return BBox(
        x=float(data["x"]),
        y=float(data["y"]),
        width=float(data["width"]),
        height=float(data["height"]),
    )


def pin_layout_from_dict(data: dict[str, Any]) -> PinLayout:
    return PinLayout(
        component_id=str(data["component_id"]),
        pin_name=str(data["pin_name"]),
        net_name=str(data["net_name"]),
        side=data["side"],
        offset_x=float(data["offset_x"]),
        offset_y=float(data["offset_y"]),
    )


def component_layout_from_dict(data: dict[str, Any]) -> ComponentLayout:
    return ComponentLayout(
        id=str(data["id"]),
        type=str(data["type"]),
        role=data.get("role"),
        grid_x=float(data["grid_x"]),
        grid_y=float(data["grid_y"]),
        orientation=data["orientation"],
        value_label=data.get("value_label"),
        display_label=data.get("display_label"),
        pins=[pin_layout_from_dict(item) for item in data.get("pins", [])],
        bbox=bbox_from(data.get("bbox")),
    )


def wire_segment_from_dict(data: dict[str, Any]) -> WireSegment:
    return WireSegment(
        start=point_from(data["start"]),
        end=point_from(data["end"]),
        net_name=str(data["net_name"]),
        kind=data.get("kind", "wire"),
    )


def wire_route_from_dict(data: dict[str, Any]) -> WireRoute:
    segments = data.get("segments")
    return WireRoute(
        net_name=str(data["net_name"]),
        connected_pins=[str(item) for item in data.get("connected_pins", [])],
        waypoints=[point_from(item) for item in data.get("waypoints", [])],
        segments=[wire_segment_from_dict(item) for item in segments] if segments else None,
    )


def label_layout_from_dict(data: dict[str, Any]) -> LabelLayout:
    return LabelLayout(
        id=str(data["id"]),
        text=str(data["text"]),
        owner_id=data.get("owner_id"),
        grid_x=float(data["grid_x"]),
        grid_y=float(data["grid_y"]),
        anchor=data.get("anchor", "middle"),
        bbox=bbox_from(data.get("bbox")),
    )


def layout_plan_from_dict(data: dict[str, Any]) -> LayoutPlan:
    return LayoutPlan(
        circuit_id=str(data["circuit_id"]),
        canvas_width=int(data["canvas_width"]),
        canvas_height=int(data["canvas_height"]),
        grid_size=int(data["grid_size"]),
        components=[component_layout_from_dict(item) for item in data.get("components", [])],
        wires=[wire_route_from_dict(item) for item in data.get("wires", [])],
        labels=[label_layout_from_dict(item) for item in data.get("labels", [])],
        net_to_pins={
            str(net): [str(pin) for pin in pins]
            for net, pins in data.get("net_to_pins", {}).items()
        },
        component_pin_nets={
            str(component): {str(pin): str(net) for pin, net in pin_map.items()}
            for component, pin_map in data.get("component_pin_nets", {}).items()
        },
        topology_signature=str(data["topology_signature"]),
        renderer=data.get("renderer", "schem_forge.svg.v1"),
        warnings=[str(item) for item in data.get("warnings", [])],
    )

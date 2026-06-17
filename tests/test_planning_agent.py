from optcpv import Circuit, Component, GridPlacementHint, PlanningHints, SemanticPlanningClient, draw_artifact
from optcpv.planner import plan_layout
from optcpv.vector_critic import critique_layout


class FakePlanningClient(SemanticPlanningClient):
    def __init__(self, hints: PlanningHints) -> None:
        self.hints = hints
        self.calls = 0

    def propose_hints(self, circuit):
        self.calls += 1
        return self.hints


def test_valid_planning_hints_preserve_or_improve_layout_score() -> None:
    circuit = _planning_fixture()
    base = plan_layout(circuit)
    hints = PlanningHints(
        recognized_topology="non-inverting amplifier",
        confidence=0.8,
        placement_hints={
            "VIN": GridPlacementHint("VIN", 0, 0, "right"),
            "U1": GridPlacementHint("U1", 1, 0, "right"),
            "Rf": GridPlacementHint("Rf", 1, -1, "left", "feedback"),
            "Rg": GridPlacementHint("Rg", 1, 1, "down", "ground_leg"),
            "VOUT": GridPlacementHint("VOUT", 2, 0, "right"),
        },
        local_terminal_policy={"gnd": "local_symbol_only"},
        routing_rules=("all wires must be Manhattan",),
    )
    client = FakePlanningClient(hints)

    planned = plan_layout(circuit, planning_client=client)

    assert client.calls == 1
    assert critique_layout(planned).score <= critique_layout(base).score


def test_invalid_planning_hint_falls_back_to_deterministic_layout() -> None:
    circuit = _planning_fixture()
    base = plan_layout(circuit)
    bad = PlanningHints(
        placement_hints={
            "VIN": GridPlacementHint("VIN", 0, 0, "right"),
            "NOPE": GridPlacementHint("NOPE", 1, 0, "right"),
        }
    )

    planned = plan_layout(circuit, planning_hints=bad)

    assert [(component.id, component.x, component.y) for component in planned.components] == [
        (component.id, component.x, component.y) for component in base.components
    ]
    assert "planning_hints: accepted" not in planned.warnings


def test_planning_hints_cannot_route_terminal_nets() -> None:
    circuit = _planning_fixture()
    bad = PlanningHints(
        placement_hints={"VIN": GridPlacementHint("VIN", 0, 0, "right")},
        local_terminal_policy={"gnd": "global_rail"},
        routing_rules=("route gnd as a global bus",),
    )

    planned = plan_layout(circuit, planning_hints=bad)

    assert "planning_hints: accepted" not in planned.warnings
    assert "gnd" not in {wire.net for wire in planned.wires}


def test_planning_hints_cannot_change_topology() -> None:
    raw_hints = {
        "placement_hints": {
            "VIN": {"col": 0, "row": 0, "orientation": "right"},
            "NEW_COMPONENT": {"col": 1, "row": 0, "orientation": "right"},
        }
    }

    planned = plan_layout(_planning_fixture(), planning_hints=raw_hints)

    assert "NEW_COMPONENT" not in {component.id for component in planned.components}
    assert "planning_hints: accepted" not in planned.warnings


def test_public_artifact_api_works_without_planning_client() -> None:
    artifact = draw_artifact(_planning_fixture())

    assert artifact.svg
    assert artifact.semantic_plan


def _planning_fixture() -> Circuit:
    return Circuit(
        id="planning_non_inverting",
        motif="non_inverting_op_amp",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}),
            Component(id="VOUT", type="output", pins={"in": "vout"}),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, role="feedback"),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, role="gain"),
        ],
    )

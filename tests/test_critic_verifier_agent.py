from __future__ import annotations

import copy

import pytest

from backend.app.schem_forge import agent as agent_module
from backend.app.schem_forge.agent import (
    LayoutPatch,
    LayoutPatchError,
    MockLLMClient,
    apply_layout_patch,
    generate_beautiful_schematic,
)
from backend.app.schem_forge.critic import critique_layout
from backend.app.schem_forge.models import Point, WireRoute
from backend.app.schem_forge.planner import plan_circuit
from backend.app.schem_forge.renderer import render_layout
from backend.app.schem_forge.verifier import ElectricalTopologyError, verify_equivalence


def test_critic_detects_component_overlap(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    plan.components[1].grid_x = plan.components[2].grid_x
    plan.components[1].grid_y = plan.components[2].grid_y

    report = critique_layout(plan, render_layout(plan))

    assert report.breakdown["component_overlap"] >= 1000


def test_critic_detects_wire_through_component(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    plan.wires.append(
        WireRoute(
            net_name="TEST",
            connected_pins=[],
            waypoints=[Point(7.0, 7.0), Point(11.0, 7.0)],
        )
    )

    report = critique_layout(plan, render_layout(plan))

    assert "wire_crosses_component_body" in report.breakdown


def test_critic_detects_label_overlap(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    r1 = next(component for component in plan.components if component.id == "R1")
    label = next(label for label in plan.labels if label.owner_id == "R1")
    label.grid_x = r1.grid_x
    label.grid_y = r1.grid_y

    report = critique_layout(plan, render_layout(plan))

    assert "label_overlaps_component" in report.breakdown
    assert report.violations[0].suggested_patch is not None


def test_verifier_accepts_equivalent_layout(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)

    assert verify_equivalence(voltage_divider_ir_fixture, plan) is True


def test_verifier_rejects_changed_component_id(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    plan.components[0].id = "MUTATED"

    with pytest.raises(ElectricalTopologyError):
        verify_equivalence(voltage_divider_ir_fixture, plan)


def test_verifier_rejects_changed_component_type(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    plan.components[0].type = "capacitor"

    with pytest.raises(ElectricalTopologyError):
        verify_equivalence(voltage_divider_ir_fixture, plan)


def test_verifier_rejects_changed_pin_net(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    plan.components[0].pins[0].net_name = "OTHER_NET"

    with pytest.raises(ElectricalTopologyError):
        verify_equivalence(voltage_divider_ir_fixture, plan)


def test_restricted_patch_cannot_mutate_topology(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)

    with pytest.raises(LayoutPatchError):
        apply_layout_patch(
            voltage_divider_ir_fixture,
            plan,
            {"component_pin_nets": {"R1": {"a": "SHORT"}}},
        )


def test_instrumentation_planner_returns_valid_topology(instrumentation_ir_fixture: dict) -> None:
    plan = plan_circuit(instrumentation_ir_fixture)

    assert verify_equivalence(instrumentation_ir_fixture, plan) is True


def test_mock_patch_improves_synthetic_bad_layout(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)
    label = next(label for label in plan.labels if label.owner_id == "R1")
    component = next(component for component in plan.components if component.id == "R1")
    label.grid_x = component.grid_x
    label.grid_y = component.grid_y
    before = critique_layout(plan, render_layout(plan))

    patch = MockLLMClient().propose_patch(plan, before, render_layout(plan))
    candidate = apply_layout_patch(voltage_divider_ir_fixture, plan, patch)
    after = critique_layout(candidate, render_layout(candidate))

    assert after.total_score < before.total_score


def test_illegal_json_patch_is_rejected(voltage_divider_ir_fixture: dict) -> None:
    plan = plan_circuit(voltage_divider_ir_fixture)

    with pytest.raises(LayoutPatchError):
        apply_layout_patch(
            voltage_divider_ir_fixture,
            plan,
            [{"op": "replace", "path": "/components/0/id", "value": "BAD"}],
        )


def test_best_layout_retained_when_patch_is_worse(monkeypatch, voltage_divider_ir_fixture: dict) -> None:
    class WorseClient:
        def propose_patch(self, layout_plan, critic_report, rendered):
            label = next(label for label in layout_plan.labels if label.owner_id == "R1")
            component = next(component for component in layout_plan.components if component.id == "R1")
            return LayoutPatch(
                move_label=[
                    {"id": label.id, "grid_x": component.grid_x, "grid_y": component.grid_y}
                ]
            )

    monkeypatch.setattr(agent_module, "ACCEPTABLE_SCORE", -1)
    result = generate_beautiful_schematic(
        voltage_divider_ir_fixture,
        max_iterations=2,
        llm_client=WorseClient(),
    )

    assert result.critic_report.total_score == 0
    assert result.improved is False
    assert result.debug_log[-1]["candidate_score"] > result.debug_log[-1]["previous_score"]

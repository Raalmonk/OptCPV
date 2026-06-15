"""Standalone schematic compiler for analog tutoring schematics."""

from __future__ import annotations

from typing import Any

from .agent import AgentResult, GeminiLLMClient, LayoutPatch, MockLLMClient, generate_beautiful_schematic
from .adapters import circuit_problem_to_schem_forge_ir
from .artifact import SchematicArtifact, build_schematic_artifact
from .critic import CriticReport, critique_layout
from .planner import (
    plan_bridge_or_wheatstone,
    plan_circuit,
    plan_instrumentation_amplifier,
    plan_non_inverting_op_amp,
    plan_rc_low_pass,
    plan_voltage_divider,
)
from .renderer import render_layout
from .verifier import ElectricalTopologyError, verify_equivalence


def compile_schematic_for_generator(
    circuit_ir: Any,
    max_iterations: int = 5,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Small integration wrapper for future schematic_generator.py adoption."""

    result = generate_beautiful_schematic(
        circuit_ir,
        max_iterations=max_iterations,
        llm_client=llm_client,
    )
    return {
        "svg": result.svg,
        "layout": result.layout,
        "geometry": result.geometry,
        "critic_report": result.critic_report,
        "artifact": result.artifact,
        "iterations": result.iterations,
    }


__all__ = [
    "AgentResult",
    "CriticReport",
    "ElectricalTopologyError",
    "GeminiLLMClient",
    "LayoutPatch",
    "MockLLMClient",
    "SchematicArtifact",
    "build_schematic_artifact",
    "circuit_problem_to_schem_forge_ir",
    "compile_schematic_for_generator",
    "critique_layout",
    "generate_beautiful_schematic",
    "plan_bridge_or_wheatstone",
    "plan_circuit",
    "plan_instrumentation_amplifier",
    "plan_non_inverting_op_amp",
    "plan_rc_low_pass",
    "plan_voltage_divider",
    "render_layout",
    "verify_equivalence",
]

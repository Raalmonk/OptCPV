"""Artifact generation service for API endpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from backend.app.schem_forge import MockLLMClient, circuit_problem_to_schem_forge_ir
from backend.app.schem_forge.agent import generate_beautiful_schematic


InputFormat = Literal["auto", "schem_forge_ir", "citt"]


def normalize_input_circuit(circuit: dict[str, Any], input_format: InputFormat = "auto") -> dict[str, Any]:
    payload = deepcopy(circuit)
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Circuit payload must contain a non-empty components list.")

    if input_format == "citt":
        return circuit_problem_to_schem_forge_ir(payload)
    if input_format == "schem_forge_ir":
        _validate_schem_forge_ir(payload)
        return payload

    has_nodes = any(isinstance(component, dict) and "nodes" in component for component in components)
    has_pins = any(isinstance(component, dict) and "pins" in component for component in components)
    if has_nodes and not has_pins:
        return circuit_problem_to_schem_forge_ir(payload)
    if has_pins:
        _validate_schem_forge_ir(payload)
        return payload
    raise ValueError(
        "Could not infer input format. Use components[].nodes for CiTT payloads "
        "or components[].pins for schem_forge IR."
    )


def generate_schematic_payload(
    circuit_ir: dict[str, Any],
    max_iterations: int = 5,
    use_mock_agent: bool = True,
) -> dict[str, Any]:
    llm_client = MockLLMClient() if use_mock_agent else None
    result = generate_beautiful_schematic(
        circuit_ir,
        max_iterations=max_iterations,
        llm_client=llm_client,
    )
    return {
        "status": "ok",
        "artifact": result.artifact.to_dict(),
        "svg": result.svg,
        "critic": result.critic_report.to_dict(),
        "warnings": list(result.layout.warnings),
        "input_ir": deepcopy(circuit_ir),
    }


def _validate_schem_forge_ir(circuit_ir: dict[str, Any]) -> None:
    components = circuit_ir.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("schem_forge IR must contain a non-empty components list.")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError("Each component must be an object.")
        if not isinstance(component.get("pins"), dict):
            raise ValueError("schem_forge IR components must include a pins object.")

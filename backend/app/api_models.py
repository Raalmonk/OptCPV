"""Pydantic request/response models for the schem_forge product API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


InputFormat = Literal["auto", "schem_forge_ir", "citt"]


class FromIRRequest(BaseModel):
    circuit: dict[str, Any]
    input_format: InputFormat = "auto"
    max_iterations: int = Field(default=5, ge=0, le=20)
    use_mock_agent: bool = True


class FromTextRequest(BaseModel):
    prompt: str = Field(min_length=1)
    style: str = "textbook"
    max_iterations: int = Field(default=5, ge=0, le=20)
    use_mock_agent: bool = True


class ErrorResponse(BaseModel):
    status: str
    message: str


class SchematicResponse(BaseModel):
    status: str = "ok"
    artifact: dict[str, Any]
    svg: str
    critic: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    input_ir: dict[str, Any]

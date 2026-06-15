"""Image-to-circuit parsing service helpers."""

from __future__ import annotations

from typing import Any

from .text_to_ir import GeminiCircuitParser


def image_parser_available() -> bool:
    return GeminiCircuitParser.is_configured()


def parse_image_to_ir(image_bytes: bytes, prompt: str | None = None) -> dict[str, Any]:
    return GeminiCircuitParser().parse_image(image_bytes, prompt=prompt)

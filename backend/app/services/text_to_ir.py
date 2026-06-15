"""Text and Gemini-backed circuit parsing boundaries.

Parsers in this module return structured circuit IR only. SVG and artifact
generation always stays inside schem_forge's deterministic renderer pipeline.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.app.schem_forge.examples import (
    instrumentation_amp_ir,
    non_inverting_op_amp_ir,
    rc_low_pass_ir,
    voltage_divider_ir,
)


_DOTENV_LOADED = False


def load_local_dotenv() -> None:
    """Load a simple local .env file for development without overriding env vars."""

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    if os.getenv("SCHEM_FORGE_SKIP_DOTENV") == "1":
        return

    repo_root = Path(__file__).resolve().parents[3]
    candidates = []
    for path in (Path.cwd() / ".env", repo_root / ".env"):
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class CircuitParserError(RuntimeError):
    """Raised when a parser is available but cannot produce valid circuit IR."""


class CircuitParserUnavailable(RuntimeError):
    """Raised when a requested parser backend is not configured or importable."""


class UnrecognizedDemoPrompt(CircuitParserError):
    """Raised when deterministic demo recognition does not match a prompt."""


class CircuitParser(ABC):
    @abstractmethod
    def parse_text(self, prompt: str) -> dict[str, Any]:
        """Return schem_forge-compatible circuit IR."""

    @abstractmethod
    def parse_image(self, image_bytes: bytes, prompt: str | None = None) -> dict[str, Any]:
        """Return schem_forge-compatible circuit IR parsed from an image."""


class DeterministicDemoParser(CircuitParser):
    """Small no-network parser for product demos and local development."""

    def parse_text(self, prompt: str) -> dict[str, Any]:
        normalized = prompt.lower().replace("_", " ").replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)

        if "instrumentation amplifier" in normalized or "instrumentation amp" in normalized:
            return deepcopy(instrumentation_amp_ir())
        if "non inverting op amp" in normalized or "non inverting opamp" in normalized:
            return deepcopy(non_inverting_op_amp_ir())
        if "rc low pass" in normalized or "low pass" in normalized or "lowpass" in normalized:
            return deepcopy(rc_low_pass_ir())
        if "voltage divider" in normalized:
            return deepcopy(voltage_divider_ir())

        raise UnrecognizedDemoPrompt("Prompt did not match a deterministic demo circuit.")

    def parse_image(self, image_bytes: bytes, prompt: str | None = None) -> dict[str, Any]:
        raise CircuitParserUnavailable(
            "Image-to-circuit parsing requires GEMINI_API_KEY or GOOGLE_API_KEY."
        )


class GeminiCircuitParser(CircuitParser):
    """Optional Gemini parser that must return structured circuit IR, never SVG."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        load_local_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not self.api_key:
            raise CircuitParserUnavailable("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")

    @staticmethod
    def is_configured() -> bool:
        load_local_dotenv()
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    def parse_text(self, prompt: str) -> dict[str, Any]:
        instruction = (
            "Convert the user's circuit description into schem_forge circuit IR JSON only. "
            "Use component objects with id, type, role when useful, display_label/value_label "
            "when useful, and pins mapping pin names to net names. Do not include SVG, HTML, "
            "layout coordinates, markdown, or prose. Supported motifs include voltage_divider, "
            "rc_low_pass, non_inverting_op_amp, and instrumentation_amplifier. "
            f"User description: {prompt}"
        )
        text = self._generate_text(instruction)
        return self._extract_ir_json(text)

    def parse_image(self, image_bytes: bytes, prompt: str | None = None) -> dict[str, Any]:
        instruction = (
            "Read this circuit image and return schem_forge circuit IR JSON only. "
            "Use components with id, type, optional role, and pins mapping pin names to net names. "
            "Do not return SVG, HTML, coordinates, markdown, or prose."
        )
        if prompt:
            instruction += f" Additional user hint: {prompt}"
        text = self._generate_image_text(image_bytes, instruction)
        return self._extract_ir_json(text)

    def _client(self) -> Any:
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise CircuitParserUnavailable(
                "Gemini parsing requires the optional google-genai package."
            ) from exc
        return genai.Client(api_key=self.api_key)

    def _generate_text(self, instruction: str) -> str:
        client = self._client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=instruction,
            )
        except Exception as exc:  # pragma: no cover - network/client boundary
            raise CircuitParserError(f"Gemini text parsing failed: {exc}") from exc
        return str(getattr(response, "text", "") or "")

    def _generate_image_text(self, image_bytes: bytes, instruction: str) -> str:
        try:
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise CircuitParserUnavailable(
                "Gemini image parsing requires the optional google-genai package."
            ) from exc

        client = self._client()
        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            response = client.models.generate_content(
                model=self.model,
                contents=[instruction, image_part],
            )
        except Exception as exc:  # pragma: no cover - network/client boundary
            raise CircuitParserError(f"Gemini image parsing failed: {exc}") from exc
        return str(getattr(response, "text", "") or "")

    @staticmethod
    def _extract_ir_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1).strip()
        if not stripped.startswith("{"):
            object_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if object_match:
                stripped = object_match.group(0)

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CircuitParserError("Parser did not return valid JSON.") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("components"), list):
            raise CircuitParserError("Parser JSON must be an object with a components list.")
        if "svg" in payload or "artifact" in payload:
            raise CircuitParserError("Parser returned display output; expected circuit IR only.")
        return payload

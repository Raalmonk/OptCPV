from __future__ import annotations

import inspect
import json
import xml.etree.ElementTree as ET

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schem_forge.citt_examples import (
    citt_bme_instrumentation_amplifier_payload,
    citt_voltage_divider_payload,
)
from backend.app.schem_forge.examples import non_inverting_op_amp_ir, voltage_divider_ir


def _patch_testclient_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    original_init = httpx.Client.__init__
    if "app" in inspect.signature(original_init).parameters:
        return

    def compatible_init(self, *args, app=None, **kwargs):
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", compatible_init)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SCHEM_FORGE_SKIP_DOTENV", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    _patch_testclient_httpx(monkeypatch)
    return TestClient(app)


def _assert_ok_artifact(payload: dict) -> None:
    assert payload["status"] == "ok"
    assert payload["svg"].startswith("<svg")
    assert payload["artifact"]["svg"].startswith("<svg")
    assert payload["artifact"]["zoom_presets"]
    assert payload["artifact"]["focus_regions"]
    assert payload["artifact"]["components"]
    assert payload["artifact"]["nets"]
    assert payload["critic"]["fatal_count"] == 0
    assert "total_score" in payload["critic"]
    assert payload["input_ir"]["components"]
    assert json.dumps(payload["artifact"])
    assert ET.fromstring(payload["svg"]).tag.endswith("svg")
    assert ET.fromstring(payload["artifact"]["svg"]).tag.endswith("svg")


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "schem_forge"}


def test_from_ir_accepts_schem_forge_native_voltage_divider(client: TestClient) -> None:
    response = client.post(
        "/v1/schematic/from-ir",
        json={"input_format": "auto", "circuit": voltage_divider_ir()},
    )

    assert response.status_code == 200
    _assert_ok_artifact(response.json())


def test_from_ir_accepts_citt_style_voltage_divider(client: TestClient) -> None:
    response = client.post(
        "/v1/schematic/from-ir",
        json={"input_format": "auto", "circuit": citt_voltage_divider_payload()},
    )

    assert response.status_code == 200
    payload = response.json()
    _assert_ok_artifact(payload)
    assert payload["input_ir"]["motif"] == "voltage_divider"


def test_schematic_endpoint_uses_schem_forge_for_op_amp(client: TestClient) -> None:
    response = client.post("/schematic", json=non_inverting_op_amp_ir())

    assert response.status_code == 200
    payload = response.json()
    _assert_ok_artifact(payload)
    assert payload["input_ir"]["motif"] == "non_inverting_op_amp"
    assert "grid_fallback" not in " ".join(payload["warnings"])


def test_schematic_endpoint_uses_schem_forge_for_bme_template(client: TestClient) -> None:
    response = client.post(
        "/schematic",
        json={"input_format": "auto", "circuit": citt_bme_instrumentation_amplifier_payload()},
    )

    assert response.status_code == 200
    payload = response.json()
    _assert_ok_artifact(payload)
    assert payload["input_ir"]["motif"] == "instrumentation_amplifier"
    assert payload["critic"]["total_score"] == 0
    assert "grid_fallback" not in " ".join(payload["warnings"])


def test_schematic_endpoint_rejects_student_facing_grid_fallback(client: TestClient) -> None:
    response = client.post(
        "/schematic",
        json={
            "id": "unknown_demo",
            "components": [
                {
                    "id": "X1",
                    "type": "mystery_block",
                    "display_label": "X1",
                    "pins": {"a": "A", "b": "B"},
                },
                {
                    "id": "X2",
                    "type": "mystery_block",
                    "display_label": "X2",
                    "pins": {"a": "A", "b": "B"},
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["status"] == "unsupported_motif"
    assert "grid fallback" in response.json()["message"]


def test_from_text_recognizes_deterministic_voltage_divider(client: TestClient) -> None:
    response = client.post(
        "/v1/schematic/from-text",
        json={"prompt": "Draw a voltage divider"},
    )

    assert response.status_code == 200
    _assert_ok_artifact(response.json())


def test_from_text_unknown_prompt_without_gemini_returns_controlled_error(client: TestClient) -> None:
    response = client.post(
        "/v1/schematic/from-text",
        json={"prompt": "Draw my mysterious custom sensor circuit"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "needs_parser",
        "message": "Text parsing requires GEMINI_API_KEY or a recognized demo prompt.",
    }


def test_from_image_without_gemini_returns_controlled_unavailable(client: TestClient) -> None:
    response = client.post(
        "/v1/schematic/from-image",
        files={"file": ("schematic.png", b"not really an image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "vision_backend_unavailable",
        "message": "Image-to-circuit parsing requires GEMINI_API_KEY or GOOGLE_API_KEY.",
    }

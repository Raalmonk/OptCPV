import sys
import types as pytypes

from optcpv.critic import critique_parts
from optcpv.examples import instrumentation_amplifier
from optcpv.patch import LayoutPatch
from optcpv.planner import plan_layout
from optcpv.raster import rasterize_svg
from optcpv.renderer import render_svg_layers
from optcpv.vision_agent import GeminiVisionClient, HeuristicVisionClient, VisualFeedback, heuristic_feedback


def test_heuristic_feedback_matches_gemini_shape() -> None:
    circuit = instrumentation_amplifier()
    layout = plan_layout(circuit)
    layers = render_svg_layers(layout)
    report = critique_parts(circuit, layout, layers.final_svg, layers=layers).combined_report

    feedback = heuristic_feedback(circuit, layout, report)
    data = feedback.to_dict()

    assert isinstance(feedback, VisualFeedback)
    assert set(data) == {
        "pass",
        "score",
        "recognized_circuit",
        "topology_errors",
        "visual_errors",
        "label_errors",
        "specific_fixes",
    }
    assert data["specific_fixes"]


def test_heuristic_vision_client_records_feedback_and_patch() -> None:
    circuit = instrumentation_amplifier()
    layout = plan_layout(circuit)
    layers = render_svg_layers(layout)
    reports = critique_parts(circuit, layout, layers.final_svg, layers=layers)
    client = HeuristicVisionClient()

    patch = client.propose_patch(circuit, layout, layers.final_svg, rasterize_svg(layers.final_svg), reports.combined_report)

    assert isinstance(patch, LayoutPatch)
    assert client.last_feedback is not None
    assert client.last_feedback.specific_fixes


def test_gemini_vision_client_sends_png_and_parses_patch(monkeypatch) -> None:
    calls = {}

    class FakePart:
        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"data": data, "mime_type": mime_type}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            return pytypes.SimpleNamespace(text='{"move_label":[{"label_id":"label:U1","x":4.5,"y":2.0}]}')

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = pytypes.SimpleNamespace(Part=FakePart)
    fake_google = pytypes.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    circuit = instrumentation_amplifier()
    layout = plan_layout(circuit)
    layers = render_svg_layers(layout)
    report = critique_parts(circuit, layout, layers.final_svg, layers=layers).combined_report
    raster = rasterize_svg(layers.final_svg)

    patch = GeminiVisionClient(api_key="test-key").propose_patch(circuit, layout, layers.final_svg, raster, report)

    assert calls["model"] == "gemini-3.5-flash"
    assert calls["config"] == {"response_mime_type": "application/json"}
    assert len(calls["contents"]) == 2
    assert calls["contents"][1]["mime_type"] == "image/png"
    assert calls["contents"][1]["data"].startswith(b"\x89PNG")
    assert patch.move_label[0].label_id == "label:U1"

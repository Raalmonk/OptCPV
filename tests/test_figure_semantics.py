from optcpv.figure_semantics import (
    FigureSemanticChecker,
    FigureSemanticDraft,
    FigureSemanticInterpreter,
    FigureVisualEvidence,
    analyze_figure_semantics,
    build_figure_check_prompt,
    build_figure_identification_prompt,
)
from optcpv.textbook_surrogate import TextbookFigureInterpreter


def test_semantic_layer_identifies_plot_information_before_overlay() -> None:
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "p",
            "page": 1,
            "caption": "Figure p Frequency-response curves for catheter sensor system with and without bubbles.",
            "categories": ["waveform_or_plot"],
            "primary_category": "waveform_or_plot",
            "is_likely_circuit": False,
        }
    )

    check = analyze_figure_semantics(
        card=card,
        evidence=FigureVisualEvidence(width=960, height=480, plot_region_area_fraction=0.4, plot_region_count=1),
    )

    assert check.final_image_kind == "plot"
    assert check.overlay_allowed is False
    assert check.draft.plot_grammar.plot_type == "frequency_response"
    assert check.draft.plot_grammar.independent_variable == "frequency"
    assert "FIGURE_SEMANTIC_DRAFT" in check.draft.structured_text
    assert "FIGURE_SEMANTIC_CHECK" in check.structured_text


def test_semantic_layer_applies_emg_circuit_grammar() -> None:
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "e",
            "page": 1,
            "caption": "Figure e EMG circuit with electrode inputs, bandpass amplifier, full-wave rectifier, and integrator.",
            "categories": ["schematic_circuit", "bioelectric_recording"],
            "primary_category": "schematic_circuit",
            "is_likely_circuit": True,
        }
    )

    check = analyze_figure_semantics(
        card=card,
        evidence=FigureVisualEvidence(
            width=960,
            height=480,
            line_segment_count=12,
            component_kind_counts={"op_amp": 2, "resistor": 4, "capacitor": 2},
        ),
    )

    assert check.final_image_kind == "schematic_circuit"
    assert check.overlay_allowed is True
    assert check.draft.subject == "emg"
    assert check.draft.circuit_grammar.grammar_family == "emg_frontend"


def test_semantic_layer_writes_auditable_two_pass_prompts() -> None:
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "b",
            "page": 2,
            "caption": "Figure b Simplified block diagram of an EMG controller.",
            "categories": ["system_block_diagram"],
            "primary_category": "system_block_diagram",
            "is_likely_circuit": True,
        }
    )
    evidence = FigureVisualEvidence(width=640, height=320, line_segment_count=8)

    layer_one = build_figure_identification_prompt(card=card, evidence=evidence)
    check = analyze_figure_semantics(card=card, evidence=evidence)
    layer_two = build_figure_check_prompt(card=card, evidence=evidence, draft=check.draft)

    assert "BASIC_FIGURE_GRAMMAR" in layer_one
    assert "Do not infer R/C/L/op_amp" in layer_one
    assert "layer_one_output" in layer_two
    assert check.draft.layer_one_input == layer_one
    assert check.layer_two_input == layer_two
    assert check.final_image_kind == "block_diagram"
    assert check.overlay_allowed is True
    assert check.draft.circuit_grammar.grammar_family == "functional_block_diagram"


def test_semantic_layer_calls_two_gemini_shaped_passes() -> None:
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "x",
            "page": 3,
            "caption": "Figure x Frequency response of an amplifier.",
            "categories": ["waveform_or_plot"],
            "primary_category": "waveform_or_plot",
            "is_likely_circuit": False,
        }
    )
    client = _FakeFigureSemanticClient()

    check = analyze_figure_semantics(card=card, evidence=FigureVisualEvidence(plot_region_count=1), client=client)

    assert client.calls == ["identify", "check"]
    assert check.final_image_kind == "plot"
    assert "Frequency response" in client.identify_prompt
    assert "plot_or_waveform" in client.check_prompt


class _FakeFigureSemanticClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.identify_prompt = ""
        self.check_prompt = ""

    def identify(self, prompt, *, image_bytes=None, card=None, evidence=None):
        del image_bytes
        self.calls.append("identify")
        self.identify_prompt = prompt
        return FigureSemanticInterpreter().interpret(card=card, evidence=evidence).to_dict()

    def check(self, prompt, *, image_bytes=None, card=None, evidence=None, draft: FigureSemanticDraft | None = None):
        del image_bytes, card, evidence
        self.calls.append("check")
        self.check_prompt = prompt
        assert draft is not None
        return FigureSemanticChecker().check(draft).to_dict()

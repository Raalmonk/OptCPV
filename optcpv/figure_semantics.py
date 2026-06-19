"""Two-pass figure grammar and semantic checking for textbook images.

This module is the gate in front of image-backed overlays.  CV may offer
evidence such as line runs or candidate symbol shapes, but it is not allowed to
decide that a flowchart rectangle is a resistor.  The first pass identifies the
input/figure type and emits plot/circuit grammar.  The second pass checks that
decision before the overlay extractor is allowed to create wires or buttons.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol

from .textbook_surrogate import FigureStructureCard


DEFAULT_GEMINI_FIGURE_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FIGURE_FALLBACK_MODELS = ("gemini-2.5-flash",)

BASIC_FIGURE_GRAMMAR = """
BASIC_FIGURE_GRAMMAR
input_kind:
  - problem_statement_text: text-only exercise or request.
  - standalone_image: an image without surrounding problem context.
  - textbook_figure_crop: cropped textbook figure with caption/card metadata.
  - full_problem_with_figure: problem text and figure are both visible.
  - complete_page: full textbook page with body text and figures.
  - unknown: insufficient context.
figure_kind:
  - schematic_circuit: electrical schematic made of circuit symbols and nets.
  - block_diagram: labeled functional/process/control blocks, arrows, or signal-flow boxes.
  - plot_or_waveform: axes, curves, traces, spectra, Bode plots, ECG/EMG waveforms, calibration curves.
  - mixed_circuit_plot: a figure with both schematic primitives and plot panels.
  - anatomy_or_context: body/photo/instrument/context illustration, not an electrical graph.
  - table_or_text: mainly text/table/equations.
  - unknown_or_text: unknown or insufficient visual/semantic evidence.
plot_grammar:
  - plot_type, independent_variable, dependent_variable, signal_or_quantity, information.
circuit_grammar:
  - grammar_family, expected_blocks, supported_symbols, signal_flow, notes.
hard_rules:
  - Classify the figure before extracting nodes, wires, or components.
  - Do not convert flowchart/block-diagram rectangles into resistors, capacitors, inductors, or op-amps.
  - In block diagrams, rectangles are functional blocks/buttons; internal text is semantic evidence.
  - In plots/waveforms, axes and curves are data, not nets or components.
  - Only schematic_circuit, block_diagram, or mixed_circuit_plot with surviving circuit primitives may create overlay targets.
  - Anatomy/photos/context panels are external visual context unless an explicit schematic is present.
""".strip()


@dataclass(frozen=True)
class FigureVisualEvidence:
    width: int = 0
    height: int = 0
    dark_density: float = 0.0
    wire_density: float = 0.0
    plot_region_area_fraction: float = 0.0
    plot_region_count: int = 0
    line_segment_count: int = 0
    component_kind_counts: dict[str, int] = field(default_factory=dict)

    @property
    def real_component_count(self) -> int:
        return sum(
            int(self.component_kind_counts.get(kind, 0))
            for kind in ("op_amp", "resistor", "capacitor", "inductor")
        )

    @property
    def functional_region_count(self) -> int:
        return sum(
            int(self.component_kind_counts.get(kind, 0))
            for kind in ("functional_block", "functional_block_or_plot_region", "component_region")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dark_density": round(self.dark_density, 5),
            "wire_density": round(self.wire_density, 5),
            "plot_region_area_fraction": round(self.plot_region_area_fraction, 5),
            "plot_region_count": self.plot_region_count,
            "line_segment_count": self.line_segment_count,
            "component_kind_counts": dict(sorted(self.component_kind_counts.items())),
            "real_component_count": self.real_component_count,
            "functional_region_count": self.functional_region_count,
        }


@dataclass(frozen=True)
class PlotGrammar:
    plot_type: str = "none"
    independent_variable: str = "unknown"
    dependent_variable: str = "unknown"
    signal_or_quantity: str = "unknown"
    information: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "plot_type": self.plot_type,
            "independent_variable": self.independent_variable,
            "dependent_variable": self.dependent_variable,
            "signal_or_quantity": self.signal_or_quantity,
            "information": self.information,
        }


@dataclass(frozen=True)
class CircuitGrammar:
    grammar_family: str = "none"
    expected_blocks: tuple[str, ...] = ()
    supported_symbols: tuple[str, ...] = ()
    signal_flow: str = "unknown"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "grammar_family": self.grammar_family,
            "expected_blocks": list(self.expected_blocks),
            "supported_symbols": list(self.supported_symbols),
            "signal_flow": self.signal_flow,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FigureSemanticDraft:
    input_kind: str
    figure_kind: str
    subject: str
    plot_grammar: PlotGrammar
    circuit_grammar: CircuitGrammar
    overlay_policy: str
    confidence: float
    reasons: tuple[str, ...]
    visual_evidence: FigureVisualEvidence
    layer_one_input: str = ""
    raw_layer_one_output: dict[str, Any] = field(default_factory=dict)

    @property
    def structured_text(self) -> str:
        return "\n".join(
            [
                "FIGURE_SEMANTIC_DRAFT",
                f"input_kind: {self.input_kind}",
                f"figure_kind: {self.figure_kind}",
                f"subject: {self.subject}",
                f"overlay_policy: {self.overlay_policy}",
                f"confidence: {self.confidence:.3f}",
                f"reasons: {', '.join(self.reasons) or 'none'}",
                f"plot_grammar: {json.dumps(self.plot_grammar.to_dict(), sort_keys=True, separators=(',', ':'))}",
                f"circuit_grammar: {json.dumps(self.circuit_grammar.to_dict(), sort_keys=True, separators=(',', ':'))}",
                f"visual_evidence: {json.dumps(self.visual_evidence.to_dict(), sort_keys=True, separators=(',', ':'))}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_kind": self.input_kind,
            "figure_kind": self.figure_kind,
            "subject": self.subject,
            "plot_grammar": self.plot_grammar.to_dict(),
            "circuit_grammar": self.circuit_grammar.to_dict(),
            "overlay_policy": self.overlay_policy,
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
            "visual_evidence": self.visual_evidence.to_dict(),
            "layer_one_input": self.layer_one_input,
            "raw_layer_one_output": dict(self.raw_layer_one_output),
            "structured_text": self.structured_text,
        }


@dataclass(frozen=True)
class FigureSemanticCheck:
    final_image_kind: str
    overlay_allowed: bool
    circuit_confidence: float
    overlay_policy: str
    reject_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    draft: FigureSemanticDraft
    layer_two_input: str = ""
    raw_layer_two_output: dict[str, Any] = field(default_factory=dict)

    @property
    def structured_text(self) -> str:
        return "\n".join(
            [
                "FIGURE_SEMANTIC_CHECK",
                f"final_image_kind: {self.final_image_kind}",
                f"overlay_allowed: {str(self.overlay_allowed).lower()}",
                f"overlay_policy: {self.overlay_policy}",
                f"circuit_confidence: {self.circuit_confidence:.3f}",
                f"reject_reasons: {', '.join(self.reject_reasons) or 'none'}",
                f"warnings: {', '.join(self.warnings) or 'none'}",
                "check_contract: only circuit_diagram, block_diagram, or mixed_circuit_plot with remaining circuit primitives may create overlay targets",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_image_kind": self.final_image_kind,
            "overlay_allowed": self.overlay_allowed,
            "circuit_confidence": round(self.circuit_confidence, 3),
            "overlay_policy": self.overlay_policy,
            "reject_reasons": list(self.reject_reasons),
            "warnings": list(self.warnings),
            "draft": self.draft.to_dict(),
            "layer_one_input": self.draft.layer_one_input,
            "layer_one_output": self.draft.to_dict(),
            "layer_two_input": self.layer_two_input,
            "layer_two_output": dict(self.raw_layer_two_output),
            "structured_text": self.structured_text,
        }


class FigureSemanticClient(Protocol):
    """Boundary for real Gemini or the local two-pass surrogate."""

    def identify(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        card: FigureStructureCard | None = None,
        evidence: FigureVisualEvidence,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def check(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        card: FigureStructureCard | None = None,
        evidence: FigureVisualEvidence,
        draft: FigureSemanticDraft,
    ) -> dict[str, Any]:
        raise NotImplementedError


class LocalFigureSemanticClient:
    """Local deterministic stand-in for the two Gemini calls."""

    def identify(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        card: FigureStructureCard | None = None,
        evidence: FigureVisualEvidence,
    ) -> dict[str, Any]:
        del prompt, image_bytes
        return FigureSemanticInterpreter().interpret(card=card, evidence=evidence).to_dict()

    def check(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        card: FigureStructureCard | None = None,
        evidence: FigureVisualEvidence,
        draft: FigureSemanticDraft,
    ) -> dict[str, Any]:
        del prompt, image_bytes, card, evidence
        return FigureSemanticChecker().check(draft).to_dict()


class GeminiFigureSemanticClient:
    """Optional Gemini-backed figure semantics client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        fallback_models: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optcpv[vision] to use GeminiFigureSemanticClient.") from exc
        env_api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
        self._client = genai.Client(api_key=env_api_key) if env_api_key else genai.Client()
        self._model = model or os.getenv("OPTCPV_GEMINI_FIGURE_MODEL", DEFAULT_GEMINI_FIGURE_MODEL)
        self._fallback_models = tuple(fallback_models) if fallback_models is not None else _fallback_models_from_env()
        self._types = types

    def identify(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        card: FigureStructureCard | None = None,
        evidence: FigureVisualEvidence,
    ) -> dict[str, Any]:
        del card, evidence
        return self._json_response(prompt, image_bytes=image_bytes)

    def check(
        self,
        prompt: str,
        *,
        image_bytes: bytes | None = None,
        card: FigureStructureCard | None = None,
        evidence: FigureVisualEvidence,
        draft: FigureSemanticDraft,
    ) -> dict[str, Any]:
        del card, evidence, draft
        return self._json_response(prompt, image_bytes=image_bytes)

    def _json_response(self, prompt: str, *, image_bytes: bytes | None) -> dict[str, Any]:
        contents: list[Any] = [prompt]
        if image_bytes is not None:
            contents.append(self._types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        models = tuple(dict.fromkeys((self._model, *self._fallback_models)))
        last_error: Exception | None = None
        for model in models:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config={"response_mime_type": "application/json"},
                )
                return _json_object((getattr(response, "text", "") or "").strip())
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("No Gemini figure-semantic models were configured.")


def figure_semantic_client_from_env() -> FigureSemanticClient | None:
    """Return a real Gemini client only when explicitly enabled."""

    from .planning_agent import load_dotenv_if_present

    load_dotenv_if_present()
    client = _key(os.getenv("OPTCPV_FIGURE_SEMANTIC_CLIENT"))
    enabled = _truthy(os.getenv("OPTCPV_USE_GEMINI_FIGURE_SEMANTICS")) or client == "gemini"
    if not enabled:
        return None
    return GeminiFigureSemanticClient()


def analyze_figure_semantics(
    *,
    card: FigureStructureCard | None,
    evidence: FigureVisualEvidence,
    image_path: str | Path | None = None,
    client: FigureSemanticClient | None = None,
) -> FigureSemanticCheck:
    """Run the two-pass figure grammar gate and return the checked result."""

    selected_client = client or figure_semantic_client_from_env() or LocalFigureSemanticClient()
    image_bytes = _read_image_bytes(image_path)
    layer_one_input = build_figure_identification_prompt(card=card, evidence=evidence)
    raw_draft = selected_client.identify(layer_one_input, image_bytes=image_bytes, card=card, evidence=evidence)
    draft = _draft_from_dict(raw_draft, evidence=evidence)
    draft = replace(draft, layer_one_input=layer_one_input, raw_layer_one_output=dict(raw_draft))

    layer_two_input = build_figure_check_prompt(card=card, evidence=evidence, draft=draft)
    raw_check = selected_client.check(
        layer_two_input,
        image_bytes=image_bytes,
        card=card,
        evidence=evidence,
        draft=draft,
    )
    check = _check_from_dict(raw_check, draft=draft)
    return replace(check, layer_two_input=layer_two_input, raw_layer_two_output=dict(raw_check))


def build_figure_identification_prompt(
    *,
    card: FigureStructureCard | None,
    evidence: FigureVisualEvidence,
) -> str:
    payload = {
        "task": "Layer 1: identify input kind, figure kind, plot grammar, circuit grammar, and overlay policy. Return JSON only.",
        "role": "You are the semantic figure interpreter before any CV overlay.",
        "basic_figure_grammar": BASIC_FIGURE_GRAMMAR,
        "figure_card": card.to_dict() if card is not None else None,
        "visual_evidence": evidence.to_dict(),
        "output_schema": _layer_one_schema(),
        "instructions": [
            "Read the caption/text first; then use visual evidence only as support.",
            "If the caption says block diagram, flow chart, system, detector chain, logic chain, or controller blocks, classify as block_diagram unless a true schematic panel is also explicit.",
            "If the figure is axes, curves, ECG/EMG traces, Bode/frequency response, calibration curve, spectrum, or waveform, classify as plot_or_waveform and fill plot_grammar.",
            "Do not infer R/C/L/op_amp from ordinary rectangles, labels, arrows, axes, or plot traces.",
            "If schematic_circuit, fill circuit_grammar with expected electrical blocks and supported schematic symbols.",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_figure_check_prompt(
    *,
    card: FigureStructureCard | None,
    evidence: FigureVisualEvidence,
    draft: FigureSemanticDraft,
) -> str:
    payload = {
        "task": "Layer 2: check the Layer 1 figure semantics before overlay. Return JSON only.",
        "role": "You are the semantic checker. Reject contradictions and protect plots/block diagrams from component hallucination.",
        "basic_figure_grammar": BASIC_FIGURE_GRAMMAR,
        "figure_card": card.to_dict() if card is not None else None,
        "visual_evidence": evidence.to_dict(),
        "layer_one_output": draft.to_dict(),
        "output_schema": _layer_two_schema(),
        "checklist": [
            "Reject overlay for plot_or_waveform, anatomy_or_context, table_or_text, unknown_or_text.",
            "Allow block_diagram overlay only as functional block/arrows/buttons; never R/C/L/op_amp symbol buttons.",
            "Allow schematic_circuit overlay only when semantic text and visible primitives agree.",
            "If a schematic caption is contradicted by plot axes/curves and no real components remain, classify final_image_kind as plot.",
            "For mixed_circuit_plot, require remaining schematic primitives after plot masks.",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


class FigureSemanticInterpreter:
    """First Gemini-like pass: identify what the figure is before overlay."""

    def interpret(self, *, card: FigureStructureCard | None, evidence: FigureVisualEvidence) -> FigureSemanticDraft:
        caption = (card.caption if card else "").lower()
        categories = set(card.categories if card else ())
        primary = card.primary_category if card else ""
        cues = set(card.component_cues if card else ())
        families = set(card.visual_families if card else ())
        input_kind = _input_kind(card, caption)
        subject = _subject_from_text(caption, cues, families)
        plot_grammar = _plot_grammar_from_caption(caption)
        reasons: list[str] = []

        caption_plot = _caption_looks_plot_like(caption) or primary == "waveform_or_plot" or "waveform_or_plot" in categories
        caption_waveform = _caption_looks_waveform_like(caption)
        anatomy = "anatomy_or_physiology" in categories or "electrode_context" in families
        physical_context = anatomy or _caption_looks_physical_context(caption)
        table_or_text = primary in {"table", "text"} or "table" in categories
        category_block = (
            (primary == "system_block_diagram" or "system_block_diagram" in categories)
            and evidence.functional_region_count > 0
            and not caption_plot
            and not caption_waveform
            and not physical_context
        )
        caption_block = _caption_claims_block(caption) or category_block
        category_circuit = "schematic_circuit" in categories and not caption_plot and not physical_context and evidence.real_component_count > 0
        caption_circuit = _caption_claims_schematic_circuit(caption) or category_circuit
        photo_like = evidence.dark_density > 0.16 and evidence.line_segment_count == 0 and not caption_circuit and not caption_block

        if table_or_text:
            figure_kind = "table_or_text"
            reasons.append("card_describes_table_or_text")
        elif caption_block and not _caption_explicitly_says_circuit_panel(caption):
            figure_kind = "block_diagram"
            reasons.append("caption_or_card_describes_block_diagram")
        elif physical_context and not caption_circuit:
            figure_kind = "anatomy_or_context"
            reasons.append("physical_or_anatomy_context")
        elif caption_plot and caption_circuit and evidence.plot_region_area_fraction >= 0.08:
            figure_kind = "mixed_circuit_plot"
            reasons.append("caption_mentions_circuit_and_plot_region")
        elif caption_plot and not caption_circuit:
            figure_kind = "plot_or_waveform"
            reasons.append("caption_describes_plot_or_waveform")
        elif caption_waveform and evidence.real_component_count == 0 and not caption_circuit:
            figure_kind = "plot_or_waveform"
            reasons.append("caption_describes_signal_trace")
        elif photo_like:
            figure_kind = "anatomy_or_context"
            reasons.append("photo_or_panel_context")
        elif evidence.plot_region_area_fraction >= 0.08 and not caption_circuit:
            figure_kind = "plot_or_waveform"
            reasons.append("visual_plot_region_without_circuit_claim")
        elif caption_circuit:
            figure_kind = "schematic_circuit"
            reasons.append("caption_or_card_describes_schematic_circuit")
        elif anatomy:
            figure_kind = "anatomy_or_context"
            reasons.append("anatomy_or_photo_context")
        elif evidence.real_component_count >= 2 and evidence.line_segment_count >= 2:
            figure_kind = "schematic_circuit"
            reasons.append("visual_components_and_wires_support_circuit")
        elif evidence.plot_region_count > 0:
            figure_kind = "plot_or_waveform"
            reasons.append("visual_plot_region_detected")
        else:
            figure_kind = "unknown_or_text"
            reasons.append("insufficient_semantic_evidence")

        circuit_grammar = _circuit_grammar_from_context(caption, cues, families, evidence, figure_kind)
        overlay_policy, confidence = _draft_overlay_policy(figure_kind, caption, evidence, circuit_grammar)
        return FigureSemanticDraft(
            input_kind=input_kind,
            figure_kind=figure_kind,
            subject=subject,
            plot_grammar=plot_grammar,
            circuit_grammar=circuit_grammar,
            overlay_policy=overlay_policy,
            confidence=confidence,
            reasons=tuple(dict.fromkeys(reasons)),
            visual_evidence=evidence,
        )


class FigureSemanticChecker:
    """Second Gemini-like pass: check the draft before overlay targets exist."""

    def check(self, draft: FigureSemanticDraft) -> FigureSemanticCheck:
        warnings: list[str] = []
        reject_reasons: list[str] = []
        image_kind = "non_circuit"
        overlay_allowed = False
        confidence = draft.confidence

        if draft.figure_kind == "schematic_circuit":
            if draft.visual_evidence.line_segment_count == 0 and draft.visual_evidence.real_component_count == 0:
                reject_reasons.append("semantic_circuit_claim_without_visual_primitives")
                confidence = min(confidence, 0.32)
            else:
                image_kind = "schematic_circuit"
                overlay_allowed = True
        elif draft.figure_kind == "block_diagram":
            image_kind = "block_diagram"
            overlay_allowed = draft.visual_evidence.line_segment_count > 0 or draft.visual_evidence.functional_region_count > 0
            confidence = min(max(confidence, 0.52), 0.86)
            warnings.append("block_diagram_buttons_must_be_functional_blocks_not_rcl_or_opamps")
            if not overlay_allowed:
                reject_reasons.append("block_diagram_without_overlay_primitives")
        elif draft.figure_kind == "mixed_circuit_plot":
            if draft.visual_evidence.line_segment_count > 0 and draft.visual_evidence.real_component_count > 0:
                image_kind = "schematic_circuit"
                overlay_allowed = True
                warnings.append("plot_subregions_must_remain_masked")
            else:
                image_kind = "plot"
                reject_reasons.append("plot_mask_left_no_circuit_primitives")
                confidence = min(confidence, 0.39)
        elif draft.figure_kind == "plot_or_waveform":
            image_kind = "plot"
            reject_reasons.append("figure_semantics_is_plot_or_waveform")
            confidence = min(confidence, 0.39)
        elif draft.figure_kind == "anatomy_or_context":
            image_kind = "anatomy_or_context"
            reject_reasons.append("figure_semantics_is_anatomy_or_context")
            confidence = min(confidence, 0.39)
        else:
            image_kind = "non_circuit"
            reject_reasons.append("figure_semantics_is_unknown_or_text")
            confidence = min(confidence, 0.32)

        if overlay_allowed and draft.visual_evidence.plot_region_area_fraction >= 0.08:
            warnings.append("overlay_allowed_only_after_plot_region_mask")
        policy = "interactive_overlay" if overlay_allowed else "skip_non_circuit"
        return FigureSemanticCheck(
            final_image_kind=image_kind,
            overlay_allowed=overlay_allowed,
            circuit_confidence=max(0.0, min(0.98, confidence)),
            overlay_policy=policy,
            reject_reasons=tuple(dict.fromkeys((*draft.reasons, *reject_reasons))),
            warnings=tuple(dict.fromkeys(warnings)),
            draft=draft,
        )


def _draft_overlay_policy(
    figure_kind: str,
    caption: str,
    evidence: FigureVisualEvidence,
    circuit_grammar: CircuitGrammar,
) -> tuple[str, float]:
    if figure_kind in {"plot_or_waveform", "anatomy_or_context", "unknown_or_text", "table_or_text"}:
        return "skip_non_circuit", 0.18 if figure_kind in {"unknown_or_text", "table_or_text"} else 0.3
    if figure_kind == "block_diagram":
        confidence = 0.6
        if _caption_claims_block(caption):
            confidence += 0.12
        if evidence.line_segment_count >= 3:
            confidence += 0.08
        return "functional_block_overlay", min(0.86, confidence)
    if figure_kind == "mixed_circuit_plot":
        if evidence.line_segment_count > 0 and evidence.real_component_count > 0:
            return "partial_circuit_overlay", 0.72
        return "skip_non_circuit", 0.35
    confidence = 0.5
    if _caption_claims_schematic_circuit(caption):
        confidence += 0.18
    if evidence.line_segment_count >= 3:
        confidence += 0.14
    if evidence.real_component_count >= 2:
        confidence += 0.14
    if circuit_grammar.grammar_family not in {"generic_circuit", "none"}:
        confidence += 0.1
    return "interactive_overlay", min(0.98, confidence)


def _input_kind(card: FigureStructureCard | None, caption: str) -> str:
    if card is None:
        return "standalone_image"
    if _looks_like_problem_statement(caption):
        return "full_problem_with_figure"
    return "textbook_figure_crop"


def _subject_from_text(caption: str, cues: set[str], families: set[str]) -> str:
    text = " ".join((caption, *cues, *families))
    if any(token in text for token in ("emg", "electromyogram", "electromyography", "myoelectric")):
        return "emg"
    if any(token in text for token in ("ecg", "electrocardiograph", "electrocardiogram")):
        return "ecg"
    if any(token in text for token in ("eeg", "electroencephalogram")):
        return "eeg"
    if "voltage clamp" in text or "tevc" in text:
        return "voltage_clamp"
    if "pressure" in text:
        return "pressure_measurement"
    if "x-ray" in text or "radiograph" in text:
        return "medical_imaging"
    return "generic_biomedical_instrumentation"


def _plot_grammar_from_caption(caption: str) -> PlotGrammar:
    if not caption:
        return PlotGrammar()
    if any(token in caption for token in ("bode", "frequency response", "frequency-response", "gain versus frequency", "versus frequency")):
        return PlotGrammar("frequency_response", "frequency", "gain/phase/amplitude", "filter_or_amplifier_response", "Frequency-domain behavior, not circuit topology.")
    if any(token in caption for token in ("ecg", "emg", "waveform", "trace", "recording strip", "action potential")):
        return PlotGrammar("time_series_waveform", "time", "voltage/amplitude", "bioelectric_signal", "Signal morphology over time.")
    if any(token in caption for token in ("characteristic", "i-v", "current", "voltage", "threshold")):
        return PlotGrammar("characteristic_curve", "input or voltage", "output or current", "device_or_circuit_characteristic", "Transfer or device characteristic.")
    if any(token in caption for token in ("spectrum", "wavelength", "spectral")):
        return PlotGrammar("spectrum", "wavelength/frequency", "intensity/transmission", "optical_spectrum", "Spectral content or transmission.")
    if any(token in caption for token in ("impedance", "function of frequency")):
        return PlotGrammar("impedance_curve", "frequency", "impedance", "electrode_or_sensor_impedance", "Impedance behavior over frequency.")
    if any(token in caption for token in ("sensitivity", "linearity", "drift", "curve")):
        return PlotGrammar("calibration_curve", "input/condition", "output/sensitivity", "static_or_calibration_behavior", "Calibration or static response.")
    return PlotGrammar("plot_or_curve", "unknown", "unknown", "measured_or_simulated_quantity", "Plot semantics inferred from caption.")


def _circuit_grammar_from_context(
    caption: str,
    cues: set[str],
    families: set[str],
    evidence: FigureVisualEvidence,
    figure_kind: str,
) -> CircuitGrammar:
    text = " ".join((caption, *cues, *families))
    if figure_kind == "block_diagram":
        return CircuitGrammar(
            "functional_block_diagram",
            ("source_or_sensor", "processing_blocks", "controller_or_output"),
            ("functional_block", "arrow_or_signal_line"),
            "left-to-right or feedback block signal flow",
            ("Block rectangles are functional blocks; never classify them as R/C/L/op_amp symbols.",),
        )
    if figure_kind in {"plot_or_waveform", "anatomy_or_context", "table_or_text", "unknown_or_text"}:
        return CircuitGrammar("none", (), (), "none", ("No circuit grammar applies to this figure kind.",))
    symbols = [kind for kind, count in sorted(evidence.component_kind_counts.items()) if count > 0]
    notes: list[str] = []
    if "emg" in text or "electromyogram" in text:
        return CircuitGrammar(
            "emg_frontend",
            ("electrode_inputs", "high_gain_bandpass_amplifier", "rectifier_or_integrator_optional"),
            tuple(symbols),
            "electrodes -> gain/filter -> activity or output",
            ("EMG waveform plots are not circuit overlays unless an amplifier/rectifier/integrator circuit is present.",),
        )
    if "ecg" in text or "electrocardiograph" in text:
        return CircuitGrammar(
            "ecg_frontend",
            ("electrode_inputs", "instrumentation_amplifier", "filtering_or_isolation"),
            tuple(symbols),
            "body electrodes -> amplifier/filter -> recording/output",
            (),
        )
    if "op amp" in text or "op-amp" in text or "feedback" in text:
        return CircuitGrammar("op_amp_feedback", ("input_network", "op_amp", "feedback_network", "output"), tuple(symbols), "input -> op amp -> output with feedback loop", ())
    if "filter" in text:
        return CircuitGrammar("filter_circuit", ("input", "r_c_l_network", "output"), tuple(symbols), "input -> frequency-selective network -> output", ())
    if "bridge" in text or "wheatstone" in text:
        return CircuitGrammar("sensor_bridge", ("bridge_arms", "sensor_element", "differential_output"), tuple(symbols), "excitation -> bridge -> differential output", ())
    if symbols:
        notes.append("generic symbol grammar inferred from visible component candidates")
    return CircuitGrammar("generic_circuit", (), tuple(symbols), "unknown", tuple(notes))


def _caption_looks_plot_like(caption: str) -> bool:
    return any(
        token in caption
        for token in (
            "plot",
            "simulation",
            "response",
            "frequency",
            "bode",
            "characteristic",
            "threshold",
            "curve",
            "sensitivity",
            "linearity",
            "drift",
            "versus",
            "function of",
            "spectrum",
            "waveforms",
            "transient",
            "gain versus",
            "dependence",
            "relationship",
            "as a function",
        )
    )


def _caption_looks_waveform_like(caption: str) -> bool:
    return any(token in caption for token in ("ecg", "emg", "electrocardiogram", "electromyogram", "waveform", "trace", "distorted", "recording strip"))


def _caption_claims_schematic_circuit(caption: str) -> bool:
    return any(
        token in caption
        for token in (
            "circuit design",
            "circuit diagram",
            "circuit with",
            "ecg circuit",
            "emg circuit",
            "amplifier circuit",
            "filter circuit",
            "bridge circuit",
            "schematic",
            "equivalent electrical",
            "electrical model",
            "circuit model",
            "op amp",
            "op-amp",
            "rectifier",
            "wheatstone",
            "protection scheme",
        )
    )


def _caption_claims_block(caption: str) -> bool:
    return any(token in caption for token in ("block diagram", "flow chart", "flowchart", "signal chain", "logic diagram", "system diagram", "controller"))


def _caption_explicitly_says_circuit_panel(caption: str) -> bool:
    return bool(re.search(r"\b(circuit|schematic)\b", caption)) and not _caption_claims_block(caption)


def _looks_like_problem_statement(caption: str) -> bool:
    return bool(re.search(r"\b(draw|design|calculate|what is|find|assume|given|derive|show)\b", caption))


def _caption_looks_physical_context(caption: str) -> bool:
    return any(
        token in caption
        for token in (
            "patient",
            "skin",
            "artery",
            "arterial",
            "tonometer",
            "probe",
            "microelectrode",
            "electrode fabricated",
            "silicon",
            "magnetic-field pickup",
            "lead wires",
            "mechanical",
            "surface of skin",
            "body",
        )
    )


def _draft_from_dict(raw: dict[str, Any], *, evidence: FigureVisualEvidence) -> FigureSemanticDraft:
    plot = _plot_from_dict(_dict(raw.get("plot_grammar")))
    circuit = _circuit_from_dict(_dict(raw.get("circuit_grammar")))
    visual_raw = _dict(raw.get("visual_evidence"))
    visual = _visual_from_dict(visual_raw) if visual_raw else evidence
    return FigureSemanticDraft(
        input_kind=str(raw.get("input_kind") or "unknown"),
        figure_kind=str(raw.get("figure_kind") or "unknown_or_text"),
        subject=str(raw.get("subject") or "generic_biomedical_instrumentation"),
        plot_grammar=plot,
        circuit_grammar=circuit,
        overlay_policy=str(raw.get("overlay_policy") or "skip_non_circuit"),
        confidence=float(raw.get("confidence") or 0.0),
        reasons=tuple(str(item) for item in raw.get("reasons") or ()),
        visual_evidence=visual,
    )


def _check_from_dict(raw: dict[str, Any], *, draft: FigureSemanticDraft) -> FigureSemanticCheck:
    return FigureSemanticCheck(
        final_image_kind=str(raw.get("final_image_kind") or "non_circuit"),
        overlay_allowed=bool(raw.get("overlay_allowed", False)),
        circuit_confidence=float(raw.get("circuit_confidence") or 0.0),
        overlay_policy=str(raw.get("overlay_policy") or "skip_non_circuit"),
        reject_reasons=tuple(str(item) for item in raw.get("reject_reasons") or ()),
        warnings=tuple(str(item) for item in raw.get("warnings") or ()),
        draft=draft,
    )


def _plot_from_dict(raw: dict[str, Any]) -> PlotGrammar:
    return PlotGrammar(
        plot_type=str(raw.get("plot_type") or "none"),
        independent_variable=str(raw.get("independent_variable") or "unknown"),
        dependent_variable=str(raw.get("dependent_variable") or "unknown"),
        signal_or_quantity=str(raw.get("signal_or_quantity") or "unknown"),
        information=str(raw.get("information") or ""),
    )


def _circuit_from_dict(raw: dict[str, Any]) -> CircuitGrammar:
    return CircuitGrammar(
        grammar_family=str(raw.get("grammar_family") or "none"),
        expected_blocks=tuple(str(item) for item in raw.get("expected_blocks") or ()),
        supported_symbols=tuple(str(item) for item in raw.get("supported_symbols") or ()),
        signal_flow=str(raw.get("signal_flow") or "unknown"),
        notes=tuple(str(item) for item in raw.get("notes") or ()),
    )


def _visual_from_dict(raw: dict[str, Any]) -> FigureVisualEvidence:
    return FigureVisualEvidence(
        width=int(raw.get("width") or 0),
        height=int(raw.get("height") or 0),
        dark_density=float(raw.get("dark_density") or 0.0),
        wire_density=float(raw.get("wire_density") or 0.0),
        plot_region_area_fraction=float(raw.get("plot_region_area_fraction") or 0.0),
        plot_region_count=int(raw.get("plot_region_count") or 0),
        line_segment_count=int(raw.get("line_segment_count") or 0),
        component_kind_counts={str(key): int(value) for key, value in _dict(raw.get("component_kind_counts")).items()},
    )


def _layer_one_schema() -> dict[str, Any]:
    return {
        "input_kind": "problem_statement_text|standalone_image|textbook_figure_crop|full_problem_with_figure|complete_page|unknown",
        "figure_kind": "schematic_circuit|block_diagram|plot_or_waveform|mixed_circuit_plot|anatomy_or_context|table_or_text|unknown_or_text",
        "subject": "short domain label such as ecg, emg, voltage_clamp, pressure_measurement",
        "plot_grammar": PlotGrammar().to_dict(),
        "circuit_grammar": CircuitGrammar().to_dict(),
        "overlay_policy": "interactive_overlay|functional_block_overlay|partial_circuit_overlay|skip_non_circuit",
        "confidence": "number 0..1",
        "reasons": ["short evidence-backed reasons"],
        "visual_evidence": FigureVisualEvidence().to_dict(),
    }


def _layer_two_schema() -> dict[str, Any]:
    return {
        "final_image_kind": "schematic_circuit|block_diagram|plot|anatomy_or_context|non_circuit",
        "overlay_allowed": "boolean",
        "circuit_confidence": "number 0..1",
        "overlay_policy": "interactive_overlay|skip_non_circuit",
        "reject_reasons": ["short reasons"],
        "warnings": ["short warnings"],
    }


def _read_image_bytes(image_path: str | Path | None) -> bytes | None:
    if image_path is None:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    return path.read_bytes()


def _json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("GeminiFigureSemanticClient returned an empty response.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("GeminiFigureSemanticClient must return a JSON object.")
    return value


def _fallback_models_from_env() -> tuple[str, ...]:
    raw = os.getenv("OPTCPV_GEMINI_FIGURE_FALLBACK_MODELS", "")
    if not raw.strip():
        return DEFAULT_GEMINI_FIGURE_FALLBACK_MODELS
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _key(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")


def _truthy(value: str | None) -> bool:
    return _key(value) in {"1", "true", "yes", "on", "gemini"}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "BASIC_FIGURE_GRAMMAR",
    "CircuitGrammar",
    "DEFAULT_GEMINI_FIGURE_MODEL",
    "FigureSemanticCheck",
    "FigureSemanticChecker",
    "FigureSemanticClient",
    "FigureSemanticDraft",
    "FigureSemanticInterpreter",
    "FigureVisualEvidence",
    "GeminiFigureSemanticClient",
    "LocalFigureSemanticClient",
    "PlotGrammar",
    "analyze_figure_semantics",
    "build_figure_check_prompt",
    "build_figure_identification_prompt",
    "figure_semantic_client_from_env",
]

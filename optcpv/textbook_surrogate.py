"""Local textbook-backed Gemini middle layer.

This module plays two Gemini-like roles without spending model quota:

1. TextbookFigureInterpreter turns extracted textbook figure records into
   structured visual grammar cards.
2. TextbookStructurePlanner retrieves those cards for a circuit and emits
   topology-safe planning hints.

The output is intentionally structured text plus OptCPV hint objects. Real
Gemini clients can receive the same text later, while the local optimizer can
use the hints immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

from .grammar import infer_schematic_intent, planning_hints_from_intent
from .models import Circuit, Component, CriticReport, LayoutPlan
from .planning_agent import SemanticPlanningClient
from .planning_hints import GridPlacementHint, RoutePolicyHint, SchematicLayoutHints


DEFAULT_TEXTBOOK_CORPUS_DIR = "textbook_circuit_corpus"


@dataclass(frozen=True)
class FigureStructureCard:
    figure_id: str
    page: int
    image: str | None
    caption: str
    primary_category: str
    categories: tuple[str, ...]
    is_likely_circuit: bool
    visual_families: tuple[str, ...]
    component_cues: tuple[str, ...]
    layout_principles: tuple[str, ...]
    route_principles: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def token_text(self) -> str:
        return " ".join(
            [
                self.figure_id,
                self.caption,
                self.primary_category,
                *self.categories,
                *self.visual_families,
                *self.component_cues,
                *self.layout_principles,
                *self.route_principles,
            ]
        )

    @property
    def structured_text(self) -> str:
        lines = [
            f"FIGURE_CARD {self.figure_id} page={self.page}",
            f"caption: {_one_line(self.caption)}",
            f"category: {self.primary_category} likely_circuit={str(self.is_likely_circuit).lower()}",
            f"visual_families: {', '.join(self.visual_families) or 'generic'}",
            f"component_cues: {', '.join(self.component_cues) or 'none'}",
            f"layout_principles: {'; '.join(self.layout_principles) or 'standard left-to-right schematic'}",
            f"route_principles: {'; '.join(self.route_principles) or 'orthogonal textbook routing'}",
        ]
        if self.metrics:
            lines.append(f"image_metrics: {json.dumps(self.metrics, sort_keys=True, separators=(',', ':'))}")
        if self.image:
            lines.append(f"image: {self.image}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "page": self.page,
            "image": self.image,
            "caption": self.caption,
            "primary_category": self.primary_category,
            "categories": list(self.categories),
            "is_likely_circuit": self.is_likely_circuit,
            "visual_families": list(self.visual_families),
            "component_cues": list(self.component_cues),
            "layout_principles": list(self.layout_principles),
            "route_principles": list(self.route_principles),
            "metrics": dict(self.metrics),
            "structured_text": self.structured_text,
        }


class TextbookFigureInterpreter:
    """Convert figure records and image metrics into structured grammar cards."""

    def card_from_record(self, record: dict[str, Any], corpus_dir: Path | None = None) -> FigureStructureCard:
        caption = str(record.get("caption", ""))
        categories = tuple(str(item) for item in record.get("categories", ()))
        image = _optional_str(record.get("image"))
        metrics = _image_metrics(corpus_dir / image) if corpus_dir is not None and image else {}
        component_cues = _component_cues(caption, categories)
        visual_families = _visual_families(caption, categories, component_cues)
        return FigureStructureCard(
            figure_id=str(record.get("figure_id", "")),
            page=int(record.get("page", 0) or 0),
            image=image,
            caption=caption,
            primary_category=str(record.get("primary_category", "")),
            categories=categories,
            is_likely_circuit=bool(record.get("is_likely_circuit", False)),
            visual_families=visual_families,
            component_cues=component_cues,
            layout_principles=_layout_principles(visual_families, component_cues, metrics),
            route_principles=_route_principles(visual_families, component_cues),
            metrics=metrics,
        )


class TextbookCorpusIndex:
    """Load and retrieve structured cards from the extracted textbook corpus."""

    def __init__(self, corpus_dir: str | Path = DEFAULT_TEXTBOOK_CORPUS_DIR, *, include_all_figures: bool = True) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.include_all_figures = include_all_figures
        self.interpreter = TextbookFigureInterpreter()
        self.cards = self._load_cards()

    @classmethod
    def from_env(cls) -> "TextbookCorpusIndex":
        return cls(os.getenv("OPTCPV_TEXTBOOK_CORPUS_DIR", DEFAULT_TEXTBOOK_CORPUS_DIR))

    def retrieve(self, circuit: Circuit, *, limit: int = 8, current_report: CriticReport | None = None) -> list[FigureStructureCard]:
        query_tokens = _circuit_tokens(circuit)
        if current_report is not None:
            query_tokens.update(_report_tokens(current_report))
        scored = [(self._score(card, query_tokens), card) for card in self.cards]
        scored = [(score, card) for score, card in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], item[1].page, item[1].figure_id))
        return [card for _, card in scored[:limit]]

    def style_guide(self, cards: list[FigureStructureCard] | None = None) -> dict[str, Any]:
        selected = cards if cards is not None else self.cards
        family_counts: dict[str, int] = {}
        component_counts: dict[str, int] = {}
        for card in selected:
            for family in card.visual_families:
                family_counts[family] = family_counts.get(family, 0) + 1
            for cue in card.component_cues:
                component_counts[cue] = component_counts.get(cue, 0) + 1
        return {
            "source": "textbook_corpus",
            "card_count": len(selected),
            "family_counts": dict(sorted(family_counts.items())),
            "component_counts": dict(sorted(component_counts.items())),
            "global_principles": [
                "Use clean black-and-white textbook line art.",
                "Keep electrical signal flow left-to-right unless a feedback loop intentionally returns around the outside.",
                "Use local ground, supply, and reference symbols instead of long global rails.",
                "Route op-amp feedback outside the triangle body.",
                "Keep biological/anatomical artwork as external context; do not create electrical components unless the netlist contains them.",
                "Prefer compact subfigures with readable labels over decorative layout.",
            ],
        }

    def _load_cards(self) -> list[FigureStructureCard]:
        index_name = "figures.jsonl" if self.include_all_figures else "likely_circuit_figures.jsonl"
        index_path = self.corpus_dir / "indexes" / index_name
        if not index_path.exists():
            return []
        cards = []
        for record in _read_jsonl(index_path):
            cards.append(self.interpreter.card_from_record(record, self.corpus_dir))
        return cards

    def _score(self, card: FigureStructureCard, query_tokens: set[str]) -> float:
        card_tokens = _tokens(card.token_text)
        overlap = len(query_tokens & card_tokens)
        score = float(overlap)
        if card.is_likely_circuit:
            score += 1.5
        if "schematic" in query_tokens and "schematic_circuit" in card.categories:
            score += 1.0
        if "op_amp" in query_tokens and "op_amp" in card.component_cues:
            score += 3.0
        if {"ecg", "electrode"} & query_tokens and {"ecg", "electrode"} & set(card.component_cues):
            score += 2.5
        if {"emg", "electromyogram", "electromyography"} & query_tokens and (
            "emg" in card.component_cues or "emg_frontend" in card.visual_families
        ):
            score += 3.2
        if {"right_leg_drive", "rld"} & query_tokens and "right_leg_drive" in card.visual_families:
            score += 4.0
        if {"voltage_clamp", "tevc"} & query_tokens and "voltage_clamp" in card.visual_families:
            score += 4.0
        if {"diode", "protection"} & query_tokens and {"diode", "protection"} & set(card.component_cues):
            score += 2.5
        return score


class TextbookStructurePlanner:
    """Turn retrieved textbook structure cards into topology-safe hints."""

    def propose(
        self,
        circuit: Circuit,
        cards: list[FigureStructureCard],
        *,
        current_layout: LayoutPlan | None = None,
        current_report: CriticReport | None = None,
    ) -> SchematicLayoutHints:
        base = _base_hints(circuit)
        card_families = {family for card in cards for family in card.visual_families}
        card_cues = {cue for card in cards for cue in card.component_cues}
        placements = list(base.placements)
        placements = _apply_textbook_placement_overrides(circuit, placements, card_families, card_cues)
        route_policies = list(base.route_policies)
        route_policies.extend(_textbook_route_policies(circuit, card_families, card_cues, current_report))
        local_terminal_policy = dict(base.local_terminal_policy)
        local_terminal_policy.update(_local_terminal_policy(circuit))
        return base.with_updates(
            recognized_topology=_recognized_topology(circuit, cards, base),
            confidence=max(base.confidence, 0.72 if cards else 0.55),
            tutor_explanation=_tutor_explanation(cards),
            source="textbook_surrogate",
            placements=tuple(_dedupe_placements(placements)),
            route_policies=tuple(_dedupe_route_policies(route_policies)),
            local_terminal_policy=local_terminal_policy,
        )


class TextbookSurrogatePlanningClient(SemanticPlanningClient):
    """Local two-layer Gemini surrogate using extracted textbook figure cards."""

    def __init__(self, corpus_dir: str | Path | None = None, *, retrieval_limit: int = 8) -> None:
        self.corpus = TextbookCorpusIndex(corpus_dir or os.getenv("OPTCPV_TEXTBOOK_CORPUS_DIR", DEFAULT_TEXTBOOK_CORPUS_DIR))
        self.planner = TextbookStructurePlanner()
        self.retrieval_limit = retrieval_limit
        self.last_cards: list[FigureStructureCard] = []
        self.last_middle_layer_text = ""

    def propose_hints(self, circuit: Circuit, reference_image: bytes | None = None) -> SchematicLayoutHints:
        cards = self.corpus.retrieve(circuit, limit=self.retrieval_limit)
        self.last_cards = cards
        self.last_middle_layer_text = self._middle_layer_text(circuit, cards, mode="initial")
        return self.planner.propose(circuit, cards)

    def refine_hints(
        self,
        circuit: Circuit,
        layout: LayoutPlan,
        svg: str,
        critic_report: CriticReport,
        reference_image: bytes | None = None,
    ) -> SchematicLayoutHints | None:
        cards = self.corpus.retrieve(circuit, limit=self.retrieval_limit, current_report=critic_report)
        self.last_cards = cards
        self.last_middle_layer_text = self._middle_layer_text(
            circuit,
            cards,
            mode="refinement",
            critic_report=critic_report,
        )
        return self.planner.propose(circuit, cards, current_layout=layout, current_report=critic_report)

    def _middle_layer_text(
        self,
        circuit: Circuit,
        cards: list[FigureStructureCard],
        *,
        mode: str,
        critic_report: CriticReport | None = None,
    ) -> str:
        payload = [
            f"GEMINI_MIDDLE_LAYER mode={mode}",
            f"circuit_id: {circuit.id}",
            f"circuit_tokens: {', '.join(sorted(_circuit_tokens(circuit))) or 'none'}",
            "style_guide:",
            json.dumps(self.corpus.style_guide(cards), indent=2, sort_keys=True),
            "retrieved_figure_cards:",
        ]
        payload.extend(card.structured_text for card in cards)
        if critic_report is not None:
            payload.extend(
                [
                    "local_critic_report:",
                    json.dumps(critic_report.to_dict(), indent=2, sort_keys=True),
                ]
            )
        payload.append("planner_instruction: emit only legal OptCPV SchematicLayoutHints using existing component ids and nets.")
        return "\n\n".join(payload)


def build_structured_cards(corpus_dir: str | Path = DEFAULT_TEXTBOOK_CORPUS_DIR, *, include_all_figures: bool = True) -> list[dict[str, Any]]:
    """Return serializable structured cards for offline corpus generation."""

    return [card.to_dict() for card in TextbookCorpusIndex(corpus_dir, include_all_figures=include_all_figures).cards]


def _base_hints(circuit: Circuit) -> SchematicLayoutHints:
    intent = infer_schematic_intent(circuit)
    return planning_hints_from_intent(circuit, intent) or SchematicLayoutHints(
        recognized_topology=circuit.motif or circuit.id,
        confidence=0.4,
        source="textbook_surrogate",
    )


def _apply_textbook_placement_overrides(
    circuit: Circuit,
    placements: list[GridPlacementHint],
    card_families: set[str],
    card_cues: set[str],
) -> list[GridPlacementHint]:
    by_id = {placement.component_id: placement for placement in placements}
    opamps = [component for component in circuit.components if _is_opamp(component)]
    inputs = [component for component in circuit.components if _is_input(component)]
    outputs = [component for component in circuit.components if _is_output(component)]
    if len(opamps) >= 3 and ("multi_opamp_stage" in card_families or "instrumentation_amplifier" in card_families):
        for index, opamp in enumerate(opamps):
            lane = -1 if index == 0 and len(opamps) > 1 else 1 if index == 1 and len(opamps) > 1 else 0
            by_id[opamp.id] = _placement_for(by_id, opamp, 2 + index // 2, lane, "RIGHT", "textbook_opamp_stage")
        for index, component in enumerate(inputs):
            lane = -1 if index == 0 else 1 if index == 1 else index
            by_id[component.id] = _placement_for(by_id, component, 0, lane, "RIGHT", "left_input_port")
        for component in outputs:
            by_id[component.id] = _placement_for(by_id, component, 4 + max(0, len(opamps) // 2), 0, "RIGHT", "right_output_port")
    if "right_leg_drive" in card_families or any("right_leg" in _identity(component) or "rld" in _identity(component) for component in circuit.components):
        bottom_lane = 5
        if len(opamps) <= 2:
            for component in opamps:
                identity = _identity(component)
                net_text = _key(" ".join(component.pins.values()))
                if any(token in f"{identity} {net_text}" for token in ("right_leg", "rld", "aux", "common_mode")):
                    continue
                stage = by_id.get(component.id, GridPlacementHint(component.id, 1, 0)).stage_x
                by_id[component.id] = _placement_for(by_id, component, stage, 0, "RIGHT", "main_signal_stage")
        common_mode_sense = [
            component
            for component in circuit.components
            if _is_common_mode_sense(component)
        ]
        for index, component in enumerate(common_mode_sense):
            lane = -5 if index == 0 else 3 if index == 1 else index + 2
            by_id[component.id] = _placement_for(by_id, component, 1, lane, "DOWN", "common_mode_sense")
        for component in circuit.components:
            if _is_common_mode_sense(component):
                continue
            identity = _identity(component)
            net_text = _key(" ".join(component.pins.values()))
            if any(token in f"{identity} {net_text}" for token in ("right_leg", "rld", "aux", "common_mode")):
                stage = by_id.get(component.id, GridPlacementHint(component.id, 3, bottom_lane)).stage_x
                by_id[component.id] = _placement_for(by_id, component, stage, bottom_lane, "RIGHT", "bottom_auxiliary_loop")
        for component in circuit.components:
            if _is_ground(component):
                stage = max(2, by_id.get(component.id, GridPlacementHint(component.id, 2, bottom_lane + 2)).stage_x)
                by_id[component.id] = _placement_for(by_id, component, stage, bottom_lane + 2, "RIGHT", "local_reference_symbol")
    if "electrode_context" in card_families:
        for component in inputs:
            by_id[component.id] = _placement_for(by_id, component, 0, by_id.get(component.id, GridPlacementHint(component.id, 0, 0)).lane_y, "RIGHT", "electrode_or_body_context")
    return list(by_id.values())


def _textbook_route_policies(
    circuit: Circuit,
    card_families: set[str],
    card_cues: set[str],
    current_report: CriticReport | None,
) -> list[RoutePolicyHint]:
    policies: list[RoutePolicyHint] = []
    for component in circuit.components:
        if _is_opamp(component):
            output_net = _opamp_output_net(component)
            if output_net and _has_feedback_to(component, output_net, circuit.components):
                policies.append(RoutePolicyHint(net=output_net, net_role="feedback", policy="top_feedback_corridor"))
        identity = _identity(component)
        if any(token in identity for token in ("feedback", "fb", "rf")):
            for net in component.pins.values():
                if not _is_local_net(net):
                    policies.append(RoutePolicyHint(net=net, net_role="feedback", policy="top_feedback_corridor"))
        if any(token in identity for token in ("right_leg", "rld", "aux")):
            for net in component.pins.values():
                if not _is_local_net(net):
                    policies.append(RoutePolicyHint(net=net, net_role="right_leg_drive", policy="bottom_auxiliary_corridor"))
    if "right_leg_drive" in card_families:
        for net in _all_nets(circuit):
            net_key = _key(net)
            if "electrode" in net_key:
                continue
            if any(token in net_key for token in ("right_leg", "rld", "common_mode", "cm")):
                policies.append(RoutePolicyHint(net=net, net_role="right_leg_drive", policy="bottom_auxiliary_corridor"))
    if current_report is not None:
        for violation in current_report.violations:
            if violation.code in {"feedback_crosses_opamp_body", "wire_through_component"} and violation.subject:
                net = violation.subject.split(":", 1)[0]
                if net in _all_nets(circuit) and not _is_local_net(net):
                    policies.append(RoutePolicyHint(net=net, net_role="feedback", policy="top_feedback_corridor"))
    return policies


def _local_terminal_policy(circuit: Circuit) -> dict[str, str]:
    return {net: "local_symbol_only" for net in _all_nets(circuit) if _is_local_net(net)}


def _recognized_topology(circuit: Circuit, cards: list[FigureStructureCard], base: SchematicLayoutHints) -> str:
    if cards:
        families = list(dict.fromkeys(family for card in cards[:4] for family in card.visual_families))
        if families:
            return f"textbook_surrogate:{','.join(families[:4])}"
    return base.recognized_topology or circuit.motif or circuit.id


def _tutor_explanation(cards: list[FigureStructureCard]) -> str:
    if not cards:
        return "Textbook surrogate used deterministic schematic grammar without corpus matches."
    refs = ", ".join(f"{card.figure_id} p.{card.page}" for card in cards[:4])
    return f"Textbook surrogate retrieved figure grammar from {refs} and converted it into topology-safe stage, lane, and route hints."


def _component_cues(caption: str, categories: tuple[str, ...]) -> tuple[str, ...]:
    text = _key(" ".join((caption, *categories)))
    cues = []
    mapping = {
        "op_amp": ("op_amp", "opamp", "one_amp", "amplifier", "instrumentation"),
        "feedback": ("feedback", "gain", "cmrr", "common_mode"),
        "voltage_clamp": ("voltage clamp", "voltage_clamp", "two-electrode voltage clamp", "two_electrode_voltage_clamp", "tevc", "command voltage"),
        "electrode": ("electrode", "lead", "patient", "skin"),
        "ecg": ("ecg", "electrocardiograph", "electrocardiogram"),
        "emg": ("emg", "electromyogram", "electromyography", "myoelectric", "muscle activity", "muscular contraction"),
        "diode": ("diode", "zener", "rectifier", "limiting"),
        "capacitor": ("capacitor", "capacitance", "filter"),
        "resistor": ("resistor", "resistance", "impedance"),
        "sensor": ("sensor", "transducer", "photodiode", "thermistor", "strain"),
        "bridge": ("bridge", "wheatstone"),
        "protection": ("protection", "transient", "defibrillation", "isolation"),
        "block": ("block", "system", "processor", "display"),
    }
    for cue, keywords in mapping.items():
        if any(_key(keyword) in text for keyword in keywords):
            cues.append(cue)
    return tuple(cues)


def _visual_families(caption: str, categories: tuple[str, ...], component_cues: tuple[str, ...]) -> tuple[str, ...]:
    text = _key(" ".join((caption, *categories, *component_cues)))
    families = []
    checks = {
        "single_opamp_feedback": ("inverting", "noninverting", "one_op_amp", "one_amp", "feedback"),
        "multi_opamp_stage": ("two additional op amps", "three_op_amp", "instrumentation", "multiple op", "multi"),
        "instrumentation_amplifier": ("differential amplifier", "instrumentation amplifier", "cmrr"),
        "voltage_clamp": ("voltage clamp", "voltage_clamp", "two-electrode voltage clamp", "two_electrode_voltage_clamp", "2-electrode voltage clamp", "tevc", "command voltage"),
        "right_leg_drive": ("right_leg", "driven_right_leg", "common_mode interference"),
        "electrode_context": ("electrode", "patient", "skin", "body"),
        "protection_clamp": ("protection", "voltage_limiting", "diode", "zener", "defibrillation"),
        "isolation_barrier": ("isolation", "barrier", "isolator", "transformer"),
        "sensor_bridge": ("bridge", "wheatstone", "strain"),
        "active_filter": ("filter", "bandpass", "low_pass", "high_pass"),
        "current_source": ("current_source", "howland", "constant current"),
        "system_block": ("block diagram", "system_block_diagram"),
        "emg_frontend": ("emg", "electromyogram", "electromyography", "myoelectric", "muscle activity"),
        "bioelectric_recording": ("ecg", "eeg", "emg", "biopotential", "bioelectric_recording"),
    }
    for family, keywords in checks.items():
        if any(_key(keyword) in text for keyword in keywords):
            families.append(family)
    if "schematic_circuit" in categories and not families:
        families.append("generic_textbook_schematic")
    return tuple(families)


def _layout_principles(visual_families: tuple[str, ...], component_cues: tuple[str, ...], metrics: dict[str, Any]) -> tuple[str, ...]:
    principles = ["left-to-right signal flow", "right-facing op-amp triangles", "labels close to their owning symbols"]
    if "multi_opamp_stage" in visual_families or "instrumentation_amplifier" in visual_families:
        principles.extend(["parallel input stages aligned in upper/lower lanes", "difference or output stage to the right"])
    if "electrode_context" in visual_families:
        principles.append("body or electrode context stays at the far left as an external source context")
    if "emg_frontend" in visual_families:
        principles.append("EMG front ends use electrode inputs followed by high-gain bandpass filtering and rectification/integration when activity is measured")
    if "voltage_clamp" in visual_families:
        principles.append("command voltage and sensed membrane voltage feed a differential amplifier that drives the current electrode")
    if "right_leg_drive" in visual_families:
        principles.append("auxiliary common-mode feedback loop uses a lower lane")
    if "isolation_barrier" in visual_families:
        principles.append("split input and output domains with a clear vertical isolation boundary")
    if "system_block" in visual_families:
        principles.append("use compact labeled blocks connected by orthogonal arrows or wires")
    if metrics.get("aspect") == "wide":
        principles.append("prefer horizontal compression over tall stacking")
    return tuple(dict.fromkeys(principles))


def _route_principles(visual_families: tuple[str, ...], component_cues: tuple[str, ...]) -> tuple[str, ...]:
    principles = ["orthogonal Manhattan wiring", "avoid running wires through component bodies"]
    if "single_opamp_feedback" in visual_families or "feedback" in component_cues:
        principles.append("feedback returns outside the op-amp body using top or bottom corridor")
    if "voltage_clamp" in visual_families:
        principles.append("sense path stays visually separate from drive/current path")
    if "right_leg_drive" in visual_families:
        principles.append("right-leg/common-mode feedback routes through bottom auxiliary corridor")
    if "protection_clamp" in visual_families:
        principles.append("clamp and protection branches use short local shunts to reference symbols")
    if "sensor_bridge" in visual_families:
        principles.append("bridge arms remain symmetric around the sensor/output nodes")
    if "emg_frontend" in visual_families:
        principles.append("EMG signal path preserves electrode input, high-pass/low-pass filtering, gain, and optional rectifier-integrator stages in order")
    return tuple(dict.fromkeys(principles))


def _image_metrics(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        image = Image.open(path).convert("L")
    except OSError:
        return {}
    width, height = image.size
    if width <= 0 or height <= 0:
        return {}
    sample = image.resize((min(width, 320), max(1, int(height * min(width, 320) / width))))
    histogram = sample.histogram()
    dark = sum(histogram[:200])
    density = dark / max(1, sample.width * sample.height)
    ratio = width / height
    aspect = "wide" if ratio >= 1.25 else "tall" if ratio <= 0.8 else "balanced"
    return {"width": width, "height": height, "aspect_ratio": round(ratio, 3), "aspect": aspect, "ink_density": round(density, 4)}


def _circuit_tokens(circuit: Circuit) -> set[str]:
    circuit_text = " ".join([circuit.id, circuit.motif or "", circuit.title or ""])
    tokens = _tokens(circuit_text)
    if any(token in _key(circuit_text) for token in ("voltage_clamp", "two_electrode_voltage_clamp", "tevc", "command_voltage")):
        tokens.add("voltage_clamp")
    for component in circuit.components:
        identity = _identity(component)
        tokens.update(_tokens(identity))
        if _is_opamp(component):
            tokens.add("op_amp")
            tokens.add("schematic")
        if _is_input(component):
            tokens.add("input")
        if _is_output(component):
            tokens.add("output")
        for net in component.pins.values():
            tokens.update(_tokens(net))
            if any(token in _key(net) for token in ("voltage_clamp", "tevc", "command_voltage", "clamp_command")):
                tokens.add("voltage_clamp")
            if any(token in _key(net) for token in ("ecg", "eeg", "emg", "electrode", "skin", "patient", "body")):
                tokens.add("electrode")
            if any(token in _key(net) for token in ("right_leg", "rld")):
                tokens.add("right_leg_drive")
    return tokens


def _report_tokens(report: CriticReport) -> set[str]:
    tokens = set()
    for violation in report.violations:
        tokens.update(_tokens(violation.code))
        if violation.subject:
            tokens.update(_tokens(violation.subject))
    return tokens


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9_]+", _key(text)) if len(token) >= 2}


def _identity(component: Component) -> str:
    return _key(" ".join(str(item or "") for item in (component.id, component.type, component.label, component.role, component.value)))


def _placement_for(
    by_id: dict[str, GridPlacementHint],
    component: Component,
    stage: int,
    lane: int,
    orientation: str,
    role: str,
) -> GridPlacementHint:
    existing = by_id.get(component.id)
    confidence = max(0.6, existing.confidence if existing else 0.0)
    return GridPlacementHint(component.id, stage, lane, orientation, role, confidence)


def _dedupe_placements(placements: list[GridPlacementHint]) -> list[GridPlacementHint]:
    result: dict[str, GridPlacementHint] = {}
    for placement in placements:
        result[placement.component_id] = placement
    return list(result.values())


def _dedupe_route_policies(policies: list[RoutePolicyHint]) -> list[RoutePolicyHint]:
    result: dict[tuple[str | None, str], RoutePolicyHint] = {}
    for policy in policies:
        result[(policy.net, policy.policy)] = policy
    return list(result.values())


def _has_feedback_to(opamp: Component, output_net: str, components: list[Component]) -> bool:
    input_nets = {net for pin, net in opamp.pins.items() if _pin_kind(pin) in {"+", "-"}}
    for component in components:
        if component.id == opamp.id:
            continue
        nets = set(component.pins.values())
        if output_net in nets and nets & input_nets:
            return True
    return False


def _opamp_output_net(component: Component) -> str | None:
    for pin, net in component.pins.items():
        if _pin_kind(pin) in {"out", "output", "o"}:
            return net
    return None


def _pin_kind(pin_name: str) -> str:
    compact = _key(pin_name).replace("_", "")
    if pin_name in {"+", "-"}:
        return pin_name
    if compact in {"plus", "noninverting", "noninv", "inp", "vp"}:
        return "+"
    if compact in {"minus", "inverting", "inv", "inn", "vn"}:
        return "-"
    return compact


def _is_opamp(component: Component) -> bool:
    identity = _identity(component)
    return any(token in identity for token in ("op_amp", "opamp", "operational_amplifier"))


def _is_input(component: Component) -> bool:
    identity = _identity(component)
    return "input" in identity or "source" in identity or component.type.lower() in {"input", "source", "voltage_source"}


def _is_output(component: Component) -> bool:
    identity = _identity(component)
    return "output" in identity or component.type.lower() == "output"


def _is_ground(component: Component) -> bool:
    identity = _identity(component)
    return "ground" in identity or _key(component.type) in {"ground", "gnd"}


def _is_common_mode_sense(component: Component) -> bool:
    identity = _identity(component)
    net_text = _key(" ".join(component.pins.values()))
    if "common_mode_sense" in identity or "cm_sense" in identity:
        return True
    return _is_type_name(component, "resistor") and "common_mode" in net_text and any(
        token in net_text for token in ("ecg", "lead", "electrode", "skin", "patient")
    )


def _is_type_name(component: Component, type_name: str) -> bool:
    return _key(component.type) == _key(type_name)


def _all_nets(circuit: Circuit) -> set[str]:
    return {net for component in circuit.components for net in component.pins.values()}


def _is_local_net(net: str) -> bool:
    key = _key(net)
    return key in {"gnd", "ground", "vcc", "vee", "vdd", "vss", "ref"} or key.startswith(("vcc", "vee", "vdd", "vss"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if isinstance(raw, dict):
                records.append(raw)
    return records


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


__all__ = [
    "DEFAULT_TEXTBOOK_CORPUS_DIR",
    "FigureStructureCard",
    "TextbookCorpusIndex",
    "TextbookFigureInterpreter",
    "TextbookStructurePlanner",
    "TextbookSurrogatePlanningClient",
    "build_structured_cards",
]

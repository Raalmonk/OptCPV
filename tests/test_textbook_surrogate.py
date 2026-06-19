from pathlib import Path

from optcpv import Circuit, Component
from optcpv import draw_optimized_artifact
from optcpv.planning_agent import planning_client_from_env
from optcpv.textbook_surrogate import TextbookCorpusIndex, TextbookFigureInterpreter, TextbookSurrogatePlanningClient


CORPUS_DIR = Path("textbook_circuit_corpus")


def _differential_fixture() -> Circuit:
    return Circuit(
        id="p1_37_like_electrode_differential_amp",
        motif="bioelectric_differential_amplifier",
        components=[
            Component(id="E_PLUS", type="input", pins={"out": "skin_plus"}, label="E+"),
            Component(id="E_MINUS", type="input", pins={"out": "skin_minus"}, label="E-"),
            Component(id="R5", type="resistor", pins={"a": "skin_plus", "b": "vin_plus"}, label="R5"),
            Component(id="R6", type="resistor", pins={"a": "skin_minus", "b": "vin_minus"}, label="R6"),
            Component(id="U1", type="op_amp", pins={"+": "vin_plus", "-": "vin_minus", "out": "vout"}, label="U1"),
            Component(id="R2", type="resistor", pins={"a": "vout", "b": "vin_minus"}, label="R2", role="feedback"),
            Component(id="R4", type="resistor", pins={"a": "vin_plus", "b": "gnd"}, label="R4"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="Vout"),
        ],
    )


def _right_leg_fixture() -> Circuit:
    return Circuit(
        id="chapter6_right_leg_drive_surrogate",
        motif="ecg_right_leg_drive",
        components=[
            Component(id="E_PLUS", type="input", pins={"out": "ecg_plus"}),
            Component(id="E_MINUS", type="input", pins={"out": "ecg_minus"}),
            Component(id="U1", type="op_amp", pins={"+": "ecg_plus", "-": "fb", "out": "ecg_out"}),
            Component(id="RF", type="resistor", pins={"a": "ecg_out", "b": "fb"}, role="feedback"),
            Component(id="A_AUX", type="op_amp", pins={"+": "gnd", "-": "common_mode", "out": "right_leg_drive"}, role="right_leg_drive"),
            Component(id="R_RL", type="resistor", pins={"a": "right_leg_drive", "b": "right_leg_electrode"}),
            Component(id="RL", type="output", pins={"in": "right_leg_electrode"}, role="body_terminal"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def test_textbook_corpus_retrieves_relevant_cards() -> None:
    corpus = TextbookCorpusIndex(CORPUS_DIR)

    cards = corpus.retrieve(_differential_fixture(), limit=6)

    assert cards
    assert any(card.figure_id in {"1.12", "P1.37"} for card in cards)
    assert any("electrode_context" in card.visual_families for card in cards)


def test_textbook_interpreter_recognizes_voltage_clamp_family() -> None:
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "4.x",
            "page": 1,
            "caption": "Figure 4.x Two-electrode voltage clamp with command voltage, buffer amp, differential amplifier, and current electrode.",
            "categories": ["schematic_circuit"],
            "primary_category": "schematic_circuit",
            "is_likely_circuit": True,
        }
    )

    assert "voltage_clamp" in card.component_cues
    assert "voltage_clamp" in card.visual_families
    assert any("sense path" in principle for principle in card.route_principles)


def test_textbook_interpreter_recognizes_emg_frontend_family() -> None:
    card = TextbookFigureInterpreter().card_from_record(
        {
            "figure_id": "6.x",
            "page": 1,
            "caption": "Figure 6.x EMG circuit with electrode inputs, high-gain bandpass amplifier, rectifier, and integrator for muscle activity.",
            "categories": ["schematic_circuit", "bioelectric_recording"],
            "primary_category": "schematic_circuit",
            "is_likely_circuit": True,
        }
    )

    assert "emg" in card.component_cues
    assert "emg_frontend" in card.visual_families
    assert any("rectifier" in principle for principle in card.route_principles)


def test_textbook_surrogate_planner_emits_structured_middle_layer_text() -> None:
    client = TextbookSurrogatePlanningClient(CORPUS_DIR, retrieval_limit=6)

    hints = client.propose_hints(_right_leg_fixture())

    assert "GEMINI_MIDDLE_LAYER mode=initial" in client.last_middle_layer_text
    assert "FIGURE_CARD" in client.last_middle_layer_text
    assert hints.source == "textbook_surrogate"
    assert any(policy.policy == "bottom_auxiliary_corridor" for policy in hints.route_policies)
    assert hints.local_terminal_policy.get("gnd") == "local_symbol_only"


def test_textbook_surrogate_can_be_selected_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPTCPV_PLANNING_CLIENT", "textbook")
    monkeypatch.setenv("OPTCPV_TEXTBOOK_CORPUS_DIR", str(CORPUS_DIR))

    client = planning_client_from_env()

    assert isinstance(client, TextbookSurrogatePlanningClient)


def test_textbook_surrogate_planning_client_runs_end_to_end() -> None:
    client = TextbookSurrogatePlanningClient(CORPUS_DIR, retrieval_limit=6)

    artifact = draw_optimized_artifact(_right_leg_fixture(), max_iterations=1, planning_client=client)

    assert artifact.svg
    assert artifact.planning_hints_used
    assert artifact.planning_hints_used["source"] == "textbook_surrogate"
    assert "FIGURE_CARD" in client.last_middle_layer_text

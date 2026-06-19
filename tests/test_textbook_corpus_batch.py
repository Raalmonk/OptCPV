from pathlib import Path

from optcpv.textbook_surrogate import TextbookCorpusIndex
from tools.run_textbook_corpus_batch import circuit_fixture_for_card, run_batch


CORPUS_DIR = Path("textbook_circuit_corpus")


def test_batch_fixture_keeps_system_diagrams_as_system_blocks() -> None:
    card = TextbookCorpusIndex(CORPUS_DIR).cards[0]

    circuit, fixture = circuit_fixture_for_card(card, 0)

    assert card.figure_id == "1.1"
    assert fixture == "system_block"
    assert any(component.type == "filter_block" for component in circuit.components)


def test_batch_runner_checks_cards_and_renders_likely_circuit_subset(tmp_path) -> None:
    summary = run_batch(corpus_dir=CORPUS_DIR, out_dir=tmp_path, max_cards=5)

    assert summary["total_cards"] == 5
    assert summary["image_passes"] == 5
    assert summary["structured_text_passes"] == 5
    assert summary["render_attempts"] == 2
    assert summary["render_errors"] == 0
    assert summary["middle_layers_generated"] == 2
    assert (tmp_path / "results.jsonl").exists()
    assert (tmp_path / "summary.json").exists()

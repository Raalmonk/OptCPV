from optcpv import draw_optimized_artifact
from optcpv.planner import plan_layout
from tools.run_textbook_corpus_batch import _active_filter_fixture, _bioelectric_fixture, _bridge_fixture


def test_active_filter_uses_grammar_without_dropping_opamp() -> None:
    artifact = draw_optimized_artifact(_active_filter_fixture("regression"), max_iterations=4)

    assert "U1" in artifact.components
    assert "CIN" in artifact.components
    assert artifact.critic_report["hard_fail"] is False
    assert artifact.critic_report["score"] == 0


def test_sensor_bridge_frontend_gets_bridge_motiflet_layout() -> None:
    layout = plan_layout(_bridge_fixture("regression"))
    artifact = draw_optimized_artifact(_bridge_fixture("regression"), max_iterations=4)

    assert layout.support.matched_motifs == ("sensor_bridge_frontend",)
    assert artifact.critic_report["hard_fail"] is False
    assert artifact.critic_report["score"] <= 3


def test_bioelectric_pair_uses_differential_frontend_motiflet() -> None:
    layout = plan_layout(_bioelectric_fixture("regression"))
    artifact = draw_optimized_artifact(_bioelectric_fixture("regression"), max_iterations=4)

    assert layout.support.matched_motifs == ("single_opamp_differential_frontend",)
    assert artifact.critic_report["hard_fail"] is False
    assert artifact.critic_report["score"] == 0

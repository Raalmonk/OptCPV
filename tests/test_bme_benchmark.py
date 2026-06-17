from examples.bme_analog_200 import _has_failures, build_cases, select_contact_sheet_cases, summarize
from optcpv import draw_optimized_artifact


def test_bme_benchmark_summary_clusters_failures() -> None:
    results = [
        {
            "id": "pass",
            "input_type": "text",
            "family": "ecg",
            "local_score": 0,
            "gemini_sim_score": 0,
            "local_hard_fail": False,
            "gemini_sim_hard_fail": False,
            "local_pass": True,
            "gemini_sim_pass": True,
            "local_violations": [],
            "gemini_sim_violations": [],
        },
        {
            "id": "fail",
            "input_type": "image",
            "family": "ecg",
            "local_score": 18,
            "gemini_sim_score": 18,
            "local_hard_fail": False,
            "gemini_sim_hard_fail": False,
            "local_pass": False,
            "gemini_sim_pass": False,
            "local_violations": ["label_visual_collision", "label_wire_overlap"],
            "gemini_sim_violations": ["label_visual_collision"],
        },
    ]

    summary = summarize(results)

    assert summary["local_passes"] == 1
    assert summary["gemini_sim_total"] == 2
    assert summary["gemini_sim_passes"] == 1
    assert summary["failure_ids"] == ["fail"]
    assert summary["failure_violation_clusters"] == [
        {"code": "label_visual_collision", "count": 1},
        {"code": "label_wire_overlap", "count": 1},
    ]
    assert summary["gemini_sim_failure_clusters"] == [{"code": "label_visual_collision", "count": 1}]
    assert summary["score_histogram"]["11-25"] == 1
    assert summary["total_render_seconds"] == 0.0
    assert summary["mean_case_seconds"] == 0.0
    assert _has_failures(summary)


def test_bme_benchmark_local_only_summary_skips_gemini_failure_gate() -> None:
    results = [
        {
            "id": "pass",
            "input_type": "text",
            "family": "ecg",
            "local_score": 0,
            "gemini_sim_score": None,
            "local_hard_fail": False,
            "gemini_sim_hard_fail": None,
            "local_pass": True,
            "gemini_sim_pass": None,
            "local_violations": [],
            "gemini_sim_violations": [],
        }
    ]

    summary = summarize(results)

    assert summary["local_passes"] == 1
    assert summary["gemini_sim_total"] == 0
    assert summary["gemini_sim_passes"] == 0
    assert summary["gemini_sim_pass_rate"] is None
    assert not _has_failures(summary)


def test_bme_benchmark_start_index_shifts_case_variants() -> None:
    cases = build_cases(text_count=1, image_count=1, adversarial=True, start_index=500)

    assert cases[0]["id"] == "bme_text_501_adv"
    assert cases[1]["id"] == "bme_image_501_adv"
    assert "Variant 26" in cases[0]["input"]["prompt"]
    assert "Variant 26" in cases[1]["input"]["caption"]


def test_contact_sheet_selection_prioritizes_failures_and_families() -> None:
    cases = build_cases(text_count=3, image_count=0)
    results = []
    for index, case in enumerate(cases):
        results.append(
            {
                "id": case["id"],
                "input_type": case["input_type"],
                "family": case["llm_conversion"]["family"],
                "local_score": 22 if index == 1 else 0,
                "gemini_sim_score": 0,
                "local_hard_fail": False,
                "gemini_sim_hard_fail": False,
                "local_pass": index != 1,
                "gemini_sim_pass": True,
                "local_violations": ["wire_crossings"] if index == 1 else [],
                "gemini_sim_violations": [],
            }
        )

    selected = select_contact_sheet_cases(cases, results, count=3)

    assert [result["id"] for _case, result in selected][:1] == [cases[1]["id"]]
    assert len({result["family"] for _case, result in selected}) == 3


def test_contact_sheet_selection_balances_input_types() -> None:
    cases = build_cases(text_count=2, image_count=2)
    results = []
    for case in cases:
        results.append(
            {
                "id": case["id"],
                "input_type": case["input_type"],
                "family": case["llm_conversion"]["family"],
                "local_score": 0,
                "gemini_sim_score": 0,
                "local_hard_fail": False,
                "gemini_sim_hard_fail": False,
                "local_pass": True,
                "gemini_sim_pass": True,
                "local_violations": [],
                "gemini_sim_violations": [],
            }
        )

    selected = select_contact_sheet_cases(cases, results, count=2)

    assert {result["input_type"] for _case, result in selected} == {"text", "image"}


def test_adversarial_native_motif_labels_are_compact_in_svg() -> None:
    case = build_cases(text_count=1, image_count=0, adversarial=True, start_index=1000)[0]

    artifact = draw_optimized_artifact(case["circuit"], max_iterations=3)

    assert "stage" not in artifact.svg.lower()
    assert "biomed" not in artifact.svg.lower()

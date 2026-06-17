"""Build and run a BME analog-circuit stress benchmark.

The benchmark records two ingestion surfaces:
- text prompts that a local LLM layer converts into OptCPV IR
- image-referenced prompts that stand in for schematic/image inputs

The source URLs are used as topic inspiration. We intentionally synthesize the
actual case text instead of copying problem statements verbatim.
"""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, replace
from math import ceil
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcpv import Circuit, Component, HeuristicVisionClient, draw_artifact, draw_optimized_artifact
from optcpv.raster import rasterize_svg


DEFAULT_TEXT_COUNT = 250
DEFAULT_IMAGE_COUNT = 250
DEFAULT_OUT_DIR = Path("generated/bme_analog_500")


TEXT_SOURCE_TOPICS = [
    {
        "url": "https://open.oregonstate.education/aandb/chapter/2-4-op-amp-circuits/",
        "topic": "op amp gain and feedback circuits",
    },
    {
        "url": "https://open.oregonstate.education/aandb/chapter/6-2-sallen-key-filters/",
        "topic": "active low-pass and Sallen-Key filters",
    },
    {
        "url": "https://www.ti.com/lit/an/sloa060/sloa060.pdf",
        "topic": "active filter design and op amp filter stages",
    },
    {
        "url": "https://www.analog.com/en/resources/analog-dialogue/articles/ecg-front-end-design-is-simplified-with-microconverter.html",
        "topic": "ECG front-end gain, filtering, and right-leg-drive context",
    },
    {
        "url": "https://www.ti.com/lit/an/sbaa160a/sbaa160a.pdf",
        "topic": "ECG analog front-end measurements",
    },
    {
        "url": "https://www.analog.com/en/resources/analog-dialogue/articles/transimpedance-amplifier-role-in-vital-signs-monitoring.html",
        "topic": "photodiode transimpedance amplifiers for vital signs",
    },
    {
        "url": "https://www.analog.com/en/resources/analog-dialogue/articles/wheatstone-bridge-sensors.html",
        "topic": "Wheatstone bridge sensor signal conditioning",
    },
    {
        "url": "https://www.ti.com/lit/an/sboa247/sboa247.pdf",
        "topic": "instrumentation amplifiers and bridge sensors",
    },
    {
        "url": "https://www.analog.com/en/resources/technical-articles/input-filtering-for-biopotential-signal-chain.html",
        "topic": "biopotential input filters",
    },
    {
        "url": "https://www.allaboutcircuits.com/textbook/semiconductors/chpt-8/active-filters/",
        "topic": "active filters and biomedical signal conditioning",
    },
]


IMAGE_SOURCE_TOPICS = [
    {
        "url": "https://commons.wikimedia.org/wiki/File:Op-Amp_Non-Inverting_Amplifier.svg",
        "topic": "non-inverting op amp image input",
    },
    {
        "url": "https://commons.wikimedia.org/wiki/File:Op-Amp_Inverting_Amplifier.svg",
        "topic": "feedback amplifier image input",
    },
    {
        "url": "https://commons.wikimedia.org/wiki/File:Wheatstonebridge.svg",
        "topic": "bridge sensor schematic image input",
    },
    {
        "url": "https://commons.wikimedia.org/wiki/File:RC_Divider.svg",
        "topic": "RC filter schematic image input",
    },
    {
        "url": "https://commons.wikimedia.org/wiki/File:Sallen-Key_Lowpass_Generic.svg",
        "topic": "active low-pass image input",
    },
    {
        "url": "https://www.analog.com/en/resources/analog-dialogue/articles/transimpedance-amplifier-role-in-vital-signs-monitoring.html",
        "topic": "photodiode TIA image input",
    },
    {
        "url": "https://www.analog.com/en/resources/analog-dialogue/articles/wheatstone-bridge-sensors.html",
        "topic": "bridge sensor front-end image input",
    },
    {
        "url": "https://www.ti.com/lit/an/sbaa160a/sbaa160a.pdf",
        "topic": "ECG front-end image input",
    },
    {
        "url": "https://www.ti.com/lit/an/sboa247/sboa247.pdf",
        "topic": "instrumentation amplifier image input",
    },
    {
        "url": "https://open.oregonstate.education/aandb/chapter/6-2-sallen-key-filters/",
        "topic": "cascaded filter image input",
    },
]


TEXT_FAMILIES = [
    ("ecg_inamp", "Design an ECG instrumentation amplifier front end with high input impedance."),
    ("emg_bandpass", "Sketch an EMG analog preamplifier followed by active filtering."),
    ("eeg_low_noise", "Draw a low-noise EEG gain chain with cascaded op amp stages."),
    ("eog_frontend", "Create an EOG preamp with DC blocking and moderate gain."),
    ("ppg_tia", "Convert a photodiode PPG signal into voltage and filter it."),
    ("pulse_ox_chain", "Build a pulse-oximeter analog receive chain."),
    ("pressure_bridge", "Condition a pressure-sensor Wheatstone bridge."),
    ("strain_bridge", "Amplify a strain-gauge bridge for respiration belts."),
    ("thermistor_divider", "Draw a thermistor-style sensing divider with output node."),
    ("gsr_divider", "Draw a galvanic skin response divider and output buffer."),
    ("baseline_highpass", "Create a baseline-wander high-pass section."),
    ("anti_alias_lowpass", "Create an anti-aliasing low-pass section."),
    ("notch_chain", "Build a mains-noise rejection chain around op amps."),
    ("right_leg_drive", "Sketch an ECG common-mode feedback/right-leg-drive style loop."),
    ("active_electrode", "Draw an active electrode buffer and gain stage."),
    ("isolation_prep", "Draw the analog front-end before an isolation barrier."),
    ("bioimpedance", "Create a respiration/bioimpedance analog gain chain."),
    ("temperature_bridge", "Condition a resistive temperature bridge."),
    ("force_plate_bridge", "Draw bridge conditioning for a force/pressure plate."),
    ("multi_channel_afe", "Create a multi-channel biomedical analog front end."),
]


IMAGE_FAMILIES = [
    ("img_ecg_three_opamp", "Image shows a three-op-amp ECG instrumentation amplifier."),
    ("img_emg_filter_stack", "Image shows an EMG preamp followed by filter blocks."),
    ("img_eeg_cascade", "Image shows cascaded low-noise op amp gain stages."),
    ("img_ppg_tia", "Image shows a photodiode transimpedance amplifier schematic."),
    ("img_pulse_ox", "Image shows a pulse-ox receive analog chain."),
    ("img_bridge_pressure", "Image shows a Wheatstone bridge sensor conditioner."),
    ("img_bridge_strain", "Image shows strain gauge bridge amplification."),
    ("img_rc_filter", "Image shows a simple RC biomedical signal filter."),
    ("img_notch", "Image shows a notch/filter chain with op amps."),
    ("img_multi_opamp", "Image shows a large 7-8 op amp analog front-end."),
    ("img_thermistor", "Image shows a resistive temperature sensing divider."),
    ("img_gsr", "Image shows a skin-conductance divider/buffer."),
    ("img_active_electrode", "Image shows an active electrode buffer."),
    ("img_right_leg", "Image shows a right-leg-drive feedback block."),
    ("img_anti_alias", "Image shows anti-alias low-pass conditioning."),
    ("img_bioimpedance", "Image shows bioimpedance receive amplification."),
    ("img_force_bridge", "Image shows force-plate bridge conditioning."),
    ("img_summing", "Image shows a summing/averaging op amp front-end."),
    ("img_isolation", "Image shows pre-isolation analog conditioning."),
    ("img_multichannel", "Image shows multi-channel analog acquisition."),
]


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(
        text_count=args.text_count,
        image_count=args.image_count,
        adversarial=args.adversarial,
        start_index=args.start_index,
    )
    results = run_cases(cases, include_gemini_sim=not args.local_only)
    summary = summarize(results)

    (out_dir / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.contact_sheet:
        write_contact_sheet(cases, results, out_dir=out_dir, count=args.contact_sheet_count)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_failure and _has_failures(summary):
        raise SystemExit(1)


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--text-count", type=int, default=DEFAULT_TEXT_COUNT)
    parser.add_argument("--image-count", type=int, default=DEFAULT_IMAGE_COUNT)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--adversarial", action="store_true", help="Mutate cases with dirty motif/label/topology hints.")
    parser.add_argument("--fail-on-failure", action="store_true", help="Exit non-zero if any local/Gemini-sim case fails.")
    parser.add_argument("--start-index", type=int, default=0, help="Shift deterministic case variants by this zero-based index.")
    parser.add_argument("--contact-sheet", action="store_true", help="Write an optimized PNG QA contact sheet.")
    parser.add_argument("--contact-sheet-count", type=int, default=12, help="Number of representative cases to render.")
    parser.add_argument("--local-only", action="store_true", help="Skip the Gemini-sim comparison path for faster local gates.")
    return parser.parse_args()


def build_cases(
    *,
    text_count: int = DEFAULT_TEXT_COUNT,
    image_count: int = DEFAULT_IMAGE_COUNT,
    adversarial: bool = False,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for offset in range(text_count):
        index = start_index + offset
        family_id, prompt = TEXT_FAMILIES[index % len(TEXT_FAMILIES)]
        source = TEXT_SOURCE_TOPICS[index % len(TEXT_SOURCE_TOPICS)]
        circuit = _convert_prompt_to_circuit(f"bme_text_{index + 1:03d}", family_id, index)
        if adversarial:
            circuit = _adversarial_circuit(circuit, family_id, index)
        cases.append(
            {
                "id": circuit.id,
                "input_type": "text",
                "source_url": source["url"],
                "source_topic": source["topic"],
                "input": {
                    "prompt": f"{prompt} Variant {index // len(TEXT_FAMILIES) + 1}.",
                    "constraints": ["left-to-right signal flow", "clear biomedical front-end labels", "no hidden topology changes"],
                },
                "llm_conversion": _conversion_note(circuit, family_id, adversarial=adversarial),
                "circuit": _circuit_dict(circuit),
            }
        )

    for offset in range(image_count):
        index = start_index + offset
        family_id, caption = IMAGE_FAMILIES[index % len(IMAGE_FAMILIES)]
        source = IMAGE_SOURCE_TOPICS[index % len(IMAGE_SOURCE_TOPICS)]
        circuit = _convert_prompt_to_circuit(f"bme_image_{index + 1:03d}", family_id, index)
        if adversarial:
            circuit = _adversarial_circuit(circuit, family_id, index)
        cases.append(
            {
                "id": circuit.id,
                "input_type": "image",
                "source_url": source["url"],
                "source_topic": source["topic"],
                "input": {
                    "image_reference": source["url"],
                    "caption": f"{caption} Variant {index // len(IMAGE_FAMILIES) + 1}.",
                    "visible_hints": ["op amp symbols", "passive feedback/gain network", "biomedical signal labels"],
                },
                "llm_conversion": _conversion_note(circuit, family_id, adversarial=adversarial),
                "circuit": _circuit_dict(circuit),
            }
        )
    return cases


def run_cases(cases: list[dict[str, Any]], *, include_gemini_sim: bool = True) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        case_start = time.perf_counter()
        circuit = _circuit_from_dict(case["circuit"])
        start = draw_artifact(circuit)
        local = draw_optimized_artifact(circuit, max_iterations=5)
        gemini_sim = (
            draw_optimized_artifact(circuit, max_iterations=5, vision_client=HeuristicVisionClient())
            if include_gemini_sim
            else None
        )
        result = {
            "id": case["id"],
            "input_type": case["input_type"],
            "family": case["llm_conversion"]["family"],
            "start_score": start.critic_report["score"],
            "local_score": local.critic_report["score"],
            "gemini_sim_score": gemini_sim.critic_report["score"] if gemini_sim else None,
            "start_hard_fail": start.critic_report["hard_fail"],
            "local_hard_fail": local.critic_report["hard_fail"],
            "gemini_sim_hard_fail": gemini_sim.critic_report["hard_fail"] if gemini_sim else None,
            "local_pass": _passes(local),
            "gemini_sim_pass": _passes(gemini_sim) if gemini_sim else None,
            "local_violations": [violation["code"] for violation in local.critic_report["violations"]],
            "gemini_sim_violations": (
                [violation["code"] for violation in gemini_sim.critic_report["violations"]]
                if gemini_sim
                else []
            ),
            "duration_seconds": round(time.perf_counter() - case_start, 4),
            "optimization_log": local.optimization_log,
        }
        results.append(result)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    for result in results:
        _bump(by_type, result["input_type"], result["local_pass"])
        _bump(by_family, result["family"], result["local_pass"])
    failures = [result for result in results if not result["local_pass"]]
    gemini_sim_results = [result for result in results if result.get("gemini_sim_pass") is not None]
    gemini_sim_failures = [result for result in gemini_sim_results if not result["gemini_sim_pass"]]
    local_passes = sum(1 for result in results if result["local_pass"])
    gemini_sim_passes = sum(1 for result in gemini_sim_results if result["gemini_sim_pass"])
    hard_failures = sum(1 for result in results if result["local_hard_fail"])
    scores = [result["local_score"] for result in results]
    durations = [float(result.get("duration_seconds", 0.0)) for result in results]
    gemini_sim_total = len(gemini_sim_results)
    return {
        "total": len(results),
        "local_passes": local_passes,
        "gemini_sim_total": gemini_sim_total,
        "gemini_sim_passes": gemini_sim_passes,
        "hard_failures": hard_failures,
        "local_pass_rate": round(local_passes / max(1, len(results)), 4),
        "gemini_sim_pass_rate": round(gemini_sim_passes / max(1, gemini_sim_total), 4) if gemini_sim_total else None,
        "by_type": by_type,
        "by_family": by_family,
        "failure_ids": [result["id"] for result in failures[:30]],
        "gemini_sim_failure_ids": [result["id"] for result in gemini_sim_failures[:30]],
        "failure_violation_clusters": _violation_clusters(failures, "local_violations"),
        "gemini_sim_failure_clusters": _violation_clusters(gemini_sim_failures, "gemini_sim_violations"),
        "score_histogram": _score_histogram(scores),
        "max_local_score": max(scores or [0]),
        "mean_local_score": round(sum(scores) / max(1, len(scores)), 3),
        "total_render_seconds": round(sum(durations), 3),
        "mean_case_seconds": round(sum(durations) / max(1, len(durations)), 4),
        "max_case_seconds": round(max(durations or [0.0]), 4),
    }


def _bump(bucket: dict[str, dict[str, int]], key: str, passed: bool) -> None:
    row = bucket.setdefault(key, {"total": 0, "pass": 0, "fail": 0})
    row["total"] += 1
    row["pass" if passed else "fail"] += 1


def _has_failures(summary: dict[str, Any]) -> bool:
    gemini_sim_total = summary.get("gemini_sim_total", summary["total"])
    return (
        summary["local_passes"] != summary["total"]
        or (gemini_sim_total > 0 and summary["gemini_sim_passes"] != gemini_sim_total)
        or summary["hard_failures"] > 0
    )


def _violation_clusters(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(code for result in results for code in result[key])
    return [{"code": code, "count": count} for code, count in counter.most_common()]


def _score_histogram(scores: list[float]) -> dict[str, int]:
    buckets = {"0": 0, "1-3": 0, "4-10": 0, "11-25": 0, "26-100": 0, ">100": 0}
    for score in scores:
        if score == 0:
            buckets["0"] += 1
        elif score <= 3:
            buckets["1-3"] += 1
        elif score <= 10:
            buckets["4-10"] += 1
        elif score <= 25:
            buckets["11-25"] += 1
        elif score <= 100:
            buckets["26-100"] += 1
        else:
            buckets[">100"] += 1
    return buckets


def write_contact_sheet(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    out_dir: Path,
    count: int = 12,
) -> dict[str, Any]:
    selected = select_contact_sheet_cases(cases, results, count=count)
    if not selected:
        metadata = {"total_selected": 0, "items": []}
        (out_dir / "contact_sheet.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    from PIL import Image, ImageDraw, ImageFont

    thumb_w, thumb_h = 480, 320
    header_h = 58
    cell_w, cell_h = thumb_w, thumb_h + header_h
    cols = min(3, max(1, len(selected)))
    rows = ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    items: list[dict[str, Any]] = []

    for index, (case, result) in enumerate(selected):
        row, col = divmod(index, cols)
        x0, y0 = col * cell_w, row * cell_h
        circuit = _circuit_from_dict(case["circuit"])
        artifact = draw_optimized_artifact(circuit, max_iterations=5)
        raster = rasterize_svg(artifact.svg, output_width=thumb_w, output_height=thumb_h)
        thumb = Image.fromarray(raster.rgba).convert("RGB")
        sheet.paste(thumb, (x0, y0 + header_h))
        header = f"{case['id']} | {result['family']} | score {result['local_score']}"
        gemini_text = _gemini_status_text(result["gemini_sim_pass"])
        subheader = f"{case['input_type']} | local {_pass_text(result['local_pass'])} | {gemini_text}"
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline=(210, 210, 210), width=1)
        draw.text((x0 + 8, y0 + 8), _clip_text(header, 68), fill=(20, 20, 20), font=font)
        draw.text((x0 + 8, y0 + 30), _clip_text(subheader, 76), fill=(80, 80, 80), font=font)
        items.append(
            {
                "id": case["id"],
                "input_type": case["input_type"],
                "family": result["family"],
                "local_score": result["local_score"],
                "gemini_sim_score": result["gemini_sim_score"],
                "local_pass": result["local_pass"],
                "gemini_sim_pass": result["gemini_sim_pass"],
                "local_violations": result["local_violations"],
                "source_url": case["source_url"],
            }
        )

    image_path = out_dir / "contact_sheet.png"
    json_path = out_dir / "contact_sheet.json"
    sheet.save(image_path)
    metadata = {"total_selected": len(items), "image": image_path.name, "items": items}
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def select_contact_sheet_cases(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    count: int = 12,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if count <= 0:
        return []
    case_by_id = {case["id"]: case for case in cases}
    result_by_id = {result["id"]: result for result in results}
    selected_ids: list[str] = []

    def add(result: dict[str, Any]) -> None:
        if result["id"] in case_by_id and result["id"] not in selected_ids and len(selected_ids) < count:
            selected_ids.append(result["id"])

    ranked = sorted(
        results,
        key=lambda result: (
            result["local_pass"] and result["gemini_sim_pass"],
            -float(result["local_score"]),
            result["family"],
            result["id"],
        ),
    )
    balanced = _interleave_input_types(ranked)
    for result in balanced:
        if not result["local_pass"] or not result["gemini_sim_pass"]:
            add(result)

    seen_types: set[str] = set()
    for result in balanced:
        input_type = result.get("input_type", "")
        if input_type in seen_types:
            continue
        seen_types.add(input_type)
        add(result)

    seen_families: set[str] = set()
    for result in balanced:
        if result["family"] in seen_families:
            continue
        seen_families.add(result["family"])
        add(result)

    for result in balanced:
        add(result)

    return [(case_by_id[item_id], result_by_id[item_id]) for item_id in selected_ids]


def _interleave_input_types(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        input_type = result.get("input_type", "")
        if input_type not in buckets:
            order.append(input_type)
            buckets[input_type] = []
        buckets[input_type].append(result)

    interleaved: list[dict[str, Any]] = []
    while any(buckets[input_type] for input_type in order):
        for input_type in order:
            if buckets[input_type]:
                interleaved.append(buckets[input_type].pop(0))
    return interleaved


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _pass_text(value: bool | None, *, skipped: str = "skipped") -> str:
    if value is None:
        return skipped
    return "pass" if value else "fail"


def _gemini_status_text(value: bool | None) -> str:
    if value is None:
        return "gemini-sim skipped"
    return f"gemini-sim {_pass_text(value)}"


def _convert_prompt_to_circuit(case_id: str, family: str, index: int) -> Circuit:
    if "divider" in family or family in {"thermistor_divider", "gsr_divider", "img_thermistor", "img_gsr"}:
        return _voltage_divider(case_id, family, index)
    if "bridge" in family or family in {"pressure_bridge", "strain_bridge", "temperature_bridge", "force_plate_bridge", "img_force_bridge"}:
        return _bridge(case_id, family, index)
    if "rc" in family or "highpass" in family or "lowpass" in family or "anti_alias" in family:
        return _rc_low_pass(case_id, family, index)
    if "inamp" in family or "three_opamp" in family:
        return _instrumentation_amplifier(case_id, family, index)
    if "multi" in family or "cascade" in family or "chain" in family or family in {"eeg_low_noise", "pulse_ox_chain", "img_pulse_ox", "img_eeg_cascade", "img_multi_opamp", "img_multichannel", "multi_channel_afe"}:
        return _opamp_chain(case_id, family, 4 + (index % 5))
    if family in {"ppg_tia", "img_ppg_tia", "active_electrode", "img_active_electrode", "right_leg_drive", "img_right_leg"}:
        return _opamp_chain(case_id, family, 2 + (index % 3))
    if family in {"notch_chain", "img_notch", "emg_bandpass", "img_emg_filter_stack", "bioimpedance", "img_bioimpedance", "isolation_prep", "img_isolation", "img_summing"}:
        return _opamp_chain(case_id, family, 3 + (index % 4))
    return _non_inverting(case_id, family, index)


def _voltage_divider(case_id: str, family: str, index: int) -> Circuit:
    return Circuit(
        id=case_id,
        motif="voltage_divider",
        title=f"{family} sensing divider",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_input_label(family)),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label=f"Rsense{index % 5 + 1}"),
            Component(id="R2", type="resistor", pins={"a": "vout", "b": "gnd"}, label=f"Rref{index % 4 + 1}"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="ADC"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def _rc_low_pass(case_id: str, family: str, index: int) -> Circuit:
    return Circuit(
        id=case_id,
        motif="rc_low_pass",
        title=f"{family} passive filter",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_input_label(family)),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"}, label=f"R{index % 9 + 1}"),
            Component(id="C1", type="capacitor", pins={"a": "vout", "b": "gnd"}, label=f"C{index % 7 + 1}"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="ADC"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def _bridge(case_id: str, family: str, index: int) -> Circuit:
    labels = ["RA", "RB", "RC", "RD"] if index % 2 else ["Rtop", "Rbot", "Rsense", "Rref"]
    return Circuit(
        id=case_id,
        motif="bridge_or_wheatstone",
        title=f"{family} bridge conditioner",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label="EXC"),
            Component(id="R1", type="resistor", pins={"a": "vin", "b": "vl"}, label=labels[0]),
            Component(id="R2", type="resistor", pins={"a": "vl", "b": "gnd"}, label=labels[1]),
            Component(id="R3", type="resistor", pins={"a": "vin", "b": "vr"}, label=labels[2]),
            Component(id="R4", type="resistor", pins={"a": "vr", "b": "gnd"}, label=labels[3]),
            Component(id="VOUT", type="output", pins={"in": "vl"}, label="VBR"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def _non_inverting(case_id: str, family: str, index: int) -> Circuit:
    return Circuit(
        id=case_id,
        motif="non_inverting_op_amp",
        title=f"{family} single op amp gain",
        components=[
            Component(id="VIN", type="input", pins={"out": "vin"}, label=_input_label(family)),
            Component(id="U1", type="op_amp", pins={"+": "vin", "-": "vm", "out": "vout"}, label=f"U{index % 8 + 1}"),
            Component(id="Rf", type="resistor", pins={"a": "vout", "b": "vm"}, label="Rf", role="feedback"),
            Component(id="Rg", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rg", role="gain"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="ADC"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def _instrumentation_amplifier(case_id: str, family: str, index: int) -> Circuit:
    return Circuit(
        id=case_id,
        motif="instrumentation_amplifier",
        title=f"{family} three op amp front end",
        components=[
            Component(id="INP", type="input", pins={"out": "vinp"}, label="E+"),
            Component(id="INN", type="input", pins={"out": "vinn"}, label="E-"),
            Component(id="U1", type="op_amp", pins={"+": "vinp", "-": "n1", "out": "o1"}, label=f"A{index % 4 + 1}"),
            Component(id="U2", type="op_amp", pins={"+": "vinn", "-": "n2", "out": "o2"}, label=f"B{index % 4 + 1}"),
            Component(id="U3", type="op_amp", pins={"+": "n3", "-": "n4", "out": "vout"}, label=f"C{index % 4 + 1}"),
            Component(id="R1", type="resistor", pins={"a": "o1", "b": "n1"}, label="R1"),
            Component(id="R2", type="resistor", pins={"a": "o2", "b": "n2"}, label="R2"),
            Component(id="Rg", type="resistor", pins={"a": "n1", "b": "n2"}, label="Rg", role="gain"),
            Component(id="R3", type="resistor", pins={"a": "o1", "b": "n3"}, label="R3"),
            Component(id="R4", type="resistor", pins={"a": "n3", "b": "gnd"}, label="R4"),
            Component(id="R5", type="resistor", pins={"a": "o2", "b": "n4"}, label="R5"),
            Component(id="R6", type="resistor", pins={"a": "n4", "b": "vout"}, label="R6"),
            Component(id="VOUT", type="output", pins={"in": "vout"}, label="ADC"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def _opamp_chain(case_id: str, family: str, stages: int) -> Circuit:
    components = [
        Component(id="VIN", type="input", pins={"out": "vin"}, label=_input_label(family)),
        Component(id="VOUT", type="output", pins={"in": f"o{stages}"}, label="ADC"),
        Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
    ]
    previous = "vin"
    for index in range(1, stages + 1):
        output = f"o{index}"
        summing = f"fb{index}"
        components.extend(
            [
                Component(id=f"U{index}", type="op_amp", pins={"+": previous, "-": summing, "out": output}, label=f"U{index}"),
                Component(id=f"Rf{index}", type="resistor", pins={"a": output, "b": summing}, label=f"Rf{index}", role="feedback"),
                Component(id=f"Rg{index}", type="resistor", pins={"a": summing, "b": "gnd"}, label=f"Rg{index}", role="gain"),
            ]
        )
        previous = output
    return Circuit(id=case_id, motif="op_amp_network", title=f"{family} {stages}-stage op amp network", components=components)


def _adversarial_circuit(circuit: Circuit, family: str, index: int) -> Circuit:
    components = [_adversarial_label(component, index) for component in circuit.components]
    opamps = [component for component in components if _component_key(component.type) in {"op_amp", "opamp"}]
    resistors = [component for component in components if "resistor" in _component_key(component.type)]
    capacitors = [component for component in components if "capacitor" in _component_key(component.type)]
    motif = circuit.motif

    if len(opamps) >= 2:
        motif = ["bridge", "voltage_divider", "rc_low_pass", "noninv", "ina", "op_amp_network"][index % 6]
        components = _with_monitor_outputs(components, opamps, index)
        if index % 5 == 0:
            components.append(Component(id=f"AGND{index}", type="ground", pins={"gnd": "gnd"}, label="ANALOG_GND"))
    elif len(opamps) == 1:
        motif = ["op_amp_network", "ina", "bridge", "noninv"][index % 4]
    elif resistors and capacitors:
        motif = ["voltage_divider", "bridge", "rc_low_pass"][index % 3]
    elif len(resistors) >= 4:
        motif = ["voltage_divider", "rc_low_pass", "bridge"][index % 3]
    elif len(resistors) == 2:
        motif = ["bridge", "rc_low_pass", "voltage_divider"][index % 3]

    return Circuit(
        id=f"{circuit.id}_adv",
        motif=motif,
        title=f"{circuit.title or family} adversarial input",
        components=components,
    )


def _adversarial_label(component: Component, index: int) -> Component:
    if index % 4 != 0:
        return component
    if _component_key(component.type) in {"ground", "gnd"}:
        return component
    base = component.label or component.value or component.id
    label = f"{base}_stage_{index % 17 + 1}_biomed_front_end"
    return replace(component, label=label)


def _with_monitor_outputs(components: list[Component], opamps: list[Component], index: int) -> list[Component]:
    if index % 3 != 0:
        return components
    monitored = opamps[min(len(opamps) - 1, max(0, len(opamps) // 2))]
    output_net = _opamp_output_net(monitored)
    if output_net is None:
        return components
    output_id = f"VMON{index}"
    if any(component.id == output_id for component in components):
        return components
    return [
        *components,
        Component(
            id=output_id,
            type="output",
            pins={"in": output_net},
            label=f"MONITOR_STAGE_{index % 9 + 1}",
        ),
    ]


def _opamp_output_net(component: Component) -> str | None:
    for pin_name, net in component.pins.items():
        if _component_key(pin_name) in {"out", "output", "o", "vout"}:
            return net
    return None


def _component_key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


def _input_label(family: str) -> str:
    if "ecg" in family:
        return "ECG"
    if "emg" in family:
        return "EMG"
    if "eeg" in family:
        return "EEG"
    if "ppg" in family or "pulse" in family:
        return "PPG"
    return "VIN"


def _conversion_note(circuit: Circuit, family: str, *, adversarial: bool = False) -> dict[str, Any]:
    return {
        "family": family,
        "converter": "codex-local-llm-gemini-sim",
        "notes": [
            "Converted source-inspired prompt into OptCPV Circuit IR.",
            "Preserved topology through component ids, pins, and named nets.",
            "Gemini layer is simulated locally by OptCPV critic feedback.",
            "Adversarial mutations add dirty motif/label/topology hints." if adversarial else "No adversarial mutation applied.",
        ],
        "component_count": len(circuit.components),
        "motif": circuit.motif,
        "adversarial": adversarial,
    }


def _passes(artifact) -> bool:
    return artifact.critic_report["score"] <= 10 and not artifact.critic_report["hard_fail"]


def _circuit_dict(circuit: Circuit) -> dict[str, Any]:
    return {
        "id": circuit.id,
        "motif": circuit.motif,
        "title": circuit.title,
        "components": [asdict(component) for component in circuit.components],
    }


def _circuit_from_dict(data: dict[str, Any]) -> Circuit:
    return Circuit(
        id=data["id"],
        motif=data.get("motif"),
        title=data.get("title"),
        components=[Component(**component) for component in data["components"]],
    )


if __name__ == "__main__":
    main()

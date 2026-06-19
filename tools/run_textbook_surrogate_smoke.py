"""Generate local textbook-surrogate planning smoke artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcpv import Circuit, Component, TextbookSurrogatePlanningClient, draw_artifact
from optcpv.raster import rasterize_svg


OUT_DIR = Path("generated/textbook_surrogate_smoke")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = TextbookSurrogatePlanningClient("textbook_circuit_corpus", retrieval_limit=6)
    summaries = []
    for circuit in (voltage_clamp_like(), right_leg_drive_like()):
        artifact = draw_artifact(circuit, planning_client=client)
        summaries.append(_write_artifact(circuit, artifact, client))
    _write_json(OUT_DIR / "summary.json", summaries)
    _write_contact_sheet(summaries, OUT_DIR / "contact_sheet.png")
    print(json.dumps(summaries, indent=2))


def voltage_clamp_like() -> Circuit:
    return Circuit(
        id="textbook_voltage_clamp_two_opamp",
        title="Two-electrode voltage clamp with buffer and differential amplifier",
        components=[
            Component(id="VC", type="input", pins={"out": "vc"}, label="Vc"),
            Component(id="BUF", type="op_amp", pins={"+": "vm", "-": "sense", "out": "sense"}, label="Buffer Amp"),
            Component(id="DIFF", type="op_amp", pins={"+": "vc", "-": "sense", "out": "drive"}, label="Diff Amp"),
            Component(id="A", type="ammeter", pins={"a": "drive", "b": "icl"}, label="A"),
            Component(id="RO", type="resistor", pins={"a": "icl", "b": "vm"}, label="Ro", role="current_electrode_output_resistance"),
            Component(id="RM", type="resistor", pins={"a": "vm", "b": "gnd"}, label="Rm"),
            Component(id="VM", type="output", pins={"in": "vm"}, label="Vm"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
        ],
    )


def right_leg_drive_like() -> Circuit:
    return Circuit(
        id="textbook_right_leg_drive",
        motif="ecg_right_leg_drive",
        title="ECG main channel with driven right leg auxiliary loop",
        components=[
            Component(id="E_PLUS", type="input", pins={"out": "ecg_plus"}, label="E+"),
            Component(id="VCM", type="input", pins={"out": "common_mode"}, label="Vcm"),
            Component(id="U1", type="op_amp", pins={"+": "ecg_plus", "-": "fb", "out": "ecg_out"}, label="U1"),
            Component(id="RF", type="resistor", pins={"a": "ecg_out", "b": "fb"}, label="Rf", role="feedback"),
            Component(id="A_AUX", type="op_amp", pins={"+": "gnd", "-": "common_mode", "out": "right_leg_drive"}, label="Aaux", role="right_leg_drive"),
            Component(id="R_RL", type="resistor", pins={"a": "right_leg_drive", "b": "right_leg_electrode"}, label="Rrl"),
            Component(id="RL", type="output", pins={"in": "right_leg_electrode"}, label="RL", role="body_terminal"),
            Component(id="GND", type="ground", pins={"gnd": "gnd"}),
        ],
    )


def _write_artifact(circuit: Circuit, artifact, client: TextbookSurrogatePlanningClient) -> dict:
    stem = circuit.id
    svg_path = OUT_DIR / f"{stem}.svg"
    png_path = OUT_DIR / f"{stem}.png"
    middle_path = OUT_DIR / f"{stem}.middle_layer.txt"
    json_path = OUT_DIR / f"{stem}.artifact.json"

    svg_path.write_text(artifact.svg, encoding="utf-8")
    middle_path.write_text(client.last_middle_layer_text, encoding="utf-8")
    _write_json(
        json_path,
        {
            "layout_support": artifact.layout_support,
            "planning_hints_used": artifact.planning_hints_used,
            "critic_report": artifact.critic_report,
            "optimization_log": artifact.optimization_log,
            "warnings": artifact.warnings,
        },
    )
    raster = rasterize_svg(artifact.svg, output_width=artifact.viewbox["width"], output_height=artifact.viewbox["height"])
    Image.fromarray(raster.rgba).save(png_path)

    report = artifact.critic_report or {}
    return {
        "id": circuit.id,
        "score": report.get("score"),
        "hard_fail": report.get("hard_fail"),
        "planning_hints_source": (artifact.planning_hints_used or {}).get("source"),
        "retrieved_figures": [card.figure_id for card in client.last_cards],
        "svg": str(svg_path),
        "png": str(png_path),
        "middle_layer": str(middle_path),
        "artifact": str(json_path),
    }


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_contact_sheet(summaries: list[dict], path: Path) -> None:
    thumb_width = 760
    thumb_height = 360
    gutter = 28
    header = 46
    sheet = Image.new("RGB", (thumb_width + 2 * gutter, len(summaries) * (thumb_height + header + gutter) + gutter), "white")
    draw = ImageDraw.Draw(sheet)
    for index, summary in enumerate(summaries):
        y = gutter + index * (thumb_height + header + gutter)
        title = f"{summary['id']} | score {summary['score']} | hard={summary['hard_fail']} | hints={summary['planning_hints_source']}"
        draw.text((gutter, y), title, fill=(17, 24, 39))
        image = Image.open(summary["png"]).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(image, (gutter + (thumb_width - image.width) // 2, y + header))
    sheet.save(path)


if __name__ == "__main__":
    main()

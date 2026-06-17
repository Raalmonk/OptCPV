"""Generate reproducible seven-op-amp ECG-style demo artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcpv import Circuit, Component, draw_artifact, draw_optimized_artifact
from optcpv.raster import rasterize_svg


OUTPUT_DIR = Path("generated/ecg_7opamp_demo")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        opamp_chain("ecg_7opamp_clean_chain", input_label="ECG IN", output_label="ECG OUT"),
        opamp_chain("ecg_7opamp_chain", input_label="ECG IN", output_label="ECG OUT"),
        opamp_chain_with_monitors(),
        verbose_label_chain(),
    ]

    summaries = []
    for circuit in cases:
        summaries.append(_export_case(circuit))
    _write_contact_sheet(summaries, OUTPUT_DIR / "contact_sheet_all.png")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def opamp_chain(circuit_id: str, *, input_label: str = "VIN", output_label: str = "VOUT") -> Circuit:
    components = [
        Component(id="VIN", type="input", pins={"out": "vin"}, label=input_label),
        Component(id="VOUT", type="output", pins={"in": "o7"}, label=output_label),
        Component(id="GND", type="ground", pins={"gnd": "gnd"}, label="GND"),
    ]
    previous = "vin"
    for index in range(1, 8):
        output = f"o{index}"
        feedback = f"fb{index}"
        components.extend(
            [
                Component(id=f"U{index}", type="op_amp", pins={"+": previous, "-": feedback, "out": output}, label=f"U{index}"),
                Component(
                    id=f"Rf{index}",
                    type="resistor",
                    pins={"a": output, "b": feedback},
                    label=f"Rf{index}",
                    role="feedback",
                ),
                Component(id=f"Rg{index}", type="resistor", pins={"a": feedback, "b": "gnd"}, label=f"Rg{index}", role="gain"),
            ]
        )
        previous = output
    return Circuit(id=circuit_id, motif="op_amp_network", title="Seven-op-amp ECG gain chain", components=components)


def opamp_chain_with_monitors() -> Circuit:
    circuit = opamp_chain("ecg_7opamp_chain_with_monitors", input_label="ECG IN", output_label="ECG OUT")
    return Circuit(
        id=circuit.id,
        motif=circuit.motif,
        title="Seven-op-amp ECG gain chain with monitor outputs",
        components=[
            *circuit.components,
            Component(id="VMON3", type="output", pins={"in": "o3"}, label="VMON3"),
            Component(id="VMON5", type="output", pins={"in": "o5"}, label="VMON5"),
        ],
    )


def verbose_label_chain() -> Circuit:
    circuit = opamp_chain("ecg_7opamp_verbose_labels", input_label="ECG INPUT NODE", output_label="ECG OUTPUT NODE")
    verbose_components = []
    for component in circuit.components:
        label = component.label
        if component.type in {"op_amp", "resistor"}:
            label = f"{component.id}_ecg_front_end_gain_stage"
        verbose_components.append(
            Component(
                id=component.id,
                type=component.type,
                pins=dict(component.pins),
                label=label,
                role=component.role,
                value=component.value,
            )
        )
    return Circuit(id=circuit.id, motif=circuit.motif, title=circuit.title, components=verbose_components)


def _export_case(circuit: Circuit) -> dict:
    raw = draw_artifact(circuit)
    optimized = draw_optimized_artifact(circuit, max_iterations=3)
    svg_path = OUTPUT_DIR / f"{circuit.id}.optimized.svg"
    png_path = OUTPUT_DIR / f"{circuit.id}.optimized.png"
    json_path = OUTPUT_DIR / f"{circuit.id}.artifact.json"
    raw_svg_path = OUTPUT_DIR / f"{circuit.id}.raw.svg"

    raw_svg_path.write_text(raw.svg, encoding="utf-8")
    svg_path.write_text(optimized.svg, encoding="utf-8")
    _write_png(optimized.svg, png_path, optimized.viewbox["width"], optimized.viewbox["height"])

    payload = _artifact_payload(optimized)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = optimized.critic_report or {}
    raw_report = raw.critic_report or {}
    return {
        "id": circuit.id,
        "component_count": len(circuit.components),
        "op_amp_count": sum(1 for component in circuit.components if "op" in component.type),
        "raw_score": raw_report.get("score", 0),
        "optimized_score": report.get("score", 0),
        "hard_fail": report.get("hard_fail", False),
        "violation_codes": [violation["code"] for violation in report.get("violations", [])],
        "violations": report.get("violations", []),
        "layout_support": optimized.layout_support,
        "optimization_log": optimized.optimization_log,
        "svg": str(svg_path),
        "png": str(png_path),
        "json": str(json_path),
    }


def _artifact_payload(artifact) -> dict:
    return {
        "components": artifact.components,
        "nets": artifact.nets,
        "labels": artifact.labels,
        "viewbox": artifact.viewbox,
        "layout_support": artifact.layout_support,
        "critic_report": artifact.critic_report,
        "vector_report": artifact.vector_report,
        "cv_report": artifact.cv_report,
        "combined_report": artifact.combined_report,
        "optimization_log": artifact.optimization_log,
        "warnings": artifact.warnings,
    }


def _write_png(svg: str, path: Path, width: int, height: int) -> None:
    raster = rasterize_svg(svg, output_width=width, output_height=height)
    Image.fromarray(raster.rgba).save(path)


def _write_contact_sheet(summaries: list[dict], path: Path) -> None:
    thumb_width = 760
    thumb_height = 330
    gutter = 28
    header = 46
    cols = 2
    rows = (len(summaries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width + (cols + 1) * gutter, rows * (thumb_height + header) + (rows + 1) * gutter), "white")
    draw = ImageDraw.Draw(sheet)
    for index, summary in enumerate(summaries):
        row, col = divmod(index, cols)
        x = gutter + col * (thumb_width + gutter)
        y = gutter + row * (thumb_height + header + gutter)
        title = f"{summary['id']} | score {summary['optimized_score']} | hard={summary['hard_fail']}"
        draw.text((x, y), title, fill=(17, 24, 39))
        image = Image.open(summary["png"]).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(image, (x + (thumb_width - image.width) // 2, y + header))
    sheet.save(path)


if __name__ == "__main__":
    main()


"""Extract textbook figure crops and problem statements into a corpus folder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


DEFAULT_PDF = (
    "Medical Instrumentation Application and Design, 5th Edition -- John G_ Webster, Amit J_ Nimunkar -- 5, 2020 -- "
    "John Wiley & Sons, Incorporated -- 9781119457312.pdf"
)
DEFAULT_OUTPUT = "textbook_circuit_corpus"
PDFTOPPM = "/Users/Raalm/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pdftoppm"
CAPTION_RE = re.compile(r"\bFigure\s+([A-Z]?\d+(?:\.\d+)+(?:\([a-z]\))?)", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+)\b", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\bFigure\s+([A-Z]?\d+(?:\.\d+)+)", re.IGNORECASE)


FIGURE_KEYWORDS = {
    "schematic_circuit": [
        "circuit",
        "op-amp",
        "op amp",
        "amplifier",
        "filter",
        "rectifier",
        "integrator",
        "differentiator",
        "comparator",
        "bridge",
        "wheatstone",
        "resistor",
        "capacitor",
        "diode",
        "transistor",
        "impedance",
        "voltage clamp",
        "current source",
        "charge amplifier",
        "equivalent circuit",
    ],
    "bioelectric_recording": [
        "ecg",
        "eeg",
        "emg",
        "eog",
        "biopotential",
        "electrode",
        "lead",
        "recording system",
        "right-leg",
        "right leg",
    ],
    "sensor_or_transducer": [
        "sensor",
        "transducer",
        "strain gage",
        "thermistor",
        "thermocouple",
        "photodiode",
        "piezoelectric",
        "lvdt",
    ],
    "system_block_diagram": [
        "system",
        "block",
        "instrumentation",
        "processor",
        "display",
        "controller",
        "feedback control",
    ],
    "waveform_or_plot": [
        "waveform",
        "plot",
        "response",
        "curve",
        "characteristic",
        "spectrum",
        "frequency response",
        "calibration",
    ],
    "anatomy_or_physiology": [
        "heart",
        "blood",
        "nerve",
        "muscle",
        "cell",
        "axon",
        "lung",
        "tissue",
        "organ",
    ],
}

PROBLEM_KEYWORDS = {
    "design_circuit": ["design", "circuit", "op-amp", "op amp", "amplifier", "filter", "comparator", "rectifier"],
    "calculate": ["calculate", "find", "determine", "derive", "what is", "value"],
    "plot_or_sketch": ["plot", "sketch", "draw"],
    "simulate": ["ltspice", "simulation", "simulate"],
    "bioelectric": ["ecg", "eeg", "emg", "eog", "electrode", "biopotential"],
    "sensor": ["sensor", "transducer", "thermistor", "thermocouple", "strain", "lvdt"],
}


@dataclass(frozen=True)
class TextLine:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    first_font: str = ""
    first_size: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "x0": self.x0,
            "top": self.top,
            "x1": self.x1,
            "bottom": self.bottom,
            "first_font": self.first_font,
            "first_size": self.first_size,
        }


def main() -> None:
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--limit-pages", type=int, default=0, help="Debug limit; 0 means all pages.")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    out_dir = Path(args.out).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    from pypdf import PdfReader

    _reset_output(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "problems").mkdir(parents=True, exist_ok=True)
    (out_dir / "indexes").mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    outline = _outline_items(reader)
    chapters = _chapter_index(outline)
    problem_sections = _problem_sections(reader, outline, chapters)

    figure_records = _extract_figures(
        pdf_path=pdf_path,
        out_dir=out_dir,
        dpi=args.dpi,
        page_limit=args.limit_pages,
        skip_render=args.skip_render,
    )
    problem_records = _extract_problems(reader, out_dir, problem_sections)

    manifest = {
        "source_pdf": str(pdf_path),
        "output_dir": str(out_dir),
        "pages": len(reader.pages),
        "chapters": chapters,
        "counts": {
            "figures": len(figure_records),
            "likely_circuit_figures": sum(1 for item in figure_records if item["is_likely_circuit"]),
            "problems": len(problem_records),
            "circuit_or_design_problems": sum(1 for item in problem_records if item["is_circuit_or_design_problem"]),
        },
        "indexes": {
            "figures": "indexes/figures.jsonl",
            "likely_circuit_figures": "indexes/likely_circuit_figures.jsonl",
            "problems": "indexes/problems.jsonl",
            "circuit_or_design_problems": "indexes/circuit_or_design_problems.jsonl",
            "classified": "classified",
        },
        "classification_notes": {
            "figure_classifier": "keyword-based first pass over captions; all figure crops are retained for later review.",
            "problem_classifier": "keyword-based first pass over problem statements; all problem text is retained.",
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_jsonl(out_dir / "indexes" / "figures.jsonl", figure_records)
    _write_jsonl(out_dir / "indexes" / "likely_circuit_figures.jsonl", [item for item in figure_records if item["is_likely_circuit"]])
    _write_jsonl(out_dir / "indexes" / "problems.jsonl", problem_records)
    _write_jsonl(
        out_dir / "indexes" / "circuit_or_design_problems.jsonl",
        [item for item in problem_records if item["is_circuit_or_design_problem"]],
    )
    _write_category_catalogs(out_dir, figure_records, problem_records)
    _write_readme(out_dir, manifest)
    print(json.dumps(manifest["counts"], indent=2))


def _reset_output(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def _outline_items(reader: PdfReader) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def walk(raw_items: list[Any], depth: int = 0) -> None:
        for item in raw_items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            title = str(getattr(item, "title", str(item))).replace("\r", " ").strip()
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                page = None
            items.append({"title": title, "page": page, "depth": depth})

    try:
        walk(reader.outline)
    except Exception:
        return []
    return items


def _chapter_index(outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for item in outline:
        match = CHAPTER_RE.match(item["title"])
        if match and item["page"] is not None:
            chapters.append({"chapter": int(match.group(1)), "title": item["title"], "page": item["page"]})
    for index, chapter in enumerate(chapters):
        next_page = chapters[index + 1]["page"] if index + 1 < len(chapters) else None
        chapter["end_page"] = None if next_page is None else next_page - 1
    return chapters


def _problem_sections(reader: PdfReader, outline: list[dict[str, Any]], chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    chapter_for_page = {chapter["page"]: chapter for chapter in chapters}
    ordered_chapters = sorted(chapters, key=lambda item: item["page"])
    for item in outline:
        if item["page"] is None or item["title"].strip().upper() != "PROBLEMS":
            continue
        chapter = _nearest_chapter_before(ordered_chapters, item["page"])
        if chapter is None:
            continue
        end_page = _first_heading_page(reader, item["page"], chapter.get("end_page") or len(reader.pages), "REFERENCES")
        if end_page is None:
            end_page = chapter.get("end_page") or item["page"]
        sections.append(
            {
                "chapter": chapter["chapter"],
                "chapter_title": chapter["title"],
                "start_page": item["page"],
                "end_page": max(item["page"], end_page - 1),
            }
        )
    return sections


def _nearest_chapter_before(chapters: list[dict[str, Any]], page: int) -> dict[str, Any] | None:
    result = None
    for chapter in chapters:
        if chapter["page"] <= page:
            result = chapter
        else:
            break
    return result


def _first_heading_page(reader: PdfReader, start: int, end: int, heading: str) -> int | None:
    target = heading.upper()
    for page_number in range(start, min(end, len(reader.pages)) + 1):
        text = reader.pages[page_number - 1].extract_text() or ""
        lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
        if target in lines:
            return page_number
    return None


def _extract_figures(
    *,
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    page_limit: int,
    skip_render: bool,
) -> list[dict[str, Any]]:
    import pdfplumber

    records: list[dict[str, Any]] = []
    page_cache = out_dir / "_page_cache"
    page_cache.mkdir(parents=True, exist_ok=True)
    rendered_pages: dict[int, Path] = {}
    seen_ids: set[tuple[str, int]] = set()

    with pdfplumber.open(str(pdf_path)) as doc:
        page_count = len(doc.pages) if page_limit <= 0 else min(page_limit, len(doc.pages))
        for page_index in range(page_count):
            page_number = page_index + 1
            page = doc.pages[page_index]
            lines = _extract_lines(page)
            for line_index, line in enumerate(lines):
                if not _is_caption_line(line.text, line):
                    continue
                for match in CAPTION_RE.finditer(line.text):
                    figure_id = match.group(1)
                    key = (figure_id.lower(), page_number)
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    caption_lines = _caption_lines(lines, line_index)
                    caption = _clean(" ".join(item.text for item in caption_lines))
                    crop = _caption_crop_bbox(page, lines, line_index, caption_lines)
                    classification = _classify_figure(caption)
                    figure_dir = out_dir / "figures" / _safe_id(f"figure_{figure_id}_p{page_number:03d}")
                    figure_dir.mkdir(parents=True, exist_ok=True)
                    image_rel = None
                    if not skip_render:
                        rendered = rendered_pages.get(page_number)
                        if rendered is None:
                            rendered = _render_page(pdf_path, page_number, page_cache, dpi)
                            rendered_pages[page_number] = rendered
                        image_rel = str((figure_dir / "crop.png").relative_to(out_dir))
                        _crop_page_image(rendered, page.width, page.height, crop, out_dir / image_rel)
                    record = {
                        "figure_id": figure_id,
                        "page": page_number,
                        "caption": caption,
                        "caption_bbox": _merge_line_bbox(caption_lines),
                        "crop_bbox": {"x0": crop[0], "top": crop[1], "x1": crop[2], "bottom": crop[3]},
                        "image": image_rel,
                        "categories": classification["categories"],
                        "primary_category": classification["primary_category"],
                        "is_likely_circuit": classification["is_likely_circuit"],
                        "confidence": classification["confidence"],
                    }
                    _write_json(figure_dir / "metadata.json", record)
                    records.append(record)
    shutil.rmtree(page_cache, ignore_errors=True)
    return records


def _extract_lines(page) -> list[TextLine]:
    words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False, extra_attrs=["fontname", "size"])
    if not words:
        return []
    words = sorted(words, key=lambda item: (item["top"], item["x0"]))
    rows: list[list[dict[str, Any]]] = []
    for word in words:
        if not rows or abs(word["top"] - _row_top(rows[-1])) > 3.0:
            rows.append([word])
        else:
            rows[-1].append(word)
    lines = []
    for row in rows:
        row = sorted(row, key=lambda item: item["x0"])
        text = " ".join(str(word["text"]) for word in row)
        lines.append(
            TextLine(
                text=text,
                x0=min(float(word["x0"]) for word in row),
                top=min(float(word["top"]) for word in row),
                x1=max(float(word["x1"]) for word in row),
                bottom=max(float(word["bottom"]) for word in row),
                first_font=str(row[0].get("fontname") or ""),
                first_size=float(row[0].get("size") or 0.0),
            )
        )
    return lines


def _row_top(row: list[dict[str, Any]]) -> float:
    return sum(float(word["top"]) for word in row) / len(row)


def _caption_lines(lines: list[TextLine], start_index: int) -> list[TextLine]:
    result = [lines[start_index]]
    start_top = lines[start_index].top
    for line in lines[start_index + 1 : start_index + 8]:
        if line.top - start_top > 110:
            break
        if _looks_like_new_section(line.text):
            break
        result.append(line)
        if line.text.endswith(".") and len(result) >= 2:
            break
    return result


def _looks_like_new_section(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    if re.match(r"^\d+\.\d+\s+[A-Z]", cleaned):
        return True
    return cleaned.isupper() and len(cleaned) > 8 and "FIGURE" not in cleaned


def _caption_crop_bbox(page, lines: list[TextLine], caption_index: int, caption_lines: list[TextLine]) -> tuple[float, float, float, float]:
    caption_top = min(line.top for line in caption_lines)
    caption_bottom = max(line.bottom for line in caption_lines)
    x0 = 24.0
    x1 = page.width - 24.0
    graphic_crop = _graphics_anchored_crop(page, lines, caption_index, caption_top, caption_bottom)
    if graphic_crop is not None:
        return graphic_crop
    tall_figure_band = min(430.0, page.height * 0.72)
    caption_margin = 36.0
    if caption_top < page.height * 0.38:
        next_caption_top = min([line.top for line in lines[caption_index + 1 :] if _is_caption_line(line.text, line)] or [page.height])
        top = max(0.0, caption_top - 28.0)
        bottom = min(page.height, next_caption_top - 8.0, caption_bottom + tall_figure_band)
    else:
        next_caption_top = min([line.top for line in lines[caption_index + 1 :] if _is_caption_line(line.text, line)] or [page.height])
        previous_caption_bottom = max(
            [line.bottom for line in lines[:caption_index] if _is_caption_line(line.text, line)] or [0.0]
        )
        top = max(previous_caption_bottom + 10.0, caption_top - tall_figure_band, 0.0)
        bottom = min(caption_bottom + caption_margin, next_caption_top - 8.0, page.height)
    if bottom - top < 120:
        top = max(0.0, caption_top - tall_figure_band)
        bottom = min(page.height, caption_bottom + caption_margin)
    return (x0, top, x1, bottom)


def _graphics_anchored_crop(
    page,
    lines: list[TextLine],
    caption_index: int,
    caption_top: float,
    caption_bottom: float,
) -> tuple[float, float, float, float] | None:
    graphic_boxes = _page_graphic_boxes(page)
    if not graphic_boxes:
        return None
    x0 = 24.0
    x1 = page.width - 24.0
    next_caption_top = min([line.top for line in lines[caption_index + 1 :] if _is_caption_line(line.text, line)] or [page.height])
    previous_caption_bottom = max([line.bottom for line in lines[:caption_index] if _is_caption_line(line.text, line)] or [0.0])

    above_clusters = _graphic_clusters_in_band(
        graphic_boxes,
        max(previous_caption_bottom + 8.0, 0.0),
        caption_top - 3.0,
    )
    below_clusters = _graphic_clusters_in_band(
        graphic_boxes,
        caption_bottom + 3.0,
        min(next_caption_top - 8.0, page.height),
    )
    above = max(above_clusters, key=lambda box: box[3]) if above_clusters else None
    below = min(below_clusters, key=lambda box: box[1]) if below_clusters else None
    if above is None and below is None:
        return None
    above_distance = caption_top - above[3] if above is not None else float("inf")
    below_distance = below[1] - caption_bottom if below is not None else float("inf")

    if above_distance <= below_distance:
        top = max(0.0, above[1] - 14.0)
        bottom = min(page.height, caption_bottom + 18.0, next_caption_top - 8.0)
    else:
        top = max(0.0, caption_top - 28.0)
        bottom = min(page.height, below[3] + 14.0, next_caption_top - 8.0)
    if bottom - top >= 100:
        return (x0, top, x1, bottom)
    return None


def _page_graphic_boxes(page) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for collection_name in ("lines", "curves", "rects", "images"):
        for item in getattr(page, collection_name, []) or []:
            try:
                x0 = float(item["x0"])
                x1 = float(item["x1"])
                top = float(item["top"])
                bottom = float(item["bottom"])
            except (KeyError, TypeError, ValueError):
                continue
            if x1 - x0 < 0.5 and bottom - top < 0.5:
                continue
            boxes.append((x0, top, x1, bottom))
    return boxes


def _graphic_clusters_in_band(
    boxes: list[tuple[float, float, float, float]],
    band_top: float,
    band_bottom: float,
) -> list[tuple[float, float, float, float]]:
    in_band = [box for box in boxes if box[3] >= band_top and box[1] <= band_bottom]
    if not in_band:
        return []
    clusters: list[tuple[float, float, float, float]] = []
    vertical_gap = 55.0
    for box in sorted(in_band, key=lambda item: item[1]):
        clipped = (box[0], max(box[1], band_top), box[2], min(box[3], band_bottom))
        if clipped[3] - clipped[1] < 0.5:
            continue
        if not clusters or clipped[1] > clusters[-1][3] + vertical_gap:
            clusters.append(clipped)
            continue
        previous = clusters[-1]
        clusters[-1] = (
            min(previous[0], clipped[0]),
            min(previous[1], clipped[1]),
            max(previous[2], clipped[2]),
            max(previous[3], clipped[3]),
        )
    return [box for box in clusters if (box[2] - box[0]) >= 20.0 and (box[3] - box[1]) >= 15.0]


def _is_caption_line(text: str, line: TextLine | None = None) -> bool:
    cleaned = text.strip()
    match = CAPTION_RE.match(cleaned)
    if not match:
        return False
    if line is not None and not _has_caption_marker_font(line):
        return False
    after = cleaned[match.end() :].strip().lower()
    after = after.lstrip(".:;,- ")
    if not after:
        return False
    reference_starters = (
        "figure ",
        "shows",
        "show",
        "is ",
        "are ",
        "changes",
        "changed",
        "illustrates",
        "depicts",
        "contains",
        "presents",
        "describes",
        "can ",
        "may ",
        "must ",
        "will ",
        "appears",
        "provides",
        "travels",
    )
    return not after.startswith(reference_starters)


def _has_caption_marker_font(line: TextLine) -> bool:
    font = line.first_font.lower()
    if "antiqueolive" in font or "black" in font:
        return True
    return 6.5 <= line.first_size <= 8.6 and line.x0 <= 64.0


def _render_page(pdf_path: Path, page_number: int, page_cache: Path, dpi: int) -> Path:
    prefix = page_cache / f"page_{page_number:04d}"
    output = prefix.with_suffix(".png")
    if output.exists():
        return output
    subprocess.run(
        [PDFTOPPM, "-png", "-singlefile", "-r", str(dpi), "-f", str(page_number), "-l", str(page_number), str(pdf_path), str(prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return output


def _crop_page_image(
    rendered_page: Path,
    page_width: float,
    page_height: float,
    crop: tuple[float, float, float, float],
    output: Path,
) -> None:
    from PIL import Image

    image = Image.open(rendered_page).convert("RGB")
    width, height = image.size
    x0, top, x1, bottom = crop
    box = (
        max(0, int(round(x0 / page_width * width))),
        max(0, int(round(top / page_height * height))),
        min(width, int(round(x1 / page_width * width))),
        min(height, int(round(bottom / page_height * height))),
    )
    image.crop(box).save(output)


def _extract_problems(reader: PdfReader, out_dir: Path, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    for section in sections:
        raw_text = "\n".join(_page_text(reader, page) for page in range(section["start_page"], section["end_page"] + 1))
        raw_text = _problem_section_text(raw_text)
        records = _split_problem_records(raw_text, section)
        chapter_path = out_dir / "problems" / f"chapter_{section['chapter']:02d}.jsonl"
        _write_jsonl(chapter_path, records)
        all_records.extend(records)
    return all_records


def _page_text(reader: PdfReader, page_number: int) -> str:
    if page_number < 1 or page_number > len(reader.pages):
        return ""
    return reader.pages[page_number - 1].extract_text() or ""


def _problem_section_text(text: str) -> str:
    upper = text.upper()
    start = upper.find("PROBLEMS")
    if start >= 0:
        text = text[start + len("PROBLEMS") :]
    upper = text.upper()
    end = upper.find("REFERENCES")
    if end >= 0:
        text = text[:end]
    return text


def _split_problem_records(text: str, section: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    problem_id_re = re.compile(rf"(?m)(?=^\s*({section['chapter']}\.\d+)\s+)")
    starts = [(match.start(), match.group(1)) for match in problem_id_re.finditer(normalized)]
    records: list[dict[str, Any]] = []
    for index, (start, problem_id) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(normalized)
        block = normalized[start:end].strip()
        if not block:
            continue
        statement = _clean(block)
        classification = _classify_problem(statement)
        records.append(
            {
                "problem_id": problem_id,
                "chapter": section["chapter"],
                "chapter_title": section["chapter_title"],
                "start_page": section["start_page"],
                "end_page": section["end_page"],
                "statement": statement,
                "figure_refs": sorted(set(FIGURE_REF_RE.findall(statement))),
                "categories": classification["categories"],
                "primary_category": classification["primary_category"],
                "is_circuit_or_design_problem": classification["is_circuit_or_design_problem"],
                "confidence": classification["confidence"],
            }
        )
    return records


def _classify_figure(text: str) -> dict[str, Any]:
    lowered = text.lower()
    categories = [category for category, keywords in FIGURE_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    if not categories:
        categories = ["uncategorized_figure"]
    likely = bool({"schematic_circuit", "bioelectric_recording", "sensor_or_transducer"} & set(categories))
    if "waveform_or_plot" in categories and "schematic_circuit" not in categories:
        likely = False
    primary = categories[0]
    confidence = min(0.95, 0.35 + 0.15 * len(categories) + (0.25 if likely else 0.0))
    return {"categories": categories, "primary_category": primary, "is_likely_circuit": likely, "confidence": round(confidence, 3)}


def _classify_problem(text: str) -> dict[str, Any]:
    lowered = text.lower()
    categories = [category for category, keywords in PROBLEM_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    if not categories:
        categories = ["uncategorized_problem"]
    likely = bool({"design_circuit", "bioelectric", "sensor"} & set(categories))
    primary = categories[0]
    confidence = min(0.95, 0.35 + 0.12 * len(categories) + (0.25 if likely else 0.0))
    return {
        "categories": categories,
        "primary_category": primary,
        "is_circuit_or_design_problem": likely,
        "confidence": round(confidence, 3),
    }


def _merge_line_bbox(lines: list[TextLine]) -> dict[str, float]:
    return {
        "x0": min(line.x0 for line in lines),
        "top": min(line.top for line in lines),
        "x1": max(line.x1 for line in lines),
        "bottom": max(line.bottom for line in lines),
    }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_category_catalogs(out_dir: Path, figure_records: list[dict[str, Any]], problem_records: list[dict[str, Any]]) -> None:
    classified = out_dir / "classified"
    figures_root = classified / "figures_by_primary"
    likely_root = classified / "likely_circuit_figures_by_primary"
    problems_root = classified / "problems_by_primary"
    circuit_problems_root = classified / "circuit_or_design_problems_by_primary"

    figure_groups = _group_by_primary(figure_records)
    likely_groups = _group_by_primary([item for item in figure_records if item["is_likely_circuit"]])
    problem_groups = _group_by_primary(problem_records)
    circuit_problem_groups = _group_by_primary([item for item in problem_records if item["is_circuit_or_design_problem"]])

    for root, groups, link_images in (
        (figures_root, figure_groups, True),
        (likely_root, likely_groups, True),
        (problems_root, problem_groups, False),
        (circuit_problems_root, circuit_problem_groups, False),
    ):
        for category, records in sorted(groups.items()):
            category_dir = root / _safe_id(category)
            category_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(category_dir / "index.jsonl", records)
            if link_images:
                images_dir = category_dir / "images"
                images_dir.mkdir(exist_ok=True)
                for record in records:
                    image = record.get("image")
                    if not image:
                        continue
                    src = out_dir / image
                    dest = images_dir / _safe_id(f"{record['figure_id']}_p{record['page']:03d}.png")
                    _link_or_copy(src, dest)

    lines = [
        "# Classified Corpus",
        "",
        "Browsable category views generated from the main JSONL indexes.",
        "",
        "Figure directories contain `index.jsonl` plus image links in `images/`.",
        "Problem directories contain `index.jsonl` records grouped by primary category.",
        "",
        "## Counts",
        "",
        f"- Figure primary categories: {len(figure_groups)}",
        f"- Likely circuit figure primary categories: {len(likely_groups)}",
        f"- Problem primary categories: {len(problem_groups)}",
        f"- Circuit/design problem primary categories: {len(circuit_problem_groups)}",
        "",
    ]
    (classified / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _group_by_primary(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("primary_category") or "uncategorized"), []).append(record)
    return groups


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    relative_source = os.path.relpath(src, dest.parent)
    try:
        dest.symlink_to(relative_source)
    except OSError:
        shutil.copy2(src, dest)


def _write_readme(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Textbook Circuit Corpus",
        "",
        "Generated from the local textbook PDF for OptCPV schematic extraction and visual-layout regression.",
        "",
        f"- Source PDF: `{Path(manifest['source_pdf']).name}`",
        f"- Pages: {manifest['pages']}",
        f"- Figures: {manifest['counts']['figures']}",
        f"- Likely circuit figures: {manifest['counts']['likely_circuit_figures']}",
        f"- Problems: {manifest['counts']['problems']}",
        f"- Circuit/design problems: {manifest['counts']['circuit_or_design_problems']}",
        "",
        "Indexes:",
        "",
        "- `indexes/figures.jsonl`",
        "- `indexes/likely_circuit_figures.jsonl`",
        "- `indexes/problems.jsonl`",
        "- `indexes/circuit_or_design_problems.jsonl`",
        "- `classified/`",
        "",
        "Each `figures/figure_*/metadata.json` stores the page, caption, crop bbox, classification, and crop image path.",
        "`classified/` groups records by primary category and links likely circuit images into browsable category folders.",
        "The classifier is intentionally a first pass; later OptCPV/Gemini review should use these records as training and regression material.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

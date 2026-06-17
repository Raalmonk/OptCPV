"""Label layout helpers shared by planners and renderers."""

from __future__ import annotations

import re


def wrap_label_lines(text: str, *, max_chars: int = 14, max_lines: int = 3) -> list[str]:
    normalized = re.sub(r"[_/]+", " ", text.strip())
    words = [word for word in normalized.split() if word]
    if not words:
        return [text]

    lines: list[str] = []
    current = ""
    for word in words:
        chunks = _word_chunks(word, max_chars)
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current) + 1 + len(chunk) <= max_chars:
                current = f"{current} {chunk}"
            else:
                lines.append(current)
                current = chunk
    if current:
        lines.append(current)

    if len(lines) <= max_lines:
        return lines
    kept = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :])
    kept.append(_ellipsize(tail, max_chars))
    return kept


def label_bbox_size(text: str) -> tuple[float, float]:
    lines = wrap_label_lines(text)
    width = max(0.7, 0.145 * max(len(line) for line in lines) + 0.35)
    height = max(0.38, 0.36 * len(lines) + 0.06)
    return width, height


def component_display_label(
    component_id: str,
    component_type: str,
    label: str | None = None,
    value: str | None = None,
) -> str:
    text = label or value or component_id
    if _key(component_id).startswith("vmon") or "monitor" in _key(text):
        return component_id
    if len(text) <= 18:
        return text
    stage_index = _key(text).find("_stage_")
    if stage_index > 0:
        prefix = text[:stage_index].strip("_-/ ")
        if prefix and len(prefix) <= 12:
            return prefix
    normalized_text = _key(text)
    normalized_id = _key(component_id)
    if normalized_text == normalized_id or normalized_text.startswith(f"{normalized_id}_"):
        return component_id
    if _is_symbolic_component(component_type):
        return component_id
    return text


def display_label_text(text: str) -> str:
    return "\n".join(wrap_label_lines(text))


def _word_chunks(word: str, max_chars: int) -> list[str]:
    if len(word) <= max_chars:
        return [word]
    return [word[index : index + max_chars] for index in range(0, len(word), max_chars)]


def _ellipsize(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max(1, max_chars - 3)]}..."


def _key(value: str | None) -> str:
    return (value or "").lower().replace("-", "_").replace(" ", "_")


def _is_symbolic_component(component_type: str) -> bool:
    key = _key(component_type)
    return "op_amp" in key or "opamp" in key or key in {"resistor", "capacitor"} or key.startswith(("r", "c"))

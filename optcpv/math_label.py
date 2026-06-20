"""Small SVG text renderer for schematic math labels.

This is intentionally not a full TeX engine. Schematic labels mostly need
subscripts, superscripts, and a few common unit/symbol commands while remaining
plain SVG text that can be searched and styled.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True)
class MathSegment:
    text: str
    role: str = "normal"


COMMAND_SYMBOLS = {
    "Omega": "\u03a9",
    "ohm": "\u03a9",
    "mu": "\u03bc",
    "micro": "\u03bc",
    "alpha": "\u03b1",
    "beta": "\u03b2",
    "gamma": "\u03b3",
    "Delta": "\u0394",
    "delta": "\u03b4",
    "pi": "\u03c0",
    "tau": "\u03c4",
    "times": "\u00d7",
    "cdot": "\u00b7",
    "degree": "\u00b0",
}


def svg_math_text(text: str) -> str:
    return "".join(_segment_svg(segment) for segment in math_segments(text))


def svg_math_line_tspan(text: str, *, x: float, y: float) -> str:
    return f'<tspan x="{x:.1f}" y="{y:.1f}">{svg_math_text(text)}</tspan>'


def math_segments(text: str) -> list[MathSegment]:
    normalized = _strip_math_delimiters(text)
    segments, _ = _parse(normalized, 0, stop_at=None)
    return _merge_segments(segments)


def plain_math_text(text: str) -> str:
    return "".join(segment.text for segment in math_segments(text))


def _segment_svg(segment: MathSegment) -> str:
    if not segment.text:
        return ""
    if segment.role == "sub":
        return (
            '<tspan baseline-shift="sub" font-size="70%">'
            f"{escape(segment.text)}</tspan>"
        )
    if segment.role == "sup":
        return (
            '<tspan baseline-shift="super" font-size="70%">'
            f"{escape(segment.text)}</tspan>"
        )
    return escape(segment.text)


def _strip_math_delimiters(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == "$" and stripped[-1] == "$":
        return stripped.strip("$").strip()
    if stripped.startswith("\\(") and stripped.endswith("\\)"):
        return stripped[2:-2].strip()
    return text


def _parse(text: str, index: int, *, stop_at: str | None) -> tuple[list[MathSegment], int]:
    segments: list[MathSegment] = []
    i = index
    while i < len(text):
        char = text[i]
        if stop_at is not None and char == stop_at:
            return segments, i + 1
        if char in {"_", "^"}:
            role = "sub" if char == "_" else "sup"
            value, i = _read_script_value(text, i + 1)
            if value:
                segments.append(MathSegment(plain_math_text(value), role))
            else:
                segments.append(MathSegment(char))
            continue
        if char == "\\":
            command_segments, i = _read_command(text, i + 1)
            segments.extend(command_segments)
            continue
        if char == "{":
            inner, i = _parse(text, i + 1, stop_at="}")
            segments.extend(inner)
            continue
        if char == "}":
            return segments, i + 1
        segments.append(MathSegment(char))
        i += 1
    return segments, i


def _read_script_value(text: str, index: int) -> tuple[str, int]:
    i = index
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return "", i
    if text[i] == "{":
        return _read_group_text(text, i + 1)
    if text[i] == "\\":
        command_segments, next_i = _read_command(text, i + 1)
        return "".join(segment.text for segment in command_segments), next_i
    return text[i], i + 1


def _read_group_text(text: str, index: int) -> tuple[str, int]:
    depth = 1
    i = index
    start = index
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return text[start:], len(text)


def _read_command(text: str, index: int) -> tuple[list[MathSegment], int]:
    if index >= len(text):
        return [MathSegment("\\")], index

    if text[index] in {"_", "^", "{", "}", "$", "\\"}:
        return [MathSegment(text[index])], index + 1
    if text[index] in {",", " "}:
        return [MathSegment(" ")], index + 1

    start = index
    while index < len(text) and text[index].isalpha():
        index += 1
    command = text[start:index]
    if not command:
        return [MathSegment(text[start])], start + 1

    if command in {"mathrm", "text", "operatorname"} and index < len(text) and text[index] == "{":
        group, next_i = _read_group_text(text, index + 1)
        return math_segments(group), next_i
    if command in COMMAND_SYMBOLS:
        return [MathSegment(COMMAND_SYMBOLS[command])], index
    return [MathSegment(command)], index


def _merge_segments(segments: list[MathSegment]) -> list[MathSegment]:
    merged: list[MathSegment] = []
    for segment in segments:
        if not segment.text:
            continue
        if merged and merged[-1].role == segment.role:
            previous = merged[-1]
            merged[-1] = MathSegment(previous.text + segment.text, previous.role)
        else:
            merged.append(segment)
    return merged

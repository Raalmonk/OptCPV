"""Fixed-frame SVG rasterization for visual criticism."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


EVAL_WIDTH = 1200
EVAL_HEIGHT = 800


@dataclass(frozen=True)
class RasterImage:
    width: int
    height: int
    rgba: "object"
    bgr: "object"
    gray: "object"


def rasterize_svg(
    svg: str,
    *,
    output_width: int = EVAL_WIDTH,
    output_height: int = EVAL_HEIGHT,
) -> RasterImage:
    import cv2
    import numpy as np
    from PIL import Image

    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=output_width,
            output_height=output_height,
        )
        image = _white_background(Image.open(BytesIO(png_bytes)).convert("RGBA"))
    except Exception:
        image = _fallback_raster(svg, output_width, output_height)
    rgba = np.asarray(image)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    return RasterImage(width=output_width, height=output_height, rgba=rgba, bgr=bgr, gray=gray)


def _fallback_raster(svg: str, width: int, height: int):
    """Small Pillow fallback for CI hosts missing libcairo.

    CairoSVG remains a core dependency and is used when the platform library is
    available. This fallback keeps the OpenCV critic operational for simple SVG
    primitives in constrained local environments.
    """

    import re
    from html import unescape

    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    clean_svg = _remove_hidden(svg)
    viewbox = _viewbox(clean_svg) or (0.0, 0.0, float(width), float(height))

    def transform(x: float, y: float, local_scale: float, offset_x: float, offset_y: float) -> tuple[float, float]:
        vx, vy, vw, vh = viewbox
        scale = min(width / max(vw, 1.0), height / max(vh, 1.0))
        pad_x = (width - vw * scale) / 2.0
        pad_y = (height - vh * scale) / 2.0
        return (
            (x * local_scale + offset_x - vx) * scale + pad_x,
            (y * local_scale + offset_y - vy) * scale + pad_y,
        )

    def stroke_width(attrs: dict[str, str], local_scale: float) -> int:
        _vx, _vy, vw, vh = viewbox
        scale = min(width / max(vw, 1.0), height / max(vh, 1.0)) * local_scale
        raw = attrs.get("stroke-width", "2")
        try:
            return max(1, int(float(raw) * scale))
        except ValueError:
            return 2

    def draw_fragment(fragment: str, local_scale: float, offset_x: float, offset_y: float) -> None:
        for rect in re.finditer(r"<rect\b([^>]*)>", fragment):
            attrs = _effective_attrs(_attrs(rect.group(1)))
            if _is_hidden(attrs):
                continue
            x = float(attrs.get("x", 0))
            y = float(attrs.get("y", 0))
            w = float(attrs.get("width", width))
            h = float(attrs.get("height", height))
            p0 = transform(x, y, local_scale, offset_x, offset_y)
            p1 = transform(x + w, y + h, local_scale, offset_x, offset_y)
            box = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
            fill = _paint(attrs.get("fill", "black"))
            stroke = _paint(attrs.get("stroke", "none"))
            if fill is not None:
                draw.rectangle(box, fill=fill)
            if stroke is not None:
                draw.rectangle(box, outline=stroke, width=stroke_width(attrs, local_scale))

        for line in re.finditer(r"<line\b([^>]*)>", fragment):
            attrs = _effective_attrs(_attrs(line.group(1)))
            if _is_hidden(attrs):
                continue
            draw.line(
                [
                    transform(float(attrs.get("x1", 0)), float(attrs.get("y1", 0)), local_scale, offset_x, offset_y),
                    transform(float(attrs.get("x2", 0)), float(attrs.get("y2", 0)), local_scale, offset_x, offset_y),
                ],
                fill=_paint(attrs.get("stroke", "black")) or "black",
                width=stroke_width(attrs, local_scale),
            )

        for poly in re.finditer(r"<polyline\b([^>]*)>", fragment):
            attrs = _effective_attrs(_attrs(poly.group(1)))
            if _is_hidden(attrs):
                continue
            pairs = _point_pairs(attrs.get("points", ""))
            if len(pairs) > 1:
                draw.line(
                    [transform(x, y, local_scale, offset_x, offset_y) for x, y in pairs],
                    fill=_paint(attrs.get("stroke", "black")) or "black",
                    width=stroke_width(attrs, local_scale),
                )

        for poly in re.finditer(r"<polygon\b([^>]*)>", fragment):
            attrs = _effective_attrs(_attrs(poly.group(1)))
            if _is_hidden(attrs):
                continue
            pairs = [transform(x, y, local_scale, offset_x, offset_y) for x, y in _point_pairs(attrs.get("points", ""))]
            if len(pairs) > 1:
                fill = _paint(attrs.get("fill", "black"))
                stroke = _paint(attrs.get("stroke", "black"))
                if fill is not None:
                    draw.polygon(pairs, fill=fill)
                if stroke is not None:
                    draw.line([*pairs, pairs[0]], fill=stroke, width=stroke_width(attrs, local_scale))

        for path in re.finditer(r"<path\b([^>]*)>", fragment):
            attrs = _effective_attrs(_attrs(path.group(1)))
            if _is_hidden(attrs):
                continue
            for subpath in _path_subpaths(attrs.get("d", "")):
                pairs = [transform(x, y, local_scale, offset_x, offset_y) for x, y in subpath]
                if len(pairs) > 1:
                    draw.line(pairs, fill=_paint(attrs.get("stroke", "black")) or "black", width=stroke_width(attrs, local_scale))

        for circle in re.finditer(r"<circle\b([^>]*)>", fragment):
            attrs = _effective_attrs(_attrs(circle.group(1)))
            if _is_hidden(attrs):
                continue
            cx = float(attrs.get("cx", 0))
            cy = float(attrs.get("cy", 0))
            r = float(attrs.get("r", 3))
            p0 = transform(cx - r, cy - r, local_scale, offset_x, offset_y)
            p1 = transform(cx + r, cy + r, local_scale, offset_x, offset_y)
            fill = _paint(attrs.get("fill", "black"))
            stroke = _paint(attrs.get("stroke", "black"))
            draw.ellipse(
                [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])],
                fill=fill,
                outline=stroke,
                width=stroke_width(attrs, local_scale),
            )

        for text in re.finditer(r"<text\b([^>]*)>(.*?)</text>", fragment, re.DOTALL):
            attrs = _effective_attrs(_attrs(text.group(1)))
            if _is_hidden(attrs):
                continue
            value = unescape(re.sub(r"<[^>]+>", "", text.group(2))).strip()
            if not value:
                continue
            fill = _paint(attrs.get("fill", "black"))
            if fill is None:
                continue
            raw_x = _tspan_x(text.group(2), float(attrs.get("x", 0)))
            raw_y = float(attrs.get("y", 0)) + _tspan_dy(text.group(2))
            x, y = transform(raw_x, raw_y, local_scale, offset_x, offset_y)
            font_size = max(8, int(_font_size(attrs) * local_scale * (height / max(viewbox[3], 1.0))))
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()
            text_box = draw.textbbox((0, 0), value, font=font)
            tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
            if attrs.get("text-anchor") == "middle":
                x -= tw / 2
            elif attrs.get("text-anchor") == "end":
                x -= tw
            draw.text((x, y - th / 2), value, fill=fill, font=font)

    for fragment, local_scale, offset_x, offset_y in _scaled_fragments(clean_svg):
        draw_fragment(fragment, local_scale, offset_x, offset_y)

    return _white_background(image)


def _attrs(text: str) -> dict[str, str]:
    import re

    return {match.group(1): match.group(3) for match in re.finditer(r"""([\w:-]+)=("|')([^"']*)\2""", text)}


def _remove_hidden(svg: str) -> str:
    import re

    hidden_group = r"""<g\b(?=[^>]*(?:id=["']optcpv-metadata["']|opacity=["']0["']|display=["']none["']|visibility=["']hidden["']))[^>]*>.*?</g>"""
    return re.sub(hidden_group, "", svg, flags=re.DOTALL)


def _viewbox(svg: str) -> tuple[float, float, float, float] | None:
    import re

    match = re.search(r"<svg\b[^>]*\bviewBox=(\"|')([^\"']+)\1", svg)
    if not match:
        return None
    values = _numbers(match.group(2))
    if len(values) != 4:
        return None
    return tuple(values)  # type: ignore[return-value]


def _first_scale(svg: str) -> float:
    import re

    values = [float(match.group(1)) for match in re.finditer(r"scale\(\s*([0-9.]+)", svg)]
    return values[0] if values else 1.0


def _scaled_fragments(svg: str) -> list[tuple[str, float, float, float]]:
    import re

    fragments: list[tuple[str, float, float, float]] = []
    cursor = 0
    pattern = re.compile(r"""<g\b[^>]*transform=["']([^"']+)["'][^>]*>(.*?)</g>""", re.DOTALL)
    for match in pattern.finditer(svg):
        if match.start() > cursor:
            fragments.append((svg[cursor : match.start()], 1.0, 0.0, 0.0))
        scale, offset_x, offset_y = _simple_transform(match.group(1))
        fragments.append((match.group(2), scale, offset_x, offset_y))
        cursor = match.end()
    if cursor < len(svg):
        fragments.append((svg[cursor:], 1.0, 0.0, 0.0))
    return fragments or [(svg, 1.0, 0.0, 0.0)]


def _simple_transform(transform: str) -> tuple[float, float, float]:
    import re

    text = transform.replace(",", " ")
    combo = re.fullmatch(
        r"\s*translate\(\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s*\)\s*scale\(\s*([0-9.]+)\s*\)\s*translate\(\s*(-?[0-9.]+)\s+(-?[0-9.]+)\s*\)\s*",
        text,
    )
    if combo:
        tx, ty, scale, px, py = (float(item) for item in combo.groups())
        return scale, tx + scale * px, ty + scale * py
    scale = re.fullmatch(r"\s*scale\(\s*([0-9.]+)\s*\)\s*", text)
    if scale:
        return float(scale.group(1)), 0.0, 0.0
    translate = re.fullmatch(r"\s*translate\(\s*(-?[0-9.]+)(?:\s+(-?[0-9.]+))?\s*\)\s*", text)
    if translate:
        return 1.0, float(translate.group(1)), float(translate.group(2) or 0.0)
    return 1.0, 0.0, 0.0


def _effective_attrs(attrs: dict[str, str]) -> dict[str, str]:
    effective: dict[str, str] = {}
    for class_name in attrs.get("class", "").split():
        effective.update(_CLASS_STYLES.get(class_name, {}))
    effective.update(_style_attrs(attrs.get("style", "")))
    effective.update({key: value for key, value in attrs.items() if key != "style"})
    return effective


_CLASS_STYLES = {
    "wire": {"fill": "none", "stroke": "#111827", "stroke-width": "2.2"},
    "component": {"fill": "#fffaf0", "stroke": "#111827", "stroke-width": "2"},
    "symbol": {"fill": "none", "stroke": "#111827", "stroke-width": "2"},
    "pin": {"fill": "#111827", "stroke": "none"},
    "label": {"fill": "#111827", "font-size": "14", "text-anchor": "middle"},
    "terminal-label": {"fill": "#374151", "font-size": "12", "text-anchor": "middle"},
}


def _style_attrs(style: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in style.split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        result[key.strip()] = value.strip()
    if "font" in result and "font-size" not in result:
        import re

        match = re.search(r"([0-9.]+)px", result["font"])
        if match:
            result["font-size"] = match.group(1)
    return result


def _is_hidden(attrs: dict[str, str]) -> bool:
    return attrs.get("opacity") == "0" or attrs.get("display") == "none" or attrs.get("visibility") == "hidden"


def _paint(value: str | None):
    if value is None:
        return None
    compact = value.strip().lower()
    if compact in {"none", "transparent"}:
        return None
    if compact in {"black", "#000", "#000000"}:
        return "black"
    if compact in {"white", "#fff", "#ffffff"}:
        return "white"
    if compact.startswith("#"):
        rgb = _hex_color(compact)
        if rgb is not None:
            return "white" if sum(rgb) / 3 >= 235 else "black"
    if compact.startswith("rgb"):
        nums = _numbers(compact)
        if len(nums) >= 3:
            return "white" if sum(nums[:3]) / 3 >= 235 else "black"
    return "black"


def _hex_color(value: str) -> tuple[int, int, int] | None:
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def _point_pairs(text: str) -> list[tuple[float, float]]:
    values = _numbers(text)
    return list(zip(values[0::2], values[1::2]))


def _numbers(text: str) -> list[float]:
    import re

    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _path_subpaths(path: str) -> list[list[tuple[float, float]]]:
    import re

    tokens = re.findall(r"[MLHVZmlhvz]|-?\d+(?:\.\d+)?", path.replace(",", " "))
    paths: list[list[tuple[float, float]]] = []
    points: list[tuple[float, float]] = []
    command = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command in {"Z", "z"} and points:
                points.append(start)
            continue
        try:
            if command in {"M", "L", "m", "l"} and index + 1 < len(tokens):
                x, y = float(tokens[index]), float(tokens[index + 1])
                if command.islower():
                    x += current[0]
                    y += current[1]
                current = (x, y)
                if command in {"M", "m"}:
                    if len(points) > 1:
                        paths.append(points)
                    points = []
                    start = current
                    command = "l" if command == "m" else "L"
                points.append(current)
                index += 2
                continue
            if command in {"H", "h"}:
                x = float(tokens[index]) + (current[0] if command == "h" else 0.0)
                current = (x, current[1])
                points.append(current)
                index += 1
                continue
            if command in {"V", "v"}:
                y = float(tokens[index]) + (current[1] if command == "v" else 0.0)
                current = (current[0], y)
                points.append(current)
                index += 1
                continue
        except ValueError:
            pass
        index += 1
    if len(points) > 1:
        paths.append(points)
    return paths


def _path_points(path: str) -> list[tuple[float, float]]:
    return [point for subpath in _path_subpaths(path) for point in subpath]


def _font_size(attrs: dict[str, str]) -> float:
    try:
        return float(attrs.get("font-size", "14").replace("px", ""))
    except ValueError:
        return 14.0


def _tspan_x(inner: str, fallback: float) -> float:
    import re

    match = re.search(r"<tspan\b[^>]*\bx=(\"|')([^\"']+)\1", inner)
    if not match:
        return fallback
    values = _numbers(match.group(2))
    return values[0] if values else fallback


def _tspan_dy(inner: str) -> float:
    import re

    match = re.search(r"<tspan\b[^>]*\bdy=(\"|')([^\"']+)\1", inner)
    if not match:
        return 0.0
    values = _numbers(match.group(2))
    return values[0] if values else 0.0


def _white_background(image):
    from PIL import Image

    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image)

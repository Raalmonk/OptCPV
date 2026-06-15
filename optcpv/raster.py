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

    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for rect in re.finditer(r"<rect\b([^>]*)>", svg):
        attrs = _attrs(rect.group(1))
        fill = attrs.get("fill", "black")
        if fill in {"none", "transparent"}:
            continue
        x = float(attrs.get("x", 0))
        y = float(attrs.get("y", 0))
        w = float(attrs.get("width", width))
        h = float(attrs.get("height", height))
        draw.rectangle([x, y, x + w, y + h], fill=_color(fill))

    for line in re.finditer(r"<line\b([^>]*)>", svg):
        attrs = _attrs(line.group(1))
        draw.line(
            [
                (float(attrs.get("x1", 0)), float(attrs.get("y1", 0))),
                (float(attrs.get("x2", 0)), float(attrs.get("y2", 0))),
            ],
            fill=_color(attrs.get("stroke", "black")),
            width=max(1, int(float(attrs.get("stroke-width", 2)))),
        )

    for poly in re.finditer(r"<polyline\b([^>]*)>", svg):
        attrs = _attrs(poly.group(1))
        points = []
        for item in attrs.get("points", "").replace(",", " ").split():
            try:
                points.append(float(item))
            except ValueError:
                pass
        pairs = list(zip(points[0::2], points[1::2]))
        if len(pairs) > 1:
            draw.line(pairs, fill=_color(attrs.get("stroke", "black")), width=max(1, int(float(attrs.get("stroke-width", 2)))))

    for circle in re.finditer(r"<circle\b([^>]*)>", svg):
        attrs = _attrs(circle.group(1))
        cx = float(attrs.get("cx", 0))
        cy = float(attrs.get("cy", 0))
        r = float(attrs.get("r", 3))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_color(attrs.get("fill", "black")), outline="black")

    return _white_background(image)


def _attrs(text: str) -> dict[str, str]:
    import re

    return {match.group(1): match.group(2) for match in re.finditer(r'([\w:-]+)="([^"]*)"', text)}


def _color(value: str) -> str:
    if value in {"black", "#000", "#000000", "#111827"}:
        return "black"
    if value in {"white", "#fff", "#ffffff"}:
        return "white"
    return "black"


def _white_background(image):
    from PIL import Image

    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image)

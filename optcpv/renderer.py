"""Default SVG rendering entry point."""

from __future__ import annotations

from .models import LayoutPlan
from .renderers.schemdraw_backend import SchemdrawRenderer
from .renderers.svg_postprocess import render_debug_svg


def render_svg(layout: LayoutPlan, *, style: str = "textbook", backend: str = "schemdraw") -> str:
    if backend == "debug":
        return render_debug_svg(layout, style=style)
    return SchemdrawRenderer().render(layout, style=style)

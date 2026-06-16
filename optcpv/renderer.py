"""Default SVG rendering entry point."""

from __future__ import annotations

from .models import LayoutPlan, RenderedLayers
from .renderers.schemdraw_backend import SchemdrawRenderer
from .renderers.svg_postprocess import render_debug_svg, render_layer_svg


def render_svg(layout: LayoutPlan, *, style: str = "textbook", backend: str = "schemdraw") -> str:
    if backend == "debug":
        return render_debug_svg(layout, style=style)
    return SchemdrawRenderer().render(layout, style=style)


def render_svg_layers(
    layout: LayoutPlan,
    *,
    style: str = "textbook",
    backend: str = "schemdraw",
    final_svg: str | None = None,
) -> RenderedLayers:
    return RenderedLayers(
        final_svg=final_svg if final_svg is not None else render_svg(layout, style=style, backend=backend),
        wires_svg=render_layer_svg(layout, "wires", style=style),
        components_svg=render_layer_svg(layout, "components", style=style),
        labels_svg=render_layer_svg(layout, "labels", style=style),
    )

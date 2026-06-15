from optcpv import draw_svg
from optcpv.examples import voltage_divider
from optcpv.raster import EVAL_HEIGHT, EVAL_WIDTH, rasterize_svg


def test_svg_rasterizes_to_fixed_dimensions() -> None:
    raster = rasterize_svg(draw_svg(voltage_divider()))

    assert raster.width == EVAL_WIDTH
    assert raster.height == EVAL_HEIGHT
    assert raster.gray.shape == (EVAL_HEIGHT, EVAL_WIDTH)

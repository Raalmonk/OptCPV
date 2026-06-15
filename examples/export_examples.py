"""Export the bundled OptCPV examples to SVG files."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optcpv import draw_svg
from optcpv.examples import EXAMPLES


def main() -> None:
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)
    for name, factory in EXAMPLES.items():
        svg = draw_svg(factory())
        (output_dir / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(output_dir / f"{name}.svg")


if __name__ == "__main__":
    main()

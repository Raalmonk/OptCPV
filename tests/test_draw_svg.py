import subprocess
import sys
import xml.etree.ElementTree as ET

from optcpv import draw_svg
from optcpv.examples import EXAMPLES


def test_import_optcpv_without_app_dependencies() -> None:
    code = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fastapi', 'uvicorn', 'google'}:
            raise ImportError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
import optcpv
print(optcpv.__all__)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)

    assert "draw_svg" in result.stdout


def test_examples_render_valid_svg() -> None:
    for name, factory in EXAMPLES.items():
        svg = draw_svg(factory())
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg"), name
        assert 'data-renderer="optcpv.svg"' in svg
        assert "data-net-name" in svg


def test_every_component_gets_data_component_id() -> None:
    circuit = EXAMPLES["instrumentation_amplifier"]()
    svg = draw_svg(circuit)

    for component in circuit.components:
        assert f'data-component-id="{component.id}"' in svg

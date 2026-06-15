from __future__ import annotations

import copy

import pytest

from backend.app.schem_forge.examples import (
    instrumentation_amp_ir,
    non_inverting_op_amp_ir,
    rc_low_pass_ir,
    voltage_divider_ir,
)


@pytest.fixture
def voltage_divider_ir_fixture() -> dict:
    return voltage_divider_ir()


@pytest.fixture
def rc_low_pass_ir_fixture() -> dict:
    return rc_low_pass_ir()


@pytest.fixture
def non_inverting_ir_fixture() -> dict:
    return non_inverting_op_amp_ir()


@pytest.fixture
def instrumentation_ir_fixture() -> dict:
    return instrumentation_amp_ir()


@pytest.fixture
def copied_voltage_divider(voltage_divider_ir_fixture: dict) -> dict:
    return copy.deepcopy(voltage_divider_ir_fixture)

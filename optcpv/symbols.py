"""Shared symbol geometry used by planners and renderers."""

from __future__ import annotations


# Schemdraw Opamp(leads=True).right() geometry in OptCPV layout units.
# The planner treats component.x/component.y as the left input lead reference.
OPAMP_INPUT_LEAD_X = 0.0
OPAMP_INPUT_LEAD_Y = 0.625
OPAMP_LEAD_LENGTH = 0.625
OPAMP_OUTPUT_LEAD_X = 3.4150635094610964
OPAMP_TIP_X = OPAMP_OUTPUT_LEAD_X - OPAMP_LEAD_LENGTH
OPAMP_HALF_HEIGHT = 1.25


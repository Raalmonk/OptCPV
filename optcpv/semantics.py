"""Topology-semantic helpers shared by planners, renderers, and critics."""

from __future__ import annotations

import re

from .models import NetClass


GROUND_NETS = {"0", "GND", "GROUND", "AGND", "DGND", "PGND", "SGND"}
REFERENCE_NETS = {"REF", "VREF", "REFERENCE"}
POSITIVE_SUPPLY_NETS = {
    "VCC",
    "VDD",
    "AVCC",
    "AVDD",
    "DVCC",
    "DVDD",
    "+V",
    "+VCC",
    "+VDD",
    "+5V",
    "+3V3",
    "+3.3V",
    "5V",
    "3V3",
    "3.3V",
}
NEGATIVE_SUPPLY_NETS = {
    "VEE",
    "VSS",
    "AVEE",
    "AVSS",
    "-V",
    "-VCC",
    "-VDD",
    "-5V",
    "-3V3",
    "-3.3V",
}

GROUND_PIN_NAMES = {"gnd", "ground", "agnd", "dgnd", "pgnd", "sgnd", "vref", "ref"}
POSITIVE_SUPPLY_PIN_NAMES = {"v+", "vpos", "vp", "vcc", "vdd", "+v", "+vs", "vs+", "vcc+", "avdd", "avcc"}
NEGATIVE_SUPPLY_PIN_NAMES = {"v-", "vneg", "vn", "vee", "vss", "-v", "-vs", "vs-", "vee-", "avss", "avee"}


def classify_net(net: str | None) -> NetClass:
    """Classify an electrical net for schematic topology, not simulation."""

    normalized = _canonical_net(net)
    if normalized in GROUND_NETS:
        return NetClass.GROUND
    if normalized in REFERENCE_NETS:
        return NetClass.REFERENCE
    if normalized in POSITIVE_SUPPLY_NETS:
        return NetClass.POSITIVE_SUPPLY
    if normalized in NEGATIVE_SUPPLY_NETS:
        return NetClass.NEGATIVE_SUPPLY
    if normalized.startswith("+") and _looks_like_voltage(normalized[1:]):
        return NetClass.POSITIVE_SUPPLY
    if normalized.startswith("-") and _looks_like_voltage(normalized[1:]):
        return NetClass.NEGATIVE_SUPPLY
    if normalized.startswith("VCC") or normalized.startswith("VDD"):
        return NetClass.POSITIVE_SUPPLY
    if normalized.startswith("VEE") or normalized.startswith("VSS"):
        return NetClass.NEGATIVE_SUPPLY
    if normalized.startswith("N$") or normalized.startswith("_"):
        return NetClass.INTERNAL
    return NetClass.SIGNAL


def is_local_terminal_net(net: str | None) -> bool:
    return classify_net(net) in {
        NetClass.GROUND,
        NetClass.POSITIVE_SUPPLY,
        NetClass.NEGATIVE_SUPPLY,
        NetClass.REFERENCE,
    }


def terminal_label(net: str | None, net_class: NetClass | None = None) -> str:
    text = (net or "").strip()
    if text:
        return text
    net_class = net_class or classify_net(net)
    labels = {
        NetClass.GROUND: "GND",
        NetClass.REFERENCE: "REF",
        NetClass.POSITIVE_SUPPLY: "VCC",
        NetClass.NEGATIVE_SUPPLY: "VEE",
    }
    return labels.get(net_class, "")


def terminal_type_for_net(net: str | None) -> str:
    return classify_net(net).value


def preferred_terminal_direction(net: str | None) -> str:
    net_class = classify_net(net)
    if net_class == NetClass.POSITIVE_SUPPLY:
        return "up"
    return "down"


def is_positive_supply_pin(pin_name: str | None, net: str | None = None) -> bool:
    return _pin_key(pin_name) in POSITIVE_SUPPLY_PIN_NAMES or classify_net(net) == NetClass.POSITIVE_SUPPLY


def is_negative_supply_pin(pin_name: str | None, net: str | None = None) -> bool:
    return _pin_key(pin_name) in NEGATIVE_SUPPLY_PIN_NAMES or classify_net(net) == NetClass.NEGATIVE_SUPPLY


def is_reference_pin(pin_name: str | None, net: str | None = None) -> bool:
    return _pin_key(pin_name) in GROUND_PIN_NAMES or classify_net(net) in {NetClass.GROUND, NetClass.REFERENCE}


def _canonical_net(net: str | None) -> str:
    value = (net or "").strip().upper()
    value = value.replace(" ", "")
    value = value.replace("_", "")
    value = value.replace("PLUS", "+")
    value = value.replace("MINUS", "-")
    return value


def _pin_key(pin_name: str | None) -> str:
    return (pin_name or "").strip().lower().replace("_", "").replace(" ", "")


def _looks_like_voltage(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?V|\d+V\d+", value))

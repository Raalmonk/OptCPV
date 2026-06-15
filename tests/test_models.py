from optcpv import Circuit, Component
from optcpv.models import circuit_from_any


def test_circuit_from_dict_accepts_minimal_ir() -> None:
    circuit = circuit_from_any(
        {
            "id": "demo",
            "motif": "voltage_divider",
            "components": [
                {"id": "R1", "type": "resistor", "pins": {"a": "vin", "b": "vout"}},
            ],
        }
    )

    assert isinstance(circuit, Circuit)
    assert circuit.components == [Component(id="R1", type="resistor", pins={"a": "vin", "b": "vout"})]


def test_circuit_from_dict_rejects_missing_components() -> None:
    try:
        circuit_from_any({"id": "empty", "components": []})
    except ValueError as exc:
        assert "components" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

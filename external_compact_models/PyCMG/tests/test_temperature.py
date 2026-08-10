"""
Temperature Verification Tests

Verifies PyCMG temperature handling against NGSPICE ground truth.

Run: pytest tests/test_temperature.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import TECHNOLOGIES, requires_osdi

TEMP_OUTPUTS = ["id", "gm", "gds"]

# ASAP7 covers wide temperature range; TSMC7 skips -40C (convergence issues)
TEMP_CASES = [
    ("ASAP7", -40.0),
    ("ASAP7", 85.0),
    ("ASAP7", 125.0),
    ("TSMC7", 85.0),
    ("TSMC7", 125.0),
]


def _temp_bias(vdd: float, device: str) -> dict[str, float]:
    """Saturation bias point for temperature tests."""
    if device == "nmos":
        return {"d": vdd / 2, "g": vdd / 2, "s": 0.0, "e": 0.0}
    else:
        return {"d": vdd * 0.3, "g": vdd * 0.3, "s": vdd, "e": vdd}


@requires_osdi
@pytest.mark.parametrize("tech_name,temp_c", TEMP_CASES,
                         ids=[f"{t[0]}_T{t[1]}" for t in TEMP_CASES])
@pytest.mark.parametrize("device", ["nmos", "pmos"])
def test_temperature(tech_name: str, temp_c: float, device: str) -> None:
    """Test DC currents and derivatives match NGSPICE at non-default temperatures."""
    vdd = TECHNOLOGIES[tech_name]["vdd"]
    bias = _temp_bias(vdd, device)

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"temp_{tech_name}_{device}_{temp_c}",
            outputs=TEMP_OUTPUTS,
            temp_c=temp_c,
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

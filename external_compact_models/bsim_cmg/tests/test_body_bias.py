"""
Body Bias Verification Tests

Verifies model accuracy with non-zero body bias (bulk terminal voltage)
for both NMOS and PMOS devices across all 5 technologies.

Run: pytest tests/test_body_bias.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import TECHNOLOGIES, TECH_NAMES, requires_osdi

BIAS_TYPES = ["reverse", "forward"]
BODY_OUTPUTS = ["id", "gm", "gds", "gmb", "ie", "qg", "qd"]


def _body_bias(vdd: float, device: str, bias_type: str) -> dict[str, float]:
    """Build bias dict for body-bias test."""
    if device == "nmos":
        ve = -0.1 if bias_type == "reverse" else 0.1
        return {"d": vdd / 2, "g": vdd / 2, "s": 0.0, "e": ve}
    else:
        ve = vdd + 0.1 if bias_type == "reverse" else vdd - 0.1
        return {"d": vdd * 0.3, "g": vdd * 0.3, "s": vdd, "e": ve}


@requires_osdi
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("bias_type", BIAS_TYPES)
def test_body_bias(tech_name: str, device: str, bias_type: str) -> None:
    """Test DC outputs match NGSPICE with non-zero body bias."""
    vdd = TECHNOLOGIES[tech_name]["vdd"]
    bias = _body_bias(vdd, device, bias_type)

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"body_bias_{tech_name}_{device}_{bias_type}",
            outputs=BODY_OUTPUTS,
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

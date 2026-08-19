"""
Core-Voltage Vt Variant Verification Tests

Verifies PyCMG vs NGSPICE agreement for all core-voltage threshold voltage
variants across technologies.

Run: pytest tests/test_vt_variants.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import (
    ALL_TECHNOLOGIES, CORE_VT_NAMES, requires_osdi,
    standard_bias_points, REGION_NAMES,
)

VT_OUTPUTS = ["id", "ig", "is", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]


@requires_osdi
@pytest.mark.parametrize("tech_name", CORE_VT_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("region", REGION_NAMES)
def test_vt_variant(tech_name: str, device: str, region: str) -> None:
    """DC verification for core-voltage Vt variants."""
    vdd = ALL_TECHNOLOGIES[tech_name]["vdd"]
    bias = standard_bias_points(vdd, device)[region]

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"vt_{tech_name}_{device}_{region}",
            outputs=VT_OUTPUTS,
            check_off_state=(region == "off"),
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

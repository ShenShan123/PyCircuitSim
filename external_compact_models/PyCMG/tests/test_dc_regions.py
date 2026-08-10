"""
DC Operating Region Tests

Verifies model accuracy across voltage-ratio-defined operating regions
for both NMOS and PMOS devices across all 5 technologies.

Run: pytest tests/test_dc_regions.py -v
"""

from __future__ import annotations

import pytest

from tests.helpers import run_dc_comparison
from tests.conftest import (
    TECH_NAMES, requires_osdi, standard_bias_points, REGION_NAMES,
)


DC_OUTPUTS = ["id", "ig", "is", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]


@requires_osdi
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("region", REGION_NAMES)
def test_dc_region(tech_name: str, device: str, region: str) -> None:
    """Test DC currents and derivatives match NGSPICE in operating region."""
    from tests.conftest import TECHNOLOGIES
    vdd = TECHNOLOGIES[tech_name]["vdd"]
    bias = standard_bias_points(vdd, device)[region]

    try:
        run_dc_comparison(
            tech_name, device, bias,
            tag=f"region_{tech_name}_{device}_{region}",
            outputs=DC_OUTPUTS,
            check_off_state=(region == "off"),
            check_ids=(region != "off"),
        )
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

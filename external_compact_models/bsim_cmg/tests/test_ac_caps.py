"""
AC Capacitance Verification Tests

Compares PyCMG's condensed capacitance matrix (cgg, cgd, cgs, cdg, cdd) from
eval_dc() against NGSPICE's @n1[cXX] operating-point variables.

PyCMG extracts capacitances via _condense_caps() which builds the full reactive
Jacobian (dQ/dV) and applies Schur complement condensation to eliminate internal
nodes. NGSPICE performs an equivalent condensation internally for OSDI models.

This test verifies that:
1. PyCMG's _condense_caps() produces correct capacitance values
2. Sign conventions match between PyCMG and NGSPICE
3. Capacitance condensation (internal node elimination) is consistent

Run: pytest tests/test_ac_caps.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pycmg import Model, Instance
from tests.helpers import (
    OSDI_PATH, run_ngspice_ac, assert_close,
    ABS_TOL_C, REL_TOL_CAP,
)
from tests.conftest import TECHNOLOGIES, TECH_NAMES, get_tech_modelcard


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
@pytest.mark.parametrize("tech_name", TECH_NAMES, ids=TECH_NAMES)
def test_pmos_ac_caps(tech_name: str) -> None:
    """Compare PyCMG PMOS capacitances against NGSPICE at saturation bias."""
    tech = TECHNOLOGIES[tech_name]
    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "pmos")
    except FileNotFoundError:
        pytest.skip(f"No PMOS modelcard for {tech_name}")
    vdd = tech["vdd"]

    # PMOS saturation: Vs=Vdd, Vd and Vg pulled low
    # ve=0.0 for PMOS: exercises deep reverse body bias (Vbs = -Vdd).
    # Standard zero body bias (ve=vdd) is tested in test_temperature.py and test_body_bias.py.
    vd = vdd * 0.3
    vg = vdd * 0.3
    vs = vdd
    ve = 0.0

    ng = run_ngspice_ac(
        modelcard, model_name, inst_params,
        vd, vg, vs, ve,
        tag=f"ac_{tech_name}_pmos",
    )

    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": ve})

    prefix = f"{tech_name}/pmos"
    cap_keys = ["cgg", "cgd", "cgs", "cdg", "cdd"]
    for cap in cap_keys:
        assert_close(
            f"{prefix}/{cap}",
            py[cap], ng[cap],
            abs_tol=ABS_TOL_C,
            rel_tol=REL_TOL_CAP,
        )


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI binary not built")
@pytest.mark.parametrize("tech_name", TECH_NAMES, ids=TECH_NAMES)
@pytest.mark.parametrize("bias", ["linear", "saturation"], ids=["linear", "saturation"])
def test_nmos_ac_caps_regions(tech_name: str, bias: str) -> None:
    """Compare NMOS capacitances at different operating regions."""
    tech = TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, "nmos")
    vdd = tech["vdd"]

    if bias == "linear":
        vd, vg = 0.1 * vdd, vdd  # Linear region: low Vds, high Vgs
    else:
        vd, vg = vdd, 0.8 * vdd  # Saturation: high Vds

    ng = run_ngspice_ac(
        modelcard, model_name, inst_params,
        vd, vg, 0.0, 0.0,
        tag=f"ac_{tech_name}_nmos_{bias}",
    )

    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py = inst.eval_dc({"d": vd, "g": vg, "s": 0.0, "e": 0.0})

    prefix = f"{tech_name}/nmos/{bias}"
    for cap in ["cgg", "cgd", "cgs", "cdg", "cdd"]:
        assert_close(f"{prefix}/{cap}", py[cap], ng[cap],
                     abs_tol=ABS_TOL_C, rel_tol=REL_TOL_CAP)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

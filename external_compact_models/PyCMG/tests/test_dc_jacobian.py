"""
DC Jacobian Verification Tests

Compares PyCMG's condensed 4x4 analytical Jacobian against NGSPICE's
numerical Jacobian computed via central finite-difference perturbation.

Run: pytest tests/test_dc_jacobian.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from pycmg import Model, Instance
from tests.helpers import (
    OSDI_PATH, run_ngspice_op, assert_close,
    ABS_TOL_G, REL_TOL_JAC,
)
from tests.conftest import (
    TECHNOLOGIES, TECH_NAMES, get_tech_modelcard,
    requires_osdi, standard_bias_points, REGION_NAMES,
)


def compute_numerical_jacobian_central(
    modelcard: Path, model_name: str, inst_params: Dict[str, float],
    op: Dict[str, float], delta: float = 1e-6, temp_c: float = 27.0,
    tag_prefix: str = "jac",
) -> np.ndarray:
    """Compute 4x4 Jacobian via central finite-difference perturbation.

    Uses central differencing for O(delta^2) accuracy:
        J[:,j] = (I(V+delta_j) - I(V-delta_j)) / (2*delta)
    """
    op_keys = ["d", "g", "s", "e"]
    current_keys = ["id", "ig", "is", "ie"]
    n = 4
    J = np.zeros((n, n))

    for j, op_key in enumerate(op_keys):
        fwd_op = dict(op)
        fwd_op[op_key] = op[op_key] + delta
        fwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            fwd_op["d"], fwd_op["g"], fwd_op["s"], fwd_op["e"],
            temp_c, tag=f"{tag_prefix}_fwd_{op_key}",
        )
        fwd_I = np.array([fwd[k] for k in current_keys])

        bwd_op = dict(op)
        bwd_op[op_key] = op[op_key] - delta
        bwd = run_ngspice_op(
            modelcard, model_name, inst_params,
            bwd_op["d"], bwd_op["g"], bwd_op["s"], bwd_op["e"],
            temp_c, tag=f"{tag_prefix}_bwd_{op_key}",
        )
        bwd_I = np.array([bwd[k] for k in current_keys])

        J[:, j] = (fwd_I - bwd_I) / (2.0 * delta)

    return J


@requires_osdi
@pytest.mark.parametrize("tech_name", TECH_NAMES)
@pytest.mark.parametrize("device", ["nmos", "pmos"])
@pytest.mark.parametrize("region", REGION_NAMES)
def test_dc_jacobian_full_matrix(tech_name: str, device: str, region: str) -> None:
    """Compare condensed 4x4 Jacobian matrix against NGSPICE numerical Jacobian."""
    tech = TECHNOLOGIES[tech_name]

    try:
        modelcard, model_name, inst_params = get_tech_modelcard(tech_name, device)
    except FileNotFoundError:
        pytest.skip(f"No {device} modelcard for {tech_name}")

    vdd = tech["vdd"]
    op = standard_bias_points(vdd, device)[region]

    # NGSPICE: numerical Jacobian via central differencing
    ng_J = compute_numerical_jacobian_central(
        modelcard, model_name, inst_params, op,
        tag_prefix=f"jac_{tech_name}_{device}_{region}",
    )

    # PyCMG: analytical condensed Jacobian
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    inst = Instance(model, params=inst_params)
    py_J = inst.get_jacobian_matrix(op)

    # Compare each entry
    terminals = ["d", "g", "s", "e"]
    for i, term_i in enumerate(terminals):
        for j, term_j in enumerate(terminals):
            label = f"{tech_name}/{device}/{region}/d(I{term_i})/d(V{term_j})"
            assert_close(
                label, py_J[i, j], ng_J[i, j],
                abs_tol=ABS_TOL_G, rel_tol=REL_TOL_JAC,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

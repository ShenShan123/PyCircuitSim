"""Tests for the sensitivity analysis module."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
ASAP7_MODELCARD = ROOT / "modelcards" / "ASAP7" / "7nm_TT_160803.pm"

SKIP_NO_OSDI = pytest.mark.skipif(
    not OSDI_PATH.exists(), reason="OSDI not built"
)

INST_PARAMS = {"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0}


@SKIP_NO_OSDI
def test_enumerate_model_params():
    """Should return a non-empty list of ParamInfo with known params."""
    from pycmg import Model, Instance
    from pycmg.sensitivity import enumerate_model_params

    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    # Must create an Instance first to populate the model buffer with
    # modelcard parameter values (Model alone doesn't apply them).
    _inst = Instance(model, params=INST_PARAMS)
    params = enumerate_model_params(model.descriptor, model.model)

    assert len(params) > 0

    names = {p.name for p in params}
    # Known BSIM-CMG process parameters should be present
    assert "toxe" in names or "eot" in names

    # Geometry/instance params should be excluded
    assert "devtype" not in names
    assert "l" not in names
    assert "nfin" not in names
    assert "tfin" not in names

    # All values should be non-zero
    for p in params:
        assert abs(p.value) > 1e-30


@SKIP_NO_OSDI
def test_enumerate_model_params_all_real():
    """All returned parameters should be real-valued model parameters."""
    from pycmg import Model, Instance
    from pycmg.sensitivity import ParamInfo, enumerate_model_params

    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    _inst = Instance(model, params=INST_PARAMS)
    params = enumerate_model_params(model.descriptor, model.model)

    for p in params:
        assert isinstance(p, ParamInfo)
        assert isinstance(p.value, float)
        assert isinstance(p.name, str)
        assert p.name == p.name.lower()  # Should be lowercase


@SKIP_NO_OSDI
def test_sensitivity_smoke():
    """compute_sensitivity should return valid results for all 3 categories."""
    from pycmg.sensitivity import OUTPUT_CATEGORIES, compute_sensitivity

    result = compute_sensitivity(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(ASAP7_MODELCARD),
        model_name="nmos_rvt",
        inst_params=INST_PARAMS,
        vdd=0.9,
        device_type="nmos",
        delta_fraction=0.05,
        top_n=9,
        verbose=0,
    )

    # Should have analyzed some parameters
    assert len(result.param_names) > 0
    assert len(result.sensitivities) > 0

    # Rankings should exist for all categories
    for cat in OUTPUT_CATEGORIES:
        assert cat in result.rankings
        assert len(result.rankings[cat]) > 0
        assert len(result.rankings[cat]) <= 9

    # Bias points should be populated
    assert len(result.bias_points) == 4
    assert result.delta_fraction == 0.05


@SKIP_NO_OSDI
def test_sensitivity_result_structure():
    """SensitivityResult fields should be internally consistent."""
    from pycmg.sensitivity import compute_sensitivity
    from pycmg.sweep import OUTPUT_KEYS

    result = compute_sensitivity(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(ASAP7_MODELCARD),
        model_name="nmos_rvt",
        inst_params=INST_PARAMS,
        vdd=0.9,
        device_type="nmos",
        delta_fraction=0.05,
        top_n=5,
        verbose=0,
    )

    # param_names should match sensitivities keys
    assert set(result.param_names) == set(result.sensitivities.keys())

    # Each sensitivity dict should have all OUTPUT_KEYS
    for param_name, sens in result.sensitivities.items():
        for key in OUTPUT_KEYS:
            assert key in sens, f"Missing {key} in sensitivities for {param_name}"

    # Ranked params should be subset of analyzed params
    for cat, ranked in result.rankings.items():
        for p in ranked:
            assert p in result.sensitivities


@SKIP_NO_OSDI
def test_sensitivity_format_table():
    """format_sensitivity_table should produce non-empty output."""
    from pycmg.sensitivity import compute_sensitivity, format_sensitivity_table

    result = compute_sensitivity(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(ASAP7_MODELCARD),
        model_name="nmos_rvt",
        inst_params=INST_PARAMS,
        vdd=0.9,
        device_type="nmos",
        delta_fraction=0.05,
        top_n=3,
        verbose=0,
    )

    for cat in ["iv", "qv", "cv"]:
        table = format_sensitivity_table(result, cat)
        assert len(table) > 0
        assert "===" in table
        assert "Rank" in table


@SKIP_NO_OSDI
def test_sensitivity_known_param_ranked():
    """Physical params like phig/eot should appear in top 9 for IV."""
    from pycmg.sensitivity import compute_sensitivity

    result = compute_sensitivity(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(ASAP7_MODELCARD),
        model_name="nmos_rvt",
        inst_params=INST_PARAMS,
        vdd=0.9,
        device_type="nmos",
        delta_fraction=0.05,
        top_n=9,
        verbose=0,
    )

    iv_top = set(result.rankings["iv"])
    # phig (work function) and eot (oxide thickness) are fundamental
    # process parameters that strongly affect drain current
    assert "phig" in iv_top or "eot" in iv_top or "easub" in iv_top, (
        f"Expected a major process param in IV top 9, got: {iv_top}"
    )


@SKIP_NO_OSDI
def test_sensitivity_different_categories():
    """I-V, Q-V, C-V rankings should generally differ (they measure different things)."""
    from pycmg.sensitivity import compute_sensitivity

    result = compute_sensitivity(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(ASAP7_MODELCARD),
        model_name="nmos_rvt",
        inst_params=INST_PARAMS,
        vdd=0.9,
        device_type="nmos",
        delta_fraction=0.05,
        top_n=9,
        verbose=0,
    )

    # At least one of the top-9 lists should differ from another
    iv_set = set(result.rankings["iv"])
    qv_set = set(result.rankings["qv"])
    cv_set = set(result.rankings["cv"])
    # Not all three should be identical
    assert not (iv_set == qv_set == cv_set), (
        "All three category rankings are identical — likely a bug"
    )

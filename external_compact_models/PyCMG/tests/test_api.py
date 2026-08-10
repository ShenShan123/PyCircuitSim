"""
PyCMG API Tests

Quick smoke tests for public API validation.
Tests the Python ctypes wrapper interface without NGSPICE comparison.

TEST COVERAGE:
- Module imports and public API availability
- Modelcard parsing (SPICE number format, parameter extraction)
- Model/Instance creation and initialization
- DC evaluation output format and keys
- Transient evaluation basic functionality
- Parameter updates and temperature sweeps

NOTE: These tests verify API correctness only. For numerical verification
comparing PyCMG vs NGSPICE (both using the same OSDI binary), see:
- tests/test_integration.py - Direct PyCMG vs NGSPICE comparison
- tests/test_asap7.py - Comprehensive ASAP7 PVT verification

Run: pytest tests/test_api.py -v
Duration: ~5 seconds
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pycmg
from pycmg import Model, Instance, parse_modelcard, parse_number_with_suffix

from tests.conftest import OSDI_PATH
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

# ASAP7 modelcard configuration
ASAP7_DIR = ROOT / "modelcards" / "ASAP7"
ASAP7_MODELCARD_OVERRIDE = os.environ.get("ASAP7_MODELCARD")


def _get_test_modelcard():
    """Get a test modelcard - prefer ASAP7, otherwise create minimal one."""
    import re
    import tempfile

    # Try ASAP7 first
    if ASAP7_MODELCARD_OVERRIDE:
        override = Path(ASAP7_MODELCARD_OVERRIDE)
        if override.is_file():
            return str(override), _find_asap7_model(override)
        if override.is_dir():
            cards = sorted(override.glob("*.pm"))
            if cards:
                return str(cards[0]), _find_asap7_model(cards[0])

    if ASAP7_DIR.exists():
        cards = sorted(ASAP7_DIR.glob("*.pm"))
        if cards:
            return str(cards[0]), _find_asap7_model(cards[0])

    # Create minimal test modelcard
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.lib', delete=False)
    tmpfile.write("""* Minimal BSIM-CMG test modelcard
.model nmos1 bsimcmg
+ BULKMOD = 1
+ CAPMOD = 0
+ COREMOD = 0
+ CGEOMOD = 0
+ DEVTYPE = 1
+ GEOMOD = 0
+ GIDLMOD = 1
+ IGBMOD = 0
+ IGCMOD = 1
+ IIMOD = 0
+ NGATE  = 0
+ NQSMOD = 0
+ RDSMOD = 0
+ RGATEMOD = 0
+ RGEOMOD = 0
+ NSEG = 5
+ SDTERM = 0
+ SHMOD = 0
+ EOT  = 1.50E-09
+ EOTBOX  = 1.40E-07
+ EPSROX  = 3.9
+ EPSRSP  = 3.9
+ EPSRSUB  = 11.9
+ L  = 2.50E-08
+ TFIN  = 1.40E-08
+ FPITCH  = 4.00E-08
+ HFIN  = 3.00E-08
+ NBODY  = 1.00E+22
+ NSD  = 2.00E+26
+ TNOM = 25
""")
    tmpfile.close()
    return tmpfile.name, "nmos1"


def _find_asap7_model(modelcard_path: Path) -> str:
    """Find the first level=72 nmos model in an ASAP7 modelcard."""
    import re
    text = modelcard_path.read_text()
    for line in text.splitlines():
        if line.strip().lower().startswith(".model"):
            parts = line.split()
            if len(parts) >= 3:
                name = parts[1]
                mtype = parts[2].lower()
                if "nmos" in mtype:
                    rest = " ".join(parts[3:])
                    if "level=72" in rest.lower() or re.search(r"\blevel\s*=\s*72\b", rest, re.I):
                        return name
    # Fallback to first nmos model
    for line in text.splitlines():
        if line.strip().lower().startswith(".model"):
            parts = line.split()
            if len(parts) >= 3 and "nmos" in parts[2].lower():
                return parts[1]
    raise ValueError(f"No NMOS model found in {modelcard_path}")


def test_import() -> None:
    """Test that pycmg module can be imported."""
    assert hasattr(pycmg, "Model")
    assert hasattr(pycmg, "Instance")
    assert hasattr(pycmg, "parse_modelcard")
    assert hasattr(pycmg, "parse_number_with_suffix")


def test_parse_number_with_suffix() -> None:
    """Test SPICE number parsing with unit suffixes."""
    parse = parse_number_with_suffix

    # Basic numbers
    assert parse("1") == 1.0
    assert parse("0.5") == 0.5
    assert parse("1e-9") == 1e-9

    # Metric suffixes
    assert parse("1t") == pytest.approx(1e12)
    assert parse("1g") == pytest.approx(1e9)
    assert parse("1meg") == pytest.approx(1e6)
    assert parse("1k") == pytest.approx(1e3)
    assert parse("1m") == pytest.approx(1e-3)
    assert parse("1u") == pytest.approx(1e-6)
    assert parse("1n") == pytest.approx(1e-9)
    assert parse("1p") == pytest.approx(1e-12)
    assert parse("1f") == pytest.approx(1e-15)
    assert parse("1a") == pytest.approx(1e-18)


def test_parse_modelcard_basic(tmp_path: Any) -> None:
    """Test basic modelcard parsing."""
    card = tmp_path / "test.lib"
    card.write_text("""
* Test modelcard
.model nmos1 bsimcmg l=16n tfin=8n
+ eot=1.5n tox=2.0n
""")

    parsed = parse_modelcard(str(card), target_model_name="nmos1")
    assert parsed.name == "nmos1"
    assert "l" in parsed.params
    assert "tfin" in parsed.params
    assert parsed.params["l"] == parse_number_with_suffix("16n")
    assert parsed.params["tfin"] == parse_number_with_suffix("8n")


def test_parse_modelcard_level72(tmp_path: Any) -> None:
    """Test parsing level=72 NMOS model."""
    card = tmp_path / "asap7.pm"
    card.write_text("""
* ASAP7 modelcard
.model nmos_lvt nmos level=72 l=14n tfin=7n
+ tox=1.5n
.model pmos_lvt pmos level=72 l=16n tfin=8n
+ tox=2.0n
""")

    parsed = parse_modelcard(str(card), target_model_name="nmos_lvt")
    assert parsed.name == "nmos_lvt"
    assert parsed.params["l"] == parse_number_with_suffix("14n")
    assert parsed.params["tfin"] == parse_number_with_suffix("7n")


def test_parse_modelcard_first_valid_when_no_target(tmp_path: Any) -> None:
    """Test that first valid model is used when no target specified."""
    card = tmp_path / "multi.pm"
    card.write_text("""
.model bad1 nmos level=71 l=10n
.model good1 bsimcmg l=20n tfin=9n
.model good2 nmos level=72 l=30n
""")

    parsed = parse_modelcard(str(card))
    assert parsed.name == "good1"
    assert parsed.params["l"] == parse_number_with_suffix("20n")


def test_model_init_signature() -> None:
    """Test Model class signature."""
    import inspect
    sig = inspect.signature(Model.__init__)
    params = sig.parameters
    assert "osdi_path" in params
    assert "modelcard_path" in params
    assert "model_name" in params


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_model_creation() -> None:
    """Test creating a Model instance."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        assert model is not None
    finally:
        # Clean up temp file if it's one we created
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_instance_creation() -> None:
    """Test creating an Instance with geometry parameters."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={
            "L": 16e-9,
            "TFIN": 8e-9,
            "NFIN": 2.0,
        })
        assert inst is not None
        assert inst.internal_node_count() >= 0
        assert inst.state_count() >= 0
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_eval_dc_smoke() -> None:
    """Test DC evaluation returns all expected outputs."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        result = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})

        # Check all expected output keys exist
        expected_keys = [
            "id", "ig", "is", "ie", "ids",  # Currents
            "qg", "qd", "qs", "qb",  # Charges
            "gm", "gds", "gmb",      # Derivatives
            "cgg", "cgd", "cgs", "cdg", "cdd",  # Capacitances
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], float), f"{key} is not a float"
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_eval_tran_smoke() -> None:
    """Test transient evaluation."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        result = inst.eval_tran(
            nodes={"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0},
            time=1e-9,
            delta_t=1e-12
        )

        # Check transient outputs
        expected_keys = ["id", "ig", "is", "ie", "ids", "qg", "qd", "qs", "qb"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], float)
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_set_params() -> None:
    """Test updating instance parameters."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        # Update NFIN parameter
        inst.set_params({"NFIN": 4.0}, allow_rebind=False)

        # Verify instance still works
        result = inst.eval_dc({"d": 0.05, "g": 0.8, "s": 0.0, "e": 0.0})
        assert "id" in result
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_temperature_sweep() -> None:
    """Test evaluation at multiple temperatures."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)

        temperatures = [223.15, 273.15, 323.15, 373.15, 398.15]  # -40C to 125C
        for temp_k in temperatures:
            inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0},
                          temperature=temp_k)
            result = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
            assert "id" in result
            assert isinstance(result["id"], float)
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_get_jacobian_matrix():
    """Test that Instance exposes condensed 4x4 Jacobian matrix."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        J = inst.get_jacobian_matrix({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})

        # Condensed to 4 external terminals: d, g, s, e
        assert J.shape == (4, 4), f"Expected (4,4), got {J.shape}"
        assert isinstance(J, np.ndarray)
        assert np.all(np.isfinite(J)), f"Non-finite entries in Jacobian: {J}"

        # gds = dId/dVd should be positive in saturation
        # (terminal ordering from sim.terminal_indices)
        assert J.sum() != 0.0, "Jacobian is all zeros"
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Negative / error tests
# ---------------------------------------------------------------------------

def test_parse_number_edge_cases() -> None:
    """Test parse_number_with_suffix with edge-case inputs."""
    parse = parse_number_with_suffix

    # Negative numbers
    assert parse("-1.5") == -1.5
    assert parse("-3e-9") == -3e-9

    # Zero
    assert parse("0") == 0.0
    assert parse("0.0") == 0.0

    # Large scientific notation
    assert parse("1e+22") == pytest.approx(1e22)
    assert parse("1.00E+22") == pytest.approx(1e22)

    # Suffix with scientific notation should be unusual but parseable
    assert parse("1.5e-9") == pytest.approx(1.5e-9)


def test_parse_modelcard_empty_file(tmp_path) -> None:
    """Test parse_modelcard with empty file raises RuntimeError."""
    card = tmp_path / "empty.lib"
    card.write_text("")
    with pytest.raises(RuntimeError, match="no .* model found"):
        parse_modelcard(str(card))


def test_parse_modelcard_no_valid_model(tmp_path) -> None:
    """Test parse_modelcard with file containing no valid BSIM-CMG model."""
    card = tmp_path / "invalid.lib"
    card.write_text("""
* Only comments
* No .model blocks
+ some continuation line
""")
    with pytest.raises(RuntimeError, match="no .* model found"):
        parse_modelcard(str(card))


def test_parse_modelcard_wrong_target(tmp_path) -> None:
    """Test parse_modelcard when target model name doesn't exist."""
    card = tmp_path / "test.lib"
    card.write_text("""
.model nmos1 bsimcmg l=16n
""")
    with pytest.raises(RuntimeError, match="no nonexistent model found"):
        parse_modelcard(str(card), target_model_name="nonexistent")


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_eval_dc_multiple_calls_consistent() -> None:
    """Verify that calling eval_dc multiple times on the same Instance gives consistent results."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        nodes = {"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0}
        r1 = inst.eval_dc(nodes)
        r2 = inst.eval_dc(nodes)

        # Results should be identical (no stale state)
        for key in r1:
            assert r1[key] == pytest.approx(r2[key], abs=1e-20), (
                f"Inconsistent {key}: {r1[key]} vs {r2[key]}"
            )
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_eval_dc_different_bias_points() -> None:
    """Verify eval_dc works at different bias points on the same Instance."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        # Saturation
        r_sat = inst.eval_dc({"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0})
        assert abs(r_sat["id"]) > 1e-12, "No current in saturation"

        # Off state
        r_off = inst.eval_dc({"d": 0.5, "g": 0.0, "s": 0.0, "e": 0.0})
        # Off current should be much smaller than saturation current
        assert abs(r_off["id"]) < abs(r_sat["id"]), "Off current >= saturation current"

        # Linear region
        r_lin = inst.eval_dc({"d": 0.1, "g": 0.8, "s": 0.0, "e": 0.0})
        assert abs(r_lin["id"]) > 1e-12, "No current in linear"
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="missing OSDI build artifact")
def test_eval_tran_negative_delta_t() -> None:
    """Verify eval_tran raises on negative delta_t."""
    modelcard_path, model_name = _get_test_modelcard()
    try:
        model = Model(str(OSDI_PATH), modelcard_path, model_name)
        inst = Instance(model, params={"L": 16e-9, "TFIN": 8e-9, "NFIN": 2.0})

        with pytest.raises(RuntimeError, match="delta_t must be positive"):
            inst.eval_tran(
                nodes={"d": 0.5, "g": 0.8, "s": 0.0, "e": 0.0},
                time=1e-9,
                delta_t=-1e-12,
            )
    finally:
        if modelcard_path.startswith("/tmp/") and "tmp" in modelcard_path:
            Path(modelcard_path).unlink(missing_ok=True)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

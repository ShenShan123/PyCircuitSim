"""
Tests for MOSFET Level 1 (Shichman-Hodges) model.

This module tests the implementation of NMOS and PMOS transistors
using the Level 1 compact model (Shichman-Hodges equations).
"""
import pytest
import numpy as np
from pycircuitsim.models.mosfet import NMOS, PMOS


class TestNMOSCreation:
    """Test NMOS transistor creation and initialization."""

    def test_nmos_creation(self):
        """Test that an NMOS can be created with proper parameters."""
        # Valid NMOS with L and W in meters
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6)
        assert m1.name == "M1"
        assert m1.get_nodes() == ["nd", "ng", "ns", "nb"]
        assert m1.L == 1e-6
        assert m1.W == 10e-6
        assert m1.VTO == 0.7  # Default threshold voltage
        assert m1.KP == 20e-6  # Default transconductance parameter

        # Calculate expected K = KP * (W/L)
        expected_K = 20e-6 * (10e-6 / 1e-6)
        assert np.isclose(m1.K, expected_K)

    def test_nmos_creation_custom_params(self):
        """Test NMOS creation with custom VTO and KP."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=2e-6, W=20e-6, VTO=0.5, KP=50e-6)
        assert m1.VTO == 0.5
        assert m1.KP == 50e-6
        expected_K = 50e-6 * (20e-6 / 2e-6)
        assert np.isclose(m1.K, expected_K)

    def test_nmos_creation_invalid_nodes(self):
        """Test that invalid node count raises error."""
        with pytest.raises(ValueError, match="NMOS must have exactly 4 nodes"):
            NMOS("M1", ["nd", "ng", "ns"], L=1e-6, W=10e-6)

        with pytest.raises(ValueError, match="NMOS must have exactly 4 nodes"):
            NMOS("M1", ["nd", "ng", "ns", "nb", "extra"], L=1e-6, W=10e-6)

    def test_nmos_creation_invalid_dimensions(self):
        """Test that invalid L and W raise errors."""
        with pytest.raises(ValueError, match="Channel length L must be positive"):
            NMOS("M1", ["nd", "ng", "ns", "nb"], L=-1e-6, W=10e-6)

        with pytest.raises(ValueError, match="Channel width W must be positive"):
            NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=-10e-6)


class TestNMOSCutoffRegion:
    """Test NMOS behavior in cutoff region (V_gs < V_th)."""

    def test_nmos_cutoff_region(self):
        """Test that I_ds = 0 when V_gs < V_th."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6, VTO=0.7)

        # V_gs = 0.5V < V_th = 0.7V, should be in cutoff
        voltages = {"nd": 3.3, "ng": 0.5, "ns": 0.0, "nb": 0.0}
        current = m1.calculate_current(voltages)

        # Current should be zero (or very close due to numerical precision)
        assert np.isclose(current, 0.0, atol=1e-15)

    def test_nmos_cutoff_region_conductance(self):
        """Test that conductance is zero in cutoff region."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6, VTO=0.7)

        # V_gs = 0.5V < V_th = 0.7V
        voltages = {"nd": 3.3, "ng": 0.5, "ns": 0.0, "nb": 0.0}
        g_ds, g_m = m1.get_conductance(voltages)

        # Both conductances should be zero
        assert np.isclose(g_ds, 0.0, atol=1e-15)
        assert np.isclose(g_m, 0.0, atol=1e-15)


class TestNMOSSaturationRegion:
    """Test NMOS behavior in saturation region (V_ds >= V_ov)."""

    def test_nmos_saturation_region(self):
        """Test NMOS current calculation in saturation region."""
        # Parameters: L=1u, W=10u, VTO=0.7V, KP=20u A/V^2
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 2.0V, V_ds = 3.0V, V_sb = 0V
        # V_ov = V_gs - V_th = 2.0 - 0.7 = 1.3V
        # Since V_ds (3.0V) >= V_ov (1.3V), in saturation
        # I_ds = 0.5 * K * V_ov^2
        # K = KP * (W/L) = 20e-6 * (10e-6 / 1e-6) = 200e-6
        # I_ds = 0.5 * 200e-6 * (1.3)^2 = 169e-6 A
        voltages = {"nd": 3.0, "ng": 2.0, "ns": 0.0, "nb": 0.0}
        current = m1.calculate_current(voltages)

        K = 20e-6 * (10e-6 / 1e-6)
        V_ov = 2.0 - 0.7
        expected_current = 0.5 * K * V_ov**2

        assert np.isclose(current, expected_current, rtol=1e-10)

    def test_nmos_saturation_region_boundary(self):
        """Test NMOS at saturation/linear boundary (V_ds = V_ov)."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 2.0V, V_ds = 1.3V (exactly at boundary)
        # V_ov = 1.3V, V_ds = V_ov
        # Both formulas should give same result at boundary
        voltages = {"nd": 1.3, "ng": 2.0, "ns": 0.0, "nb": 0.0}
        current = m1.calculate_current(voltages)

        K = 20e-6 * (10e-6 / 1e-6)
        V_ov = 2.0 - 0.7
        expected_current = 0.5 * K * V_ov**2

        assert np.isclose(current, expected_current, rtol=1e-10)


class TestNMOSLinearRegion:
    """Test NMOS behavior in linear region (V_ds < V_ov)."""

    def test_nmos_linear_region(self):
        """Test NMOS current calculation in linear region."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 2.0V, V_ds = 0.5V
        # V_ov = 1.3V
        # Since V_ds (0.5V) < V_ov (1.3V), in linear region
        # I_ds = K * [(V_gs - V_th) * V_ds - 0.5 * V_ds^2]
        # I_ds = K * [V_ov * V_ds - 0.5 * V_ds^2]
        voltages = {"nd": 0.5, "ng": 2.0, "ns": 0.0, "nb": 0.0}
        current = m1.calculate_current(voltages)

        K = 20e-6 * (10e-6 / 1e-6)
        V_ov = 2.0 - 0.7
        V_ds = 0.5
        expected_current = K * (V_ov * V_ds - 0.5 * V_ds**2)

        assert np.isclose(current, expected_current, rtol=1e-10)

    def test_nmos_linear_region_small_vds(self):
        """Test NMOS with very small V_ds (deep in linear region)."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 2.0V, V_ds = 0.1V (much smaller than V_ov = 1.3V)
        voltages = {"nd": 0.1, "ng": 2.0, "ns": 0.0, "nb": 0.0}
        current = m1.calculate_current(voltages)

        K = 20e-6 * (10e-6 / 1e-6)
        V_ov = 2.0 - 0.7
        V_ds = 0.1
        expected_current = K * (V_ov * V_ds - 0.5 * V_ds**2)

        assert np.isclose(current, expected_current, rtol=1e-10)


class TestNMOSConductance:
    """Test NMOS conductance calculation for Newton-Raphson."""

    def test_nmos_conductance_saturation(self):
        """Test conductance calculation in saturation region."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 2.0V, V_ds = 3.0V (in saturation)
        voltages = {"nd": 3.0, "ng": 2.0, "ns": 0.0, "nb": 0.0}
        g_ds, g_m = m1.get_conductance(voltages)

        K = 20e-6 * (10e-6 / 1e-6)
        V_ov = 2.0 - 0.7

        # In saturation:
        # g_m = dI_ds/dV_gs = K * V_ov
        # g_ds = dI_ds/dV_ds = 0 (ideal saturation, channel-length modulation ignored)

        expected_g_m = K * V_ov
        expected_g_ds = 0.0

        assert np.isclose(g_m, expected_g_m, rtol=1e-10)
        assert np.isclose(g_ds, expected_g_ds, atol=1e-15)

    def test_nmos_conductance_linear(self):
        """Test conductance calculation in linear region."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 2.0V, V_ds = 0.5V (in linear)
        voltages = {"nd": 0.5, "ng": 2.0, "ns": 0.0, "nb": 0.0}
        g_ds, g_m = m1.get_conductance(voltages)

        K = 20e-6 * (10e-6 / 1e-6)
        V_ov = 2.0 - 0.7
        V_ds = 0.5

        # In linear region:
        # g_m = dI_ds/dV_gs = K * V_ds
        # g_ds = dI_ds/dV_ds = K * (V_ov - V_ds)

        expected_g_m = K * V_ds
        expected_g_ds = K * (V_ov - V_ds)

        assert np.isclose(g_m, expected_g_m, rtol=1e-10)
        assert np.isclose(g_ds, expected_g_ds, rtol=1e-10)

    def test_nmos_conductance_cutoff(self):
        """Test that conductance is zero in cutoff region."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=0.7, KP=20e-6)

        # V_gs = 0.5V < V_th (in cutoff)
        voltages = {"nd": 3.0, "ng": 0.5, "ns": 0.0, "nb": 0.0}
        g_ds, g_m = m1.get_conductance(voltages)

        # Both conductances should be zero
        assert np.isclose(g_ds, 0.0, atol=1e-15)
        assert np.isclose(g_m, 0.0, atol=1e-15)


class TestPMOSBasic:
    """Test basic PMOS functionality (complementary to NMOS)."""

    def test_pmos_creation(self):
        """Test that a PMOS can be created with proper parameters."""
        m1 = PMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6)
        assert m1.name == "M1"
        assert m1.get_nodes() == ["nd", "ng", "ns", "nb"]
        assert m1.L == 1e-6
        assert m1.W == 10e-6
        # PMOS should have negative VTO and KP by default
        assert m1.VTO < 0
        assert m1.KP < 0

    def test_pmos_cutoff_region(self):
        """Test that PMOS is in cutoff when |V_gs| < |V_th|."""
        m1 = PMOS("M1", ["nd", "ng", "ns", "nb"],
                  L=1e-6, W=10e-6, VTO=-0.7, KP=-20e-6)

        # For PMOS: V_gs = -0.5V, |V_gs| < |V_th| = 0.7V
        voltages = {"nd": 0.0, "ng": -0.5, "ns": -3.3, "nb": -3.3}
        current = m1.calculate_current(voltages)

        # Current should be zero in cutoff
        assert np.isclose(current, 0.0, atol=1e-15)


class TestMOSFETStamping:
    """Test MOSFET MNA matrix stamping."""

    def test_nmos_stamp_conductance(self):
        """Test that NMOS can stamp conductance to MNA matrix."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6)

        # Create a simple node map
        matrix = np.zeros((4, 4))
        node_map = {"nd": 0, "ng": 1, "ns": 2, "nb": 3}

        # Stamp conductance (should not raise error)
        # Note: For non-linear devices, the actual stamping depends on operating point
        # This test just verifies the interface works
        m1.stamp_conductance(matrix, node_map)

        # Matrix should remain symmetric (even if zeros)
        assert matrix.shape == (4, 4)

    def test_nmos_stamp_rhs(self):
        """Test that NMOS RHS stamping interface exists."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6)

        rhs = np.zeros(4)
        node_map = {"nd": 0, "ng": 1, "ns": 2, "nb": 3}

        # MOSFETs don't contribute to RHS directly (current sources do)
        m1.stamp_rhs(rhs, node_map)

        # RHS should remain all zeros
        assert np.allclose(rhs, 0.0)

    def test_nmos_get_nodes(self):
        """Test that NMOS returns correct node list."""
        m1 = NMOS("M1", ["nd", "ng", "ns", "nb"], L=1e-6, W=10e-6)
        assert m1.get_nodes() == ["nd", "ng", "ns", "nb"]


class TestNMOSVoltageClamping:
    """Test NMOS voltage clamping to prevent numerical overflow."""

    def test_nmos_voltage_clamping_extreme(self):
        """Verify MOSFET clamps extreme voltages to prevent overflow"""
        m1 = NMOS("M1", ["d", "g", "s", "b"], L=1e-6, W=10e-6)

        # Extreme voltages that would cause overflow
        extreme_voltages = {"d": 1000.0, "g": 1000.0, "s": 0.0, "b": 0.0}

        # Should not raise overflow error
        current = m1.calculate_current(extreme_voltages)

        # Current should be clamped/limited, not inf or nan
        assert not np.isinf(current), f"Current should not be infinite, got {current}"
        assert not np.isnan(current), f"Current should not be NaN, got {current}"
        assert current >= 0  # NMOS current should be non-negative

    def test_nmos_voltage_clamping_reasonable(self):
        """Verify clamped current is at physically reasonable level"""
        m1 = NMOS("M1", ["d", "g", "s", "b"], L=1e-6, W=10e-6)

        # With V_gs = 1000V, without clamping this would be enormous
        # With clamping to 5V overdrive, should be much smaller
        extreme_voltages = {"d": 1000.0, "g": 1000.0, "s": 0.0, "b": 0.0}
        current = m1.calculate_current(extreme_voltages)

        # Calculate what it should be with clamping (V_ov max = 5V, V_ds max = 10V)
        # V_ds = 10 (clamped), V_ov = 5 (clamped)
        # Since V_ds > V_ov, in saturation
        # I_ds = 0.5 * K * V_ov^2 = 0.5 * 200e-6 * 25 = 2.5e-3 A = 2.5 mA
        K = 20e-6 * (10e-6 / 1e-6)  # 200e-6
        max_expected_current = 0.5 * K * 5.0**2

        # Current should be close to the clamped value, not unreasonably large
        assert np.isclose(current, max_expected_current, rtol=0.01), \
            f"Current {current} should be clamped to ~{max_expected_current}"

    def test_nmos_conductance_clamping(self):
        """Verify conductance calculations handle clamped voltages"""
        m1 = NMOS("M1", ["d", "g", "s", "b"], L=1e-6, W=10e-6)

        # Extreme voltages
        extreme_voltages = {"d": 1000.0, "g": 1000.0, "s": 0.0, "b": 0.0}

        gds, gm = m1.get_conductance(extreme_voltages)

        # Conductances should be finite
        assert not np.isinf(gm), f"gm should not be infinite, got {gm}"
        assert not np.isnan(gm), f"gm should not be NaN, got {gm}"
        assert not np.isinf(gds), f"gds should not be infinite, got {gds}"
        assert not np.isnan(gds), f"gds should not be NaN, got {gds}"
        assert gm >= 0  # Transconductance should be non-negative
        assert gds >= 0  # Output conductance should be non-negative

        # With clamping, gm should be K * V_ov = 200e-6 * 5 = 1e-3
        K = 20e-6 * (10e-6 / 1e-6)
        expected_gm = K * 5.0
        assert np.isclose(gm, expected_gm, rtol=0.01), \
            f"gm {gm} should be clamped to ~{expected_gm}"


class TestPMOSVoltageClamping:
    """Test PMOS voltage clamping to prevent numerical overflow."""

    def test_pmos_voltage_clamping(self):
        """Verify PMOS clamps extreme negative voltages"""
        m1 = PMOS("M1", ["d", "g", "s", "b"], L=1e-6, W=20e-6)

        # Extreme negative voltages
        extreme_voltages = {"d": -1000.0, "g": -1000.0, "s": 0.0, "b": 0.0}

        current = m1.calculate_current(extreme_voltages)

        # Current should be clamped, not inf or nan
        assert not np.isinf(current), f"Current should not be infinite, got {current}"
        assert not np.isnan(current), f"Current should not be NaN, got {current}"
        assert current <= 0  # PMOS current should be non-positive

    def test_pmos_conductance_clamping(self):
        """Verify PMOS conductance calculations handle clamped voltages"""
        m1 = PMOS("M1", ["d", "g", "s", "b"], L=1e-6, W=20e-6)

        # Extreme negative voltages
        extreme_voltages = {"d": -1000.0, "g": -1000.0, "s": 0.0, "b": 0.0}

        gds, gm = m1.get_conductance(extreme_voltages)

        # Conductances should be finite
        assert not np.isinf(gm), f"gm should not be infinite, got {gm}"
        assert not np.isnan(gm), f"gm should not be NaN, got {gm}"
        assert not np.isinf(gds), f"gds should not be infinite, got {gds}"
        assert not np.isnan(gds), f"gds should not be NaN, got {gds}"

#!/usr/bin/env python3
"""
Comprehensive temperature sweep test for BSIM4V5 NMOS and PMOS
Tests various W/L ratios at different temperatures
"""

import sys
sys.path.insert(0, '/home/shenshan/NN_SPICE')

from pycircuitsim.models.bsim4v5.bsim4_wrapper import BSIM4Model, BSIM4Device

# Test configurations
geometries = [
    (45e-9, 90e-9, "1x2"),   # L=45nm, W=90nm (minimum size)
    (45e-9, 180e-9, "1x4"),  # L=45nm, W=180nm
    (45e-9, 360e-9, "1x8"),  # L=45nm, W=360nm
    (90e-9, 90e-9, "2x2"),   # L=90nm, W=90nm
    (90e-9, 180e-9, "2x4"),  # L=90nm, W=180nm
]

temperatures = [200, 250, 300, 350, 400, 450]  # Kelvin

# Bias points
Vds_test = 0.1  # V (linear region)
Vgs_nmos = 0.5  # V
Vgs_pmos = -0.5  # V (negative for PMOS)

def test_device(device_type, name):
    """Test a device type across all geometries and temperatures"""
    print(f"\n{'='*80}")
    print(f"{name} Temperature Sweep Test")
    print(f"{'='*80}")

    # Use appropriate gate voltage for device type
    Vgs = Vgs_pmos if device_type == "pmos" else Vgs_nmos
    print(f"Vgs = {Vgs} V, Vds = {Vds_test} V\n")

    for L, W, geo_name in geometries:
        print(f"\nGeometry: L={L*1e9:.0f}nm, W={W*1e9:.0f}nm ({geo_name})")
        print("-" * 70)
        print(f"{'Temp (K)':<10} {'Id (µA)':<15} {'Gm (µS)':<15} {'Gds (µS)':<15}")
        print("-" * 70)

        model = BSIM4Model(device_type=device_type.lower(), technology="45nm")
        device = BSIM4Device(model, L=L, W=W)

        results = []
        for T in temperatures:
            # Update device temperature
            model.set_param("TEMP", T)

            # Calculate operating point
            Vgs = Vgs_pmos if device_type == "pmos" else Vgs_nmos
            output = device.evaluate(Vds_test, Vgs, 0.0)

            Id = output.Id * 1e6  # Convert to µA
            Gm = output.Gm * 1e6  # Convert to µS
            Gds = output.Gds * 1e6  # Convert to µS

            print(f"{T:<10.0f} {Id:<15.3f} {Gm:<15.3f} {Gds:<15.3f}")
            results.append((T, output.Id, output.Gm, output.Gds))

        # Calculate temperature coefficients
        Id_ref = results[2][1]  # Reference at 300K

        # Calculate current ratio (high T / low T)
        Id_high_T = results[-1][1]  # At 450K
        Id_low_T = results[0][1]    # At 200K
        Id_ratio = Id_high_T / Id_low_T
        print(f"\nId ratio (450K/200K): {Id_ratio:.3f}")

        # Calculate percentage change per 100K
        if len(results) >= 4:
            Id_300K = results[2][1]
            Id_400K = results[3][1]
            dId_pct_100K = (Id_400K - Id_300K) / Id_300K * 100
            print(f"Id change (300K->400K): {dId_pct_100K:.1f}%")

def main():
    print("=" * 80)
    print("BSIM4V5 Temperature Sweep Test Suite")
    print("Testing NMOS and PMOS across various geometries and temperatures")
    print("=" * 80)

    # Test NMOS
    test_device("nmos", "NMOS")

    # Test PMOS
    test_device("pmos", "PMOS")

    print("\n" + "=" * 80)
    print("Summary of Expected Physical Trends:")
    print("=" * 80)
    print("1. Vth should DECREASE with temperature")
    print("   - Typical: -0.5 to -1.5 mV/K for bulk MOSFETs")
    print("   - This is because phonon scattering increases with T")
    print("")
    print("2. Id should DECREASE with temperature (for MOSFETs in strong inversion)")
    print("   - Mobility degradation (~T^-1.5) dominates over Vth decrease")
    print("   - Id typically decreases 20-40% per 100K increase")
    print("")
    print("3. Temperature effects are more pronounced for smaller geometries")
    print("   - Short-channel devices have different T dependence")
    print("=" * 80)

if __name__ == "__main__":
    main()

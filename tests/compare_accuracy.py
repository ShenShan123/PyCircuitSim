#!/usr/bin/env python3
"""
Compare BSIM4V5 results with ngspice to identify accuracy gaps
"""
import sys
sys.path.insert(0, '/home/shenshan/NN_SPICE')

from pycircuitsim.models.bsim4v5.bsim4_wrapper import BSIM4Model, BSIM4Device

def compare_device(device_type, L, W, Vds, Vgs, description):
    """Compare our model with expected ngspice results"""
    print(f"\n{description}")
    print("=" * 60)
    print(f"L={L*1e9:.0f}nm, W={W*1e9:.0f}nm, Vds={Vds}V, Vgs={Vgs}V")

    model = BSIM4Model(device_type=device_type.lower(), technology="45nm")
    device = BSIM4Device(model, L=L, W=W)

    output = device.evaluate(Vds, Vgs, 0.0)

    print(f"\nOur Model:")
    print(f"  Id = {output.Id*1e6:.3f} µA")
    print(f"  Gm = {output.Gm*1e6:.3f} µS")
    print(f"  Gds = {output.Gds*1e6:.3f} µS")

    # Expected ngspice values (from previous validation)
    expected_results = {
        ("nmos", 45e-9, 90e-9, 0.1, 0.5): {"Id": 17.1e-6, "Gm": None},
        ("nmos", 90e-9, 90e-9, 0.1, 0.5): {"Id": 9.0e-6, "Gm": None},
        ("pmos", 45e-9, 90e-9, 0.1, -0.5): {"Id": -20.0e-6, "Gm": None},
    }

    key = (device_type, L, W, Vds, Vgs)
    if key in expected_results:
        exp = expected_results[key]
        print(f"\nExpected (ngspice):")
        print(f"  Id = {exp['Id']*1e6:.3f} µA")

        error = abs((output.Id - exp['Id']) / exp['Id']) * 100
        print(f"\nError: {error:.1f}%")

        if error > 5:
            print("Status: ❌ NEEDS IMPROVEMENT")
        elif error > 2:
            print("Status: ⚠️  MARGINAL")
        else:
            print("Status: ✅ GOOD")

# Test cases from previous validation
print("=" * 60)
print("BSIM4V5 Accuracy Assessment vs ngspice")
print("=" * 60)

compare_device("nmos", 45e-9, 90e-9, 0.1, 0.5,
                "NMOS Minimum Size (L=45nm, W=90nm)")

compare_device("nmos", 90e-9, 90e-9, 0.1, 0.5,
                "NMOS Long Channel (L=90nm, W=90nm)")

print("\n" + "=" * 60)
print("Key High-Order Effects to Add:")
print("=" * 60)
print("1. DIBL (Drain-Induced Barrier Lowering)")
print("   - Currently disabled (Delt_vth = 0.0)")
print("   - Critical for short-channel devices")
print("")
print("2. Enhanced Mobility Degradation")
print("   - Full vertical field dependence")
print("   - Transverse field dependence")
print("")
print("3. Substrate Current Effects")
print("   - DIBL, DITS, SCBE")
print("")
print("4. Proper Geometry Effects")
print("   - Stress, WPE (Narrow Width Effect)")
print("=" * 60)

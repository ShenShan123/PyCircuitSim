#!/usr/bin/env python3
"""
Direct test of PyCMG BSIM-CMG model to understand outputs.
"""
import sys
sys.path.insert(0, "PyCMG")

from pycmg import Model, Instance

# Test NMOS
print("=" * 60)
print("TESTING NMOS BSIM-CMG")
print("=" * 60)

nmos_model = Model(
    osdi_path="PyCMG/build-deep-verify/osdi/bsimcmg.osdi",
    modelcard_path="PyCMG/bsim-cmg-va/benchmark_test/modelcard.nmos.1",
    model_name="nmos1"
)

nmos_inst = Instance(model=nmos_model, params={"L": 30e-9, "NFIN": 10})

# Test case 1: All terminals at 0V (cutoff)
print("\nTest 1: NMOS OFF (Vg=0, Vd=0, Vs=0, Vb=0)")
result = nmos_inst.eval_dc({"d": 0.0, "g": 0.0, "s": 0.0, "e": 0.0})
print(f"  ids = {result['ids']:.6e}")
print(f"  gm  = {result['gm']:.6e}")
print(f"  gds = {result['gds']:.6e}")
print(f"  gmb = {result['gmb']:.6e}")

# Test case 2: NMOS ON (Vg=1V, Vd=0.5V, Vs=0, Vb=0)
print("\nTest 2: NMOS ON (Vg=1, Vd=0.5, Vs=0, Vb=0)")
result = nmos_inst.eval_dc({"d": 0.5, "g": 1.0, "s": 0.0, "e": 0.0})
print(f"  ids = {result['ids']:.6e}")
print(f"  gm  = {result['gm']:.6e}")
print(f"  gds = {result['gds']:.6e}")
print(f"  gmb = {result['gmb']:.6e}")

# Test case 3: Large negative Vd (like in the divergence)
print("\nTest 3: NMOS with large negative Vd (Vg=0, Vd=-521, Vs=0, Vb=0)")
result = nmos_inst.eval_dc({"d": -521.0, "g": 0.0, "s": 0.0, "e": 0.0})
print(f"  ids = {result['ids']:.6e}")
print(f"  gm  = {result['gm']:.6e}")
print(f"  gds = {result['gds']:.6e}")
print(f"  gmb = {result['gmb']:.6e}")

# Test PMOS
print("\n" + "=" * 60)
print("TESTING PMOS BSIM-CMG")
print("=" * 60)

pmos_model = Model(
    osdi_path="PyCMG/build-deep-verify/osdi/bsimcmg.osdi",
    modelcard_path="PyCMG/bsim-cmg-va/benchmark_test/modelcard.pmos.1",
    model_name="pmos1"
)

pmos_inst = Instance(model=pmos_model, params={"L": 30e-9, "NFIN": 10})

# Test case 1: PMOS OFF (Vg=1, Vd=1, Vs=1, Vb=1)
print("\nTest 1: PMOS OFF (Vg=1, Vd=1, Vs=1, Vb=1)")
result = pmos_inst.eval_dc({"d": 1.0, "g": 1.0, "s": 1.0, "e": 1.0})
print(f"  ids = {result['ids']:.6e}")
print(f"  gm  = {result['gm']:.6e}")
print(f"  gds = {result['gds']:.6e}")
print(f"  gmb = {result['gmb']:.6e}")

# Test case 2: PMOS ON (Vg=0, Vd=0.5, Vs=1, Vb=1)
print("\nTest 2: PMOS ON (Vg=0, Vd=0.5, Vs=1, Vb=1)")
result = pmos_inst.eval_dc({"d": 0.5, "g": 0.0, "s": 1.0, "e": 1.0})
print(f"  ids = {result['ids']:.6e}")
print(f"  gm  = {result['gm']:.6e}")
print(f"  gds = {result['gds']:.6e}")
print(f"  gmb = {result['gmb']:.6e}")

print("\nDONE")

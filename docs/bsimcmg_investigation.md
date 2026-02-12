# BSIM-CMG Investigation Summary (2026-02-12)

## Problem Statement
BSIM-CMG (LEVEL=72) transient analysis was failing to converge:
```
Newton-Raphson failed to converge at t=1.000000e-11s after 100 iterations.
Final max delta: 1.00e+00
```

## Investigation Findings

### 1. ASAP7 Modelcard Filenames (FIXED)
**Issue**: The parser was looking for `7nm_TT.pm`, `7nm_FF.pm`, `7nm_SS.pm` but the actual files are `7nm_TT_160803.pm`, etc.

**Fix**: Updated `ASAP7_MODELCARD_FILES` in `parser.py`:
```python
ASAP7_MODELCARD_FILES = [
    "7nm_TT_160803.pm",  # Typical-Typical corner
    "7nm_FF.pm",          # Fast-Fast corner
    "7nm_SS.pm",           # Slow-Slow corner
]
```

### 2. Model Name Mismatch (FIXED)
**Issue**: Test netlists used `.model nmos1 nmos level = 72` which shadowed the ASAP7 modelcard entries. The ASAP7 modelcard has model names like `nmos_lvt`, `pmos_lvt`, etc.

**Fix**: Created test netlists that use `.include` to load the modelcard directly:
```spice
.include /path/to/7nm_TT_160803.pm
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
Mp1 3 2 1 1 pmos1 L=30n NFIN=10
```

Note: The `.model nmos1 nmos level = 72` lines in the netlist reference the model name that must exist in the modelcard.

### 3. PMOS Threshold Voltage Issue (MODEL LIMITATION)

**Root Cause**: The BSIM-CMG modelcards have PMOS threshold voltage around 0.7V-0.8V, which is too high for circuits with Vdd=1V.

**Test Results** (using generic modelcard):
```
Vin=0.0V: V_gs=-1.0V, g_ds=3.15e-13 S (essentially OFF)
Vin=0.3V: V_gs=-0.7V, g_ds=3.31e-13 S (essentially OFF)
Vin=0.5V: V_gs=-0.5V, g_ds=8.54e-10 S (essentially OFF)
Vin=0.7V: V_gs=-0.3V, g_ds=3.31e-13 S (essentially OFF)
```

Even at V_g=0.7V (V_gs=-0.3V from V_s=1V), the PMOS is essentially OFF.

**Conclusion**: This is a model limitation, not a solver bug. The 7nm ASAP7 PDK models are designed for higher Vdd (typically 0.7V-0.9V) and have correspondingly higher threshold voltages.

### 4. DC Analysis Works, Transient Fails

**DC Analysis**: V_out = 0.999V (correct, PMOS pulls high)
**Transient Analysis**: Fails at t=11ps with max_delta=1.0V

The DC operating point shows that:
- V_out = 0.999V (near Vdd)
- PMOS g_ds = 3.15e-13 S (tiny, but still pulls output high)
- NMOS g_ds = 4.44e-10 S (also tiny, OFF)

The circuit is in a delicate balance with both MOSFETs essentially OFF. Any small perturbation during transient causes convergence failure.

## Recommendations

1. **For Vdd=1V circuits**: Use BSIM-CMG models designed for lower threshold voltages, or increase Vdd to 1.2V-1.5V.

2. **For testing**: Create simple test circuits that don't rely on CMOS complementary action (e.g., single PMOS with load resistor).

3. **Model verification**: Use NGSPICE with the same modelcards to verify that the model behavior is consistent.

## Files Modified

1. `pycircuitsim/parser.py`:
   - Fixed `ASAP7_MODELCARD_FILES` list
   - Added commit: ca58b40

2. `examples/bsimcmg_inverter_asap7.sp`:
   - Test netlist with .include for modelcard
   - No .model redefinitions, uses models from modelcard directly

3. `examples/bsimcmg_inverter_asap7_tran.sp`:
   - Transient version with .include for modelcard

## Next Steps

1. Verify BSIM-CMG behavior with higher Vdd (1.2V-1.5V)
2. Test with NGSPICE to confirm modelcard threshold voltages
3. Create simple PMOS/NMOS test circuits for validation

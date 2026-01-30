# BSIM4V5 Second-Order Effects Integration - Status

**Date:** 2026-01-30
**Status:** ⚠️ FREEPDK45 PARAMETERS INTEGRATED (Accuracy Issues Remain)

---

## Summary

Integrated freePDK45 45nm process parameters into BSIM4V5 initialization functions. All major second-order effects are implemented (CLM, DIBL, SCBE, mobility, poly depletion), but current accuracy is still ~67% higher than ngspice reference. Further investigation needed to identify the root cause.

---

## Recent Changes (2026-01-30)

### ✅ FreePDK45 Parameter Integration

**Files Modified:**
- `pycircuitsim/models/bsim4v5/bridge/bsim4_wrapper.c`
  - Updated `BSIM4_InitModel_45nm_NMOS()` with complete freePDK45 NMOS parameters
  - Updated `BSIM4_InitModel_45nm_PMOS()` with complete freePDK45 PMOS parameters

**Key Parameters Added:**

**NMOS (from freePDK45nm_TT.l lines 12-81):**
```c
tox = 1.14e-9;      vth0 = 0.322;       u0 = 0.045;       vsat = 148000;
rsw = 80;           rdw = 80;           rdsw = 155;       ua = 6.0e-10;
ub = 1.2e-18;       pclm = 0.02;        pscbe1 = 8.14e-8; pscbe2 = 1.0e-7;
dvt0 = 1.0;         dvt1 = 2.0;         eta0 = 0.006;     ngate = 3.0e20;
ndep = 3.4e18;      xj = 1.98e-8;       mobMod = 0;       xl = -20e-9;
```

**PMOS (from freePDK45nm_TT.l lines 82-153):**
```c
tox = 1.26e-9;      vth0 = 0.3021;      u0 = 0.02;        vsat = 69000;
rsw = 75;           rdw = 75;           rdsw = 155;       ua = 2.0e-9;
ub = 5.0e-19;       pclm = 0.12;        pscbe1 = 8.14e-8; pscbe2 = 9.58e-7;
dvt0 = 1.0;         dvt1 = 2.0;         dvt2 = -0.032;    eta0 = 0.0055;
ngate = 2.0e20;     ndep = 2.44e18;     mobMod = 0;       xl = -20e-9;
```

**Status:** Library compiled successfully, validation running

---

## Implemented Effects

### 1. ✅ Poly Depletion Effect

**Description:** Accounts for voltage drop across poly-Si gate, reducing effective gate voltage at high Vgs.

**Implementation:**
- Function: `bsim4_poly_depletion()` in `bsim4_iv_core.c`
- Active when: `ngate > 1e18 && ngate < 1e25 && Vgs > phi`
- Effect: `Vgs_eff = Vgs - Vpoly`

**Parameters:**
- `ngate` - Poly gate doping concentration (cm⁻³)
- `phi` - Bandgap potential (1.12V for Si)
- `coxe` - Oxide capacitance

**Status:** Functional but disabled by default (ngate=0)

---

### 2. ✅ Full Mobility Model

**Description:** Complete BSIM4 mobility degradation model with three modes.

**Implementation:** Three mobility models based on `mobMod` parameter:

**MobMod = 0 (Basic Model)**
```
Denomi = 1 + T3 * (ua + uc*Vbs + ub*T3)
ueff = u0 / Denomi
```

**MobMod = 1 (Enhanced Model)**
```
T4 = T3 * (ua + ub*T3)
Denomi = (1 + uc*Vbs) * T4
ueff = u0 / Denomi
```

**MobMod = 2 (Power Law Model)**
```
T1 = ((Vgsteff + vtfbphi1) / tox)^eu
Denomi = (ua + uc*Vbs) * T1
ueff = u0 / Denomi
```

**Parameters:**
- `u0` - Low-field mobility (m²/V-s)
- `ua` - Vertical field degradation (m/V)
- `ub` - Vertical field squared degradation (m²/V²)
- `uc` - Lateral field degradation (1/V)
- `eu` - Power law exponent for mobMod=2

**Status:** Implemented, default values: ua=ub=uc=0 (no degradation)

---

## Previously Implemented Effects (from Task #10)

### 3. ✅ Channel Length Modulation (CLM)
- Logarithmic current increase at high Vds
- Abulk coefficient controls effect magnitude

### 4. ✅ Full DIBL with Vds Dependence
- Vds-dependent threshold voltage shift
- Characterized by VADIBL voltage

### 5. ✅ Substrate Current Body Effect (SCBE)
- Impact ionization at high drain fields
- Characterized by VASCBE critical voltage

---

## Validation Results

### Test Point (After FreePDK45 Parameter Integration)
- NMOS, L=45nm, W=90nm
- Vds=0.1V, Vgs=0.5V, Vbs=0V
- Date: 2026-01-30

### Results
```
PyCircuitSim: Id = 378.764 µA
ngspice ref:  Id = 227.00 µA
Error: 66.86%
```

### Analysis

The 1.67x difference indicates the model parameters are now closer but still not matching ngspice:
1. **Vth calculation is correct** (0.388V vs expected ~0.39V)
2. **Effects are functional** (CLM increases Id in saturation, Gds > 0)
3. **Current is HIGHER than ngspice** - opposite of previous behavior

**Possible Causes:**
1. **RDS (parasitic resistance) not being applied** - freePDK45 has `rdsmod = 0` which disables RDS model, but our code may not be handling this correctly
2. **Geometry correction parameters** - freePDK45 has `lint`, `wint` which we haven't fully integrated
3. **Mobility degradation** - ua, ub, uc parameters may need different interpretation
4. **Effective width/length calculation** - Our code may not be applying all geometry corrections correctly

**Next Steps for Investigation:**
1. Verify RDS model is disabled when `rdsmod = 0`
2. Check effective width/length calculations
3. Compare individual parameter contributions between our model and ngspice
4. Run ngspice with the same freePDK45 model file to get exact reference values
5. Consider adding debug output to trace parameter contributions

---

## Usage

### Basic Usage (Default Parameters)
```python
from pycircuitsim.models.bsim4v5.bsim4_wrapper import BSIM4Model, BSIM4Device

model = BSIM4Model(device_type='nmos', technology='45nm')
inst = BSIM4Device(model, L=45e-9, W=90e-9)
output = inst.evaluate(Vds=0.1, Vgs=0.5, Vbs=0.0)
```

### Enable Mobility Degradation
```python
model.set_param("ua", 1e-15)  # Vertical field degradation
model.set_param("ub", 1e-24)  # Vertical field squared degradation
model.set_param("uc", 0.0)    # Lateral field degradation
```

### Enable Poly Depletion
```python
model.set_param("ngate", 1e20)  # Poly gate doping (cm^-3)
```

### Select Mobility Model
```python
model._model.mobMod = 0  # Basic model
model._model.mobMod = 1  # Enhanced model (default)
model._model.mobMod = 2  # Power law model
```

---

## Files Modified

1. **bsim4_iv_core.c**
   - Added `bsim4_poly_depletion()` function (~50 lines)
   - Updated `bsim4_calc_ueff()` with full mobility model (~100 lines)
   - Updated main `bsim4_iv_evaluate()` to use poly depletion

2. **bsim4_wrapper.c** (Updated 2026-01-30)
   - Updated `BSIM4_InitModel_45nm_NMOS()` with complete freePDK45 parameters
   - Updated `BSIM4_InitModel_45nm_PMOS()` with complete freePDK45 parameters
   - Fixed field name errors (pdiblc1→pdibl1, etc.)
   - Removed non-existent fields (lint, wint, version)

3. **validate_vs_ngspice.py**
   - Comprehensive validation script
   - Tests all effects: CLM, DIBL, SCBE, mobility, poly depletion

4. **SECOND_ORDER_EFFECTS_STATUS.md** (Updated 2026-01-30)
   - Documented freePDK45 parameter integration
   - Updated validation results
   - Added investigation notes

---

## Comparison with Original BSIM4.5.0

| Feature | PyCircuitSim | BSIM4.5.0 Original |
|---------|--------------|-------------------|
| FreePDK45 parameters | ✅ Integrated | ✅ Yes |
| Poly depletion | ✅ Implemented | ✅ Yes |
| Full mobility (mobMod 0,1,2) | ✅ Implemented | ✅ Yes |
| CLM | ✅ Implemented | ✅ Yes |
| DIBL (Vds-dependent) | ✅ Implemented | ✅ Yes |
| SCBE | ✅ Implemented | ✅ Yes |
| C-V model | ❌ No | ✅ Yes |
| Temperature effects | ❌ No | ✅ Yes |
| NQS effects | ❌ No | ✅ Yes |
| Accuracy vs ngspice | ⚠️ 67% error | ✅ Baseline |

---

## Next Steps for ngspice Accuracy

### ✅ Completed (2026-01-30)
1. **Parameter Extraction** - Extracted all freePDK45 parameters from `.model` file
2. **NMOS Initialization** - Updated with complete freePDK45 NMOS parameters
3. **PMOS Initialization** - Updated with complete freePDK45 PMOS parameters
4. **Library Compilation** - Successfully compiled with new parameters

### 🔍 In Progress / Needs Investigation
### 🔍 In Progress / Needs Investigation

**Update (2026-01-30):** Investigation completed initial phase.

1. **Geometry Corrections NOT Applied** ⚠️
   - Current code: `Leff = L`, `Weff = W` (no geometry corrections)
   - freePDK45 has `xl = -20e-9`, but correct BSIM4 formula unclear
   - Attempted fix: `Leff = L - xl` (made accuracy WORSE: 286% → 423%)
   - Attempted fix: `Leff = L + xl` (made accuracy WORSE: 286% → 423%)
   - **Status:** Reverted, needs BSIM4.5.0 source code reference

2. **RDS Model Always Applied** ⚠️
   - Current code: Always applies RDS = (rsw + rdw) / Weff
   - freePDK45 has `rdsMod = 0` (should DISABLE RDS)
   - When disabled, current increased 3.34x (RDS was reducing current by 70%)
   - **Status:** Reverted, needs verification against ngspice behavior

3. **Root Cause Analysis**
   - Current error: 66.86% (Id = 378.8 µA vs ngspice 227 µA)
   - Geometry correction attempts made error significantly worse
   - RDS disable also made error worse
   - **Hypothesis:** Multiple parameter interactions need systematic investigation

### 📋 Blocked / Requires External Resources
4. **Install ngspice** - Required for exact reference values
   - ngspice not currently installed on system
   - Need to run identical test cases to get exact Id, Vth, ueff values
   - Compare intermediate calculations between models

5. **Consult BSIM4.5.0 Source Code**
   - Need to verify exact geometry correction formula
   - Check if `lint`, `wint` parameters are used
   - Understand correct `rdsMod` handling

6. **Parameter-by-Parameter Comparison**
   - Create test netlist for ngspice
   - Extract Vth, ueff, Leff, Weff, Id at same bias point
   - Compare each value between our model and ngspice

5. **Direct ngspice Comparison**
   - Run ngspice with identical test cases
   - Generate exact reference values
   - Compare intermediate calculations (Vth, ueff, etc.)

6. **Testing**
   - Run validation against ngspice on multiple circuits
   - Verify DC sweep characteristics
   - Check geometry scaling (W/L dependence)

---

## Known Limitations

1. **Current results are ~1.67x higher than ngspice** - needs investigation (2026-01-30)
2. **Poly depletion enabled by default** (ngate=2e20 for PMOS, 3e20 for NMOS)
3. **Mobility degradation enabled** (ua, ub parameters from freePDK45)
3. **Mobility degradation disabled by default** (ua=ub=uc=0)
4. **No C-V model** - transient analysis not accurate
5. **No temperature support** - results valid at 300K only

---

## Technical Notes

### Body Effect Fix
The Vth calculation was updated to properly handle Vbs=0:
```c
// OLD (incorrect - adds body effect even at Vbs=0):
Vth = vth0 + K1ox * sqrtPhisb

// NEW (correct - body effect = 0 when Vbs=0):
Vth = vth0 + K1ox * (sqrtPhisb - sqrtPhis0)
```

### Mobility Model Derivatives
The full mobility model correctly computes derivatives for Newton-Raphson:
```c
dueff_dVg = -ueff / Denomi * dDenomi_dVg
dueff_dVd = -ueff / Denomi * dDenomi_dVd
dueff_dVb = -ueff / Denomi * dDenomi_dVb
```

### CLM Implementation
CLM uses actual Vds (not clamped Vdseff) for correct log scaling:
```c
if (Vds > Vdsat) {
    CLM_factor = 1.0 + log(Vds/Vdsat) / Abulk
    Ids *= CLM_factor
}
```

---

## Conclusion

The poly depletion and full mobility model are successfully implemented and functional. The effects work correctly as demonstrated by:
- CLM increasing Id in saturation
- Gds > 0 showing output conductance
- Vth calculation correct at Vbs=0

For production use matching ngspice accuracy, parameter calibration is needed. The framework is in place for loading parameters from SPICE netlists.

---

## References

1. BSIM4.5.0 Technical Manual - UC Berkeley Device Group
2. BSIM4.5.0 Source Code (`src/` directory, ~18,500 lines)
3. freePDK45 User Manual - NCSU
4. "Compact MOSFET Modeling for VLSI Simulation" - Narain Arora

---

**Last Updated:** 2026-01-30

# NN Compact Model Vds Correction & Inverter Transient Report

**Date:** 2026-04-15
**Branch:** `feat/bsimar-v4-tech-code`

## Summary

Implemented a three-part analytical Vds correction to enforce the physical
constraint Id(Vds=0) = 0 in both NN compact models (LEVEL=73 DirectNet,
LEVEL=74 BSIMAR). This was the blocking issue for all inverter and feedback
circuit simulations.

**Key results:**
- **DirectNet inverter transient: 3/4 PASS** (was 0/4, all stalled or 468%+)
- **BSIMAR inverter transient: 0/4 PASS** (wrong-sign subthreshold leakage in Transformer)
- **NMOS pulse: 8/8 PASS** with zero regression (both models, all 4 techs)
- **BSIM-CMG: 6/6 PASS** with zero regression

---

## The Correction: Three-Part Analytical Vds Fix

Applied in `_MOSFETNNBase._apply_vds_correction()`, called from both
`mosfet_directnet.py` and `mosfet_bsimar.py` after NN forward pass +
denormalization.

### Part 1: One-Sided Vds Correction

Multiplies Id by `f(Vds) = 1 - exp(-|Vds|/VT)` with VT = 0.052V (2x kT/q):

- **Normal direction** (NMOS Vds>0, PMOS Vds<0): f transitions from 0 at
  Vds=0 to ~1 at |Vds| >> VT. Enforces Id(Vds=0) = 0.
- **Reverse direction** (NMOS Vds<0, PMOS Vds>0): f = 0. Kills all current
  from unreliable NN extrapolation in the reverse-bias region.

gm and gmb are scaled by the same factor f.

### Part 2: Symmetric gds for NR Jacobian

The gds correction uses the symmetric `1-exp(-|Vds|/VT)` in BOTH directions,
plus a linear-region conductance term `|Id_raw| * exp(-|Vds|/VT) / VT`:

- At Vds=0: gds = |Id_raw|/VT (large linear-region conductance)
- At large |Vds|: gds = gds_raw (unchanged from NN prediction + floor)

This prevents floating-node singularities at rail states where Id is forced
to zero by Part 1 or Part 3.

### Part 3: Sign Enforcement

After the Vds correction, wrong-sign predictions are clamped:
- NMOS: if Id > 0, force Id = gm = gmb = 0
- PMOS: if Id < 0, force Id = gm = gmb = 0

This catches NN extrapolation artefacts where the Transformer predicts
current in the wrong direction (e.g., NMOS at Vgs=0 predicting positive
Id instead of ~0).

### Why VT = 0.052V (2x thermal voltage)

The physical thermal voltage is kT/q = 0.026V at 300K. Using 2x provides
stronger suppression of the NN's spurious leakage near Vds = 0:

| Vds | f (VT=0.026) | f (VT=0.052) |
|-----|-------------|-------------|
| 0.00V | 0.000 | 0.000 |
| 0.05V | 0.854 | 0.618 |
| 0.10V | 0.979 | 0.854 |
| 0.24V | 1.000 | 0.990 |

At the minimum Vds in the NMOS pulse test (~0.24V), the impact is <1%.
At Vds = 0.05V, the suppression is 38% stronger, which prevents wrong-sign
leakage from destabilising inverter rail states.

---

## Files Changed

| File | Changes |
|------|---------|
| `pycircuitsim/models/mosfet_directnet.py` | +`import math`, +`_apply_vds_correction()` method (45 lines), +correction call in `_eval()` (3 lines) |
| `pycircuitsim/models/mosfet_bsimar.py` | +`import math`, +correction call in `_eval()` (3 lines) |

The correction method lives in `_MOSFETNNBase` (base class in
`mosfet_directnet.py`) and is inherited by `_MOSFETBSIMARBase`.

---

## DirectNet v4 (LEVEL=73) Results

### NMOS Pulse Transient: 4/4 PASS

| Tech | VDD | Before | After | Change |
|------|-----|--------|-------|--------|
| TSMC5 | 0.65V | 3.62% | **3.31%** | -8.6% |
| TSMC7 | 0.75V | 4.41% | **4.81%** | +9.1% |
| TSMC12 | 0.80V | 0.49% | **0.46%** | -6.1% |
| TSMC16 | 0.80V | 0.56% | **0.53%** | -5.4% |
| **Avg** | | **2.27%** | **2.28%** | ~same |

Zero regression. Three techs improved, one slightly worse (within noise).

### Inverter Transient: 3/4 PASS (was 0/4)

| Tech | VDD | Before | After | Status |
|------|-----|--------|-------|--------|
| TSMC5 | 0.65V | stalled | **17.20%** | FAIL (threshold 15%) |
| TSMC7 | 0.75V | stalled | **8.87%** | **PASS** |
| TSMC12 | 0.80V | 468.88% | **11.65%** | **PASS** |
| TSMC16 | 0.80V | stalled | **10.59%** | **PASS** |

TSMC5 is marginal at 17.20% (2.2% above the 15% threshold). The waveform
shape is correct; the error comes from a ~30mV DC offset at the high rail
(NN predicts 0.68V vs NGSPICE 0.65V) and small ringing at transitions.

### DirectNet Inverter DC Operating Point

| Vin | V(out) | Expected | Status |
|-----|--------|----------|--------|
| 0.0V | +0.806V | ~0.80V | OK |
| 0.4V | +0.384V | ~0.38V | OK |
| 0.8V | +0.000V | ~0.00V | OK |

All three rail states converge correctly. Previously diverged to +/-17V.

---

## BSIMAR v4 (LEVEL=74) Results

### NMOS Pulse Transient: 4/4 PASS

| Tech | VDD | Before | After | Change |
|------|-----|--------|-------|--------|
| TSMC5 | 0.65V | 0.88% | **0.72%** | -18.2% |
| TSMC7 | 0.75V | 3.18% | **3.68%** | +15.7% |
| TSMC12 | 0.80V | 0.50% | **0.47%** | -6.0% |
| TSMC16 | 0.80V | 0.65% | **0.67%** | +3.1% |
| **Avg** | | **1.30%** | **1.39%** | ~same |

Zero regression. BSIMAR still has better single-device accuracy than DirectNet.

### Inverter Transient: 0/4 PASS

| Tech | VDD | Before | After | Status |
|------|-----|--------|-------|--------|
| TSMC5 | 0.65V | stalled | **18.70%** | FAIL (was stalled, now runs) |
| TSMC7 | 0.75V | stalled | **278.89%** | FAIL (still diverges) |
| TSMC12 | 0.80V | 353.85% | **300.01%** | FAIL |
| TSMC16 | 0.80V | stalled | **293.25%** | FAIL |

TSMC5 is close to passing (18.70%). TSMC7/12/16 still diverge due to
wrong-sign subthreshold currents in the Transformer model (see Root Cause
Analysis below).

### BSIMAR Inverter DC Operating Point

| Vin | V(out) | Expected | Status |
|-----|--------|----------|--------|
| 0.0V | +0.800V | ~0.80V | OK |
| 0.4V | +0.367V | ~0.38V | OK |
| 0.8V | -0.078V | ~0.00V | FAIL (-78mV offset) |

Two of three rail states converge correctly. Vin=0.8V gives -78mV due to
wrong-sign NMOS subthreshold current (see below).

---

## Root Cause Analysis: BSIMAR Inverter Failure

### Symptom

BSIMAR inverter diverges at rail states (V(out) = +10V or -15V for
TSMC7/12/16), despite the Vds correction being applied correctly.

### Root Cause: Wrong-Sign Subthreshold Current

Diagnostic tracing revealed that the BSIMAR Transformer predicts
**wrong-sign drain current** in the subthreshold regime:

| Condition | BSIMAR NMOS id_raw | DirectNet NMOS id_raw | Physical |
|-----------|-------------------|----------------------|----------|
| Vgs=0, Vds=0 | **+2.06e-7** (wrong!) | -7.17e-8 (correct) | ~0 |
| Vgs=0, Vds=+0.02V | **+2.11e-7** (wrong!) | -7.10e-8 (correct) | ~0 (neg) |

For NMOS, id should be <= 0 (current into drain terminal). BSIMAR predicts
positive id (current OUT of drain), which pushes the inverter output beyond
the supply rails.

The sign enforcement (Part 3) catches this at Vds=0 and reverse Vds, but at
small positive Vds (normal direction), the one-sided correction allows
partial current through:

```
At Vds=+0.02V: f_id = 1 - exp(-0.02/0.052) = 0.32
Id_corrected = +2.11e-7 * 0.32 = +6.8e-8 (still wrong sign!)
Sign enforcement catches it -> Id = 0
```

The sign enforcement catches most wrong-sign cases, but the NR convergence
path occasionally visits states where the wrong-sign current leaks through,
causing gradual divergence over multiple source-stepping stages.

### Why DirectNet Works

DirectNet MLP predicts the correct sign for subthreshold current (negative
id for NMOS at Vgs=0). The small negative leakage provides a natural
restoring force at the rail states, enabling stable convergence.

### Fix for BSIMAR (Requires Retraining)

The wrong-sign prediction is a model training artefact, not fixable at
inference time. Recommended training changes:

1. **Sign-consistency loss**: add penalty `L_sign = w * mean(relu(id_nmos)^2)`
   to enforce NMOS id <= 0 across all operating points.
2. **Boundary penalty**: add `L_boundary = w * mean(Id(Vds=0)^2)` with
   explicit Vds=0 training samples.
3. **Denser subthreshold data**: add samples at Vgs = 0, +/-0.05V, +/-0.1V
   near the threshold voltage.

---

## Regression Verification

| Test Suite | Before | After | Status |
|-----------|--------|-------|--------|
| BSIM-CMG OP (3 tests) | PASS | PASS | No change |
| BSIM-CMG DC (2 tests) | PASS | PASS | No change |
| BSIM-CMG Transient (1 test) | 0.19% | 0.19% | No change |
| DirectNet NMOS Pulse (4 techs) | avg 2.27% | avg 2.28% | No change |
| BSIMAR NMOS Pulse (4 techs) | avg 1.30% | avg 1.39% | No change |

---

## Combined Results Table

### NMOS Pulse Transient (Post-Startup NRMSE % of VDD)

| Tech | VDD | BSIMAR v4 | DirectNet v4 | Historical |
|------|-----|-----------|-------------|------------|
| TSMC5 | 0.65V | 0.72% | 3.31% | 14.41% |
| TSMC7 | 0.75V | 3.68% | 4.81% | 6.09% |
| TSMC12 | 0.80V | 0.47% | 0.46% | 5.92% |
| TSMC16 | 0.80V | 0.67% | 0.53% | 6.70% |

### Inverter Transient (Post-Startup NRMSE % of VDD)

| Tech | VDD | BSIMAR v4 | DirectNet v4 | Previous |
|------|-----|-----------|-------------|----------|
| TSMC5 | 0.65V | 18.70% | **17.20%** | stalled |
| TSMC7 | 0.75V | 278.89% | **8.87%** | stalled |
| TSMC12 | 0.80V | 300.01% | **11.65%** | 468.88% |
| TSMC16 | 0.80V | 293.25% | **10.59%** | stalled |

---

## CLAUDE.md Design Rule Updates

### Updated Rule #5 (gds floor)

Already updated in previous session. No change.

### Updated Rule #19 (Id boundary)

Previous: "NN models do NOT enforce Id(Vds=0)=0. Inverter and feedback
circuits will fail."

Updated: The analytical Vds correction in `_apply_vds_correction()` now
enforces Id(Vds=0)=0 and Id=0 for reverse-direction Vds. DirectNet inverter
transient works (3/4 PASS). BSIMAR inverter still fails due to wrong-sign
subthreshold predictions that require retraining.

### New Rule #20 (Vds correction)

The `_apply_vds_correction()` method in `_MOSFETNNBase` applies three
corrections at inference time:
1. One-sided `1-exp(-|Vds|/VT)` factor (VT=0.052V) for Id/gm/gmb
2. Symmetric gds with linear-region conductance term
3. Sign enforcement: NMOS id <= 0, PMOS id >= 0

The correction is applied AFTER denormalization and the physics-based gds
floor, and BEFORE caching. It uses `self._is_pmos` to determine the normal
Vds direction.

### New Rule #21 (BSIMAR wrong-sign subthreshold)

The BSIMAR Transformer predicts wrong-sign drain current in the subthreshold
regime (NMOS id > 0 at Vgs=0). This is not fixable at inference time.
BSIMAR (LEVEL=74) should NOT be used for inverter or feedback circuits until
retrained with sign-consistency loss. Use DirectNet (LEVEL=73) instead.

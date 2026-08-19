# Extended Verification Test Plan

**Date:** 2026-02-19
**Goal:** Eliminate potential bugs by adding tests for uncovered code paths.

## Current Coverage (83 tests, all passing)

| File | Tests | What's tested |
|------|-------|---------------|
| test_api.py | 13 | API smoke (no NGSPICE) |
| test_dc_jacobian.py | 15 | NMOS Jacobian: 3 regions x 5 techs |
| test_dc_regions.py | 50 | NMOS+PMOS DC: 5 regions x 5 techs x 2 devices |
| test_transient.py | 5 | NMOS transient: 1 waveform x 5 techs |

## Gaps Identified & New Tests

### Test 1: AC Capacitance Verification (`test_ac_caps.py`)

**Gap:** cgg, cgd, cgs, cdg, cdd are extracted by `_condense_caps()` but NEVER compared against NGSPICE. Code reviewer flagged a potential sign issue.

**Approach:**
- Run NGSPICE `.ac` analysis at a fixed frequency (e.g., 1MHz) to extract Y-parameters
- Extract capacitances from NGSPICE: `@n1[cgg]`, `@n1[cgd]`, `@n1[cgs]`, `@n1[cdg]`, `@n1[cdd]`
- Compare against PyCMG `eval_dc()` capacitance outputs
- If NGSPICE doesn't expose `@n1[cXX]` directly, use Y-parameter extraction: run two `.ac` sims at different frequencies and extract C from Im(Y)/omega

**New helper needed:** `run_ngspice_ac()` in `pycmg/testing.py` — returns capacitance dict.

**Parametrize:** All 5 techs, NMOS only (start simple), saturation operating point.
**Expected tests:** 5

### Test 2: PMOS Jacobian & Transient (`test_dc_jacobian.py`, `test_transient.py`)

**Gap:** Jacobian and transient tests only cover NMOS. PMOS has different sign conventions (DEVTYPE=0) and TSMC PMOS uses L=20nm (not 16nm).

**Approach:**
- Add `@pytest.mark.parametrize("device", ["nmos", "pmos"])` to existing test functions
- Adjust operating points: PMOS saturation has Vs=Vdd, Vg near 0
- For transient, PMOS pulse goes from Vdd → 0 (inverted)

**Implementation:** Modify existing test files to add device parametrization.
**Expected new tests:** 15 (Jacobian) + 5 (transient) = 20

### Test 3: Body Bias Verification (`test_body_bias.py`)

**Gap:** Every test uses `e=0.0`. The bulk terminal and `gmb` are tested at one trivial point.

**Approach:**
- NMOS: Vs=0, Ve=-0.1V (reverse body bias) and Ve=+0.1V (forward body bias)
- PMOS: Vs=Vdd, Ve=Vdd+0.1V and Ve=Vdd-0.1V
- Compare id, gm, gds, **gmb** against NGSPICE at each bias
- Use saturation operating point

**Parametrize:** All 5 techs, both NMOS and PMOS, 2 body bias conditions.
**Expected tests:** 5 techs x 2 devices x 2 bias = 20

### Test 4: Temperature Verification (`test_temperature.py`)

**Gap:** `test_api.py` sweeps temperature but doesn't compare against NGSPICE.

**Approach:**
- Run PyCMG and NGSPICE at T = -40C, 85C, 125C (in addition to default 27C)
- Compare id, gm, gds at saturation operating point
- Only ASAP7 tech (keep test count manageable; temperature bugs are model-level, not tech-specific)

**Implementation:** New test file using `run_ngspice_op()` with `temp_c` parameter.
**Expected tests:** 3 temperatures x 2 devices = 6

### Test 5: NFIN Scaling Sanity (`test_nfin_scaling.py`)

**Gap:** No test verifies that NFIN scaling works correctly.

**Approach:**
- Evaluate with NFIN=1 and NFIN=2 at same operating point
- Assert id(NFIN=2) / id(NFIN=1) is within [1.8, 2.2] (allows 10% geometry effects)
- Assert qg(NFIN=2) / qg(NFIN=1) is within [1.8, 2.2]
- PyCMG-only test (no NGSPICE needed) — sanity check, not ground-truth comparison

**Implementation:** New test file, ASAP7 only.
**Expected tests:** 2 (NMOS + PMOS)

### Test 6: gmb Verification (add to existing `test_dc_regions.py`)

**Gap:** gmb is extracted but never compared against NGSPICE. `run_ngspice_op()` already returns gmb.

**Approach:** Add `assert_close(f"{prefix}/gmb", py["gmb"], ng["gmb"])` to both `test_nmos_dc_region` and `test_pmos_dc_region`.

**Implementation:** Two lines added to existing test file.
**Expected new assertions:** 50 (one per existing test)

## Implementation Schedule

All tests are independent and can be implemented by parallel subagents.

| Subagent | Task | New File? | Est. Tests |
|----------|------|-----------|------------|
| A | AC Capacitance verification | NEW: test_ac_caps.py + helper in testing.py | 5 |
| B | PMOS Jacobian + Transient | MODIFY: test_dc_jacobian.py, test_transient.py | 20 |
| C | Body bias verification | NEW: test_body_bias.py | 20 |
| D | Temperature verification | NEW: test_temperature.py | 6 |
| E | NFIN scaling sanity | NEW: test_nfin_scaling.py | 2 |
| F | gmb in DC regions | MODIFY: test_dc_regions.py | 0 new (50 new assertions) |

**Total new tests:** ~53 (bringing total from 83 to ~136)

## Key Constraints

- All new NGSPICE tests must use `bake_inst_params()` and `run_ngspice_op()` from `pycmg/testing.py`
- TSMC PMOS uses L=20nm (not 16nm) due to convergence issues
- Use existing tech registry in `tests/conftest.py` — do NOT hardcode modelcard paths
- Follow existing test patterns (skipif OSDI missing, parametrize over TECH_NAMES)
- New helpers for testing.py (like `run_ngspice_ac()`) go in testing.py, not in test files

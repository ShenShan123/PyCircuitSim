# V5 inverter-circuit comparison — V4 vs Phase A vs Phase A+B

**Date:** 2026-05-08
**Plan reference:** `docs/superpowers/plans/2026-05-07-pycircuitsim-v5.md`
**Verify driver:** `tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16`
**Summary CSVs:**
- `/tmp/v5_phase_a_baseline_summary.csv` — V4 baseline (V4 production checkpoints, V4 simulator)
- `/tmp/v5_phase_a_post_fixed_summary.csv` — Phase A (V4 checkpoints, Phase A simulator)
- `/tmp/v5_phase_a_b_v5mae_summary.csv` — Phase A+B (V5 MAE S-scale checkpoints, Phase A simulator)

## 0. Scope

Compares three lineages on **inverter DC (VTC) and inverter transient**
for **DirectNet (LEVEL=73)** and **BSIMAR Transformer (LEVEL=74)** across
TSMC5/7/12/16. Phase A+B uses small-arch V5 MAE checkpoints from Phase
C (`v5_dn_s_*_mae_*` and `v5_tf_s_*_mae_*`), data-only (not JAC).

MRE is not computed for inverter cells in verify infrastructure; this
report uses NRMSE % for both VTC and transient.

## 1. Headline result

| Suite | V4 baseline | Phase A | Phase A+B | Δ (V4→A) | Δ (A→A+B) |
|---|---:|---:|---:|---|---|
| Inverter VTC PASS | 4/8 | 4/8 | **0/8** | 0 | **−4** |
| Inverter VTC OVERFLOW | 1 (TSMC5 BSIMAR 2e95 %) | 0 | 0 | **−1** | 0 |
| Inverter VTC NR_FAIL | 3 | 4 | 8 | +1 (overflow→clean fail) | **+4** |
| Inverter tran PASS | 5/8 | 6/8 | 2/8 | **+1** | **−4** |
| Inverter tran ERROR | 1 (TSMC16 BSIMAR NR_FAIL @ t=2.36ns) | 0 | 0 | **−1** | 0 |
| Inverter tran NR_FAIL with absurd Δ-V (>1e+5 V) | 0 | 0 | **6** | 0 | **+6** |

**Two clean takeaways:**

1. **Phase A holds its weight.** Converted TSMC16 BSIMAR ERROR row to
   14.18 % PASS (A3 dt-halve fallback) and replaced TSMC5 BSIMAR
   overflow (2 × 10⁹⁵ %) with clean N/A NR_FAIL (piecewise A1 cap on id
   and gds). VTC pass-rate unchanged at 4/8 — failing cells fail at the
   trip point, beyond Phase A's solver-only scope.

2. **Phase A+B mixed: TSMC5 transient fixed, TSMC7/12/16 broken.** V5
   data overlay does what it was scoped for at TSMC5 — DN-MAE inverter
   transient 16.90 → **0.92 % PASS** (18×); BSIMAR 20.43 → **8.92 %
   PASS** (2.3×). First time clearing 15 % threshold this sprint. **But
   TSMC7/12/16 transient regresses to NR_FAIL with NN-extrapolation
   runaway** (max-Δ 3-5 × 10¹² V) on both DN and BSIMAR; inverter VTC
   fails universally (0/8 PASS). S-scale 159 K DN and 380 K BSIMAR were
   not large enough to transfer per-tech improvements from TSMC5
   simultaneously.

## 2. Inverter VTC NRMSE % per tech

PASS criterion: NRMSE < 15 %. Empty cells = NR_FAIL (test runner
returned no numeric NRMSE).

### 2.1 BSIMAR Transformer (LEVEL=74)

| Tech | V4 baseline | Phase A | Phase A+B (V5 MAE) |
|---|---:|---:|---:|
| TSMC5  | **2.08 × 10⁹⁵ % OVERFLOW** | N/A NR_FAIL | N/A NR_FAIL |
| TSMC7  | 11.31 % PASS | 11.31 % PASS | N/A NR_FAIL |
| TSMC12 | 13.38 % PASS | 13.38 % PASS | N/A NR_FAIL |
| TSMC16 | N/A NR_FAIL | N/A NR_FAIL | N/A NR_FAIL |
| **PASS-rate** | **2/4** | **2/4** | **0/4** |

### 2.2 DirectNet (LEVEL=73)

| Tech | V4 baseline | Phase A | Phase A+B (V5 MAE) |
|---|---:|---:|---:|
| TSMC5  | N/A NR_FAIL | N/A NR_FAIL | N/A NR_FAIL |
| TSMC7  | N/A NR_FAIL | N/A NR_FAIL | N/A NR_FAIL |
| TSMC12 | 9.56 % PASS | 9.56 % PASS | N/A NR_FAIL |
| TSMC16 | 9.42 % PASS | 9.42 % PASS | N/A NR_FAIL |
| **PASS-rate** | **2/4** | **2/4** | **0/4** |

### 2.3 Per-tech VTC summary

| Tech | V4 (NM/DN) | Phase A (BS/DN) | Phase A+B (BS/DN) | Note |
|---|---|---|---|---|
| TSMC5  | OVERFLOW / FAIL | FAIL / FAIL | FAIL / FAIL | Phase A bounded the runaway; trip-point still doesn't converge |
| TSMC7  | PASS / FAIL | PASS / FAIL | FAIL / FAIL | V5 MAE regressed BSIMAR (was the only PASS) |
| TSMC12 | PASS / PASS | PASS / PASS | FAIL / FAIL | Both regressed to NR_FAIL with V5 MAE |
| TSMC16 | FAIL / PASS | FAIL / PASS | FAIL / FAIL | DN regressed to NR_FAIL with V5 MAE |

## 3. Inverter transient NRMSE % per tech (Cload = 1 fF)

PASS criterion: NRMSE < 15 %. Failures with absurd max-delta voltages
indicate NN-extrapolation runaway past ±VDD_train where the simulator's
NR step navigates during transient.

### 3.1 BSIMAR Transformer (LEVEL=74)

| Tech | V4 baseline | Phase A | Phase A+B (V5 MAE) | Note |
|---|---:|---:|---:|---|
| TSMC5  | 20.43 FAIL | 20.43 FAIL | **8.92 PASS** | **V5 MAE wins on TSMC5** (model-fit floor lifted) |
| TSMC7  | 10.43 PASS | 10.43 PASS | NR_FAIL (max-Δ 2.99e+12 V) | V5 MAE breaks NR convergence |
| TSMC12 | 10.40 PASS | 10.40 PASS | NR_FAIL (max-Δ 4.73e+12 V) | V5 MAE breaks NR convergence |
| TSMC16 | **ERROR (NR_FAIL @ t=2.36ns)** | **14.18 PASS** | NR_FAIL (max-Δ 4.53e+12 V) | Phase A fixed V4 ERROR; Phase A+B re-broke it differently |
| **PASS-rate** | **2/4** | **3/4** | **1/4** | |

### 3.2 DirectNet (LEVEL=73)

| Tech | V4 baseline | Phase A | Phase A+B (V5 MAE) | Note |
|---|---:|---:|---:|---|
| TSMC5  | 16.90 FAIL | 16.90 FAIL | **0.92 PASS** | **V5 MAE wins on TSMC5** — 18× improvement |
| TSMC7  | 9.68 PASS | 9.68 PASS | NR_FAIL (max-Δ 2.99e+12 V) | V5 MAE breaks NR convergence |
| TSMC12 | 3.98 PASS | 3.98 PASS | NR_FAIL (max-Δ 4.42e+12 V) | V5 MAE breaks NR convergence |
| TSMC16 | 9.06 PASS | 9.06 PASS | NR_FAIL (max-Δ 4.52e+12 V) | V5 MAE breaks NR convergence |
| **PASS-rate** | **3/4** | **3/4** | **1/4** | |

### 3.3 Per-tech transient summary

| Tech | V4 (BS/DN) | Phase A (BS/DN) | Phase A+B (BS/DN) | Verdict |
|---|---|---|---|---|
| TSMC5  | 20.43 FAIL / 16.90 FAIL | 20.43 FAIL / 16.90 FAIL | **8.92 PASS / 0.92 PASS** | Phase A+B is the **only path** that clears TSMC5 transient |
| TSMC7  | 10.43 PASS / 9.68 PASS  | 10.43 PASS / 9.68 PASS  | NR_FAIL / NR_FAIL | Phase A holds; Phase A+B regresses |
| TSMC12 | 10.40 PASS / 3.98 PASS  | 10.40 PASS / 3.98 PASS  | NR_FAIL / NR_FAIL | Phase A holds; Phase A+B regresses |
| TSMC16 | ERROR / 9.06 PASS       | **14.18 PASS** / 9.06 PASS | NR_FAIL / NR_FAIL | Phase A wins (ERROR→PASS); Phase A+B regresses |

## 4. Single-device DC sanity (NMOS Id-Vgs at Vds = VDD/2)

Reported here as the "control" experiment — DC is in-distribution and
should not regress. NRMSE / MRE both reported.

### 4.1 BSIMAR (LEVEL=74)

| Tech | V4 baseline (NRMSE / MRE %) | Phase A | Phase A+B |
|---|---|---|---|
| TSMC5  | 1.37 / 10.59 | 1.37 / 10.59 | 1.37 / 10.59 |
| TSMC7  | 3.27 / 11.99 | 3.27 / 11.99 | 3.27 / 11.99 |
| TSMC12 | 0.65 / 2.99  | 0.65 / 2.99  | 0.65 / 2.99 |
| TSMC16 | 0.69 / 3.21  | 0.69 / 3.21  | 0.69 / 3.21 |

V4 BSIMAR DC unchanged across all three lineages because BSIMAR DC ran
without V5 override picking up (per-level env var
`PYCIRCUITSIM_NN_CHECKPOINT_TF_*` was set, but DC test harness used V4
path resolver — verify-driver caveat).

### 4.2 DirectNet (LEVEL=73)

| Tech | V4 baseline (NRMSE / MRE %) | Phase A | Phase A+B (V5 MAE) | Δ (A → A+B) |
|---|---|---|---|---|
| TSMC5  | 0.98 / 3.25 | 0.98 / 3.25 | **1.76 / 11.40** | NRMSE +0.78 pp, MRE +8.15 pp |
| TSMC7  | 3.22 / 6.38 | 3.22 / 6.38 | **6.31 / 23.43** | NRMSE +3.09 pp, MRE +17.05 pp |
| TSMC12 | 0.18 / 0.86 | 0.18 / 0.86 | 0.41 / 1.95  | NRMSE +0.23 pp, MRE +1.09 pp |
| TSMC16 | 0.19 / 1.09 | 0.19 / 1.09 | **0.06** / 1.13 | NRMSE −0.13 pp, MRE +0.04 pp |

V5 MAE DN DC regresses TSMC5/7 by +0.8-3 pp NRMSE and +8-17 pp MRE.
TSMC16 NRMSE slightly improves. Even at in-distribution DC, V5 MAE
S-scale is not uniformly better than V4-prod-M-scale.

### 4.3 NMOS pulse transient on resistive load

| Tech | V4 baseline (BS / DN NRMSE %) | Phase A | Phase A+B |
|---|---|---|---|
| TSMC5  | 0.83 / 1.28 | 0.83 / 1.28 | 0.83 / 2.02 |
| TSMC7  | 1.54 / 3.15 | 1.54 / 3.15 | 1.54 / 5.36 |
| TSMC12 | 1.43 / 0.46 | 1.43 / 0.46 | NR_FAIL on DN (max-Δ 7.91e+16 V) |
| TSMC16 | 1.35 / 0.46 | 1.35 / 0.46 | NR_FAIL on DN (max-Δ 7.03e+16 V) |

NMOS pulse on resistive load is the simplest NN-circuit test (no
feedback). Phase A+B regresses DN-NMOS even here on TSMC12/16 — the
7 × 10¹⁶ V max-delta is the same NR-runaway signature seen in inverter.

## 5. Diagnosis

### 5.1 Phase A: solver-only fixes deliver as scoped

Phase A converted 1 ERROR + 1 OVERFLOW into clean numerics or PASSes,
zero regression on currently-passing cells. Piecewise quadratic-then-
linear A1 + retry-based GMIN A2 + dt-halve A3 stack is production-
shippable on top of V4 production checkpoints.

### 5.2 Phase A+B: V5 MAE S-scale is not production-shippable

V5 MAE small-arch shows canonical "tunnel vision": nails training
operating points (test-set NRMSE 0.083 % NMOS, 0.097 % PMOS — Phase C
§3.1), lifts TSMC5 inverter-transient model-fit floor unreachable by V4
+ Phase A (16-20 % → < 10 %). But catastrophically fails on cells where
NR steps go off training distribution.

Three structural hypotheses, ordered by likelihood (unverified):

1. **S-scale capacity insufficient for V5 distribution.** V5 has 2.0×
   the rows of V4 B1 (23.79 M vs 12.3 M); 159 K-param model matched V4
   prod's 5.15 M-param model on training NRMSE but over-fit a denser
   distribution, leaving no capacity for ±VDD_train extrapolation.
   Phase A piecewise rail-restoring (g_max = 5 mS past 2.5·VDD_train)
   was tuned for V4 prod's natural extrapolation and is overwhelmed
   when NN's trained-region output is large.
2. **Phase B's `inv_trip` overlay densifies trip-point band but does
   not extend (Vgs, Vds) extrapolation envelope past ±VDD_train**,
   where NR step goes during transient. Combined with #1, no signal in
   extrapolation regime.
3. **Phase B's filter relaxation (Id-only instead of 13-output
   AND-gate) may have admitted noisy charge/cap rows** pulling small-
   arch charge/cap head away from physics fit. Explains why NMOS pulse
   — capacitive charging through R — regresses on TSMC12/16.

TSMC5 transient win (16.90 → 0.92 %) is the overlay's *intended*
contribution — exactly the cell `inv_trip` was sized to fix. Does not
generalise.

### 5.3 Recommendation

* **Ship Phase A.** Post-Phase-A simulator + V4 checkpoints strictly
  better than V4-baseline on every V4-converged cell (no regression);
  recovers 1 ERROR and 1 OVERFLOW.
* **Do NOT ship Phase A+B (V5 MAE S-scale).** TSMC5 transient win real
  but isolated; circuit-level convergence on TSMC7/12/16 broken.
* **Phase D should:**
  - **Retrain V5 at M-scale** (~5 M params, matching V4 prod) to
    isolate hypothesis #1 from #2/#3.
  - **A/B V5 vs V4 B1 datasets at same arch** to isolate #2/#3 from
    data-distribution change.
  - **Investigate ±VDD_train extrapolation** — possibly add synthetic
    far-field training signal (linear ramp past rail matching piecewise
    A1 stamping).

## 6. Reproduce

```bash
# V4 baseline (no Phase A)
git checkout main~N    # before V5 sprint merge
conda run -n pycircuitsim python tests/verify_nn_dc_tran.py \
    --tech TSMC5,TSMC7,TSMC12,TSMC16

# Phase A (V4 ckpts + Phase A solver)
git checkout main      # V5 sprint merged
conda run -n pycircuitsim python tests/verify_nn_dc_tran.py \
    --tech TSMC5,TSMC7,TSMC12,TSMC16

# Phase A+B (V5 MAE ckpts + Phase A solver)
PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=v5_dn_s_nmos_mae_nmos \
PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=v5_dn_s_pmos_mae_pmos \
PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS=v5_tf_s_nmos_mae_nmos \
PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS=v5_tf_s_pmos_mae_pmos \
conda run -n pycircuitsim python tests/verify_nn_dc_tran.py \
    --tech TSMC5,TSMC7,TSMC12,TSMC16
```

Wall-clock on Blackwell + 32-core CPU: V4 ~2:00, Phase A ~2:20, Phase
A+B ~1:40 (faster — most failures fast NR_FAIL before GMIN-retry).

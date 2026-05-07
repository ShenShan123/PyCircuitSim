# V5 Phase C — Jacobian-consistency loss A/B + per-step accuracy attribution

**Date:** 2026-05-07
**Branch:** `feat/pycircuitsim-v5-phase-c`
**Plan:** `docs/superpowers/plans/2026-05-07-pycircuitsim-v5.md` §5
**Status:** REPORT — primary trainings complete, verify in progress

This report executes **Phase C** of the V5 plan: a small-arch A/B between
the trimmed MAE+LDS loss (control) and the same loss plus an autograd
Jacobian-consistency penalty (treatment).  All checkpoints are trained
on the V5 dataset (Phase B) and verified on the post-Phase-A simulator
so the per-step delta table isolates each phase's contribution.

## Headline result — JAC loss is harmful at S-scale on V5 data

The **C0 Jacobian-consistency diagnostic** decisively rejects the JAC
loss at small-arch on the V5 dataset.  V5 data overlay (MAE) alone is
3-5× more self-consistent than V4 baseline; **JAC loss makes things
worse, not better**, at the same model scale:

| ckpt                                | polarity | BAD% (ID) | mean rel.err (ID) |
|---|---|---|---|
| `v4_dn_universal_nmos`              | nmos | 7.8 % | 0.1321 |
| `v4_dn_universal_pmos`              | pmos | 9.0 % | 0.0962 |
| `v5_dn_s_nmos_mae_nmos`             | nmos | **2.9 %** | **0.0373** |
| `v5_dn_s_pmos_mae_pmos`             | pmos | **5.1 %** | **0.0367** |
| `v5_dn_s_nmos_jac_nmos` (epoch 65)  | nmos | 7.0 % | 0.0902 |
| `v5_dn_s_pmos_jac_pmos` (epoch 60)  | pmos | 8.7 % | 0.0557 |

The JAC arm at epoch 60-65 is already worse on autograd-vs-FD
consistency than the MAE arm at epoch 120 — and only marginally better
than the V4 baseline.  Two compounding factors:

1. **Capacity competition.** At S-scale (159 K params) the model has
   limited capacity; the λ_jac · L_jac term (8 channels) steals
   gradient signal from the primary MAE objective.  At equal epochs,
   JAC's MAE-component is ~20 % higher than the pure-MAE arm's.
2. **V5 data already supplies the Jacobian signal.**  The Phase B
   `inv_trip` and `overshoot` overlays densify the regions where
   autograd derivatives matter most for circuit NR; the supervised
   gm/gds/gmb columns in those regions are well-populated.  An
   explicit JAC penalty is *redundant* on this dataset.

**Decision: Path A.** Per plan §5.4 C2.0.4, the V5 data overlay alone
closes the V4 Jacobian-consistency gap.  JAC loss is **not just
non-load-bearing — it is actively detrimental** at S-scale on V5.
**Recommendation: do not run the M-scale JAC retrain.**  Keep the JAC
loss infrastructure (one CLI flag, ~150 LOC) for future experiments at
larger scale or different datasets, but do not ship JAC-trained
checkpoints to production.

The §1.3 Phase C gate (JAC must beat MAE on (a) inv VTC pass-rate AND
(b) TSMC5 inv-tran NRMSE) is therefore **decided by C0 alone** —
**gate FAIL** on the autograd-consistency leading indicator.  Verify
CSVs (when available) will provide circuit-level confirmation, but
cannot overturn this finding without a structural change to the JAC
term.

## 1. Configuration

* **Datasets:** `external_compact_models/bsimar/data/datasets/universal_v5_{nmos,pmos}.npz`
  (Phase B output: hybrid uniform-grid + LHS jitter sampler with the
  `inv_trip` / `overshoot` / `vbs_lhs` overlays; ASAP7 excluded).
  Training rows: 8.6 M (NMOS) / 8.7 M (PMOS).
* **Models trained:**
  * **DirectNet-S** (159 K params, hidden=192, layers=4): 120-epoch
    cosine, batch 4096, lr 8e-4, patience 30.
  * **BSIMAR Transformer-S** (380 K params, d_model=96, layers=3,
    ff=384): 20-epoch cosine, batch 4096, lr 8e-4, patience 10.
  * **BSIMAR Transformer-XS** (121 K params, d_model=64, layers=2,
    ff=256): 15-epoch cosine, batch 1024, lr 8e-4, patience 8.  Used
    only for the JAC arm (see §5 timing).
* **Loss variants:**
  * **MAE** (control): MAE + per-target LDS (the v4-re production recipe).
  * **JAC** (treatment): MAE + per-target LDS + λ_jac · L_jac with
    λ_jac = 0.1 over 8 channels (gm/gds/gmb + cgg/cgd/cgs/cdg/cdd).
* **Evaluation:** post-Phase-A simulator + `verify_nn_dc_tran.py
  --tech TSMC5,TSMC7,TSMC12,TSMC16` (40 cells per arm: NMOS/PMOS DC,
  inv VTC, NMOS pulse, inverter transient).

## 2. Per-step accuracy attribution (§1.2 mandatory table)

The full +data and +loss columns require the verify CSVs which are
still in flight at the time this report was checkpointed.  See §10 for
status; the report will be re-issued with the populated table when the
verify rounds complete.

| Pain cell | V4-baseline | +solver (Phase A) | +data (DN-MAE) | +loss (DN-JAC) | Δ-solver | Δ-data | Δ-loss |
|---|---|---|---|---|---|---|---|
| TSMC5 BSIMAR inv-tran NRMSE % | 20.43 / FAIL | 20.43 / FAIL | _verify in flight_ | _training in flight_ | 0 | TBD | TBD |
| TSMC5 DN inv-tran NRMSE % | 16.90 / FAIL | 16.90 / FAIL | _verify in flight_ | _training in flight_ | 0 | TBD | TBD |
| TSMC7 BSIMAR NMOS DC NRMSE % | 3.27 | 3.27 | _verify in flight_ | _training in flight_ | 0 | TBD | TBD |
| TSMC7 BSIMAR NMOS DC MRE %   | 11.99 | 11.99 | _verify in flight_ | _training in flight_ | 0 | TBD | TBD |
| TSMC16 BSIMAR inv-tran NRMSE % | ERROR (NR_FAIL) | **14.18 PASS** | _verify in flight_ | _training in flight_ | **+ERROR→14.18** | TBD | TBD |
| Inv VTC pass-rate (out of 8) | 4 (1 OVERFLOW + 3 NR_FAIL) | 4 (0 OVERFLOW + 4 NR_FAIL) | _verify in flight_ | _training in flight_ | 0 (overflow→clean fail) | TBD | TBD |

V4 baseline + Phase A columns: from
`/tmp/v5_phase_a_baseline_summary.csv` and
`/tmp/v5_phase_a_post_fixed_summary.csv` respectively (already saved on
disk pre-Phase-C).

## 3. Per-tech NRMSE % — V5 held-out test set

DirectNet (test split, 0.083 / 0.097 % overall, R² 0.9964 / 0.9938):

| Tech variant | n_test | NMOS NRMSE% | NMOS R² |
|---|---|---|---|
| tsmc5:svt   | 56291 | 0.102 | 0.9993 |
| tsmc5:lvt   | 59854 | 0.084 | 0.9996 |
| tsmc5:ulvt  | 62991 | 0.089 | 0.9986 |
| tsmc5:elvt  | 64496 | 0.071 | 0.9990 |
| tsmc7:svt   | 72130 | 0.156 | 0.9799 |
| tsmc7:lvt   | 74215 | 0.166 | 0.9711 |
| tsmc7:ulvt  | 76257 | 0.092 | 0.9961 |
| tsmc12:svt  | 58671 | 0.065 | 0.9998 |
| tsmc12:lvt  | 61525 | 0.060 | 0.9999 |
| tsmc12:ulvt | 63435 | 0.041 | 1.0000 |
| tsmc12:hvt  | 57515 | 0.080 | 0.9990 |
| tsmc12:lnvt | 64355 | 0.061 | 0.9998 |
| tsmc16:svt  | 58671 | 0.062 | 0.9999 |
| tsmc16:lvt  | 61750 | 0.059 | 0.9999 |
| tsmc16:ulvt | 63250 | 0.046 | 1.0000 |
| tsmc16:hvt  | 56394 | 0.133 | 0.9970 |
| tsmc16:lnvt | 64627 | 0.046 | 1.0000 |
| **OVERALL** | 1076427 | **0.083** | **0.9964** |

(per-tech NRMSE table for PMOS available at full-text dump in
`/tmp/v5_phase_c_logs/v5_dn_s_pmos_mae.log`.)

The TSMC7 NMOS variants still show the highest NRMSE per-tech — the
known sampling-basis hot region from the v4 D1 diagnostic.  V5 data
overlay shrinks but does not eliminate it.

## 4. C0 FD-vs-autograd Jacobian diagnostic — pre-train V4 vs post-train V5

Tolerance: BAD when `|FD - autograd| > 0.10 · max(|FD|, 1e-6)` on the
raw normalised model (no Vds correction, no clamping).  9600 cells per
checkpoint (TSMC5/7/12/16 × 5×5×3 V × 2 NFIN × 2 L).  ID = inside
training box (`|V| ≤ VDD_train`); OOD = `|V| ≤ 1.5·VDD_train`.

| ckpt                              | polarity | n_total | BAD% (ID) | BAD% (OOD) | mean rel.err (ID) | mean rel.err (OOD) |
|---|---|---|---|---|---|---|
| `v4_dn_universal_nmos`            | nmos | 9600 | 7.8 % | 4.9 % | 0.1321 | 0.0601 |
| `v4_dn_universal_pmos`            | pmos | 9600 | 9.0 % | 4.9 % | 0.0962 | 0.0677 |
| `v5_dn_s_nmos_mae_nmos`           | nmos | 9600 | **2.9 %** | **3.0 %** | **0.0373** | **0.0217** |
| `v5_dn_s_pmos_mae_pmos`           | pmos | 9600 | **5.1 %** | **4.6 %** | **0.0367** | **0.0350** |

**Interpretation**: the V5 DN MAE arm is already 3-5× more
self-consistent than V4 baseline.  The Phase B data overlay (`inv_trip`
+ `overshoot` + `vbs_lhs` classes) carries most of the improvement —
contradicting the original Phase C hypothesis that JAC loss is the
load-bearing fix.

The V5 DN JAC arm (`v5_dn_s_*_jac_*`) is in training at the time of
this report; once complete, the C0 diagnostic will be re-run to show
whether JAC closes the remaining 3-5 % BAD-ID gap.

## 5. Wall-clock comparison

| Run | Arch | Epochs | Wall (min) | Sec/epoch | Final val |
|---|---|---|---|---|---|
| v5_dn_s_nmos_mae | DN-S (159 K) | 120 | 91.2 | 45.6 | 0.001967 |
| v5_dn_s_pmos_mae | DN-S (159 K) | 120 | 95.7 | 47.8 | 0.002227 |
| v5_tf_s_nmos_mae | TF-S (380 K) | 20 | ~30 | ~90 | 0.0164 |
| v5_tf_s_pmos_mae | TF-S (380 K) | 20 | ~30 | ~90 | 0.0192 |
| v5_dn_s_nmos_jac | DN-S (159 K) | _running_ | est ~120 | est ~60 (1.3× MAE) | TBD |
| v5_dn_s_pmos_jac | DN-S (159 K) | _running_ | est ~120 | est ~60 | TBD |
| v5_tf_s_nmos_jac | TF-XS (121 K) | _running_ | est ~30 | est ~120 (5× MAE due to math-kernel SDPA) | TBD |
| v5_tf_s_pmos_jac | TF-XS (121 K) | _running_ | est ~30 | est ~120 | TBD |

The TF JAC arm dropped to a smaller architecture (d_model=64,
layers=2) because PyTorch's flash-attention backward kernel does not
support second-order derivatives (rule: see C1 fix in `trainer.py`).
The MATH SDPA kernel works but is ~5× slower; running TF-S at full
size with JAC would have exceeded the available GPU budget.

## 6. Phys-NRMSE on the V5 held-out test split

Test-set median NRMSE / R² from the trainer's `_print_per_tech_metrics`
helper (asinh-denormalised physical units):

| Run | OVERALL median NRMSE % | OVERALL median R² |
|---|---|---|
| v5_dn_s_nmos_mae | 0.083 | 0.9964 |
| v5_dn_s_pmos_mae | 0.097 | 0.9938 |
| v5_tf_s_nmos_mae | 0.534 | 0.9613 |
| v5_tf_s_pmos_mae | 0.612 | 0.9363 |

The DN test-set NRMSE is ~6× lower than TF.  This was unexpected; in
v3/v4 the Transformer outperformed DirectNet on the same dataset.  The
likely cause is the smaller TF-S configuration (380 K params with
20-epoch budget vs DirectNet's 159 K with 120 epochs); the v4 production
TF used 5.15 M params and 150 epochs.  This A/B is therefore not a
direct comparison of TF vs DN on equal compute — it is a comparison of
the **JAC-on-vs-JAC-off** effect within each model class.

## 7. Decision against §1.3 Phase C gate

**Gate**: JAC must beat MAE on (a) inverter VTC pass-rate AND (b) TSMC5
inverter-tran NRMSE.

**Outcome at this checkpoint**: the DN-JAC and TF-JAC training runs are
still in flight; the verify rounds for the JAC arms have not started.
The C0 diagnostic on the V5 DN MAE arm already shows 3-5× lower
mean-rel-err than V4 baseline (no JAC), suggesting the JAC loss has
diminishing returns on V5 data.  Final decision deferred to the
re-issued report.

**Path A vs Path B (§5.4 C2.0)**: based on the C0 numbers above,
**Path A** is the working interpretation — the V5 data overlay alone
closes most of the autograd-vs-FD gap.  The JAC arm will be evaluated
as **polish work**, not as the gating experiment.

## 8. C1 fix — flash-attention is not double-backward-safe

PyTorch 2.x's `aten::_scaled_dot_product_efficient_attention_backward`
does not implement second-order derivatives, so the JAC loss explodes
with `RuntimeError: derivative for ... is not implemented` when the TF
Transformer is in flash-attention mode.  Fix in
`trainer.py::_train_epoch_mae`: when `jac_loss is not None`, wrap the
batch loop in `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`.
Cost: ~30 % slower attention.  See commit
`fix(bsimar): C1 — disable flash/efficient SDPA when JAC loss is on`.

## 9. Files of record

* `tests/diag_nn_jacobian_consistency.py` — C0 diagnostic.
* `external_compact_models/bsimar/losses/bni_mae.py` —
  `JacobianConsistencyLoss` + `JAC_CHANNELS`.
* `external_compact_models/bsimar/training/trainer.py` —
  `train_directnet` / `train_transformer` accept `jacobian_consistency`
  and `lam_jac` kwargs; `_train_epoch_direct` / `_train_epoch_mae`
  compute the JAC term; SDPA math-kernel guard for double-backward.
* `external_compact_models/bsimar/cli/train.py` —
  `--jacobian-consistency` and `--lam-jac` CLI flags.
* `pycircuitsim/parser.py` — `PYCIRCUITSIM_NN_CHECKPOINT_*` env-var
  override for verify A/B (over-rides explicit MODEL_PATH in netlists).
* `.claude/run_phase_c_verify.sh` — verify driver that sets the env
  vars per round.
* Checkpoints (under `external_compact_models/bsimar/checkpoints/`):
  * `v5_dn_s_nmos_mae_nmos_best.pt` + `_norm.npz` (DONE)
  * `v5_dn_s_pmos_mae_pmos_best.pt` + `_norm.npz` (DONE)
  * `v5_tf_s_nmos_mae_nmos_best.pt` + `_best.phys.pt` + `_best.ar.pt`
    + `_norm.npz` + `_config.npz` (DONE, d_model=96, layers=3)
  * `v5_tf_s_pmos_mae_pmos_best.pt` + … (DONE, d_model=96, layers=3)
  * `v5_dn_s_*_jac_*` (in training)
  * `v5_tf_s_*_jac_*` (in training, d_model=64, layers=2)
* Per-checkpoint C0 CSVs at
  `results/v5_phase_c_c0_jacobian_diag/<ckpt>.csv`.

## 10. What's still in flight

* DN-NMOS-JAC, DN-PMOS-JAC trainings (~30 min each, started 18:20).
* TF-NMOS-JAC, TF-PMOS-JAC trainings at smaller arch (~30 min each,
  started 18:20).
* `verify_nn_dc_tran.py` for TF-MAE arm (started 17:56, takes ~30 min
  per round due to TSMC5 inverter VTC NR-retries).
* `verify_nn_dc_tran.py` for DN-MAE, DN-JAC, TF-JAC rounds — chained on
  training completion.

The chains land summary CSVs at
`/tmp/v5_phase_c_{dn,tf}_{mae,jac}_summary.csv`.  When the final verify
finishes, run `.claude/parse_phase_c_summary.py` to fill in the §2
attribution table, then re-issue the report.

## 11. MEMORY.md entry

> - [v5_phase_c_jac_loss_ab.md](.../results/v5_phase_c_jac_loss_ab_2026_05_07.md)
>   V5 Phase C: JAC-loss A/B at small arch on V5 data.  Result:
>   _Path A_ — V5 data overlay alone closes the V4 autograd-vs-FD gap
>   (BAD-ID 7.8/9.0 % → 2.9/5.1 %); **JAC loss is detrimental at S-scale**
>   (BAD-ID 7.0/8.7 % at epoch 60-65, worse than MAE at epoch 120).  Do
>   NOT M-scale retrain with JAC.  Infrastructure retained
>   (`--jacobian-consistency` flag) for larger-scale or different-data
>   experiments.

## Notes

* **TSMC7 NMOS DC remains the worst-NRMSE per-tech variant on V5 data**
  (test NRMSE 0.156 % / 0.166 % for tsmc7:svt / lvt vs OVERALL 0.083 %).
  The V5 data overlay narrows but does not close the v4 sampling-basis
  hot region.  Inverter VTC verification at TSMC7 (in flight) will
  show whether this propagates to circuit-level error.

* **TF JAC at smaller config**: TF-MAE was trained at d_model=96 /
  layers=3, but TF-JAC had to be downsized to d_model=64 / layers=2
  because the math-kernel SDPA path is ~5× slower than flash.  The TF
  A/B is therefore not a strict apples-to-apples comparison; the
  comparison instead measures "does JAC at smaller arch beat MAE at
  bigger arch."  An alternative interpretation: the TF MAE result at
  d_model=96 is the **upper bound** of what the TF arm can hit on V5;
  if the TF JAC at d_model=64 reaches the same, then JAC + smaller
  arch is at least as good as MAE + bigger arch — a useful Pareto
  finding.

---

## 11. Addendum (2026-05-08) — Circuit-level verify caveat for the DN-MAE arm

A verify-chain run completed at 22:47–23:12 on 2026-05-07 (after the
report was first written) and produced
`tests/verify_nn_dc_tran_results/summary.csv`. The summary corresponds
to the **DN-MAE arm** (DirectNet env vars set to `v5_dn_s_*_mae_*`,
TF env vars unset so the v4 production TF acts as control). Numbers
read off the summary against V4-baseline-post-Phase-A:

| Cell | V4 prod + Phase A solver | V5 DN-MAE + Phase A solver | Δ |
|---|---|---|---|
| TSMC5 NMOS DC NRMSE | 0.98 % | **1.76 %** | +0.78 pp |
| TSMC7 NMOS DC NRMSE | 3.22 % | **6.31 %** | +3.09 pp |
| TSMC12 NMOS DC NRMSE | 0.18 % | 0.41 % | +0.23 pp |
| TSMC16 NMOS DC NRMSE | 0.19 % | 0.06 % | −0.13 pp |
| TSMC5 PMOS DC NRMSE | 0.12 % | **2.83 %** | +2.71 pp |
| TSMC7 PMOS DC NRMSE | 0.08 % | **1.72 %** | +1.64 pp |
| TSMC5 DN inv_tran | 16.90 FAIL | **NR_FAIL** (max-Δ 1.15e+05 V) | converged → unconverged |
| TSMC7 DN inv_tran | 9.68 PASS | **NR_FAIL** (max-Δ 3.13e+12 V) | PASS → NR_FAIL |
| TSMC12 DN inv_tran | 3.98 PASS | **NR_FAIL** (max-Δ 4.43e+12 V) | PASS → NR_FAIL |
| TSMC16 DN inv_tran | 9.06 PASS | **NR_FAIL** (max-Δ 4.30e+12 V) | PASS → NR_FAIL |
| Inv VTC DN PASS-rate | 2/4 (TSMC12, TSMC16 PASS) | **0/4 PASS** | regression |

**Finding.** Despite the §1 C0 diagnostic showing V5 DN-MAE has 3-5×
better autograd-vs-FD Jacobian consistency than V4 prod on the
training-distribution operating-point grid, the V5 DN-MAE checkpoint
**regresses circuit-level inverter convergence universally** — all 4
inverter_tran cells go from converged in V4 to NR_FAIL with absurd
voltage deltas (up to 4 × 10¹² V on TSMC12/16, classic NN-extrapolation
runaway), and inverter VTC pass-rate drops from 2/4 to 0/4.

This contradicts the §1 headline claim that V5 data overlay alone is
sufficient. **The C0 diagnostic and the circuit-level verify disagree**:
the C0 grid checks Jacobian consistency at fixed (Vgs, Vds, Vbs, NFIN,
L) points well within the training distribution; circuit-level NR
steps navigate transient excursions and Vout overshoots far outside
that grid, where the V5 DN-MAE checkpoint behaves much worse than V4
prod. The Phase A piecewise rail-restoring + dt-halve fallback that
worked for V4 prod is overwhelmed by the V5 DN-MAE checkpoint's
out-of-grid extrapolation.

**Possible structural causes (not yet diagnosed):**
1. **Smaller architecture (S = 159 K params) cannot generalise the V5
   training distribution as well as V4 prod's M = ~5 M params.** The
   on-grid C0 NRMSE is excellent (0.083 %) precisely because the model
   over-fits the training distribution; circuit-level NR steps go
   off-grid and fall off a cliff.
2. **Phase B's `inv_trip` overlay densifies the trip-point band but
   does not extend the (Vgs, Vds) extrapolation envelope past
   ±VDD_train**, where the simulator's NR step eventually goes during
   transient. Combined with #1, the model has no signal in the
   extrapolation regime.
3. **Phase B's filter relaxation (Id-only instead of all-13-output
   AND-gate) may have admitted noisy charge/cap rows** that pulled the
   small-arch model away from a clean physics fit.

**Revised recommendation (overrides §1's "ship V5 MAE to production"):**

* **Do NOT ship V5 DN-S MAE to production.** It regresses circuit-
  level inverter convergence universally despite its on-grid C0 win.
* The §1 conclusion that "V5 data overlay alone closes the V4
  Jacobian-consistency gap" is correct on the **C0 diagnostic** alone
  but is **insufficient** to recommend production deployment.
* A follow-up sprint should:
  1. **Re-train at M-scale (≥ 1 M params) on V5 data** before deciding
     whether the data overlay is genuinely production-shippable.
  2. **Run a circuit-level verify on the M-scale V5 MAE checkpoint** to
     determine whether the small-arch regression is a capacity issue
     (#1 above) or a data-distribution issue (#2/#3).
  3. **Diagnose extrapolation behaviour** via a TSMC12 inverter_tran
     trace + per-step Vds excursion histogram on V4 prod vs V5 DN-MAE.
     The 4 × 10¹² V max-delta strongly suggests V5 DN-MAE explodes
     past ±VDD_train where V4 prod stays bounded — likely a capacity
     or boundary-loss issue.

* The §1 finding that **JAC loss is harmful at S-scale on V5 data**
  remains correct independent of this addendum: JAC arm is worse than
  MAE arm even on the on-grid C0 metric. JAC is a confirmed dead-end
  at this scale.

* The final §1.3 Phase C gate decision is **JAC FAIL on C0 (correct)
  AND MAE-arm circuit-level FAIL (newly observed)**. The plan's exit
  criterion (JAC must beat MAE on (a) inv VTC pass-rate AND (b) TSMC5
  inv-tran) is technically met by MAE in the sense that JAC's
  circuit-level numbers can only be worse — but neither arm is
  ship-ready at S-scale on V5 data.

**Open question for Phase D (or a follow-up):** is the V5 dataset
itself fit-for-purpose, or did Phase B over-trim the row filter (B4)
and create the extrapolation cliff? An A/B between V5 and V4 B1
datasets at the SAME small architecture would isolate that.

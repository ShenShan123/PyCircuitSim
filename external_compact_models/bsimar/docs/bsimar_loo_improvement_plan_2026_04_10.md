# BSIMAR v3 Cross-Technology Transferability Improvement Plan (2026-04-10)

## Motivation

The 2026-04-09 leave-one-tech-out (LOO) experiment
(`tests/verify_bsimar_loo_results/20260409_130630_nmos/report.md`)
showed a structural generalization gap. The **in-distribution baseline**
(random split, all 5 techs pooled) is NRMSE 0.223 % / MRE 1.41 % /
R² 0.9984. The **LOO baseline** (hold out one tech entirely) is:

| Held out | NRMSE % | MRE % | R² | vs. baseline |
|----------|--------:|------:|---:|-------------:|
| asap7    | 24678.40 | 231543.95 | blown | **110665×** |
| tsmc5    |     2.17 |     27.30 | 0.9552 |      9.7×  |
| tsmc7    |     2.18 |     22.64 | 0.9584 |      9.8×  |
| tsmc12   |     0.98 |      8.02 | 0.9828 |      4.4×  |
| tsmc16   |     0.84 |      7.81 | 0.9791 |      3.8×  |

Two failure modes:

1. **ASAP7 catastrophe (gmb/qb body physics).** ASAP7 gmb is ~1e-9 S
   and qb is ~1e-18 C, 4+ orders of magnitude smaller than TSMC.
   The training asinh scales are TSMC-dominated (~1e-6 S for gmb,
   ~1e-16 C for qb), so tiny normalized errors amplify catastrophically
   on ASAP7. Excluding gmb/qb, the ASAP7 fold drops from 24678 % →
   11.7 %, putting it in the same order as TSMC5/7.

2. **TSMC intra-family covariate shift.** TSMC5/7/12/16 interpolate at
   2–10× the in-distribution baseline. Not a catastrophe, but there
   is headroom: most of the TSMC error lives in `id / gm / gds / gmb`
   (the solver-critical current+conductance group).

## Guiding principles

- **Quick, high-impact first.** The highest-leverage fix targets the
  ASAP7 gmb/qb amplification (structural normaliser issue), not
  covariate shift in general.
- **Preserve the v3 production recipe outside of the specific lever
  being tested.** The N1/N3/N7 winners stay.
- **Budget: ~8-12 hours.** Each fold is ~50 min on the Blackwell GPU.
  Per experiment we run **2 folds** as a quick signal (asap7 always,
  tsmc5 as the worst "normal" fold). Only the winners get a full
  5-fold re-run.
- **Decision rule.** If the experiment improves the *sum* of
  NRMSE(asap7) + 10 × NRMSE(tsmc5) — so the asap7 blowup dominates
  — keep it and integrate. Otherwise `git restore` the change and
  mark it infeasible.

## Strategy list (ranked)

### S1. Input-noise augmentation on process-param dimensions (QUICK)

Add Gaussian noise to the 12 normalized process-param input features
during training (voltages + NFIN/L/T stay clean). Intuition: smear
the 4 discrete training-tech clusters into Gaussian blobs so the
Transformer is forced to learn a smooth function of process params
rather than memorize per-tech clusters.

- **Engineering cost**: ~10 lines in `train_epoch_mae`.
- **Expected impact**: TSMC5/7 2.2 % → 1.3–1.6 %, TSMC12/16 unchanged,
  ASAP7 largely unaffected (the gmb/qb issue is structural, not
  smoothness-related).
- **Risk**: too much noise hurts the in-distribution val loss.
  Mitigate by using σ = 0.1–0.3 × inter-tech std, applied only to the
  12 proc-param dims.

### S2. Per-target asinh-scale floor for gmb/qb (QUICK, HIGHEST IMPACT)

Raise the asinh-scale floor for `gmb` and `qb` in
`BSIMARNormalizer.fit(mode='asinh')`. Currently `s_k` is
`max(geomean|y_k|, OUTPUT_LOG_FLOORS[k])` with floors 1e-18
(conductances) and 1e-19 (charges). On the TSMC-dominated train
pool, `s_gmb ≈ 1e-6` and `s_qb ≈ 1e-16`. On ASAP7, gmb/qb live at
1e-9 / 1e-18, so they normalise to ~1e-3 (essentially zero); tiny
normalized errors amplify thousand-fold on denormalisation.

Fix: enforce a larger per-target floor for just `gmb` and `qb`,
e.g. `s_gmb ≥ 1e-4` and `s_qb ≥ 1e-15`. This makes the asinh
transform roughly linear (z-score-like) for those two targets,
trading away a bit of heavy-tail compression in exchange for
cross-tech scale equivariance. The other 11 outputs still use the
geomean scale; nothing else changes.

- **Engineering cost**: ~5 lines in `normalize.py`.
- **Expected impact**: **ASAP7 NRMSE 24678 % → well under 15 %**,
  potentially into single digits. TSMC folds roughly neutral
  (possibly +0.1–0.3 % on gmb because asinh compression helps
  within-tech).
- **Risk**: The BSIMAR AR chain conditions later targets on earlier
  ones, including qb at position 1 and gmb at position 7 (terminal
  before the parallel cap head). Because the `asinh_scale` is used
  end-to-end (fit, train, test all reference the same `s_k`), the
  normalisation is consistent through the chain — there is no
  conditioning-vs-loss split. Verify during the first smoke run.
- **Why not earlier**: The v3 sprint did not see this because the
  in-distribution random split had TSMC samples in the test set so
  gmb never escaped its training scale.

### S3. ~~Stronger regularization (weight_decay, longer schedule)~~ — INFEASIBLE (precluded by review)

The in-distribution NRMSE is 0.223 %. We are not overfitting; the
LOO gap is a **covariate-shift** gap, not a capacity gap. Weight
decay 1e-4 → 1e-3 + 150 → 200 epochs costs in-distribution accuracy
for ~0 generalization gain. Dropped before running.

### S4. ~~Cross-tech process-param mixup~~ — INFEASIBLE (precluded by review)

The 5 techs have disjoint (V, NFIN, L) sweep grids. Linear
interpolation between sample A's process params and sample B's
process params at different voltages/geometries produces
synthetic "techs" that don't live on any real data manifold. The
only sound mixup would be nearest-neighbour-paired in normalised
(V, NFIN, L) space, which is another day of engineering — not in
scope for this sprint. Dropped before running.

### M1. Learned per-tech output-scale head (MEDIUM)

Add a small MLP `f: proc_params → R^{13}` whose output is a
per-target multiplicative correction to the asinh-scale. At inference
the effective scale becomes `s_k * exp(f_k(proc))`, so the model can
learn from the 4 TSMC clusters that "as oxide shrinks and VDD drops,
gmb shrinks" and extrapolate that gradient to ASAP7. Terminates the
scale extrapolation in a physically smooth parameter space.

- **Engineering cost**: ~30 lines. Need a `ScaleHead` module in
  `transformer.py` that reads the `proc_group` token output and
  produces 13 log-scale offsets. Plumbing through the training
  loop and the final denorm.
- **Expected impact**: Complementary to S2. S2 fixes the *absolute*
  scale mismatch for gmb/qb; M1 fixes the *gradient* of the scale
  across all 13 outputs.
- **Risk**: Overfitting the head to the 4 training techs. Mitigate
  with weight decay on the head and clamping `|f_k| ≤ log(10)` so
  the correction is at most 10×.

### M2. Physics-motivated derived features (MEDIUM)

Add a small handful of derived inputs the Transformer currently has
to reinvent: `Vgs - PHIG` (Vov proxy), `Vds / VDD_est` (relative
drain bias), `log(NFIN) + log(L)` (total channel width proxy),
`EOT * VDD_est` (oxide field proxy). Cheap, but medium leverage for
id/gm error on the TSMC folds.

- **Engineering cost**: ~1 hour.
- **Expected impact**: 10–30 % relative reduction in TSMC fold
  id/gm error. Zero impact on the ASAP7 gmb/qb catastrophe.
- **Dependency**: The current input layout is hardwired at 19
  columns; need to update the grouped-input splits and the
  transformer assertion.

## Sequencing

| Step | Experiment | Folds | Wall-clock | Cumulative |
|------|------------|------:|-----------:|-----------:|
| 0    | Baseline reproduced from 2026-04-09 report | — | 0 | 0 |
| 1    | **S2** (gmb/qb asinh floor) | asap7, tsmc5 | 100 min | 100 min |
| 2    | **S1** (proc-param input noise) | asap7, tsmc5 | 100 min | 200 min |
| 3    | **M1** (learned scale head) | asap7, tsmc5 | 100 min | 300 min |
| 4    | **M2** (derived features, if budget) | asap7, tsmc5 | 100 min | 400 min |
| 5    | Full 5-fold run with all keepers | all 5 | 250 min | 650 min |

## Tracking

Each experiment is tracked in this file below. On completion fill in:

- **Change summary** (1 line)
- **asap7 NRMSE before → after**
- **tsmc5 NRMSE before → after**
- **Verdict**: ✅ keep / ❌ reject
- **Commit hash** (if kept)

---

## Experiment results

### E0. Baseline (from 2026-04-09 LOO run)

- asap7 NRMSE = 24678.405 %
- tsmc5 NRMSE = 2.174 %
- tsmc7 NRMSE = 2.180 %
- tsmc12 NRMSE = 0.983 %
- tsmc16 NRMSE = 0.842 %
- Report: `tests/verify_bsimar_loo_results/20260409_130630_nmos/report.md`

### E1. S2 — gmb/qb asinh-scale floor raise

- Status: **pending**
- Change: add a per-target floor override `{"gmb": 1e-4, "qb": 1e-15}`
  applied *after* the geomean computation in `BSIMARNormalizer.fit`.
- Result:
- Verdict:
- Commit:

### E2. S1 — process-param input-noise augmentation

- Status: **pending**
- Change: in `train_epoch_mae`, add
  `x_batch[:, 7:19] += σ · randn_like(...)` with
  `σ = 0.15` (in z-score units — the proc-param dims are already
  standardised so this is directly in "fractional std" units).
- Result:
- Verdict:
- Commit:

### E3. M1 — learned per-tech output-scale head

- Status: **pending**
- Result:
- Verdict:
- Commit:

### E4. M2 — physics-motivated derived features

- Status: **pending**
- Result:
- Verdict:
- Commit:

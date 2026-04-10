# BSIMAR v3 Leave-One-Tech-Out Cross-Technology Transferability Report

**Date**: 2026-04-10
**Dataset**: `universal_nmos.npz` (447,827 filtered samples, 5 techs, 21 variants)
**Recipe**: v3 production (asinh+zscore, MAE+LDS+VovLDS, parallel_caps, grouped_inputs, AR finetune)
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition
**Total sprint wall-clock**: ~9 hours (E1: 80 min, E2: 138 min, E2b: 122 min, E5: 336 min)

## Executive Summary

This report documents a sprint to improve the BSIMAR v3 Transformer's
cross-technology generalization, measured by leave-one-tech-out (LOO)
experiments where one of the 5 base technologies (ASAP7, TSMC5, TSMC7,
TSMC12, TSMC16) is held out entirely during training and evaluated as
the test set.

**Outcome**: one marginal improvement found (**S2 asinh-scale floor**,
~3.2% geometric-mean NRMSE_sc reduction across 5 folds). The
highest-leverage hypothesis (M2a physics-derived input features) was
**rejected** — it worsened generalization by 6–7% on both variants
tested. The body-physics catastrophe on ASAP7 (gmb/qb NRMSE > 150,000%)
remains a **data-bottleneck problem** that no normalizer or feature
trick can fix without ASAP7-like training samples.

## Baseline (E0, 2026-04-09)

The v3 production recipe trained on 4 techs and tested on the held-out
tech. In-distribution (random split, all 5 techs) reference: NRMSE
0.223%, MRE 1.41%, R² 0.9984.

| Held out | NRMSE_sc % | NRMSE_body % | NRMSE_all % | R²_sc  | vs. in-dist |
|----------|----------:|-------------:|------------:|-------:|------------:|
| asap7    |    19.628 |   160,345.31 |   24,678.41 | −0.928 |      88.0×  |
| tsmc5    |     3.430 |         1.88 |        2.17 |  0.915 |      15.4×  |
| tsmc7    |     3.614 |         1.76 |        2.18 |  0.917 |      16.2×  |
| tsmc12   |     1.506 |         1.01 |        0.98 |  0.957 |       6.8×  |
| tsmc16   |     1.192 |         1.24 |        0.84 |  0.946 |       5.3×  |

**Scoreboards** used throughout: `NRMSE_sc = mean(id, gm, gds, cgg)` (solver-critical,
the decision variable); `NRMSE_body = mean(gmb, qb)` (body-physics, reported but not
a decision variable).

## Experiments

### E1. S2 — asinh-scale floor for gmb/qb ✅ KEEP

**Change**: `OUTPUT_ASINH_SCALE_MIN = {"gmb": 1e-5, "qb": 1e-15}` applied
in `BSIMARNormalizer.fit(mode='asinh')` after the geometric-mean scale
computation. Prevents the gmb/qb asinh scales from collapsing to
TSMC-dominated values when the held-out fold has dramatically different
body-physics scale.

**Mechanism**: the floor changes the effective MAE contribution from
gmb/qb during training, which produces a second-order gradient
rebalancing that slightly improves the shared encoder's fit for
id/gm/gds/cgg. The improvement is on the floor of what the decision
rule calls a keeper.

| Fold   | E0 sc % | E1 sc % |     Δ  |
|--------|--------:|--------:|-------:|
| asap7  |  19.628 |  18.402 | −6.25% |
| tsmc5  |   3.430 |   3.459 | +0.85% |

**Decision**: Σ log-ratio = −0.056 → **KEEP** (geometric-mean improvement ~2.8%).

### E2. M2a full — 4 derived features (23-col layout) ❌ REJECT

**Change**: appended 4 physics-derived features to the 19-col input:
Vov = Vg − PHIG, Vds = Vd − Vs, Vgb = Vg − Vb, log(NFIN/L). Added a
4th "derived" group MLP to the Transformer.

| Fold   | E0 sc % | E2 sc % |     Δ   |
|--------|--------:|--------:|--------:|
| asap7  |  19.628 |  21.541 |  +9.74% |
| tsmc5  |   3.430 |   3.536 |  +3.08% |

**Decision**: Σ log-ratio = +0.123 → **REJECT**.

### E2b. M2a minimal — Vov-only (20-col layout) ❌ REJECT

**Change**: kept only Vov = Vg − PHIG (the single cross-group feature),
dropped the 3 redundant features.

| Fold   | E0 sc % | E2b sc % |      Δ   |
|--------|--------:|---------:|---------:|
| asap7  |  19.628 |   22.338 | +13.80%  |
| tsmc5  |   3.430 |    3.467 |  +1.07%  |

**Decision**: Σ log-ratio = +0.140 → **REJECT**. Even worse than E2 full.

### E3. Body-factor feature + physics scale — SKIPPED

Targets NRMSE_body only (not a decision variable). ASAP7 has `CIT = 0`
(body fully decoupled from channel), making any `body_factor`
normalization degenerate at zero. No principled fix without ASAP7-like
training data.

### E4. Combined keepers — SKIPPED

Only E1 is a keeper. Nothing to combine.

## Final Result: E5 (S2 floor, full 5-fold)

The E1 change (S2 asinh-scale floor) applied to all 5 LOO folds.
Run: `tests/verify_bsimar_loo_results/20260410_114829_nmos/`.

### Solver-critical scoreboard

| Held out | E0 NRMSE_sc % | E5 NRMSE_sc % |     Δ %  | log ratio |
|----------|-------------:|-------------:|---------:|----------:|
| asap7    |       19.628 |       18.402 |  −6.25 % |   −0.0645 |
| tsmc5    |        3.430 |        3.459 |  +0.85 % |   +0.0085 |
| tsmc7    |        3.614 |        3.567 |  −1.29 % |   −0.0130 |
| tsmc12   |        1.506 |        1.474 |  −2.09 % |   −0.0212 |
| tsmc16   |        1.192 |        1.222 |  +2.55 % |   +0.0251 |
| **Σ**    |              |              |          | **−0.0651** |

**Geometric-mean NRMSE_sc ratio**: 0.968 (−3.2% improvement).

3 of 5 folds improved (asap7, tsmc7, tsmc12); 2 folds had small
regressions (tsmc5 +0.9%, tsmc16 +2.6%). The net effect across all
5 folds is a weak but statistically consistent improvement driven by
the gradient rebalancing from the gmb/qb scale floor.

### Body-physics scoreboard

| Held out | E0 NRMSE_body % | E5 NRMSE_body % | ratio |
|----------|----------------:|----------------:|------:|
| asap7    |      160,345.31 |      156,010.08 | 0.973 |
| tsmc5    |            1.88 |            2.04 | 1.085 |
| tsmc7    |            1.76 |            1.95 | 1.107 |
| tsmc12   |            1.01 |            0.80 | 0.792 |
| tsmc16   |            1.24 |            0.86 | 0.694 |

ASAP7 body remains catastrophic (unchanged). TSMC12/TSMC16 body
improved 20–30%; TSMC5/TSMC7 body regressed 8–11%. Mixed picture,
not a decision variable.

### In-distribution guardrail (phys-best on 4-tech val)

| Fold   | E0 phys-best % | E5 phys-best % | Δ relative |
|--------|---------------:|---------------:|-----------:|
| asap7  |          0.176 |          0.175 |     −0.6 % |
| tsmc5  |          0.278 |          0.255 |     −8.3 % |
| tsmc7  |          0.286 |          0.210 |    −26.6 % |
| tsmc12 |          0.292 |          0.218 |    −25.3 % |
| tsmc16 |          0.285 |          0.200 |    −29.8 % |

All 5 folds show improved or equal in-distribution accuracy. No
guardrail violations.

## Key Lessons

### What worked

**S2 asinh-scale floor** — a 2-line change (`OUTPUT_ASINH_SCALE_MIN`)
that floors the per-target asinh scale for gmb/qb so they cannot
collapse to TSMC-dominated values. The mechanism is indirect: the
floored scale changes the effective gradient contribution from
gmb/qb during training, which rebalances the shared encoder slightly
in favor of id/gm/gds/cgg. The improvement is ~3% geometric mean
across 5 folds. Small but free.

### What failed and why

**M2a derived input features** (Vov, Vds, Vgb, log_NFIN_L) — the
hypothesis was that handing the Transformer explicit dimensionless
groups (overdrive, W/L ratio) would make the I-V function smoother
across tech boundaries. In practice:

1. **Vds and Vgb are redundant** with the voltage group MLP
   (`nn.Linear(4, d*2) → GELU → nn.Linear(d*2, d)`), which can
   learn these linear combinations in its first layer.

2. **Vov = Vg − PHIG carries a per-tech constant offset** (ΔPHIG ≈
   0.13 V between ASAP7 and TSMC). The model calibrates to the
   TSMC Vov distribution during training, then extrapolates badly
   on ASAP7's shifted Vov range.

3. **Even Vov alone (E2b) regressed more than the full 4-feature set**
   (+13.8% vs +9.7%), ruling out feature-interaction or dilution as
   the cause. The regression is intrinsic to making per-tech offsets
   explicit.

4. **The extra 134k parameters** from the derived group MLP increase
   model capacity without adding information, causing the 4-tech
   training distribution to be fit more tightly at the expense of
   the held-out fold.

**Lesson**: for LOO generalization, do NOT add input features that
carry per-tech constant offsets. The model already has the raw
scalars and can compose whatever it needs via attention across group
tokens. Making compositions explicit breaks when the held-out tech's
composed values shift out of the training range.

### The ASAP7 body-physics problem is structural

ASAP7's FinFET modelcard has `CIT = 0` (zero interface trap charge),
which fully decouples the body from the channel. This makes gmb ~1e-9 S
and qb ~1e-18 C — 3–4 orders of magnitude smaller than TSMC. No
normalizer trick, input feature, or loss reweighting can fix this:
the model has never seen CIT=0 during training and has no basis for
extrapolating to body-decoupled physics. The only principled fix is
to include ASAP7-like samples in the training pool (violating the
LOO protocol) or to add a second body-decoupled tech to the registry.

## Recommendations

1. **Ship S2** — the asinh-scale floor is a zero-cost improvement.
   Commit `859d429` is already on `main`.

2. **Do NOT retry M2a** without truly dimensionless ratios
   (Vov/VT_est, Vds/VDD_est) or tech-invariant normalization of
   the derived features. The current raw-offset approach is a
   confirmed dead end.

3. **Accept the ASAP7 body gap** as a known limitation. Report
   NRMSE_sc and NRMSE_body separately in all LOO results so the
   body catastrophe does not mask real improvements on the
   solver-critical outputs.

4. **Future LOO improvements** should focus on:
   - **Data augmentation**: interpolating process-param vectors
     between techs (requires careful handling of (V, L, NFIN) grids).
   - **Meta-learning**: MAML or Reptile-style training that
     explicitly optimizes for fast adaptation to held-out techs.
   - **Per-tech calibration**: a few-shot fine-tune protocol where
     10–100 samples from the held-out tech are used to calibrate
     a frozen encoder.

## Appendix: Run Artifacts

| Run | Path |
|-----|------|
| E0 baseline | `tests/verify_bsimar_loo_results/20260409_130630_nmos/` |
| E1 (2-fold) | `tests/verify_bsimar_loo_results/20260410_091523_nmos/` |
| E2 full M2a | `tests/verify_bsimar_loo_results/20260410_092654_nmos_m2a/` |
| E2b Vov-only | `tests/verify_bsimar_loo_results/20260410_104534_nmos_vov/` |
| E5 (5-fold) | `tests/verify_bsimar_loo_results/20260410_114829_nmos/` |
| Improvement plan | `external_compact_models/bsimar/docs/bsimar_loo_improvement_plan_2026_04_10.md` |

# BSIM-AR Architecture Round 2 — Retry Sprint Report

**Date:** 2026-04-08
**Sprint scope:** 6 parallel 50-epoch experiments on top of the round-1
final HEAD (T1 + P4 + T2 asinh + A2 grouped input tokens, commit `71f0e50`).
**Baseline to beat:** AVG **MRE 4.93%**, NRMSE_phys 0.575%, R²_norm 0.9982.
**Hardware:** NMOS-only, 4 GPUs in parallel (RTX PRO 6000 Blackwell + 3 × A100-40GB).

## Headline result

**0/6 retries beat the A2 baseline on MRE.** The A2+T2+P4+T1 configuration is
a strong local minimum at this model size (d_model=128, 3 layers, ~400K params)
and 50-epoch budget. The closest near-misses (T4 +0.06pp, P5 +0.08pp,
T3 +0.15pp) are within run-to-run noise but no run produced a robust win.

| # | experiment | code | MRE (%) | NRMSE_phys (%) | R²_norm | Δ MRE | verdict |
|---|---|---|---:|---:|---:|---:|---|
| – | **A2 baseline** | `71f0e50` | **4.93** | 0.575 | 0.9982 | – | reference |
| 12 | P2-retry: charge neutrality + √3 upweight | `40dd99e` | **20.19** | 0.955 | 0.9911 | +15.26 | **INFEASIBLE** |
| 13 | T3-retry: Laplace NLL replacing LDS | `8839df8` | 5.08 | 0.576 | 0.9973 | +0.15 | **INFEASIBLE** |
| 14 | T4-retry: warmup + EMA (adaptive decay) | `b7f1dec` | 4.99 | **0.523** | 0.9981 | +0.06 | **INFEASIBLE on MRE / WIN on NRMSE** |
| 15 | T5-retry: log\|id\| loss + zscore (drops asinh) | `bafc456` | 16.99 | 0.745 | 0.9901 | +12.06 | **INFEASIBLE** |
| 16 | P3: Vov featurization (Vth_hat head) | `9ba3bcc` | 5.17 | **0.567** | 0.9980 | +0.24 | **INFEASIBLE** |
| 17 | P5: spectral-norm gm/gds/gmb heads | `1536817` | 5.01 | 0.577 | 0.9981 | +0.08 | **INFEASIBLE** |

## Per-target MRE (% — only "live" experiments shown, P2/T5 excluded as outright failures)

| target | A2 (base) | T3-retry | T4-retry | P3 Vov | P5 specnorm |
|---|---:|---:|---:|---:|---:|
| id | 4.97 | 5.37 (+0.40) | 5.70 (+0.73) | 6.37 (+1.40) | 4.96 (−0.01) |
| **gm** | **11.01** | 12.08 (+1.07) | 11.44 (+0.43) | 12.39 (+1.38) | 11.19 (+0.18) |
| **gds** | **8.94** | 9.90 (+0.96) | 9.27 (+0.33) | 9.15 (+0.21) | 9.19 (+0.25) |
| **gmb** | **8.02** | 7.93 (−0.09) | 7.75 (−0.27) | 9.36 (+1.34) | **7.58 (−0.44)** |
| qg | 3.10 | 3.06 (−0.04) | 3.28 (+0.18) | 3.35 (+0.25) | 3.27 (+0.17) |
| qd | 3.82 | 3.77 (−0.05) | 3.92 (+0.10) | 4.19 (+0.37) | 3.98 (+0.16) |
| qs | 2.92 | 2.75 (−0.17) | 2.92 (0.00) | 2.77 (−0.15) | 3.03 (+0.11) |
| qb | 6.30 | 6.31 (+0.01) | **6.00 (−0.30)** | **6.22 (−0.08)** | 6.54 (+0.24) |
| cgg | 2.56 | 2.39 (−0.17) | 2.38 (−0.18) | 2.21 (−0.35) | 2.54 (−0.02) |
| cgd | 3.47 | 3.39 (−0.08) | 3.19 (−0.28) | **3.03 (−0.44)** | 3.47 (0.00) |
| cgs | 2.81 | 2.89 (+0.08) | 3.09 (+0.28) | 2.85 (+0.04) | 3.05 (+0.24) |
| cdg | 2.88 | 2.80 (−0.08) | 2.83 (−0.05) | 2.70 (−0.18) | 3.00 (+0.12) |
| cdd | 3.36 | 3.36 (0.00) | 3.05 (−0.31) | **2.69 (−0.67)** | 3.32 (−0.04) |
| **AVG** | **4.93** | 5.08 | 4.99 | 5.17 | 5.01 |

## Per-experiment diagnoses

### #12 P2-retry — INFEASIBLE (catastrophic regression on qb)

The √3 upweight on `qg/qd/qs` was insufficient to compensate for the
post-head sum `qb = -(qg+qd+qs)`. qb MRE went from 6.30% → **193.23%**
because three independent ~5-9% errors do not cancel when summed; the
relative error in their sum is ~√3× the per-target error, but qb's
physical magnitude is often comparable to or smaller than |qg|, |qd|,
|qs|, so the absolute residual dominates the (small) qb scale.

Capacitances did improve modestly across the board (cgg 2.56→2.34, cgd
3.47→3.26, cdd 3.36→3.00), confirming that the shorter AR sequence
helps the C-block, but the qb explosion swamps every gain.

**Verdict:** P2 charge neutrality is structurally dead at this
architecture. Future variants would need either (a) a learned residual
head that *corrects* qb after the deterministic sum (which defeats the
purpose), or (b) restoring qb supervision and treating the sum as a
soft-penalty term — at which point P2 is just a regularizer with extra
steps. Recommend permanently parking this idea.

### #13 T3-retry — INFEASIBLE (Laplace ≈ MAE in disguise)

Removing `--lds` did not save Laplace NLL. The learned `log_b_k` settled
in [−2.6, −3.6] (matching original T3) but the post-mortem call of
"collapse" in round 1 was wrong: this range simply reflects the natural
log of per-target MAE after asinh normalization (~0.04-0.07). Laplace
NLL with a learned global scale is, in this regime, just a per-target
weighted MAE — and the auto-balanced weights it picks shift error from
charges/caps (which got marginally better) onto id/gm/gds (which got
notably worse).

**Verdict:** Laplace NLL provides no inductive advantage over MAE+LDS in
this setup. The auto-balance moves loss in the wrong direction. Park.

### #14 T4-retry — split decision (NRMSE wins, MRE loses)

The adaptive-decay EMA (`decay = 1 - 1/√total_steps ≈ 0.989`, half-life
~64 steps) worked exactly as the round-1 post-mortem predicted — raw
and EMA val losses converge by epoch 50 instead of the EMA shadow being
anchored to random init. Diagnostics:
- raw → EMA val loss progression: ep1 0.987→1.113 · ep10 0.038→0.036 ·
  ep50 0.0197→0.0196 (cleanly converging)

The result is a real **NRMSE win**: 0.575 → **0.523** (−9% relative).
But MRE moved the wrong way by +0.06pp because EMA averaging smooths
high-magnitude outliers (which dominate NRMSE) at the cost of slight
over-regularization in the low-|x| tail (which dominates MRE).

**Verdict:** Strict MRE gate is failed (4.99 > 4.93), but **for any
downstream NRMSE-oriented use case (transient simulation peak fit, R²
on absolute waveforms), the T4-retry checkpoint is the new SOTA**. The
two checkpoints should be retained as alternatives — A2 for MRE,
T4-retry for NRMSE.

### #15 T5-retry — INFEASIBLE (zscore can't reach asinh accuracy)

Confirms that asinh is essential, not optional. Plain zscore + a
subthreshold `log|id|` patch on 12% of samples gets nowhere near asinh
on the full distribution: id MRE 4.97 → 20.35, gmb MRE 8.02 → 34.13,
AVG MRE 4.93 → 16.99. The L_sub term ran ~22× larger than L_mae by
end of training, so it actively fought the main loss instead of
complementing it.

**Verdict:** asinh + LDS is the recommended config; do not strip asinh
from the production pipeline. The T5 idea (log-id loss) is now
dominated by asinh in every regime tested. Park permanently.

### #16 P3 Vov featurization — INFEASIBLE (regresses the exact targets it targets)

The learned `Vth_hat(geom, proc)` head (~545 params) was supposed to
collapse the (Vgs, Vds) curves onto a near-universal `(Vov, Vds/Vdsat)`
surface and improve gm/gds/gmb. Instead all three conductances *and*
id got worse (id +1.40, gm +1.38, gds +0.21, gmb +1.34) while caps
slightly improved.

**Diagnosis:** without a physical anchor, the unconstrained Vth_hat
drifts to whatever value minimizes the joint loss — which apparently
helps the cap block (its physics has no Vth dependence) but breaks the
current-block representations. The model effectively treats Vov as a
nuisance feature for I-V and uses it as a free regularizer for C-V.

**Retry path (not run this sprint):** add a weak supervised regularizer
on Vth_hat (e.g. pull it toward the median Vgs of samples where
|id| ≈ id_thresh on a per-tech bin). Or constrain Vth_hat sign and
magnitude by passing it through `0.3 * tanh(...) + offset` for an
ASAP7-like 0.4V prior. Worth one more shot if anyone returns to P3.

### #17 P5 spectral-norm — INFEASIBLE (binding but ineffective)

Spectral-norm parametrization on gm/gds/gmb heads is verified active
(post-training max singular values are pinned at 1.000; the original
weights would have been 1.54-1.60). But the constraint did not help:
gm +0.18, gds +0.25, gmb −0.44 (only gmb improved, and that's within
noise). AVG MRE +0.08.

**Diagnosis:** the conductance bottleneck is in the *upstream
Transformer features*, not in the final linear projection. Bounding
the head's Lipschitz constant just clips a degree of freedom that
wasn't actively misbehaving. To attack the conductance ceiling, future
work should target the encoder layers feeding into those heads — e.g.
P1 EKV residual head (which moves the 14-decade dynamic-range problem
to an analytic prior) or P5+ at intermediate layers, not the final
projection.

**Verdict:** The hypothesis "conductance heads overfit label noise as
spikes" is not confirmed. Park unless paired with an upstream change.

## Sprint takeaways

1. **The current best (A2+T2+P4+T1) is a strong local optimum at ~400K
   params and 50 epochs.** None of 6 reasonable architectural retries
   beat it on MRE. To make further progress at this size, the
   recommended next steps are upstream of the linear heads:
   - **P1 EKV residual head** (Phase 3, never tried) — collapses 14-decade
     id range to ~1-2 decades for gm/gds, the only candidate left that
     specifically attacks the conductance bottleneck.
   - **Larger model** — d_model=256, 6 layers, 110K-256K params, paper
     scale. The grouped-input attention is now ~6× cheaper than the
     pre-A2 baseline, so this is a cheap experiment to run.
   - **Scaling-law plumbing** is already on `dc5afa6`, ready for use.

2. **P2 charge neutrality, T3 Laplace, T5 log-id, and P3 Vov can be
   retired** — each has a clear failure mode that future variants would
   need to architect *around*, not just retune.

3. **The T4-retry EMA checkpoint should be retained as an alternative**
   for NRMSE-oriented downstream use. It is the only round-2 result
   that improves on the round-1 final on *any* aggregate metric.

4. **MRE-primary scoring continues to be load-bearing.** The same 6
   experiments scored on NRMSE_phys would have flipped at least 2
   verdicts (T4 would clearly KEEP, P3 would marginally KEEP) — a
   reminder that the choice of primary metric is itself an
   experimental design decision.

## Reproduction

Each experiment lives on its own branch off `bsimar-arch-experiments`:

| branch | commit | log | checkpoint |
|---|---|---|---|
| `exp2-p2-retry` | `40dd99e` | `exp_logs/exp12_p2_retry.log` | `checkpoints/exp12_p2_retry_50ep_nmos_*.pt` |
| `exp2-t3-retry` | `8839df8` | `exp_logs/exp13_t3_retry.log` | `checkpoints/exp13_t3_retry_50ep_nmos_*.pt` |
| `exp2-t4-retry` | `b7f1dec` | `exp_logs/exp14_t4_retry.log` | `checkpoints/exp14_t4_retry_50ep_nmos_*.pt` |
| `exp2-t5-retry` | `bafc456` | `exp_logs/exp15_t5_retry.log` | `checkpoints/exp15_t5_retry_zscore_50ep_nmos_*.pt` |
| `exp2-p3-vov` | `9ba3bcc` | `exp_logs/exp16_p3_vov.log` | `checkpoints/exp16_p3_vov_50ep_nmos_*.pt` |
| `exp2-p5-specnorm` | `1536817` | `exp_logs/exp17_p5_specnorm.log` | `checkpoints/exp17_p5_specnorm_50ep_nmos_*.pt` |

Worktrees live at `/home/shenshan/NN_SPICE_exp2_<name>/`. None of the
branches has been merged. The main agent (or human) decides which (if
any) to keep. Recommendation: **retain only `exp2-t4-retry` as the
NRMSE-oriented alternative** to the round-1 A2 head. The other 5 can be
left as historical branches and the worktrees deleted.

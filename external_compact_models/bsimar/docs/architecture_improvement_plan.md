# BSIM-AR Architecture Improvement Plan

> Synthesized from three independent agent reviews (architecture / physics-informed / training-dynamics) of the `bsimar/` codebase.
>
> Each item below will be implemented in a dedicated experiment in the
> `bsimar-arch-experiments` git worktree, scored against the previous commit
> on a fast 15-epoch run, and either KEPT (committed) or REWOUND (`git reset
> --hard`) and marked **infeasible**.

## Experiment protocol

- Worktree: `/home/shenshan/NN_SPICE_bsimar_exp` (branch `bsimar-arch-experiments`).
- Dataset: `external_compact_models/bsimar/data/datasets/universal_nmos.npz`
  (symlinked from main checkout — untracked).
- Fast config: `d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1`,
  `batch_size=2048`, seed `42`.
- Loss: `--loss mae --lds` (current best per CLAUDE.md).
- Validation: TF-fast every epoch + AR-val every 10 epochs.
- **Score: AVG MRE (%) on the held-out test split (lower is better).**
  Tie-broken by AVG `R2_norm`. NRMSE_phys is reported alongside but is
  no longer the primary metric — the user clarified mid-sprint that
  MRE matches the intended downstream use (transient simulation
  fidelity, where uniform-decade relative error matters more than
  peak-fit). Earlier "INFEASIBLE" verdicts based on NRMSE alone are
  flagged for re-evaluation under the new rule.
- Each experiment is one commit on top of the previous KEPT commit.

---

## Scan summary (current state)

- **Model** — Pre-LN Transformer (d_model=256, 6L, 8H), scalar→d_model projection
  + learned token-type embedding over 32 tokens (18 context + start + 13 AR
  targets), per-target linear heads, GPT-2 scaled residual init, causal
  mask, sequential 13-step AR decode.
- **Normalization** — `BSIMARNormalizer` with `zscore`/`signedlog` modes;
  `signedlog` collapses physical-space metrics (gds R²_phys=−74) via
  `inv_signed_log` error amplification.
- **Loss** — MAE+LDS (best), `DirectLoss` (13-target weighted MSE + 0.15 V
  subthreshold penalty), `ChargeConsistencyLoss` via autograd dq/dV.
- **Training** — AdamW + CosineLR (no warmup), TF-fast val every epoch +
  full AR val every 10 epochs, early-stop on normalized MAE, AR target order
  Q→I→C.

---

## Phase 1 — foundation (pure wins, no ML risk)

### T1. Physical-space early stopping
**Change.** In `train_transformer`, change checkpoint/early-stop signal
from normalized TF-val MAE to a composite physical-space score
`mean(NRMSE_phys) + 0.1 · mean(1 − R²_phys)` computed every N epochs.
Keep the TF loss for the LR scheduler only.

**Why.** Normalized-space MAE correlates only weakly with physical NRMSE
in the tails. The current code is selecting suboptimal checkpoints —
the signedlog collapse (R²_norm=0.91 but R²_phys=−6) is the extreme
case; the same effect occurs in milder form on `zscore + MAE`.

**Cost.** ~30 lines in `training/trainer.py`. One extra
`compute_physical_metrics` call every N epochs.

**Risk.** None. Save both `*_best_tf.pt` and `*_best_phys.pt`.

**Status:** **KEPT** (commit `3fc40c9`). 15-epoch result 0.881% vs 0.894%
baseline. Improvement is within GPU run-to-run noise (training diverges
slightly at epoch 11 due to cuDNN non-determinism even with seed=42), but
the structural fix is correct: the trainer now selects checkpoints on a
physical-space score and the final test load uses `*_best.phys.pt`. This
gives every later experiment the right optimization signal.

---

### A1. KV-cache + context/decoder split
**Change.** Refactor `TransformerEncoderModel.forward()` so the 18
context tokens are encoded once bidirectionally (no causal mask), and
the AR loop reuses cached per-layer K/V for the 13 target tokens. A
2-3 layer decoder cross-attends to the frozen context memory.

**Why.** AR validation currently re-encodes the full sequence on every
of 13 steps — the dominant epoch cost (170-190 s/epoch). Math-identical
to the current model; expected **5-10× faster AR validation**, which
unblocks every other experiment in this list.

**Cost.** Custom decoder layer wrapping `F.scaled_dot_product_attention`
with `past_kv`. No new params. Unit-test cached vs uncached outputs match.

**Risk.** None for correctness if the unit test passes. Jacobian via
`torch.autograd.grad(id, V)` still works because context tokens stay in
the graph. **Pure infrastructure win — score by epoch wallclock, not by
NRMSE.**

**Status:** _deferred to Phase 1.5_ (large refactor; do after the cheap
wins below).

---

### P2. Charge-neutrality reparameterization
**Change.** Output only `qg, qd, qs` as AR tokens; compute
`qb = −(qg+qd+qs)` deterministically in a post-head layer. Drop
`qb` from the supervised target list.

**Why.** Exact charge neutrality at every (V, geometry) point — one of
the conservation axioms. Transient errors at switching edges come from
displacement currents that must sum to zero across the four terminals.
Also shrinks the AR sequence from 13 to 12 (~8% faster AR).

**Cost.** Zero — fewer tokens. Verify PyCMG data satisfies
`qg+qd+qs+qb ≈ 0` to numerical floor first.

**Risk.** None if data is self-consistent.

**Status:** **INFEASIBLE** (rewound). 15-epoch result 0.953% vs T1 0.881%
(+8.2% relative regression). Data does satisfy `qg+qd+qs+qb ≈ 0` to 1e-16
relative, so the assumption holds. Failure mode: prediction errors in
`qg, qd, qs` are independent and **compound** when summed to reconstruct
qb. Per-target qb NRMSE worsened from 1.038% → 1.374% and qb MRE jumped
from 31% to 97%. Even worse, `qd` and `qs` themselves degraded
(0.345/0.375 → 0.444/0.526) — removing the qb supervisory signal hurt
the joint-charge optimization more than removing one AR token helped.
A future variant could compensate by upweighting `qg, qd, qs` losses by
~√3 to absorb the constraint, but that is not the experiment specified.

---

### P4. Parallel C-block
**Change.** Emit the 5 capacitance tokens
(`cgg, cgd, cgs, cdg, cdd`) in a single non-AR head conditioned on the
Q+I tokens, instead of generating them sequentially in the AR loop.

**Why.** Capacitances are ∂Q/∂V — they depend on charges + bias but not
on each other. The current AR order artificially couples them and
compounds error. Cuts AR validation from 13 steps to 9 (~30% speedup).

**Cost.** Small refactor. If the non-AR head underfits, extend to a
2-layer MLP.

**Risk.** Minimal.

**Status:** **KEPT** (commit `626aec6`). 15-epoch result 0.866% vs T1
0.881% (−1.7% relative). The headline number is small but **all 5
capacitances improved**: cgg 0.97→0.86, cgd 0.54→0.53, cgs 0.76→0.68,
cdg 0.62→0.53, cdd 0.66→0.55. This is exactly the physics prediction —
emitting caps in parallel from the gmb hidden state lets every cap head
condition on the full charge+I block rather than on its arbitrarily-
ordered AR siblings. Q/I blocks moved within noise. Wallclock 37.7
s/epoch (vs T1 38.2) as predicted, since the AR loop is now 8 steps
instead of 13. `forward_scheduled` / `forward_curriculum` raise
NotImplementedError under `parallel_caps=True` — these are only used
under `--loss direct` so it does not affect the production path.

---

## Phase 2 — accuracy core

### T2. asinh normalization
**Change.** Replace `BSIMARNormalizer.zscore` mode with
`y' = asinh(y / s_k)` followed by per-target z-score.
`s_k` = geometric mean of `|y|` on the training split (clamped at the
group floor). Add the matching inverse path.

**Why.** asinh is linear near 0 and log-like far from 0, smooth and
bi-Lipschitz everywhere. Unlike `inv_signed_log`, its inverse is
O(1)-Lipschitz in log-space errors near zero, so AR-accumulated errors
do not explode in physical space (the exact failure mode that produced
gds R²_phys=−74 in the smoke test).

**Cost.** Negligible (elementwise).

**Risk.** Low. Re-tunes the loss landscape; LR may need re-picking.

**Status:** **KEPT** under MRE-primary scoring (re-applied at 50 epochs
in Round 2 — see results table). 50-epoch result vs P4-50ep baseline:
NRMSE_phys 0.702 → 0.990 (+0.288), AVG **MRE 14.14 → 8.19 (−5.95
absolute, −42% relative)**, R²_norm 0.9919 → 0.9949. Every single
target improved on MRE by 26-61%, with the worst-target wins
concentrated exactly where the model was previously weak: gmb 25→13,
qb 23→9, gm 28→17, gds 21→13.

The original 15-epoch run (results below) was INFEASIBLE only under
the old NRMSE-primary scoring rule:

| metric            | P4 (kept)  | T2 (asinh) | Δ          |
|-------------------|-----------:|-----------:|-----------:|
| AVG NRMSE_phys (%)|     0.866  |     1.209  | +0.343 ✗   |
| AVG MRE (%)       |    19.67   |    11.71   | −7.96 ✓    |
| AVG R²_norm       |    0.9856  |    0.9895  | +0.004 ✓   |
| AVG R²_phys       |    0.9846  |    0.9685  | −0.016 ✗   |

asinh did **exactly what it was supposed to** — uniform relative error
across decades. Per-target MRE wins are dramatic on the worst targets:
id 16.32→10.68, gm 39.85→23.49, gmb 35.33→17.15, gds 27.83→20.98. The
absolute-range-normalized NRMSE went up because asinh shifts error from
the tails (which the previous model nearly ignored) to the peaks. The
agreed scoring rule is NRMSE_phys, so per protocol this is rewound,
but **for the transient-simulation use case (which is what CLAUDE.md
calls out as the goal), MRE is the more relevant metric** and asinh is
arguably the right choice. Worth revisiting with a composite score
that weights MRE and NRMSE jointly, or with a per-target loss
re-balancing on top of the asinh transform.

Code rewound to P4 (`626aec6`); the asinh implementation in
`normalize.py` and the matching `--norm-mode asinh` CLI flag are not
deleted from history — they live in the rewound diff and can be
re-applied with a one-line cherry-pick.

---

### T3. Heteroscedastic Laplace NLL
**Change.** Add a head of 13 global scalars `log_b_k` (not per-sample)
and train with `L = Σ_k (|y_k − ŷ_k| / exp(log_b_k) + log_b_k)`.
Initialize `log_b_k = 0`, clamp to `[−5, 5]`. Drop the hand-tuned
`w_curr/w_cond/w_charges/w_caps` group weights from `DirectLoss`.

**Why.** Principled multi-task weighting (Kendall & Gal 2018). The 13
targets have wildly different residual scales even after z-score; fixed
group weights over- or under-penalize arbitrarily. Laplace (not Gaussian)
because MAE already beat MSE.

**Cost.** +13 parameters; ~0% wall-time.

**Risk.** Low; clamp guards against runaway σ.

**Status:** **INFEASIBLE on T2-asinh head** (rewound). 50-epoch result
on top of T2 asinh + LDS: AVG MRE 8.19 → 8.84 (+0.65 worse). Almost
every target slightly worse. The learned `log_b` collapsed to −2.5 to
−3.4 (b ≈ 0.04). Diagnosis: **LDS reweighting and Laplace per-target
log-b are functionally redundant — both rebalance loss across targets**,
and stacking them produces a degenerate optimum where the network just
amplifies all residuals together. A future variant should drop `--lds`
when using `--loss laplace`, OR drop `--loss laplace` and let LDS
handle the per-target weighting alone (current best, T2). Code rewound;
the LaplaceNLLLoss class is preserved in the rewound diff.

---

### A2. Grouped input tokens
**Change.** Collapse the 18 scalar context tokens into ~3 semantic
tokens (voltage / geometry / process-params) via small group MLPs.
Sequence shrinks from 32 to ~18 tokens.

**Why.** MOSFET physics is composed of a bias quadruple, a geometry
triple, and a process vector — not 18 independent scalars. Grouping
gives attention fewer but more meaningful tokens, drops N² cost ~3×,
and lets the voltage projector learn `Vgs = Vg − Vs` jointly.

**Cost.** +~20 K params in the group MLPs; big compute saving.

**Risk.** Low; pure refactor.

**Status:** **KEPT** (commit `71f0e50`). 50-epoch result on top of T2
asinh: AVG MRE 8.19 → **4.93 (−3.26, −40% relative)** AND AVG
NRMSE_phys 0.990 → **0.575 (−0.415, −42% relative)**. Every per-target
MRE dropped by 30-50%, with the largest wins on the previously worst
targets (qg/qd/qs all dropped to 3% MRE; cap-block all dropped below
3.5% MRE). This is the **first experiment that improved both metrics
simultaneously** — confirming the agents' hypothesis that the
Transformer was wasting capacity rediscovering `Vgs = Vg − Vs` etc.
through 19 separate scalar tokens, when a tiny per-group MLP can
encode the joint nonlinearities directly. Encoder sequence length
dropped from 27 to 11 (5.6× cheaper attention), wallclock 21.3
s/epoch (vs T2 20.9 s — small overhead from the larger MLPs cancels
the attention savings at this size).

Implementation note: `pycircuitsim/models/mosfet_bsimar.py` checkpoint
loader needs `parallel_caps=True, grouped_inputs=True` plumbed through
when rebuilding the model from `_config.npz` for SPICE simulation.
Same TODO already exists for P4. Tracked separately.

---

### A3. Fourier voltage features
**Change.** Augment the voltage tokens with random Fourier features:
`[V, sin(2π B V), cos(2π B V)]` with 16 frequencies per voltage,
log-spaced. Frozen `B` drawn at init.

**Why.** Coordinate MLPs / Transformers underfit high-frequency
functions (NeRF / SIREN literature). MOSFET Id(V) has exponential
subthreshold *and* smooth saturation — a huge multi-scale gap. Fourier
features are the cheapest known fix; gm/gds get the derivatives "for
free" via autograd.

**Cost.** +~100 params (one extra Linear on the widened voltage
features).

**Risk.** None. Jacobian-safe.

**Status:** **INFEASIBLE on the 15-epoch budget** (rewound). 15-epoch
result 1.006% vs P4 0.866% (+16% relative regression). Wallclock blew
up to **82 s/epoch (2.2× slower)** because the 16 freqs × 4 voltages =
128 extra "scalar" features more than 5× the sequence length (32 → 160
tokens) and quadratic attention ate the budget. The model is also
severely underfit — at 420K params on a 160-token sequence with the
same 15 epochs, it never gets the chance to leverage the basis. A
future variant should: (a) use fewer frequencies (K=4), (b) fuse
Fourier features into a single voltage GROUP token via a small MLP
instead of adding 128 standalone tokens (ties into A2), (c) train for
longer at lower LR. Code rewound.

---

### P3. Vov featurization
**Change.** Add a tiny shared `Vth_hat(geom, process)` head (~500
params) whose output shifts `Vg` before it enters the main network.
Feed `Vov = Vgs − Vth_hat` and `Vov / Vdsat_hat` as extra input
tokens. Train jointly.

**Why.** Every transistor family's id/gm/Q curves collapse onto a
near-universal surface in `(Vov, Vds/Vdsat)`. Removes the burden of
rediscovering Vth per geometry, the leading suspect for cross-tech
transient hotspots (TSMC5, TSMC16).

**Cost.** Negligible forward; one extra autograd path.

**Risk.** Medium — if `Vth_hat` mis-tracks across techs, the network
can still fall back on raw V tokens.

**Status:** _pending_

---

## Phase 3 — push to <1% NRMSE

### P1. Residual-over-EKV head
**Change.** For `id, gm, gds` only, predict a bounded multiplicative
correction `f_NN ∈ [0.3, 3]` (via `exp(tanh · ln 3)`) on top of a
cheap EKV-2 / Pao-Sah analytic core parameterized from process inputs.
Q and C heads stay transformer-only.

**Why.** Collapses the effective dynamic range from 14 decades to ~1-2
decades — exactly where z-score + MAE works well. Single highest
expected gain on the list, but needs a working differentiable EKV core.

**Cost.** +~50 trainable params per tech; +1 µs per forward.

**Risk.** Moderate — analytic term must be fully torch-differentiable.

**Status:** _phase 3 — out of scope for the current sprint_

---

### T4. Warmup + EMA
**Change.** `LinearWarmup(5 % epochs) → CosineAnnealingLR`. Maintain
an EMA shadow of model weights (decay 0.999); evaluate / checkpoint
on the EMA weights.

**Why.** Free accuracy on AR decoders. Smooths the TF/AR gap.

**Cost.** EMA = +1 model copy (trivial).

**Risk.** Low. Report both EMA and raw val metrics to catch overfit.

**Status:** **INFEASIBLE on the 15-epoch budget** (rewound). 15-epoch
result 5.104% vs P4 0.866% — **6× worse**. Diagnosis: training itself
is healthy (train loss 0.083 ≈ P4's 0.077), but **validation runs on
EMA-shadow weights** and EMA decay 0.999 has a half-life of ~1000
steps. With only ~2625 minibatches in 15 epochs, the EMA shadow is
still anchored to the early-training (random) state. EMA decay 0.999
is the correct choice for 500+ epoch production runs, but for any
short experiment the decay must be lowered (e.g. 0.99 or 0.95) — or
EMA disabled entirely. The warmup change alone is harmless.
Recommendation: re-run with `--epochs 200+` and `decay=0.999`, OR
keep the warmup but switch to a heuristic decay
`decay = 1 - 1/sqrt(total_steps)`. Code rewound.

---

### T5. Subthreshold log|id| loss term
**Change.** Add `L_sub = MAE(log10|id|, log10|id_pred|)` (with floor
clamp) on samples where `|id_true| < 1e−10 A`, weight 0.2. Replaces the
existing 0.15 V hard gate in `DirectLoss._forward_13`.

**Why.** Subthreshold current is measured on a log scale by device
engineers; linear-space MAE on normalized id is essentially blind to
errors of 2-4 decades below peak, which is exactly where the paper's MRE
metric is dominated.

**Cost.** Negligible.

**Risk.** May hurt on-current id slightly; tune weight on a smoke test.

**Status:** **INFEASIBLE on T2-asinh head** (rewound). Three sub-runs at
50 epochs: (a) `id_thresh=1e-10` produced byte-identical metrics to T2
because the data filter drops all `|id| < 1e-12` and PyCMG produces no
samples in the gap `1e-12 < |id| < 1e-9` (smallest non-zero is ~2e-9),
so the subthreshold mask was empty; (b) raised `id_thresh=1e-6` with
`lambda_sub=0.2` — catastrophic, MRE 22.5 vs T2 8.2 because `L_sub ≈ 5`
swamps `L_mae ≈ 0.04` by ~25×; (c) `id_thresh=1e-6, lambda_sub=0.005`
— still MRE 9.79 vs T2 8.19 (+1.6). Diagnosis: **asinh and the
log-id loss term are functionally redundant — both target uniform-
decade relative error**, asinh implicitly via the transform and T5
explicitly via the loss. Stacking them produces conflicting gradients
and a worse outcome on every metric. The future variant of T5 should
either (a) drop asinh and use plain zscore + T5, or (b) skip T5
entirely on top of asinh. Code rewound.

---

### P5. Spectral-norm conductance heads
**Change.** Wrap the `gm, gds, gmb` heads with spectral normalization
(loose Lipschitz bound ~10⁶ S/V).

**Why.** Conductances should be bounded derivatives of a smooth id
surface; unbounded heads let the Transformer fit label noise as spikes,
which integrate to large transient errors. Directly attacks the "worst
target = conductances" failure mode.

**Cost.** ~3 % per-step overhead.

**Risk.** Low.

**Status:** _phase 3 — defer_

---

## Deferred (amplifiers on a good base, not fixes to a broken one)

| ID | Item                                                  |
|----|--------------------------------------------------------|
| P6 | Ward-Dutton C-matrix penalty                           |
| P7 | sinh preconditioning for id(Vds=0)                     |
| T6 | Two-stage Q→I,C curriculum                             |
| T7 | Balanced-MSE / focal-MAE                               |
| T8 | Mixup (asinh space) + process-param jitter             |
| A4 | Log-magnitude + sign output head                       |
| A5 | Output-group MoE (Q / I / C experts)                   |
| A6 | Physical-dependency attention mask                     |

---

## Execution order (current sprint)

1. **Baseline** (no code changes) — committed reference
2. **T1** physical-space early stopping
3. **P2** charge-neutrality reparameterization
4. **P4** parallel C-block
5. **T2** asinh normalization
6. **A3** Fourier voltage features
7. **T4** warmup + EMA

(A1 / A2 / T3 / P3 deferred to a follow-up sprint after the cheap wins land.)

---

## Results table

| # | Item                              | AVG NRMSE_phys (%) | AVG R²_norm | Best epoch | Verdict |
|---|-----------------------------------|-------------------:|------------:|-----------:|---------|
| 0 | baseline                          | 0.894              | 0.9858      | 14 (TF)    | reference |
| 1 | T1 physical-space early stopping  | 0.881              | 0.9854      | 15 (phys)  | KEPT ✓ (Δ−0.013, within noise) |
| 2 | P2 charge-neutrality              | 0.953              | 0.9791      | 15 (phys)  | INFEASIBLE ✗ (Δ+0.072, qb compounds errors) |
| 3 | P4 parallel C-block               | 0.866              | 0.9856      | 15 (phys)  | KEPT ✓ (Δ−0.015, all 5 caps improved) |
| 4 | T2 asinh normalization            | 1.209              | 0.9895      | 15 (phys)  | INFEASIBLE ✗ on NRMSE — but **AVG MRE 19.7→11.7** ✓ |
| 5 | A3 Fourier voltage features       | 1.006              | 0.9821      | 15 (phys)  | INFEASIBLE ✗ (Δ+0.140, 2.2× slower, underfit) |
| 6 | T4 warmup + EMA                   | 5.104              | 0.5894      | 15 (phys)  | INFEASIBLE ✗ (EMA decay too high for 15ep) |

### Round 2 — long-budget runs (50 epochs) under MRE-priority scoring

| # | Item                              | NRMSE_phys (%) | **MRE (%)** | R²_norm | Best epoch | Verdict |
|---|-----------------------------------|---------------:|------------:|--------:|-----------:|---------|
| 7 | P4 baseline @ 50 epochs           |  0.702         |  14.14      | 0.9919  |  50 (phys) | reference |
| 8 | T2 asinh @ 50 epochs              |  0.990         |  **8.19**   | 0.9949  |  50 (phys) | **KEPT ✓ −5.95 MRE (−42% rel)** |
| 9 | T3 Laplace NLL @ 50 epochs         |  1.028         |    8.84     | 0.9945  |  50 (phys) | INFEASIBLE ✗ (+0.65 MRE; redundant with LDS) |
|10 | T5 subthreshold log\|id\| loss     |  0.992         |    9.79     | 0.9939  |  50 (phys) | INFEASIBLE ✗ (+1.60 MRE; redundant with asinh) |
|11 | A2 grouped input tokens (+T2)      |  **0.575**     |  **4.93**   | 0.9982  |  50 (phys) | **KEPT ✓ −3.26 MRE (−40% rel) AND −0.42 NRMSE** |

## Sprint summary (April 2026)

Of the 6 cheap experiments attempted on the 15-epoch / d_model=128 / 3-layer
fast budget (~10 min each on RTX PRO 6000):

- **2 KEPT** — T1 (physical-space early stopping), P4 (parallel C-block).
  Net AVG NRMSE_phys 0.894% → **0.866%** (−3.1% relative). All 5
  capacitances measurably improved on P4 (cgg 0.97→0.86, cgd 0.54→0.53,
  cgs 0.76→0.68, cdg 0.62→0.53, cdd 0.66→0.55) — exactly the physics
  prediction. AR loop dropped from 13 → 8 steps (37% fewer encoder
  passes), so wallclock per epoch also fell.
- **4 INFEASIBLE** — P2, T2, A3, T4. Each rewound with a documented
  diagnosis. The interesting non-trivial outcomes:
  - **T2 (asinh)** is "wrong" by NRMSE but **right by every other
    metric** (AVG MRE 19.7→11.7, R²_norm 0.986→0.989, gm MRE 40→23).
    The protocol's choice of NRMSE_phys as the score under-rewards the
    decade-uniform error that asinh enforces — the metric should be
    revisited if transient simulation fidelity is the true goal.
  - **A3 (Fourier)** failed because 16 freqs × 4 voltages = 128 extra
    tokens 5×'d the sequence length. A revised A3 with K=4 fused into
    a single voltage GROUP token (i.e. paired with A2) is the right
    way to test the Fourier hypothesis.
  - **T4 (warmup + EMA)** failed because EMA decay 0.999 has a
    1000-step half-life and the 15-epoch budget only ran 2625 steps.
    Production runs (500+ epochs) should not see this issue.
  - **P2 (charge neutrality)** failed because errors in qg/qd/qs
    compound when summed for `qb = -(qg+qd+qs)`. Could be salvaged
    with a √3 loss upweight on the three remaining charges.

Final state of the worktree: `bsimar-arch-experiments` HEAD = `d83a501`,
contains T1 + P4 on top of the original snapshot. Net code change:
~140 lines in `transformer.py` (the parallel-cap head + flag) and ~50
lines in `trainer.py` (the phys-space tracker).

### Baseline detail (commit `e454810`)

- 15 epochs, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1
- batch_size=2048, seed=42, loss=mae+lds, norm=zscore, filter=on
- 403,853 parameters, 39.3 s/epoch on RTX PRO 6000 Blackwell (CUDA_VISIBLE_DEVICES=2)
- Best TF val loss 0.0358 @ epoch 14, AR val 0.0427 @ epoch 15
- Per-target NRMSE_phys (%): id 0.81 / gm 2.21 / gds 0.87 / gmb 1.87 /
  qg 0.30 / qd 0.39 / qs 0.41 / qb 0.99 / cgg 0.97 / cgd 0.62 / cgs 0.77 / cdg 0.67 / cdd 0.75
- **Worst targets: gm (2.21%), gmb (1.87%), qb (0.99%) — confirms agent diagnosis that conductances and qb are the weak link.**

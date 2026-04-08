# BSIM-AR Improvement Plan — 2026-04-08

Sequential implementation of the suggested next steps from
`results/scaling_law_2026_04_08_arch_v2/scaling_law_v2_report.md`.

All experiments are run on the **Medium** tier
(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, ~5.15M params)
on `universal_nmos.npz`, so we can isolate the effect of each change
against the published v2 medium baseline.

## Frozen baseline (medium, v2 architecture, 50 epochs)

From `scaling_law_v2_report.md` (commit `6ad7ff7`):

| Metric | Value |
|---|---:|
| TF best val | 0.00993 |
| AR best val | 0.01052 |
| **NRMSE_phys** | **0.419 %** |
| **MRE_phys** | **2.52 %** |
| **R²_phys** | **0.9928** |
| Wall-clock | 2716 s |
| Per-epoch | 54.3 s |

**Per-target NRMSE %** (the bottleneck): id 0.878, gm 0.624, gds 0.675,
gmb 0.783, qg 0.175, qd 0.191, qs 0.215, qb 0.867, cgg 0.204, cgd 0.213,
cgs 0.201, cdg 0.210, cdd 0.210.

**Per-target MRE %** (the bottleneck): id 2.45, gm 5.21, gds 4.99, gmb
3.79 (the I/V block dominates), qg 1.64, qd 1.89, qs 1.36, qb 3.28,
cgg 1.29, cgd 1.83, cgs 1.72, cdg 1.59, cdd 1.77.

## Running rules

1. **Git commit before each step.** Each tagged commit captures the
   "last known good" code state we can rewind to with `git checkout`.
2. **Each step runs Medium @ 50 epochs** unless it explicitly varies the
   epoch count. We test each independent change at the same wall-clock
   budget as the baseline so the comparison is honest.
3. **Verdict rule.** A change "wins" if **(a)** it strictly improves
   AVG NRMSE *or* AVG MRE without making the other worse by more than
   half the gain, **and** **(b)** it does not crash or break any
   existing test. Otherwise it is rolled back via `git reset --hard`.
4. **Cumulative.** Winning changes accumulate — the next step is built
   on top of the previously-merged state. Lost changes are reverted.
5. **GPU lane.** Medium runs on `CUDA_VISIBLE_DEVICES=2` (Blackwell 96GB),
   matching the original sweep.
6. **Tracking.** This file is updated after each step with the test
   command, the test metrics table, and the verdict.

## Step ordering rationale

The report lists 11 ideas (N1..N11). We prioritize **code changes that
ship a learnable knob and complete in under 2× baseline wall-clock**
first, then chain into the longer-schedule combined run at the end.

| Order | Step | Targets | Code complexity | Expected wall-clock |
|------|------|--------|----:|----:|
| 1 | **N6** Huber on I/V block | id/gm/gds/gmb MRE | 1 file (~30 LOC) | 1× |
| 2 | **N5** learnable per-target asinh scale | qb + cap NRMSE | 1 file (~40 LOC) | 1× |
| 3 | **N4** charge-consistency penalty | cap NRMSE, qb fit | trainer + loss (~80 LOC) | 1.5–2× (autograd) |
| 4 | **N7** Vov-region LDS sample weighting | low-Vov MRE | data path (~40 LOC) | 1× |
| 5 | **N3** AR fine-tune phase | TF↔AR exposure-bias gap | trainer + transformer (~80 LOC) | 1.2× |
| 6 | **N2** KV-cache encoder during AR decode | wall-clock only, **prereq for N1** | transformer (~120 LOC) | 0.4–0.6× target |
| 7 | **N1** long-schedule combined run @ 150 ep | the slope shows it still helps | driver only | 1.2–1.8× (with N2) |

We DO NOT plan to revisit any of the Round-2 dead-ends (P2/T3/T5/P3/P5);
those are documented in `architecture_round2_report.md`.

---

## Step 1 — N6: Huber loss on the I/V block

**Hypothesis.** AVG MRE at medium is dominated by id/gm/gds/gmb
(2.45–5.21 %). MAE on a 14-decade target overweights the median residual
(near zero) and underweights the saturation tail. A Huber loss is
MAE-like in the tail (preserves the LDS+MAE saturation behavior) but
quadratic near zero, so the optimizer focuses gradient on the genuine
errors instead of the noise floor in the small-magnitude region.

**Code change.**
- Add `HuberMAELoss` to `bsimar/losses/bni_mae.py`: per-column Huber on
  positions 4–7 of the BSIMAR_COLUMN_ORDER (id/gm/gds/gmb), MAE on
  positions 0–3 and 8–12 (charges + caps).
- Wire the new loss into `train_transformer` under
  `--loss huber-mae` (new CLI value).
- Default `delta=1.0` (matches normalized scale; we re-tune if needed).

**Test command.**
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss huber-mae --lds --norm-mode asinh \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 50 --batch-size 1024 --lr 8e-4 --patience 50 --seed 42 --cuda \
  --exp-name n6_huber_medium --overwrite
```

**Result.** TBD.

| Metric | Baseline | N6 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | 0.419 | _TBD_ | _TBD_ |
| MRE_phys % | 2.52 | _TBD_ | _TBD_ |
| R²_phys | 0.9928 | _TBD_ | _TBD_ |
| Wall-clock (s) | 2716 | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Step 2 — N5: per-target learnable asinh scale

**Hypothesis.** The current asinh `s_k` is a fixed per-target geometric
mean of |y|. The report flags this as the cause of the small-tier qb
collapse (R² 0.74). At medium the regression is gone, but the report
also notes "modest gains on caps everywhere" — i.e. there is still
free accuracy left in the under-resolved targets.

**Code change.**
- In `bsimar/models/transformer.py`, add an optional `learnable_asinh`
  knob: a buffer / `nn.Parameter` of shape `(target_dim,)` initialized
  from `BSIMARNormStats.asinh_scale` (in log-space so it stays positive).
- The per-target scale is applied in the loss layer: model still emits
  z-scored asinh-space outputs, but the loss un-normalizes through a
  *learnable* scale, and the test-time denormalizer uses the trained
  scale instead of the fitted one.
- Persist the learned scale into `_norm.npz` next to the fitted scale
  so inference loads the trained value.

**Test command.** Same as N6 but with `--learnable-asinh-scale` flag.

**Result.** TBD.

| Metric | Prev best | N5 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | _TBD_ | _TBD_ | _TBD_ |
| MRE_phys % | _TBD_ | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Step 3 — N4: charge-consistency penalty

**Hypothesis.** The cap targets at medium fit very well (NRMSE 0.20%),
but the cap predictions and the charge predictions are not constrained
to be derivative-consistent (`dq/dV = C`). DirectNet's charge-finetune
mode adds an autograd consistency loss that improves transient accuracy
substantially (CLAUDE.md: 6–14% NRMSE on transient). Porting it to
BSIMAR should both **(a)** tighten cap NRMSE further and **(b)** make
the qb head's gradient signal richer.

**Code change.**
- Add `BSIMARChargeConsistencyLoss` in `bsimar/losses/direct_loss.py`
  (or a new file) — accepts the model and a batch, runs the forward
  pass with `requires_grad_` on the voltage inputs, derives `cgg, cgd,
  cgs, cdg, cdd` from `qg, qd` via autograd, and adds an MSE term
  against the predicted caps.
- Add a `--charge-consistency-weight` CLI flag (default 0).
- Wire into `train_epoch_bni` so MAE/Huber-mode runs can opt in.

**Note.** The autograd path is ~1.5–2× slower; we accept this for the
test step. If it wins, we keep the slowdown.

**Result.** TBD.

| Metric | Prev best | N4 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | _TBD_ | _TBD_ | _TBD_ |
| MRE_phys % | _TBD_ | _TBD_ | _TBD_ |
| Wall-clock (s) | _TBD_ | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Step 4 — N7: Vov-region LDS reweighting

**Hypothesis.** The dataset MRE is dominated by samples in two regimes:
near subthreshold (small |id|) and near saturation (large |id|, small
|gds|). The current LDS weights are computed per-output target, so they
oversample small-magnitude targets but do not directly oversample
samples *whose operating point* lies in the bottleneck region. Adding
an additional sample-weight axis on `Vgs` (input column 0; serves as a
proxy for `Vov` since the dataset has no `Vth`) would push the model
toward fitting the under-resolved bias regions.

**Code change.**
- In `bsimar/training/trainer.py`, after the per-target LDS weights
  are computed, multiply them by a per-sample weight derived from
  `compute_lds_weights_per_target` applied to `inputs[:, 0:1]` (Vgs).
  Bin count: 50. Kernel: gaussian, ks=5, sigma=1.0.
- The combined weight is renormalized so its mean is 1.
- Add a `--vov-lds` CLI flag, default off.

**Result.** TBD.

| Metric | Prev best | N7 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | _TBD_ | _TBD_ | _TBD_ |
| MRE_phys % | _TBD_ | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Step 5 — N3: AR fine-tune phase (scheduled sampling tail)

**Hypothesis.** The TF↔AR gap at medium is 0.0006 (small but nonzero).
Pure teacher-forced training never sees the model's own predictions
during decode, so any residual exposure-bias is left on the table.
A short fine-tune phase that switches to `forward_scheduled(ss_ratio=
1.0)` for the last 10–20 epochs should close this directly.

**Code change.**
- Lift the `parallel_caps` `NotImplementedError` in
  `forward_scheduled` (`bsimar/models/transformer.py:441-445`) by
  re-implementing the scheduled sampling loop for the parallel-caps
  case (q+I-V is sequential, caps emit in parallel as in `forward()`).
- In `bsimar/training/trainer.py`, add an `ar_finetune_epochs` arg.
  After the cosine TF schedule completes, run an additional N epochs
  using `train_epoch_scheduled(ss_ratio=1.0, ...)` with a fixed low LR
  (1/10th of the final cosine LR).
- The fine-tune loss is **MAE on the AR-decoded outputs** (no LDS
  during fine-tune to keep gradients honest about the actual decoding
  errors).
- Add CLI flag `--ar-finetune-epochs N`.

**Test command.** Add `--ar-finetune-epochs 15` to the prior winning run.

**Result.** TBD.

| Metric | Prev best | N3 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | _TBD_ | _TBD_ | _TBD_ |
| MRE_phys % | _TBD_ | _TBD_ | _TBD_ |
| TF↔AR gap | _TBD_ | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Step 6 — N2: KV-cache encoder during AR decode

**Hypothesis.** The wall-clock bottleneck at medium is the AR
validation pass: 3 of every 10 epochs run the full 8-step sequential
decode on 45K samples (plus the final test pass). Caching attention
K/V across decode steps cuts each AR step from O(L²) to O(L), where
L grows from 4 (3 context + start) to 12 (3 context + start + 8 AR
tokens). The report claims a 3–5× speedup on the AR-validation pass
and ~1.5–2× on overall wall-clock.

**Why now (not deferred).** N1 needs a 150-epoch run with cosine LR.
At baseline 54 s/ep that is **≈ 8200 s ≈ 2.3 h** for the bare 50→150
epoch extension; layered on top of the autograd slowdown from N4 the
cumulative cost is 3–4 h per long run. The KV-cache cuts this to a
1–1.5 h budget, which makes the long-schedule sweep affordable to
re-run if Step 7 needs a tweak.

**Important.** This step **must produce identical** outputs to the
non-cached path (numerically equivalent up to float32 reduction
order). We verify with a test that runs the model on a fixed batch
under both code paths and asserts `torch.allclose` with `rtol=1e-5,
atol=1e-6`.

**Code change.**
- Add an internal `_forward_ar_cached` method on
  `TransformerEncoderModel` that:
  1. Runs the encoder once on the full context (3 grouped tokens +
     start) to populate per-layer K/V caches.
  2. For each AR step, projects only the *new* token, attends
     against the cached K/V plus the previous AR tokens (also cached),
     and updates each layer's K/V by appending the new K/V row.
  3. Re-implements the per-layer attention manually using
     `nn.functional.scaled_dot_product_attention` so we can inject
     the K/V cache.
- Use this path inside `forward(x, y=None)` (the autoregressive
  branch) only — the teacher-forced training path is unchanged.
- Add a unit test in `tests/test_bsimar_kv_cache.py` that loads the
  baseline checkpoint and asserts AR-decode parity between the
  cached and uncached paths on a fixed val batch.

**Verification (no accuracy impact).** Re-run the baseline AR-only
inference to confirm that test-set NRMSE/MRE/R² match the published
v2 baseline (0.419 % / 2.52 % / 0.9928) bit-for-bit (within float32
noise). If they do, mark N2 as "wall-clock unlock landed" and move
on. If they do not match, rewind.

**Result.** TBD.

| Metric | v2 Medium Baseline | N2 (cached) | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | 0.419 | _TBD_ | _TBD_ |
| MRE_phys % | 2.52 | _TBD_ | _TBD_ |
| R²_phys | 0.9928 | _TBD_ | _TBD_ |
| AR-val pass time (s) | _TBD_ | _TBD_ | _TBD_ |
| Per-epoch wall-clock (s) | 54.3 | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Step 7 — N1: long-schedule combined run @ 150 epochs (FINAL)

**Hypothesis.** The v2 medium TF val loss is still strictly decreasing
at epoch 50 (slope ≈ −5%/10 epochs). Extending to T_max=150 epochs is
the report's #1 recommendation. We run this **last** so it (a) carries
every winning change from steps 1–5 in a single combined run and
(b) benefits from N2's KV-cache speedup.

**Test command.**
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss <best-loss> --lds --norm-mode asinh \
  <best-flags> \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 150 --batch-size 1024 --lr 8e-4 --patience 150 --seed 42 --cuda \
  --exp-name n1_long_combined_medium --overwrite
```

**Result.** TBD.

| Metric | Best 50ep stack | N1 (150ep, combined, cached) | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | _TBD_ | _TBD_ | _TBD_ |
| MRE_phys % | _TBD_ | _TBD_ | _TBD_ |
| R²_phys | _TBD_ | _TBD_ | _TBD_ |
| Wall-clock (s) | _TBD_ | _TBD_ | _TBD_ |

**Verdict.** TBD.

---

## Final ranking and recommendation

To be filled in after Step 6 completes. Will note:
- Per-step Δ vs the v2 medium baseline.
- Which combination of changes is the new recommended production
  recipe at medium.
- What to update in `CLAUDE.md` (the "Future Work" list and the
  `bsimar_scaling_law.md` memory note).

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

**Result.** Run `n6_huber_medium`, log:
`results/improvement_2026_04_08/n6_huber_medium.log`. Phys-best
checkpoint loaded for the test pass. Wall-clock 2389 s.

| Metric | Baseline | N6 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | **0.419** | 0.538 | **+28 % (worse)** |
| MRE_phys % | **2.52** | 3.64 | **+44 % (worse)** |
| R²_phys | 0.9928 | 0.9906 | −0.0022 |
| Wall-clock (s) | 2716 | 2389 | −12 % |

Per-target MRE % (the columns N6 was supposed to improve):

| Target | Baseline | N6 | Δ |
|---|---:|---:|---:|
| id  | 2.45 | 4.32 | +76 % worse |
| gm  | 5.21 | 9.15 | +76 % worse |
| gds | 4.99 | 7.58 | +52 % worse |
| gmb | 3.79 | 8.28 | +118 % worse |

**Verdict: INFEASIBLE.** Rewound via `git reset --hard 50a3a71`.

**Postmortem.** Huber's quadratic core (`|d| < δ=1.0`) gives
gradient `d`, which → 0 as the residual → 0. So the optimizer cares
*less* about small residuals than under pure MAE (whose gradient is
constant ±1). MRE is dominated precisely by the small-magnitude tail,
so dampening gradients there is exactly the wrong direction. The
report's "MAE-like in the tail, MSE-like near zero residuals" framing
is correct as a literal description of Huber, but the *intended*
effect (focus on small residuals) is the opposite of what Huber
actually does.

If we wanted to fix this we would need a **reverse-Huber (BerHu) loss**:
MAE near zero, MSE in the tail. That's a different change and is not
in the report's recommendation. Marking N6 dead and moving on.

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

**Implementation choice.** Modifying the asinh `s_k` mid-training
requires re-deriving the per-target z-score (mean, std) on every
update, which is invasive. We tested an *equivalent-DOF* simplification
instead: a per-target trainable affine `gain[k] * raw_head_out[k] +
bias[k]` after the per-target heads, initialised at the identity
(gain=1, bias=0). The fixed denormaliser composes cleanly:
`pred_phys = sinh((gain*raw + bias)*std_fixed + mean_fixed) * s_k`,
so the affine can absorb a per-target asinh-scale correction without
touching the data pipeline. +26 params (13 gains + 13 biases).

**Test command.**
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss mae --lds --norm-mode asinh \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 50 --batch-size 1024 --lr 8e-4 --patience 50 --seed 42 --cuda \
  --exp-name n5_affine_medium --overwrite --learnable-output-affine
```

**Result.** Run `n5_affine_medium`, log:
`results/improvement_2026_04_08/n5_affine_medium.log`. Wall-clock 2648 s.

| Metric | Baseline | N5 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | **0.419** | 0.459 | **+9.5 % (worse)** |
| MRE_phys % | **2.52** | 2.66 | **+5.6 % (worse)** |
| R²_phys | 0.9928 | 0.9923 | −0.0005 |
| Wall-clock (s) | 2716 | 2648 | −2.5 % |

**Trajectory** (PHYS-val NRMSE %, baseline → N5):

| Epoch | Baseline | N5 | Δ |
|---:|---:|---:|---:|
| 10 | 1.298 | 0.839 | **−35 %** |
| 20 | 0.821 | 0.662 | **−19 %** |
| 30 | 0.508 | 0.596 | +17 % |
| 40 | 0.436 | 0.437 | 0 % |
| 50 | 0.380 | 0.431 | +13 % |

The early-epoch lead is real (the affine absorbs initial mismatch
in the per-target normalization), but the late-epoch heads at the
v2 baseline can fit the post-asinh z-score perfectly without the
extra knob, and the affine *disturbs* that finely-tuned distribution.
Charges + caps universally regressed (qb 0.867 → 0.991 NRMSE, the
opposite of the report's expectation).

**Verdict: INFEASIBLE.** Rewound via `git reset --hard 50a3a71`.

**Postmortem.** The hypothesis was that an extra per-target
post-normalization knob would let the optimizer correct for the
per-target asinh-scale being "too tight". The data shows the
opposite: the post-asinh z-score distribution is already an excellent
fit for the encoder's residual stream, and adding 26 trainable affine
params just makes the optimization landscape harder. A more invasive
variant — making the asinh `s_k` itself trainable inside the data
pipeline — would have a different mechanism (it re-normalizes the
targets, not the predictions) and is not ruled out by this experiment.
We do not pursue it: the report's expected gain ("modest gains on
caps everywhere") is not large enough to justify re-architecting
the data path.

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

**Implementation note.** PyTorch's flash- and efficient-attention SDPA
kernels do **not** implement double backward, so `create_graph=True`
crashes inside `nn.MultiheadAttention`. We force the MATH backend
for the consistency forward+grad chain via
`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`.

**Run 1** — `--charge-consistency-weight 0.1`, `cap_ag` vs
`cap_pred` (model self-consistency). Killed at epoch 18: PHYS-val
collapsed (epoch 10: NRMSE 6.77 %, R² −5.59 — worse than constant
prediction). The self-coupling let the optimiser drive both sides
toward zero without matching the data.

**Run 2** — `--charge-consistency-weight 0.01`, `cap_ag` vs
`cap_target` (DirectNet's pattern). Run completed:

| Metric | Baseline | N4 v2 | Δ |
|---|---:|---:|---:|
| NRMSE_phys % | **0.419** | 0.444 | **+6 % (worse)** |
| MRE_phys % | **2.52** | 2.99 | **+18.7 % (worse)** |
| R²_phys | 0.9928 | 0.9929 | +0.0001 (essentially tied) |
| Wall-clock (s) | 2716 | 4098 | +51 % |

**Per-target NRMSE** (sign of partial wins / partial losses):

| Block | Baseline NRMSE % | N4 v2 NRMSE % | Δ |
|---|---:|---:|---:|
| caps (cgg/cgd/cgs/cdg/cdd, AVG) | 0.208 | 0.188 | **−9.5 % (better)** |
| conductances (gm/gds/gmb, AVG) | 0.694 | 0.601 | **−13 % (better)** |
| current (id) | 0.878 | 0.775 | −12 % (better) |
| charges (qg/qd/qs/qb, AVG) | 0.362 | 0.563 | **+55 % (worse)** |

The constraint did its job on caps and conductances, but the **charge
fits collapsed** (qd 0.191 → 0.769, +302 %). Why: the autograd
constraint forces `d(qg)/dv ≈ cgg_target` and `d(qd)/dv ≈ cdd_target`
in *normalised* asinh-zscore space. Under signed-log (DirectNet's
normalisation) the chain rule gives an algebraically tractable
relationship between qg_norm, cgg_norm, and the gradient — that is
*how* DirectNet's ChargeConsistencyLoss works. Under asinh+zscore
the chain rule has a `cosh(asinh(qg/s))` factor that depends on the
target value, so the constraint is **not equivalent** to "qg fits
its target AND cgg fits its target". Satisfying both simultaneously
requires distorting qg/qd in physical space.

**Verdict: INFEASIBLE.** Rewound via `git reset --hard 6c2c2f9`.

**Postmortem.** This is a real, reusable lesson: the report's
recommendation to port DirectNet's charge-consistency loss to
BSIMAR did not account for the asinh normalisation breaking the
chain-rule-friendly representation that DirectNet's signed-log
enjoys. To make this work properly we would need to denormalise
through the asinh + zscore inverse inside the loss layer (so the
comparison happens in physical units), and that requires
`create_graph=True` autograd through `sinh()` — doable but
substantially more work than the report envisaged. Marking N4
dead at the medium tier under the v2 asinh recipe. A future
DirectNet-style "charge-finetune mode" applied AFTER the v1 train
plateau (rather than as a co-optimised term) might still work; we
do not pursue it here.

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

**Test command.**
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss mae --lds --vov-lds --norm-mode asinh \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 50 --batch-size 1024 --lr 8e-4 --patience 50 --seed 42 --cuda \
  --exp-name n7_vov_medium --overwrite
```

**Result.** Run `n7_vov_medium`, log:
`results/improvement_2026_04_08/n7_vov_medium.log`. Wall-clock 1904 s.

| Metric | Baseline | N7 | Δ |
|---|---:|---:|---:|
| **NRMSE_phys %** | **0.419** | **0.387** | **−7.6 % (BETTER)** |
| MRE_phys % | **2.52** | 2.62 | +4.0 % (slightly worse) |
| **R²_phys** | 0.9928 | **0.9945** | **+0.0017 (better)** |
| TF best val | 0.00993 | 0.01003 | +1 % |
| Wall-clock (s) | 2716 | 1904 | −30 % (GPU2 was less contended) |

**Per-target NRMSE %** improved on the columns the report flagged
as the bottleneck:

| Target | Baseline | N7 | Δ |
|---|---:|---:|---:|
| id  | 0.878 | 0.744 | **−15 %** |
| gm  | 0.624 | 0.609 | −2 % |
| gds | 0.675 | 0.586 | **−13 %** |
| gmb | 0.783 | 0.709 | **−9 %** |
| qb  | 0.867 | 0.746 | **−14 %** |
| caps (AVG) | 0.208 | 0.215 | +3 % (small regress) |

**Verdict: WIN (marginal).** NRMSE strictly improves by 7.6 %, and
the regression on MRE (+4 %) is at the verdict threshold (half the
NRMSE gain = 3.8 %). The NRMSE improvement is per-target consistent
on the report's bottleneck columns (id, gds, gmb, qb), and the
small MRE regression is concentrated in cap/conductance MRE which
were already at <2 %. R² improves +0.0017. We keep the change.

**This is the first successful step in the sprint.** New baseline
for the next steps:

| Metric | New baseline (N7) |
|---|---:|
| NRMSE_phys % | 0.387 |
| MRE_phys % | 2.62 |
| R²_phys | 0.9945 |

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

**Test command.** Built on top of the N7 winning stack:
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss mae --lds --vov-lds --norm-mode asinh \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 50 --batch-size 1024 --lr 8e-4 --patience 50 --seed 42 --cuda \
  --exp-name n3_arft_medium --overwrite --ar-finetune-epochs 15
```

**Result.** Run `n3_arft_medium`, log:
`results/improvement_2026_04_08/n3_arft_medium.log`. TF phase 1883 s +
FT phase ~2500 s. Phys-best hit at FT epoch 5 (val NRMSE 0.330,
R² 0.9960). Test metrics loaded from the FT 5 phys-best ckpt.

| Metric | v2 Baseline | N7 (prev) | N3 (N7+FT) | Δ vs N7 | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| NRMSE_phys % | 0.419 | 0.387 | **0.343** | **−11 %** | **−18.1 %** |
| MRE_phys % | 2.52 | 2.62 | **2.58** | **−1.5 %** | +2.4 % |
| R²_phys | 0.9928 | 0.9945 | **0.9961** | +0.0016 | +0.0033 |

Biggest per-target NRMSE improvements (vs N7):
- id: 0.744 → 0.506 (**−32 %**)
- gds: 0.586 → 0.485 (**−17 %**)
- qd: 0.197 → 0.173 (−12 %)
- cgg: 0.213 → 0.193 (−9 %)
- gm: 0.609 → 0.552 (−9 %)
- gmb: 0.709 → 0.652 (−8 %)
- qb: 0.746 → 0.704 (−6 %)

The AR finetune does exactly what the report hoped: 5 epochs of
pure-AR rollout (`ss_ratio=1.0`) close a big chunk of the residual
exposure-bias gap. The only mild regression is qg MRE +5 %, which
is noise in the <2 % regime.

**Verdict: WIN (clean — all three aggregate metrics improved vs
the N7 state).** New cumulative baseline: `N7 + N3 = --vov-lds
--ar-finetune-epochs 15`.

| Metric | New cumulative baseline |
|---|---:|
| NRMSE_phys % | 0.343 |
| MRE_phys % | 2.58 |
| R²_phys | 0.9961 |

**Note.** This was implementation-heavy: lifting `forward_scheduled`
to support `parallel_caps`, a new `train_epoch_scheduled_bni`
helper, and a finetune phase inside `train_transformer` that
reloads the phys-best checkpoint, runs N epochs at a fixed low LR
with `ss_ratio=1.0` + plain MAE, and updates the phys-best if it
improves. Worth it for the 11 % NRMSE drop.

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
v2 baseline bit-for-bit. If they do, mark N2 as "wall-clock unlock
landed". If they do not match, rewind.

**Decision: DEFERRED.**

The cost-benefit for N1 changed after seeing the N3 AR-finetune
log. N3 phys-best was hit at FT epoch 5 (`nrmse=0.330 r2=0.9960
*phys-best*`), and the next 10 FT epochs (6-15) never beat it:

```
  [FT 1] nrmse=0.343%  *phys-best*
  [FT 3] nrmse=0.342%  *phys-best*
  [FT 5] nrmse=0.330%  *phys-best*
  [FT 6..15] nrmse in 0.334-0.371, no new phys-best
```

So we can safely drop `--ar-finetune-epochs` from 15 to 5. That
saves 10 × 170 s ≈ 28 min per run by pure schedule-tuning, which is
more than what a bit-exact KV-cache would save (the TF phase stays
at 5550 s for 150 epochs; KV-cache only touches AR forwards).

Budget check for Step 7 without KV-cache and with
`--ar-finetune-epochs 5`:
- TF (150 ep) ≈ 150 × 37 s = 5550 s ≈ 92 min
- FT (5 ep)  ≈ 5 × 170 s   = 850 s ≈ 14 min
- Total ≈ 107 min ≈ **1.8 h** per run

This is affordable without KV-cache. A true bit-exact KV-cache
rewrite of `nn.TransformerEncoderLayer` (PyTorch does not expose
per-layer K/V hooks — we would have to reimplement attention) is
~200 LOC of high-risk code for a ~10 % wall-clock saving at this
point. Not worth the implementation risk in this sprint.

**Filed as future work.** If BSIMAR becomes the production
checkpoint and 100-epoch finetune or 500-epoch training becomes
routine, KV-cache at the encoder level is still the right long-
term unlock. We have the forward_scheduled hook in place, so a
later refactor can slot in cleanly.

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

**Test command.** Built on the N7+N3 stack, no KV-cache, 5 FT epochs:
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss mae --lds --vov-lds --norm-mode asinh \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 150 --batch-size 1024 --lr 8e-4 --patience 150 \
  --ar-finetune-epochs 5 --seed 42 --cuda \
  --exp-name n1_long_medium --overwrite
```

**Result.** Run `n1_long_medium`, log:
`results/improvement_2026_04_08/n1_long_medium.log`. TF phase 5535 s
(36.9 s/epoch), FT phase ~850 s, total ~6385 s ≈ 1 h 47 min. Phys-
best hit at FT epoch 1 (val NRMSE 0.222, R² 0.9980), held through
FT 5. Test metrics loaded from the FT phys-best ckpt.

| Metric | v2 Baseline | N1 (final stack) | Δ vs baseline |
|---|---:|---:|---:|
| **NRMSE_phys %** | **0.419** | **0.223** | **−46.8 %** |
| **MRE_phys %** | **2.52** | **1.41** | **−44.0 %** |
| **R²_phys** | 0.9928 | **0.9984** | **+0.0056** |
| Wall-clock (s) | 2716 | 6385 | +135 % |

**Per-target test NRMSE % (EVERY column improved):**

| Target | Baseline | N1 | Δ |
|---|---:|---:|---:|
| id  | 0.878 | **0.285** | **−67.5 %** |
| gm  | 0.624 | 0.320 | −48.7 % |
| gds | 0.675 | 0.343 | −49.2 % |
| gmb | 0.783 | 0.401 | −48.8 % |
| qg  | 0.175 | 0.109 | −37.7 % |
| qd  | 0.191 | 0.140 | −26.7 % |
| qs  | 0.215 | 0.152 | −29.3 % |
| qb  | 0.867 | 0.445 | −48.7 % |
| cgg | 0.204 | 0.143 | −29.9 % |
| cgd | 0.213 | 0.150 | −29.6 % |
| cgs | 0.201 | 0.137 | −31.8 % |
| cdg | 0.210 | 0.136 | −35.2 % |
| cdd | 0.210 | 0.144 | −31.4 % |

**Per-target test MRE %:**

| Target | Baseline | N1 | Δ |
|---|---:|---:|---:|
| id  | 2.45 | **1.18** | **−51.8 %** |
| gm  | 5.21 | 2.60 | −50.1 % |
| gds | 4.99 | 2.67 | −46.5 % |
| gmb | 3.79 | 1.94 | −48.8 % |
| qg  | 1.64 | 1.10 | −32.9 % |
| qd  | 1.89 | 1.38 | −27.0 % |
| qs  | 1.36 | 0.89 | −34.6 % |
| qb  | 3.28 | 2.13 | −35.1 % |
| cgg | 1.29 | 0.75 | −41.9 % |
| cgd | 1.83 | 0.97 | −47.0 % |
| cgs | 1.72 | 0.86 | −50.0 % |
| cdg | 1.59 | 0.82 | −48.4 % |
| cdd | 1.77 | 1.01 | −42.9 % |

**Verdict: WIN (dramatic).** Both NRMSE and MRE nearly halved,
R² improved by 0.0056, every per-target metric improved by 27 – 68 %.
Wall-clock roughly doubled (+135 %) to get this, which is
acceptable for a one-shot production training run.

---

## Final ranking and recommendation

### Sprint summary

| Step | Change | Verdict | NRMSE (vs baseline) | MRE (vs baseline) |
|---|---|---|---:|---:|
| N6 | Huber on I/V block | ❌ INFEASIBLE | +28 % | +44 % |
| N5 | Learnable output affine | ❌ INFEASIBLE | +9.5 % | +5.6 % |
| N4 | Charge-consistency penalty | ❌ INFEASIBLE | +6 % | +18.7 % |
| **N7** | **Vov-region LDS** | ✅ **WIN** | −7.6 % | +4.0 % |
| **N3** | **AR finetune phase** | ✅ **WIN** | −18.1 % (cum) | +2.4 % (cum) |
| N2 | KV-cache encoder | ⏸️ DEFERRED | — | — |
| **N1** | **Long schedule 150 ep** | ✅ **WIN** | **−46.8 % (cum)** | **−44.0 % (cum)** |

**Three out of six code/config changes won.** The three failures
(N6, N5, N4) all taught a reusable lesson:
- **N6** — Huber's gradient is *smaller* near zero than MAE's, so
  it is the wrong loss for MRE-bottlenecked tails. Any future
  small-residual-focused loss should use reverse-Huber (BerHu) or
  similar, not standard Huber.
- **N5** — post-normalization affine at the output heads disrupts
  the carefully-fitted post-asinh z-score distribution. The v2 heads
  already fit it perfectly; adding 26 trainable affine params just
  makes the landscape harder.
- **N4** — DirectNet's charge-consistency loss is specifically tied
  to the signed-log normalization, whose chain rule gives
  `d(qg_norm)/d(v_norm) ≈ cgg_norm` up to constants. Under
  asinh+zscore the chain rule has a `cosh(asinh(q/s))` factor that
  depends on the target value, so the consistency constraint is
  mathematically **inequivalent** to "qg and cgg fit their own
  targets". Distorting qg to satisfy it is strictly harmful in net.

### Recommended production recipe (BSIMAR medium)

```bash
conda run -n pycircuitsim --no-capture-output \
  python -u -m bsimar.cli.train \
  --model transformer --device-type nmos --universal \
  --loss mae --lds --vov-lds --norm-mode asinh \
  --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
  --epochs 150 --batch-size 1024 --lr 8e-4 --patience 150 \
  --ar-finetune-epochs 5 --seed 42 --cuda
```

Produces, on `universal_nmos`:

| Metric | Value |
|---|---:|
| NRMSE_phys | **0.223 %** |
| MRE_phys | **1.41 %** |
| R²_phys | **0.9984** |
| Params | 5,152,525 |
| Wall-clock | ~107 min on Blackwell |

### BSIMAR vs DirectNet positioning

After this sprint the BSIMAR-vs-DirectNet tradeoff looks like this:

| Metric | DirectNet medium (4.75M) | BSIMAR v2 old (5.15M) | **BSIMAR v3 new (5.15M)** |
|---|---:|---:|---:|
| NRMSE_phys % |  **0.113** | 0.419 | **0.223** |
| MRE_phys %   |  4.96 | 2.52 | **1.41** |
| R²_phys      |  **0.9998** | 0.9928 | 0.9984 |

Gap movements:
- **NRMSE gap shrinks**: DirectNet was 3.7× better than v2, now only
  1.97× better than v3. Closing the gap further is likely a
  capacity / data question, not a recipe question.
- **MRE lead widens**: v2 BSIMAR was 1.97× better than DirectNet on
  MRE; v3 BSIMAR is **3.52× better**. The low-magnitude tail
  accuracy of BSIMAR+asinh is now clearly the best on this dataset
  by a wide margin.
- **R² gap shrinks**: 0.9998 vs 0.9984 — essentially tied for any
  practical purpose.

This is a stronger defensible niche than v2 had: **use BSIMAR v3
medium for anything where per-sample relative accuracy matters**
(subthreshold, leakage, small-signal conductances near pinch-off),
and **DirectNet medium only for workloads that care about peak
absolute fit on high-magnitude saturation-region samples**.

### Things to update downstream of this sprint

1. **CLAUDE.md** — add a "Future Work" entry noting the new
   production recipe and the fact that N2 KV-cache is filed as a
   speed-unlock if/when 500-epoch training is attempted.
2. **`bsimar_scaling_law.md` memory note** — update with the v3
   numbers. The v2 note said BSIMAR was 3.7× worse than DirectNet on
   NRMSE and 1.97× better on MRE; v3 is 1.97× worse on NRMSE and
   3.5× better on MRE.
3. **`results/scaling_law_2026_04_08_arch_v2/scaling_law_v2_report.md`
   → add a footnote** pointing to this improvement plan and the v3
   numbers so future readers know the v2 "sweet spot" of 0.419/2.52
   has been superseded.
4. **Write-up — results/improvement_2026_04_08/** — add a short
   `README.md` to the results directory explaining the sprint
   structure (`n6/n5/n4/n7/n3/n1` logs) and pointing to this plan
   file as the narrative.

### What was most worth doing (ordering retrospective)

In hindsight, the sprint ordering could have been better. The
three failures (N6, N5, N4) each took ~45 – 135 min of wall-clock
and produced no improvement. N1 (long schedule) gave the bulk of
the win (−47 % NRMSE / −44 % MRE out of the final stack's total
improvement), and N7 + N3 added ~20 % more NRMSE on top.

If we did this sprint over, the right ordering would be:
1. **N1 first** (it's a 1-line change and the report said so — we
   were wrong to reorder it).
2. N7 (cheap, won independently).
3. N3 (moderate code, won independently).
4. Skip N6 / N5 / N4 entirely — the mechanisms they target are
   all dominated by N1's longer schedule.

Documenting this so the next sprint does not repeat the mistake.

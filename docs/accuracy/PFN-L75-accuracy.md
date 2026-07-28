# PFN / TabPFN (LEVEL=75) — family report

**What it is:** a faithful scaled-down port of **TabPFN-v3**'s in-context (ICL)
tabular transformer into the `bsimar` stack, netlist `LEVEL=75`. The **research**
family: trained from scratch per-tech on the existing datasets and gated on the
unmodified V6.8.0 harness.

**Status:** `clean@small` (0.69 M params) is **11/16 strict** across
OMP ∈ {1,2,4} with zero flips — the strongest *clean* small tier on record, at
the smallest parameter count of the three families.

Cross-cutting numbers live in [`by-tech.md`](by-tech.md),
[`by-scale.md`](by-scale.md) and [`by-recipe.md`](by-recipe.md); gate definitions
in [`methodology.md`](methodology.md). This file carries only what is specific to
the in-context transformer.

**Covers:** V6.10.0 (port + full gate campaign, merged `6b3a890`) → V6.11.0
(TSMC6 sweep) → V6.13.0 (post-gds-fix re-gate) → V7.1.0 (device/AC re-gate).

---

## 1. What was ported, and what was deliberately changed

The TabPFN v3 pipeline (upstream `tabpfn_v3.py`): circular-shift feature
grouping → cell embedding → per-column **feature-distribution embedder**
(SetTransformer-style induced attention over ROWS) → **column aggregator**
(transformer over the FEATURE axis + CLS-token cross-attention readout, RoPE) →
**ICL transformer** (rows attend to context rows only) → head.

`TabPFNCompact` (`external_compact_models/bsimar/models/tabpfn.py`) keeps all
three signature stages at compact scale plus the v3 details that matter:
RMSNorm everywhere, ff_factor 2, zero-init residual outputs, query-scaled
softmax (`SoftmaxScalingMLP`), bias-free attention, trunc-normal CLS/inducing
vectors. The 7 inputs are grouped by circular shifts {1,2,4} (a perfect
difference cover mod 7) and the tech code enters as an **8th learned column
token** (`nn.Embedding`, local vocab + tail UNKNOWN, Rule 16).

**Two deliberate deviations from stock TabPFN:**

1. **Frozen learned context** instead of a user-supplied train set. During
   training each step samples a fresh K-row context from the training split
   (episodic ICL — the model genuinely learns to *read* a context). At save time
   a stratified context (tech-code × 8 asinh-|id| quantile bins) is frozen into
   checkpoint buffers `ctx_x/ctx_y/ctx_tc` — **float32; integer buffers corrupt
   under EMA**. At inference the context-side activations (per-block inducing
   hidden states + per-layer ICL K/V) are cached once per loaded checkpoint, so
   each Newton query costs only cross-attention reads.
2. **Direct 13-output value head** instead of the 5000-bin bar distribution.
   Newton-Raphson needs cheap, smooth first-order autograd (gm/gds/gmb and the
   dQ/dV caps come from the base class's autograd Jacobian), and the trainer is
   MAE-on-asinh-normalized values.

**Why not the pretrained 58 M model in the solver:** single-scalar-target,
numpy/`inference_mode` predict path, per-eval cost orders of magnitude over
budget for transient gates. It is used instead as a device-level in-context
baseline (§4).

### Integration surface

* `bsimar`: `models/tabpfn.py`, `TabPFNConfig`, `train_tabpfn` +
  `_stratified_context`, CLI `--model tabpfn` + S/M/L/XL presets + guards (aux
  losses rejected in phase 1), tag `pfn`.
* Simulator: `pycircuitsim/models/mosfet_pfn.py` — reads the **required**
  `_config.npz` sidecar (the arch cannot be rebuilt without it),
  `output_layout="standard"`, NMOS `-id` / PMOS `+id`; `solver.py` type
  enumerators; `parser.py` LEVEL=75 dispatch + per-tech cascade
  `tsmc{X}_pfn_{large,medium,small,xl}` + env pins
  `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}` + `PYCIRCUITSIM_NN_FORCE_LEVEL=75`.
* Drivers: every recipe/gate/benchmark script takes `MODEL=tabpfn`; results land
  in `results/pfn_bench/`.
* Behaviour-preserving for DN/TF: the sidecar-save guard was generalized
  (DirectNet passes `arch_config=None`), grad-clip threaded as a flag, LEVEL=73/74
  paths untouched (regression-checked: production `tsmc7_dn_large` VTC unchanged).

## 2. Presets and training

| preset | E | n_ind | dist | agg | n_cls | W | icl | heads | params | ctx K | epochs | lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **small** | 64 | 32 | 2 | 2 | 2 | 128 | 3 | 4 | **686 k** | 1024 | 80 | 6e-4 |
| medium | 96 | 32 | 3 | 3 | 2 | 192 | 4 | 6 | 2.03 M | 2048 | 150 | 5e-4 |
| large | 128 | 48 | 3 | 3 | 2 | 256 | 6 | 8 | 4.65 M | 2048 | 150 (+amp) | 4e-4 |
| **xl** (V7.1.0) | 192 | 64 | 4 | 4 | 2 | **384** | 9 | 12 | **14.86 M** | 2048 | 150 (+amp) | 3e-4 |

The **xl** tier is new in V7.1.0 and completes the family's 4-scale capacity
curve. Its 14.86 M mirrors BSIM-AR xl's 14.81 M, and W = embed_dim ×
n_cls_tokens = 384 equals BSIM-AR xl's `d_model`, so the top of the capacity
axis is directly comparable across families. lr is 3e-4 rather than large's
4e-4 because of the instability below: 5 of the 8 large-tier collapses were at
4e-4, and the xl ICL stack is 50 % deeper.

Clean recipe throughout (`--apply-filter off --swa-mode ema --seed 42`), per-tech
scope: 24 checkpoints at S/M/L (4 techs × N/P × 3 sizes), +8 for the V7.1.0 xl
tier, +8 more for TSMC6 — the TSMC7 repeat (`by-tech.md` §5). Small/medium are
dataloader-bound (~118 s/epoch on shared 4090s); large is compute-bound (~12 min
/epoch at 2 jobs/GPU) and trains a 150-epoch cosine with bf16 autocast.
EMA-buffer audit: the frozen context buffers are bit-exact through the EMA wrap.

### Large-tier training instability

**8 divergence-collapse events** across the large wave: train loss explodes
0.02 → 0.77 at epoch 25–80 and the network collapses to a constant predictor,
with grad-clip 1.0 active. 5/8 jobs at lr 4e-4, and **3 of the lr 3e-4 retries
diverged again — the instability is not purely LR-bound.** Suspect class:
unconstrained attention-logit scaling (the `SoftmaxScalingMLP` base term)
compounding through the deeper W=256 ICL stack; small/medium never diverged.

A diverged run early-stops (patience 50) and banks its pre-divergence EMA best,
which sits in the same val band (0.0020–0.0029) as the healthy completions, so
all 8 shipped checkpoints are tier-representative (final mix: 3 @4e-4, 5 @3e-4,
epochs 82–150). Stabilization (logit clamp / warmup) is future work.

### Device fidelity on the trainer's own held-out split

Rule-13 metrics in physical space, from the trainer (not the simulator), so
**unaffected by the gds fix** — but read them against `methodology.md` §8.1: the
split is a uniform row permutation over a dense per-bin lattice, so these are
optimistic relative to gate behaviour.

| tier | id MRE % (range over 8 ckpts) | all-target AVG MRE % | best val |
|---|---|---|---|
| small | 0.33 – 0.58 | 0.21 – 0.26 | ~0.0008–0.0012 |
| **medium** | **0.10 – 0.30** | **0.09 – 0.15** | ~0.0004–0.0005 |
| large | 0.39 – 1.39 | 0.39 – 0.66 | 0.0020–0.0029 |

Device fidelity peaks at **medium** — the best device-fidelity clean tier
measured in this project — and declines at `large`, where §2's instability caps
how far the 150-epoch cosine gets. Autograd ↔ finite-difference consistency: gm
and cgg agree to ~0.2 % relative.

## 3. The off-grid geometry corner — root-caused, capacity-repaired

`verify_nn_multi_tech_dc` fails the `nmos_nfin_10` Id-Vgs case on tsmc16
(small + medium) and tsmc12 (medium): NRMSE 27–31 %, a flat ~+20 %
strong-inversion over-prediction.

**NFIN = 10 is off-grid** — the tsmc12/16 datasets contain NFIN ∈ {2, 3, 4, 6,
20.9}, so the case probes the 6 → 21 interpolation gap. DirectNet's smooth global
MLP bridges it to gate precision; PFN's context-relative distribution embedding
anchors to the discrete NFIN values present in the context. **At `large` the
corner recovers to PASS (~6 % NRMSE)** — capacity does repair off-grid
interpolation, and it is the one axis where PFN's `large` tier wins.

Cheaper fix classes, both untried: denser NFIN sampling at data generation, or
**post-hoc context enrichment** — the frozen context is *data*, not
architecture, and training was episodic, so a re-frozen context is admissible
without retraining.

## 4. Pretrained-TabPFN in-context baseline (device level only)

Stock `tabpfn-v3-regressor-v3_20260506_ood.ckpt` (58 M), 10k-row stratified
context from the same train split, 4 estimators, per-target fits on the test
split (5k rows), identical asinh target space. Aggregated over the 8 (tech,
device) combos — a pure data-fitting measurement, so **unaffected by the gds
fix**:

| target | MRE % | R² | NRMSE % |
|---|---|---|---|
| id | 3.2 – 5.7 | 0.937 – 0.997 | 0.16 – 0.51 |
| qg | 0.69 – 0.90 | ≥0.9998 | ≤0.14 |
| qd | 0.91 – 1.10 | ≥0.9997 | ≤0.15 |
| qb | 0.31 – 0.85 | ≥0.997 | ≤0.43 |

Zero-training charges are impressive; current is ~10× worse than the
from-scratch PFN-small. **The from-scratch port beats the 58 M pretrained model
~10× on `id` with 1.2 % of the parameters.**

## 5. Current state and runtime

`TSMC6 ⚠` is the repeat column — the same recipe retrained on bit-identical
TSMC7 rows, scored /4, never inside the /16 (`by-tech.md` §5).

| tier | params | complex (single-run, post-fix) | strict | TSMC6 ⚠ /4 |
|---|---|---|---|---|
| **small** | **0.69 M** | **11/16** | **11/16, zero flips** | 3 |
| medium | 2.03 M | 11/16 | not swept | 2 |
| large | 4.65 M | 9/16 | not swept | 2 |
| xl | 14.86 M | *gating (V7.1.0)* | — | 3 |

Margins at `small`: tsmc12 ring 2.9–4.3 %, opamp 0.57 %, sram ≤1.5 %; tsmc16
ring 2.7–3.1 %, opamp 3.66 %, switchcap 3.8 %. `tsmc12-switchcap` sits at
5.10–5.32 % against a 5.0 % gate — a stable near-miss, the first candidate cell
for any charge-side improvement.

**Runtime** (CPU, 1 thread, cached frozen context, grouped `batch_eval`):
**15.6 ms/eval** — ~10× DirectNet, **4× faster than BSIM-AR**. A full 32-cell
evaluation (8 suites × 4 techs, PAR=16) completes in ~24 min per tier, versus
hours for BSIM-AR. The context-KV cache and the solver's batched pre-warm are
the levers.

**A distinction that has been overtaken:** PFN was the first family with no
observable OMP multistability. That was true when measured and is no longer a
differentiator — after the V6.13.0 gds fix *every* family is flip-free
(`methodology.md` §6). What PFN keeps is the smallest parameter count and the
best clean small-tier result; what it does not keep is a unique claim on
determinism.

## 6. Open work, in priority order

1. **The `xl` tier's gate result** — the first data point on whether PFN's
   *declining* capacity curve keeps declining past `large` or turns over. The
   other two families disagree here (DirectNet peaks at `large` then partially
   recovers under a curriculum at `xl`; BSIM-AR is flat), so PFN's answer is
   not predictable from them.
2. **The corridor curriculum has never been run on PFN.** `MODEL=tabpfn
   RECIPES=corroft SIZES=small` is fully wired. It is the lever that took
   BSIM-AR from 13 to 15/16 and it aims at exactly the four tsmc5/tsmc7 cells
   PFN fails. If the flip-free property survives the curriculum, PFN-small could
   contend with production at ~1/5 the parameters.
3. **Context enrichment for the NFIN gap** (re-freeze, no retrain) + denser NFIN
   sampling at data generation.
4. **`tsmc12-switchcap` is 0.3 pp from its gate.**
5. **Stabilize the large tier** (attention-logit clamp / warmup) — and watch
   whether `xl` inherits the same collapse mode.

## 7. Reproduction

```bash
# train (per tier)
MODEL=tabpfn RECIPES=clean SIZES="small medium" GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
MODEL=tabpfn RECIPES=clean SIZES="large" EXTRA_ARGS="--amp" GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
# evaluate — one tier at a time, never two recipe_eval dispatchers at once
MODEL=tabpfn RECIPES=clean SIZES="small" PAR=16 bash scripts/recipe_eval.sh
# strict OMP sweep, one cell
MODEL=tabpfn bash scripts/recipe_multirun_gate.sh clean small TSMC16 verify_complex_ring_osc
# single netlist
PYCIRCUITSIM_NN_FORCE_LEVEL=75 python main.py <deck with LEVEL=73/75 cards>
```

Campaign log: `docs/plans/2026-07-11-tabpfn-pfn-family.md`. Raw gates:
`results/pfn_bench/`, `results/a3_regate/pfn_*`, `results/v710_regate/pfn/`.
Full pretrained-baseline per-combo table: `pretrained_baseline_results.csv`
(regenerable).

# PFN / TabPFN (LEVEL=75) — Unified Accuracy Report

**Family:** PFN, a faithful scaled-down port of TabPFN-v3's in-context (ICL) tabular
transformer, netlist `LEVEL=75` — **research** family.
**Ground truth:** always NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI model.
**Consolidated:** 2026-07-24. **Covers:** V6.10.0 (port + full gate campaign) →
V6.11.0 (TSMC6 sweep).

> **This file supersedes and replaces** `docs/V6.10.0-tabpfn-pfn-report.md`. All of its
> content is carried here. The deleted file remains in git history (last present at
> commit `1fe1cdb`).

Sibling reports: `DirectNet-L73-accuracy.md` (LEVEL=73, production),
`BSIM-AR-L74-accuracy.md` (LEVEL=74). Shared methodology:
`DirectNet-L73-accuracy.md` §2 — identical gates, isolation and strict-OMP discipline.

---

## 0. Provenance

| Source campaign | Date | What it contributed |
|---|---|---|
| V6.10.0 | 2026-07-11/14 | The port itself + presets, device/complex/AC gates, pretrained baseline, runtime (branch `worktree-tabpfn-pfn`, merged `6b3a890`) |
| V6.11.0 | 2026-07-14/17 | TSMC6 clean capacity sweep (§9 — see the TSMC6≡TSMC7 correction) |

---

## 1. Headline

The TabPFN v3 architecture (Prior Labs' tabular in-context-learning transformer, local
copy at `/data2/shenshan/TabPFN-main`) was ported into the `bsimar` stack as a third NN
compact-model family, trained from scratch per-tech on the existing datasets, and gated
on the exact V6.8.0 harness.

| Metric | Result |
|---|---|
| **Best config** | **clean `small`, 0.69 M params** |
| **Complex gates** | **11/16 — STRICT across OMP∈{1,2,4} with ZERO flips** |
| Distinction | **the first family in this project with no observable OMP multistability** |
| Clean capacity curve on gates | **11 → 10 → 8 / 16** (small → medium → large) — *monotone decline* |
| Device fidelity peak | **medium** (id MRE 0.10–0.30 %) — the best device-fidelity clean tier measured in this project |
| Device CS-amp AC | 5/8 small · 0/8 medium · 5/8 large (**non-monotone**) |
| Opamp open-loop AC | 0/4 at every tier |
| Runtime | 15.6 ms/eval — ~10× DirectNet, **4× faster than BSIM-AR** |
| Large tier | **optimization-unstable** (8 divergence-collapse events) |

DirectNet (LEVEL=73) remains production; PFN is a validated research family whose sweet
spot is the **small** tier, with three identified architecture-specific behaviours:
off-grid geometry interpolation weakness (which capacity repairs), non-monotone AC, and
large-tier training instability.

---

## 2. What was ported, and what was deliberately changed

TabPFN v3 pipeline (upstream `tabpfn_v3.py`): circular-shift feature grouping → cell
embedding → per-column **feature-distribution embedder** (SetTransformer-style induced
attention over ROWS) → **column aggregator** (transformer over the FEATURE axis +
CLS-token cross-attention readout, RoPE) → **ICL transformer** (rows attend to context
rows only) → head.

`TabPFNCompact` (`external_compact_models/bsimar/models/tabpfn.py`) keeps all three
signature stages at compact scale, plus the v3 details that matter: RMSNorm everywhere,
ff_factor 2, zero-init residual outputs, query-scaled softmax (`SoftmaxScalingMLP`),
bias-free attention, trunc-normal CLS/inducing vectors. The 7 inputs are grouped by
circular shifts {1,2,4} (a perfect difference cover mod 7) and the tech code enters as
an 8th learned column token (`nn.Embedding`, local vocab + tail UNKNOWN, Rule 16).

**Two deliberate deviations from stock TabPFN:**

1. **Frozen learned context** instead of a user-supplied train set. During training,
   each step samples a fresh K-row context from the training split (episodic ICL — the
   model genuinely learns to read a context). At save time a stratified context
   (tech-code × 8 asinh-|id| quantile bins) is frozen into checkpoint buffers
   (`ctx_x/ctx_y/ctx_tc`, **float32 — integer buffers corrupt under EMA**). At inference
   the context-side activations (per-block inducing hidden states + per-layer ICL K/V)
   are cached once per loaded checkpoint; each NR query costs only cross-attention reads.
2. **Direct 13-output value head** instead of the 5000-bin bar distribution:
   Newton-Raphson needs cheap smooth first-order autograd (gm/gds/gmb and the dQ/dV caps
   come from the base class's autograd Jacobian), and the trainer is MAE-on-asinh-
   normalized values.

**Why not the pretrained 58 M model in the solver:** single-scalar-target,
numpy/`inference_mode` predict path, and per-eval cost orders of magnitude over budget
for transient gates. It is instead used as a device-level in-context baseline (§6).

### Integration surface

- `bsimar`: `models/tabpfn.py` (new), `TabPFNConfig`, `train_tabpfn` +
  `_stratified_context` (trainer), CLI `--model tabpfn` + S/M/L presets + guards (aux
  losses rejected phase-1), tag `pfn`.
- Simulator: `pycircuitsim/models/mosfet_pfn.py` (reads the **required** `_config.npz`
  sidecar, `output_layout="standard"`, NMOS `-id` / PMOS `+id` sign rules);
  `solver.py` — all three type-enumerators (Rule 5); `parser.py` — LEVEL=75 dispatch,
  per-tech resolver cascade `tsmc{X}_pfn_{large,medium,small,xl}`, stem→scope match, env
  pins `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`, `PYCIRCUITSIM_NN_FORCE_LEVEL=75`.
- Drivers: `recipe_train.sh`, `gate_matrix_iso.sh`, `recipe_eval.sh`,
  `benchmark_run_tests.sh`, `recipe_multirun_gate.sh` all take `MODEL=tabpfn` (TAG=pfn,
  PFN pins, FORCE_LEVEL=75); PFN results land in `results/pfn_bench/`.
- Behavior-preserving for DN/TF: sidecar-save guard generalized (DirectNet passes
  `arch_config=None`), grad-clip threaded as a flag, LEVEL=73/74 paths untouched
  (regression-checked: production `tsmc7_dn_large` VTC unchanged).

---

## 3. Presets and training

| preset | E | n_ind | dist | agg | n_cls | W | icl | heads | params | ctx K | epochs | lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 64 | 32 | 2 | 2 | 2 | 128 | 3 | 4 | 686k | 1024 | 80 | 6e-4 |
| medium | 96 | 32 | 3 | 3 | 2 | 192 | 4 | 6 | 2.03M | 2048 | 150 | 5e-4 |
| large | 128 | 48 | 3 | 3 | 2 | 256 | 6 | 8 | 4.65M | 2048 | 150 (+amp) | 4e-4 |

Clean recipe throughout: `--apply-filter off --swa-mode ema --seed 42`, per-tech scope
(local vocab), 24 checkpoints (4 TSMC techs × N/P × 3 sizes). Small/medium are
dataloader-bound (~118 s/epoch on shared 4090s); large is compute-bound (~12 min/epoch
at 2 jobs/GPU) → the large tier trains a 150-epoch cosine with bf16 autocast (`--amp`).
EMA-buffer audit: frozen context buffers bit-exact through the EMA wrap.

**Large-tier training instability.** 8 divergence-collapse events across the large wave
(train loss explodes 0.02→0.77 at epoch 25–80, network collapses to a constant
predictor; grad-clip 1.0 active): 5/8 jobs at lr 4e-4, and 3 of the lr 3e-4 retries
diverged again — **the instability is not purely LR-bound.** Suspect class:
unconstrained attention-logit scaling (`SoftmaxScalingMLP` base term) compounding
through the deeper W=256 ICL stack; small/medium never diverged. A diverged run
early-stops (patience 50) and banks its pre-divergence EMA best, which sits in the same
val band (0.0020–0.0029) as the healthy completions — so all 8 shipped checkpoints are
tier-representative. Final tier mix: 3 ckpts @4e-4, 5 @3e-4 (epochs 82–150).
Stabilization (logit clamp / warmup) is future work.

---

## 4. Device-level accuracy (test split, per-tech checkpoints)

Rule-13 metrics from the trainer's held-out test split (physical space):

| tier | id MRE% (range over 8 ckpts) | all-target AVG MRE% | best val |
|---|---|---|---|
| small | 0.33 – 0.58 | 0.21 – 0.26 | ~0.0008–0.0012 |
| **medium** | **0.10 – 0.30** | **0.09 – 0.15** | ~0.0004–0.0005 |
| large | 0.39 – 1.39 | 0.39 – 0.66 | 0.0020–0.0029 |

Device fidelity peaks at **medium** — the best device-fidelity clean tier measured in
this project to date — and declines at large (§3's instability caps how far the
150-epoch cosine gets). Autograd↔finite-difference consistency: gm and cgg agree to
~0.2 % relative (the gds path is guarded by the base class's floor/Vds-correction — see
the §10 caveat).

**Off-grid geometry corner (root-caused, capacity-repaired):**
`verify_nn_multi_tech_dc` fails the `nmos_nfin_10` Id-Vgs case on tsmc16 (small+medium)
and tsmc12 (medium): NRMSE 27–31 %, a flat ~+20 % strong-inversion over-prediction.
NFIN=10 is **off-grid** — the tsmc12/16 datasets contain NFIN ∈ {2,3,4,6,20.9}, so the
case probes the 6→21 interpolation gap. DirectNet's smooth global MLP bridges it to gate
precision; PFN's context-relative distribution embedding anchors to the discrete NFIN
values present in the context. **At large the corner recovers to PASS (~6 % NRMSE)** —
capacity does repair off-grid interpolation, the one axis where large wins. Cheaper
fix-classes: denser NFIN sampling in data generation, or post-hoc context enrichment
(the frozen context is *data*, not architecture — training was episodic, so a re-frozen
context is admissible without retraining). Left as future work.

---

## 5. Complex-circuit gates (4 circuits × 4 techs vs NGSPICE BSIM-CMG)

Single-run matrix (`recipe_eval.sh`, CPU-pinned, isolated scratch) — cells listed are
the **passing** ones:

| tier | tsmc5 | tsmc7 | tsmc12 | tsmc16 | total |
|---|---|---|---|---|---|
| **small** | sram, swcap | sram, swcap | ring, opamp, sram | ALL 4 | **11/16** |
| medium | sram, swcap | sram, swcap | ring, opamp, sram | ring, sram, swcap | 10/16 |
| large | sram, swcap | sram, swcap | ring, sram | ring, sram | 8/16 |

The clean capacity curve on gates **declines monotonically** (11 → 10 → 8). At large the
opamp rails on ALL four techs (tsmc12's 0.57 %-gain pass at s/m is lost) and the
tsmc12/16 switchcaps regress (5.30 % / 4.99 % + droop-fail) — value-surface basins
relocate with capacity, and the unstable optimization compounds it.

**Strict OMP∈{1,2,4} @small = 11/16 with ZERO flips** — every cell verdict identical at
all thread counts (`recipe_multirun_gate.sh`, 48 runs). No other family has shown a
flip-free strict matrix; the DN/TF campaigns repeatedly lost gates to OMP
multistability. Margins: tsmc12 ring 2.9–4.3 %, opamp 0.57 %, sram ≤1.5 %; tsmc16 ring
2.7–3.1 %, opamp 3.66 %, swcap 3.8 %.

**Failing cells and their class:**

- **tsmc5 / tsmc7 ring** (8.7–9.9 %) and **opamp** (gain railed, 100 % err): the same
  low-VDD steep-tech value-surface cells every clean recipe fails (DN clean large fails
  tsmc5-ring / tsmc7-opamp too). The V6.8.0 corridor curriculum (`corroft`) is the known
  lever — **untested on PFN** (phase 2).
- **tsmc12 switchcap**: 5.10–5.32 % vs a 5.0 % tolerance — a stable near-miss, not a
  rail; charge-side fidelity nearly suffices.
- **tsmc16 opamp rails at medium** (passed small at 3.66 %): capacity relocates NR
  basins, echoing the V6.6.5 tier-dependence finding.

Reference points (single-run clean recipes): DN small 7/16, DN large 13/16; production
DN crit30@large = 14/16 strict; BSIM-AR corroft@medium = 15/16 strict. **PFN clean small
= 11/16 strict is the strongest clean small tier on record**, but the clean ceiling
stays below the curriculum recipes.

---

## 6. Pretrained-TabPFN in-context baseline (device level only)

Stock `tabpfn-v3-regressor-v3_20260506_ood.ckpt` (58 M), 10k-row stratified context from
the same train split, 4 estimators, per-target fits on the test split (5k rows),
identical asinh target space. Aggregated over the 8 (tech, dev) combos:

| target | MRE% | R² | NRMSE% |
|---|---|---|---|
| id | 3.2 – 5.7 | 0.937 – 0.997 | 0.16 – 0.51 |
| qg | 0.69 – 0.90 | ≥0.9998 | ≤0.14 |
| qd | 0.91 – 1.10 | ≥0.9997 | ≤0.15 |
| qb | 0.31 – 0.85 | ≥0.997 | ≤0.43 |

Zero-training charges are impressive; current is ~10× worse than the from-scratch
PFN-small (0.33–0.58 % id MRE at 686 k params). **The from-scratch port beats the 58 M
pretrained model ~10× on `id` with 1.2 % of the parameters.**

---

## 7. AC small-signal and transient

- **Device CS-amp** (`verify_nn_ac`): small **5/8** (fails tsmc5-nmos 11.4 %,
  tsmc12-pmos 12.5 %, tsmc16-pmos 14.1 % magNRMSE — all magNRMSE; gain0/f3db mostly
  in-gate), medium **0/8**, large **5/8** — **non-monotone**; the medium-tier dQ/dV
  surface is the outlier, not a trend.
- **Opamp AC** (`verify_complex_opamp_ac`): **0/4 at every tier** — the harness's
  peak-gain bias probe rails (OP-MISBIAS) even where the DC opamp gate passes; the
  historically hardest cell class (BSIM-AR also fails it broadly).
- **Device transient** (`verify_nn_multi_tech_tran`): 4/4 at small and medium; 3/4 at
  large (tsmc12 fails).
- **Device DC** (`verify_nn_multi_tech_dc`): 3/4 · 2/4 · 4/4 (s/m/l — the NFIN corner, §4).

---

## 8. Runtime

CPU 1-thread per-eval (cached frozen context, `batch_eval` grouped):

| model | ms/eval |
|---|---|
| DirectNet large | 1.5 |
| **PFN small** | **15.6** |
| BSIM-AR medium | 61.5 |

The full 32-cell eval (8 suites × 4 techs, PAR=16) completes in ~24 min per tier — vs
hours for BSIM-AR. The context-KV cache + the solver's batched pre-warm are the levers;
`NN_BATCHED_EVAL=0` opt-out applies as usual.

---

## 9. TSMC6 — retired, and what it taught us

**TSMC6 was deleted from this repo on 2026-07-24.** It was never an independent
technology under BSIM-CMG — it is TSMC7 relabelled — so the V6.11.0 "TSMC6"
campaign was a *second training run on the TSMC7 data*, and its rows here were
a duplicate data point presented as a sixth technology.

Evidence (`docs/2026-07-21-systematic-audit.md` §D1, re-verified at deletion):

* `tsmc6_{nmos,pmos}.npz` were `array_equal` to `tsmc7_*` in `inputs`,
  `geometry`, `outputs` and `sample_class` over 1,816,830 / 2,187,292 rows —
  only `meta_tech_name` differed.
* The raw PDKs genuinely differ, but every differing key (`tmi_ver_lod`,
  `tmi_ver_isocpode`, `sfxmin`, `samax_c`, `wodx5akvth0`) is a TSMC
  TMI-proprietary extension with **zero occurrences** in the BSIM-CMG
  Verilog-A. Reproduced mechanically at deletion time: of 871 implemented
  parameter names parsed from the Verilog-A sources, `toxp` and `phig` are
  present and all five TMI keys are absent.
* Two LEVEL=72 Id-Vgs sweeps at identical geometry matched to the last digit.

**What was deleted:** 22 checkpoints, both datasets, `results/tsmc6_gate`, the
registry/driver/test entries, and the per-size TSMC6 tables that stood here.
They remain in git history (last present at commit `a96112a`). The raw vendor
PDK is kept but unreferenced. TSMC6 held tail codes 22-24, so nothing was
renumbered.

**The methodological result is worth keeping.** PFN read *flat* 2/4 across all three sizes on the duplicated data, agreeing with its TSMC7 rows — which, read correctly, is the reassuring case: the family that showed no TSMC6-vs-TSMC7 discrepancy is the one whose gates were already flip-free (§5). The families that disagreed across the duplicate were the ones with unstable opamp basins, which is the signature to watch for.

**The guard:** `bsimar.config.assert_tech_is_distinct(tech)` compares resolved
modelcards restricted to parameters BSIM-CMG implements, and refuses a tech
that collides with an existing one. It flags `tsmc6`↔`tsmc7` and confirms
tsmc5/7/12/16 are genuinely distinct. Run it *before* onboarding a technology —
the V6.9.0 onboarding gated TSMC6 9/9 DC and 14/14 transient, and passing those
gates told us nothing, because they were TSMC7's gates.
---

## 10. Verdict, routing, and caveats

- **Production unchanged**: DirectNet crit30@large (14/16 strict) remains the fast path;
  BSIM-AR corroft@medium (15/16 strict) the high-fidelity option.
- **PFN in one line**: best-in-class small-tier gates (11/16 strict) + total OMP
  determinism + mid-pack speed, with a clean-recipe ceiling below the curriculum
  families, a monotone-declining gate capacity curve (11/10/8), an off-grid geometry
  weakness that capacity repairs, and a large tier that is optimization-unstable.

| family | best config | params | complex strict | AC (device) | CPU ms/eval |
|---|---|---|---|---|---|
| DirectNet (73) | crit30f@large | 0.92 M | 14/16 | 4/12 | 1.5 |
| BSIM-AR (74) | corroft@medium | 1.9 M | **15/16** | 4/8 | 61.5 |
| **PFN (75)** | **clean@small** | **0.69 M** | 11/16 (**zero flips**) | 5/8 | 15.6 |

**Highest-EV next steps** (not run):

1. **`corroft` corridor curriculum @small** (warm-start clean small, corridor data +
   class-weights — fully supported by the existing wiring): the V6.8.0 lever that took
   BSIM-AR from 13 to 15/16, aimed at the 4 tsmc5/7 cells. If the zero-flip property
   survives the curriculum, PFN small could contend with production at 1/5 the params.
2. **Context enrichment for the NFIN gap** (re-freeze, no retrain) + denser NFIN
   sampling at data generation.
3. **tsmc12 switchcap sits 0.3 % from its gate** — the first candidate cell for any
   charge-side improvement.
4. **Stabilize the large tier** (attention-logit clamp / warmup).

**Standing caveat — the gds sign bug.** Every number here was measured with the
inference-side `gds` sign error present (`docs/2026-07-21-systematic-audit.md` §A3; not
shipped as of 2026-07-24). The audit measured the corruption across all three NN
families and specifically flagged `tsmc6_pfn_small_nmos` as showing the off-grid NFIN
signature at a shifted centre. Expect §7's AC numbers and the opamp rails to move when
the sign + guard-F fix lands. See `DirectNet-L73-accuracy.md` §12.2.

---

## 11. Reproduction

```bash
# train (per tier)
MODEL=tabpfn RECIPES=clean SIZES="small medium" GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
MODEL=tabpfn RECIPES=clean SIZES="large" EXTRA_ARGS="--amp" GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
# evaluate (one tier at a time — never two recipe_eval dispatchers at once)
MODEL=tabpfn RECIPES=clean SIZES="small" PAR=16 bash scripts/recipe_eval.sh
# strict OMP sweep, one cell
MODEL=tabpfn bash scripts/recipe_multirun_gate.sh clean small TSMC16 verify_complex_ring_osc
# single netlist
PYCIRCUITSIM_NN_FORCE_LEVEL=75 python main.py <deck with LEVEL=73/75 cards>
```

Campaign log: `docs/plans/2026-07-11-tabpfn-pfn-family.md`. Raw gates:
`results/pfn_bench/`. Full pretrained-baseline per-combo table:
`pretrained_baseline_results.csv` (regenerable).

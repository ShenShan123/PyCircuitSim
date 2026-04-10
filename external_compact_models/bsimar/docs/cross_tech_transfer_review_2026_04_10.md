# Cross-Technology Transfer Roadmap — Review & Revised Plan (2026-04-10)

Five-agent review of `cross_technology_transfer_roadmap_2026_04_09.md`,
incorporating experimental results from the LOO improvement sprint
(`bsimar_loo_improvement_plan_2026_04_10.md`) and the v3 production
sprint (`bsimar_improvement_plan_2026_04_08.md`).

Reviewers: Feasibility, Adversarial, Scope Guardian, Coherence,
Product Strategy.

---

## Executive Summary

The original roadmap (10 ideas, 7 stages) is well-structured as
research, but poorly targeted as product work. The five reviewers
converge on three conclusions:

1. **Zero-shot transfer is not a user requirement.** Every tech node
   that can be simulated already has the PyCMG modelcard needed to
   generate training data in minutes. Retraining the universal model
   takes ~107 min and achieves 0.2% NRMSE — 100x better than even the
   most optimistic zero-shot target. The roadmap never benchmarks
   against this 2-hour baseline.

2. **The plan conflates two fundamentally different problems.** The
   ASAP7 catastrophe (24,678% NRMSE) is a data bottleneck — body
   physics (gmb/qb) are 10^4x smaller than TSMC, and no recipe change
   on the current data can fix it. The TSMC intra-family gap
   (0.84–2.18% NRMSE) is already 7x inside the 15% simulation
   threshold. These need different solutions (more data vs. nothing),
   not a single 7-stage pipeline.

3. **Higher-priority production work is blocked.** Retraining all v3
   checkpoints, fixing broken test scripts, TSMC5 transient risk
   (14.41%), and SRAM validation are all waiting while effort goes to
   LOO benchmark optimization.

---

## Per-Stage Verdicts from the Original Roadmap

### Stage 0 — Baseline the right problem

**Verdict: DONE.** Already completed during the 2026-04-09 LOO run.

Evidence:
- `external_compact_models/bsimar/eval/loo_labels.py` — tech
  fingerprint labeller + LOO split builder.
- `tests/verify_bsimar_loo.py` — full 5-fold runner with macro-average,
  worst-tech, per-output metrics, and markdown report generation.
- Frozen baseline: `n1_long_medium_nmos_best.phys.pt` checkpoint, with
  per-fold NRMSE/MRE/R^2 in
  `tests/verify_bsimar_loo_results/20260409_130630_nmos/`.

No further work needed.

### Stage 1 — Group-aware optimization + process jitter

**Verdict: Half-dead.**

- **Process-parameter jitter: DROPPED.** The LOO sprint (S1) showed
  that ASAP7, TSMC5, and TSMC16 all have 100% of held-out samples
  outside the training input box on process-param dimensions. Noise
  inside the training blob does not move points outside it. This was
  dropped before running.

- **Group-DRO / balanced sampling: Untested but low priority.** Would
  need ~200–400 LOC of new group-DRO loss infrastructure. The target
  problem (TSMC intra-family transfer at 0.84–2.18% NRMSE) is already
  within the 15% simulation threshold. The cost-benefit ratio is poor.

- **Shadow risk not addressed in original plan:** Tech-balanced
  oversampling would duplicate ASAP7 data ~4–5x (ASAP7 has ~48
  geometry combos vs TSMC's ~200+ per tech), risking overfitting to
  ASAP7 while undertraining TSMC.

### Stage 2 — Richer technology conditioning

**Verdict: Split into two ideas of very different cost.**

- **FiLM / conditional LayerNorm: HIGH COST.** Requires rewriting
  `nn.TransformerEncoderLayer` internals (PyTorch stock layers don't
  expose LayerNorm for conditional replacement). The v3 refactor
  explicitly chose stock `TransformerEncoderLayer` with `norm_first=True`
  and GPT-2 scaled init that directly accesses internal layer weights.
  AdaLN breaks these assumptions. Estimate: multi-day rewrite.

- **Multiple process tokens: LOW COST, UNTESTED.** Split the 12-dim
  process parameter token into 2–3 sub-group tokens (e.g., transport
  [U0, VSAT, RDSW, UA, EU], oxide [EOT, TOXP, CGSL, CFS], threshold/body
  [PHIG, ETA0, CIT]). Only requires changing `_embed_context` in
  `transformer.py`. Does not change `input_dim`, does not break the
  inference chain. ~2h engineering + 100 min GPU for a 2-fold probe.
  **This is the lowest-risk untested idea in the entire roadmap.**

### Stage 3 — Constrained Vth_hat / Vov conditioning

**Verdict: Likely dead as written.**

E2b (Vov-only derived feature) already regressed +13.8% on ASAP7 in
the LOO sprint. Root cause: Vov carries per-tech constant offsets that
become OOD under leave-one-out. A learned `Vth_hat(geom, process)`
trained on 4 TSMC techs will produce undefined outputs for ASAP7
process params — the same extrapolation failure.

The roadmap's "supervised anchors or hard bounds" mitigation is
underspecified. If anchors come from modelcard PHIG values, this is
structurally identical to E2b.

**Only viable reframing:** Truly dimensionless ratios (Vov/VT_est,
Vds/VDD_est) that are tech-invariant by construction. This is a
different design than what the roadmap describes, and would need its
own hypothesis and experiment.

### Stage 4 — Pretraining + lightweight adaptation (LoRA / adapters)

**Verdict: Infrastructure gap, wrong framing.**

- Zero adapter/LoRA code exists in the codebase.
- "Self-supervised pretraining" is undefined for MOSFET data (no
  natural masking or reconstruction target).
- With N=5 techs, "pretrain on 4, adapt to 1" is just LOO training
  with a frozen backbone — structurally identical to the existing LOO
  experiment but with fewer tunable parameters.
- The `train_transformer` pipeline is hard-wired and does not support
  freezing parameter subsets.

Estimate: 1–2 weeks of infrastructure work for an approach whose
theoretical justification does not hold at N=5.

### Stage 5 — EKV residual head

**Verdict: Multi-week research project, not an experiment slot.**

Architecture conflicts identified:

1. **AR dependency chain breakage.** Charges are predicted before
   currents in BSIMAR_COLUMN_ORDER (qg, qb, qd, qs, **id**, gm, gds,
   gmb). If `id = ekv(V, params) + nn_residual`, the AR tokens for
   charges cannot condition on EKV-informed currents.

2. **Normalization chain breakage.** The simulator consumer
   (`mosfet_bsimar.py`) assumes the full output goes through
   `asinh_scale * sinh(y_zscore * out_std + out_mean)`. A hybrid
   EKV+NN output breaks this assumption.

3. **Jacobian computation breakage.** The simulator takes
   `torch.autograd.grad(id, V)` on the NN output. An EKV core must
   either participate in the autograd graph or have its analytical
   Jacobian added separately.

4. **Questionable physics prior accuracy.** EKV is a long-channel
   model. At 7nm FinFET (ASAP7), short-channel effects (velocity
   saturation, quantum confinement, DIBL, fin shape) can cause 30–50%
   EKV error. The "bounded residual" may be as large as the full
   prediction, defeating the purpose.

Estimate: 2–3 weeks minimum. Should be a standalone research proposal,
not part of a production improvement plan.

### Stage 6 — Later-stage ablations

**Verdict: Correctly deferred.** Agree with the original plan's
assessment that these belong after all earlier stages.

---

## Coherence Issues in the Original Document

These should be fixed if the roadmap is kept as a reference:

1. **"Zero-shot" vs "held-out" terminology drift.** Lines 9, 153–155
   define transfer metrics as "zero-shot." Lines 195, 197 use
   "held-out-technology" in Stage 2/3 success gates. These are
   different protocols (no target data at all vs. target validation
   split available).

2. **Frozen-split paradox.** Stage 0 says "keep one frozen split
   definition for all later comparisons." Stage 1 proposes "balanced
   technology/variant sampling" which changes the split. Which wins?

3. **No numeric thresholds in success gates.** "Metrics improve" is
   not discriminative. The LOO improvement plan's log-ratio decision
   rule (`log(asap7_new/19.628) + log(tsmc5_new/3.430) < 0 AND no
   more than 0.5% relative regression`) is the right template.

4. **"In-distribution" is redefined silently.** After Stage 0 replaces
   pooled random with leave-one-tech, "pooled in-distribution accuracy"
   (line 302) becomes meaningless.

5. **Stages 4 and 5 overlap without sequencing.** Both claim to improve
   "zero-shot transfer" but use incompatible mechanisms. Should Stage 5
   (EKV) be embedded inside a pretrained model from Stage 4, or tried
   independently? Not specified.

---

## Revised Plan

### Tier 1 — Production priorities (do this week)

These directly unblock production use and are higher-leverage than any
cross-tech transfer work.

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | Retrain v3 universal NMOS + PMOS checkpoints | ~4h GPU | Unblocks all LEVEL=74 simulation |
| 2 | Port `verify_nn_*.py` to new `NNTechConfig` API | ~1 day | Unblocks end-to-end CI |
| 3 | Investigate TSMC5 transient (14.41% NRMSE) | ~1 day | 14.41% is dangerously close to 15% FAIL threshold |

### Tier 2 — Cross-tech exploration (1–2 days max, only after Tier 1)

If cross-tech transfer remains of interest, this is the minimal
experiment set with acceptable risk.

| # | Experiment | Effort | Rationale |
|---|-----------|--------|-----------|
| 1 | Update roadmap doc to reflect LOO sprint results | 1h | Prevent re-running failed experiments (E2/E2b, S1) |
| 2 | **Multiple process tokens** — split 12-dim process into 3 sub-group tokens in `_embed_context` | ~2h eng + 100 min GPU (2-fold probe) | Lowest-risk untested idea; only changes one function; plausible mechanism (lets attention weight transport vs oxide vs threshold params differently) |
| 3 | If (2) helps: run full 5-fold LOO | ~250 min GPU | Confirm signal |
| 4 | If (2) fails: **close** the cross-tech roadmap | 0 | Accept that transfer is a data problem, not a model problem |

**Decision rule for experiment (2):**
Use the LOO sprint's log-ratio rule:
`log(asap7_new / asap7_base) + log(tsmc5_new / tsmc5_base) < 0`
AND no more than 0.5% relative regression in in-distribution val loss.
Base values: asap7 NRMSE_sc = 19.628%, tsmc5 NRMSE_sc = 3.430%.

### Tier 3 — When a new PDK arrives

The practical solution to "new tech node" onboarding:

1. Add the modelcard to PyCMG (`pycmg/tech.py` TECH_REGISTRY entry).
2. Run `generate_nn_data.py --device both --universal` to append the
   new tech's data (minutes).
3. Retrain: `python -m bsimar.cli.train --model transformer
   --device-type nmos --universal --cuda` (~107 min).
4. Repeat for PMOS.

Total wall-clock: under 4 hours, fully automated, zero architecture
changes. Result: in-distribution model at ~0.2% NRMSE, which is
10–100x better than any zero-shot transfer target.

If the new tech has fundamentally different physics (like ASAP7's
decoupled body), add 5–10 anchor samples from that tech to the
training pool before retraining. This is the only principled fix for
cross-family structural outliers.

### Ideas explicitly parked

The following ideas from the original roadmap are parked. They should
not be attempted without new evidence or a standalone research
proposal:

| Idea | Reason parked |
|------|--------------|
| Process-parameter jitter (M3) | Disproven by OOD analysis — held-out techs are 100% outside training box on process dims |
| Derived features / Vov (M4) | E2/E2b both REJECT in LOO sprint; per-tech constant offsets extrapolate badly |
| Group-DRO (M2) | Target problem (TSMC 0.84–2.18% NRMSE) is already within 15% threshold |
| FiLM / conditional LayerNorm (M3) | Requires rewriting TransformerEncoderLayer internals; high cost for uncertain benefit |
| LoRA / adapters (M5) | Zero infrastructure exists; N=5 techs is too small for meaningful pretrain/adapt split |
| EKV residual head (M6) | Multi-week effort; breaks AR chain, normalization, and Jacobian assumptions; EKV may be 30–50% wrong at 7nm |
| Spectral normalization (M8) | Head-only spectral norm already looked ineffective in prior work |
| Mixture of experts (M9) | High overfit risk with N=5 groups; the plan itself says "belongs later" |
| Physics attention mask (M10) | Current causal masking already works; low priority |

### Stop criterion

If the Tier 2 multiple-process-tokens experiment fails to improve
the log-ratio metric, conclude that **cross-technology transfer in
BSIMAR is fundamentally a data coverage problem**, not a model
architecture problem. Redirect all further effort to Tier 1
production work and the Tier 3 retrain-with-new-data workflow.

---

## Appendix: Reviewer Findings Summary

| # | Finding | Source | Confidence |
|---|---------|--------|-----------|
| 1 | Zero-shot transfer not a user requirement; "just retrain" takes 2h | Product | 0.90 |
| 2 | Two problems (ASAP7 data gap vs TSMC covariate shift) treated as one | Adversarial | 0.90 |
| 3 | Stage 0 already completed | Feasibility, Scope | 0.90 |
| 4 | Process jitter disproven by OOD analysis before any experiment ran | Adversarial | 0.92 |
| 5 | Success gates have no numeric thresholds; not discriminative | Adversarial, Coherence | 0.88 |
| 6 | TSMC transfer (0.84–2.18%) already production-usable (threshold 15%) | Scope, Adversarial | 0.85 |
| 7 | "Cross-technology transfer" is wrong framing for N=5 discrete techs | Adversarial | 0.85 |
| 8 | Do-nothing baseline stronger than document acknowledges | Adversarial | 0.85 |
| 9 | Opportunity cost: broken tests, missing PMOS ckpts, TSMC5 transient risk | Product | 0.80 |
| 10 | EKV residual breaks AR chain, normalization, and autograd Jacobian | Feasibility | 0.75 |
| 11 | FiLM/AdaLN conflicts with stock TransformerEncoderLayer architecture | Feasibility | 0.70 |
| 12 | Vth_hat structurally identical to E2b which already REJECT | Feasibility | 0.80 |
| 13 | Multiple process tokens is the lowest-risk untested idea remaining | Feasibility | 0.65 |
| 14 | Experimental hit rate ~20%; 10-idea roadmap will mostly produce rejects | Product | 0.75 |
| 15 | Terminology drift: "zero-shot" vs "held-out" used interchangeably | Coherence | 0.80 |
| 16 | Frozen-split paradox between Stage 0 and Stage 1 | Coherence | 0.80 |
| 17 | `input_dim=19` assertion blocks any feature addition without handling | Feasibility | 0.85 |
| 18 | Plan implicitly shifts BSIMAR identity from "drop-in model" to "foundation model" | Product | 0.85 |

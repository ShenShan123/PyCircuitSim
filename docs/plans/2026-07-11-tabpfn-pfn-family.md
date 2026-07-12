# V6.9.0 — TabPFN port: "PFN" compact-model family (LEVEL=75)

## Context

The user has a local copy of **TabPFN v3** (`/data2/shenshan/TabPFN-main`) — Prior Labs' tabular in-context-learning transformer — and wants it adapted into PyCircuitSim's NN compact-model stack, alongside DirectNet (LEVEL=73) and BSIM-AR (LEVEL=74). A previous experiment in that repo (`train.py`, `results_multitarget/`) already applied pretrained TabPFN in-context regression to device data with excellent single-target results (R²≈0.9999 on a capacitance), so the architecture is promising for I-V/Q-V surfaces.

**Why not use the pretrained model directly:** TabPFN v3 is ~58M params, single-scalar-target (5000-bin bar-distribution head), with a numpy/`inference_mode` predict path. In the NR inner loop of circuit gates it would be orders of magnitude too slow, non-differentiable via the sklearn path, and would need 4+ model instances (one per target). 

**Chosen adaptation:** a **scaled-down, faithful from-scratch port of the TabPFN v3 architecture** trained per-tech on our existing datasets — keeping its three signature components (per-column distribution embedder with induced row-attention; column aggregator with CLS readout over the feature axis; ICL transformer where queries attend to context rows only) — with two deliberate deviations:
1. **Frozen learned context** instead of user-supplied train set: episodic context sampling during training (genuine ICL), a stratified K-row context frozen into checkpoint buffers at save, context-side KV cached once per loaded checkpoint → per-query cost = cross-attention reads.
2. **Direct 13-output value head** instead of the 5000-bin bar distribution: NR Jacobians need cheap smooth first-order autograd, and our trainer is MAE-on-asinh-normalized values.

New identity: netlist **LEVEL=75**, CLI `--model tabpfn`, tag `pfn`, stems `tsmc{X}_pfn_{size}_{dev}`, env pins `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`, `PYCIRCUITSIM_NN_FORCE_LEVEL=75`. Train small/medium/large × 4 techs × NMOS/PMOS (24 checkpoints, clean recipe), evaluate with the exact V6.8.0 harness, report V6.9.0.

Work happens on a new branch in a **worktree** (user-requested). First implementation step: copy this plan to `docs/plans/2026-07-11-tabpfn-pfn-family.md` (project convention) and keep it updated.

## Phase 0 — Worktree + branch + setup

1. `EnterWorktree` (name `tabpfn-pfn`) → branch + isolated tree under `.claude/worktrees/`.
2. Symlink gitignored/untracked assets from the main repo into the worktree (they're absent there):
   - `external_compact_models/bsimar/checkpoints`, `external_compact_models/bsimar/data/datasets`
   - `external_compact_models/PyCMG` (untracked!), `tools/`
3. Datasets already on disk (`tsmc{5,7,12,16}_{nmos,pmos}.npz`, ~0.5 GB each) — no regeneration.

## Phase 1 — The model: `external_compact_models/bsimar/models/tabpfn.py`

`TabPFNCompact(nn.Module)` + scaled-down ports from `TabPFN-main/src/tabpfn/architectures/tabpfn_v3.py`: `RMSNorm` (dtype-matching), `RotaryEmbedding` (aggregator only, default on, base 100k), `MLP` (GELU, zero-init out), `SoftmaxScalingMLP`, `Attention`/`CrossAttention` (xavier init, zero-init out-proj), `CrossAttentionBlock`, `InducedSelfAttentionBlock` (+ cached-hidden support), `FeatureDistributionEmbedder`, `ColumnAggregator` (CLS prepend → TransformerBlocks over feature axis → CLS cross-attention readout), `ICLAttention`/`ICLTransformerBlock` (context-only K/V, cached-KV path). No GQA/compile/chunked-inplace/KV-quant ports — sizes are tiny.

**Forward** `forward(x, tech_codes=None) -> (B,13)` in `OUTPUT_COLUMN_ORDER` (rides the DirectNet dispatch path everywhere):
- Token build: circular-shift feature grouping (roll by {1,2,4}; 7 cols × group 3 → 7 tokens — a perfect difference cover mod 7; no NaN indicators) → `Linear(3,E)`; + 1 tech-embedding token (`nn.Embedding(num_tech_codes,E)`, `unknown_code_id=num_tech_codes-1`, train-time `p_unknown` dropout on both query and context codes — Rule 16) → `(R,8,E)`. Context rows additionally get `col_y_encoder=Linear(13,E)` added (out-of-place). No "missing-y" query token (v3 has none — asymmetry lives in the attention structure).
- Dist embedder: per ISAB, `hidden = block1(inducing ← ctx_tokens)`; ctx and query tokens separately `block2(· ← hidden)` (per-column, fold 8 cols into batch).
- Aggregator: per-row over 8 feature tokens → `n_cls` CLS tokens → flatten to ICL width `W = n_cls·E`.
- ICL: `ctx_seq += icl_y_encoder(ctx_y)` (`Linear(13,W)`); per layer ctx self-attention evolves ctx and records `(k,v)`; queries cross-attend + MLP residual. Queries never attend to each other (row independence — required by `batch_eval` and per-row losses).
- Head: `RMSNorm(W) → Linear(W,2W) → GELU → Linear(2W,13)`.

**Context mechanics** (the load-bearing part, validated against trainer internals):
- Buffers `ctx_x (K,7)`, `ctx_y (K,13)` (normalized), `ctx_tc (K,)` — **float32** (int buffers corrupt under EMA's non-lerp branch; cast `.long()` in forward), registered at construction, filled via `set_frozen_context(...)` **before** `_train_loop` wraps the model in `AveragedModel` (EMA of a constant buffer is exact → every saved checkpoint carries the identical context bit-exact).
- Training bank: `set_context_bank(train_x, train_y, train_tc)` stores tensors in a tiny holder with `__deepcopy__ = lambda self, memo: self` (AveragedModel deepcopies the module — a plain 0.5 GB attribute would blow up RAM). Plain attribute → never in `state_dict` → `torch.load(weights_only=True)` in `_MOSFETNNBase` stays happy.
- train mode: sample K fresh context rows per step (episodic ICL). eval mode: frozen buffers via `_get_ctx_cache()` (built lazily under `no_grad`, detached). Cache invalidation: override `train(mode)` (trainer flips train/eval every epoch) and `_apply()` (catches `.to(device)` from `_setup_gpu`).
- Frozen-context selection: stratified over local tech-code vocab × 8 asinh-|id| quantile bins, fixed seed.
- Context choice consistency across NMOS/PMOS/tech: each checkpoint has its own buffers; caches key off module instances (`_SHARED_NN_MODULES` is per-file). Per-device tech_codes only affect the query path.

**Size presets** (param counts computed from block arithmetic; comparable to TF tiers):

| preset | E | n_ind | dist | agg | n_cls | W | icl | heads | params | ctx_K | batch | epochs | pat | lr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 64 | 32 | 2 | 2 | 2 | 128 | 3 | 4 | ~0.69M | 1024 | 1024 | 80 | 25 | 6e-4 |
| medium | 96 | 32 | 3 | 3 | 2 | 192 | 4 | 6 | ~2.03M | 2048 | 1024 | 150 | 40 | 5e-4 |
| large | 128 | 48 | 3 | 3 | 2 | 256 | 6 | 8 | ~4.65M | 2048 | 1024 | 300 | 80 | 4e-4 |

## Phase 2 — Training/CLI integration (bsimar)

- `bsimar/config.py`: add `TabPFNConfig` dataclass next to `TransformerConfig` (~L196).
- `bsimar/training/trainer.py`: add `train_tabpfn(...)` mirroring `train_transformer` (L741) but `is_transformer=False` (standard column order, one-shot dispatch `model(x, tech_codes=tc)` at L104-105 — zero loop changes); install context bank + frozen context before `_train_loop`; pass `arch_config`; generalize sidecar guard L524 `if is_transformer and arch_config is not None:` → `if arch_config is not None:` (DirectNet passes None — behavior-preserving); extend grad-clip (L136-137) via a `clip_grad` flag defaulting to current behavior; support `--init-from` (strict-load pattern).
- `bsimar/cli/train.py`: `--model` choices += `tabpfn` (L289); `SIZE_PRESETS` += 3 rows (L65-101); `_make_save_prefix` tag map += `tabpfn→pfn` (L129-140); guards: reject `--sobolev/--subthresh/--charge-sobolev/--monotonic/--ekv-core` for tabpfn (phase 1); dispatch branch in `_run` (~L234).

## Phase 3 — Simulator integration (pycircuitsim)

- **New** `pycircuitsim/models/mosfet_pfn.py` mirroring `mosfet_bsimar.py`: `_MOSFETPFNBase(_MOSFETNNBase)`, model_factory reads sibling `{stem}_config.npz` and rebuilds `TabPFNCompact`; `output_layout="standard"`; assert asinh norm; no `_forward_model` override (signature matches base default and `batch_eval`'s direct call). `NMOS_PFN` returns `-id`, `PMOS_PFN` sets `_is_pmos=True`, returns `+id` (CLAUDE.md sign rules, mirror `mosfet_directnet.py:93-108`).
- `pycircuitsim/solver.py` (**mandatory** — Rule 5; missed edits = devices silently never stamped): add `NMOS_PFN/PMOS_PFN` try/except imports to `_mosfet_types()` (L117), `_pmos_types()` (L138), `_nn_mosfet_types()` (L159, batched pre-warm eligible — row-independent forward). `_has_nn_device` works via isinstance(_MOSFETNNBase).
- `pycircuitsim/parser.py`: `level_tag` map += `75→"PFN"` (L84); LEVEL=75 resolver cascade (per-tech preempt `{tech}_pfn_{large,medium,small,xl}_{dev}_best.pt` + fallbacks, mirroring 73); stem→scope match += `f"{s}_pfn_"` (L250-252); dispatch `elif level in (73,74,75)` (L823) + import branch for `NMOS_PFN/PMOS_PFN`; `FORCE_LEVEL` acceptance += 75 (L833); error string (L884).

## Phase 4 — Driver scripts

Add a `tabpfn) TAG="pfn"` case + `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}` pins + `PYCIRCUITSIM_NN_FORCE_LEVEL=75` export (mirroring the `tf` branch exactly) in: `scripts/recipe_train.sh` (L45-49), `scripts/gate_matrix_iso.sh` (L33-38 + pin block L61-69), `scripts/recipe_eval.sh`, `scripts/benchmark_run_tests.sh`, `scripts/recipe_multirun_gate.sh`. PFN results default to `results/pfn_bench/`.

## Phase 5 — Smoke tests (each gates the next)

1. Build each preset; forward/backward check (train mode w/ synthetic bank, eval mode w/ frozen ctx); assert `(B,13)`, finite `autograd.grad`, eval determinism, cache-hit == cache-miss (allclose). Param counts.
2. Tiny train: `--model tabpfn --size small --device-type nmos --tech-scope tsmc7 --max-rows 200000 --epochs 6 --cuda` → `_best.pt` + `_norm.npz` + `_config.npz` land; reload via `mosfet_pfn`, probe forward matches trainer-side.
3. Jacobian sanity: gm/gds/cgg autograd vs finite differences of `_eval` (diag-style script).
4. Parser/gate smoke: `FORCE_LEVEL=75` + PFN pins → `verify_nn_multi_tech_dc.py --tech TSMC7` + one `examples/nn_inverter_dc.sp`; check `[NN-resolver]` line; run once with ≥2 PFN devices (batch_eval) and once `NN_BATCHED_EVAL=0`; **measure per-eval CPU latency** vs DirectNet/BSIM-AR to budget gates.
5. EMA-buffer audit: after 2-epoch `--swa-mode ema` run, saved `ctx_*` == frozen context bit-exact.

## Phase 6 — Full training (24 checkpoints)

```bash
MODEL=tabpfn RECIPES=clean SIZES="small medium large" TECHS="tsmc5 tsmc7 tsmc12 tsmc16" \
  GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
```
Clean recipe (`--apply-filter off --swa-mode ema --seed 42`), lands on production slots `tsmc{X}_pfn_{size}_{dev}`, `.complete` markers, resumable. Budget: ~1.5–2.5 days on 3 shared 4090s (small ~1-2 h, medium ~4-6 h, large ~10-16 h per ckpt; early stop typically sooner).

## Phase 7 — Evaluation (exact V6.8.0 harness) + report

```bash
# 16-gate complex matrix per size (CPU-pinned, isolated; don't run concurrently with dn/tf evals):
MODEL=tabpfn SIZE=small  bash scripts/gate_matrix_iso.sh   # then medium, large
# device DC/tran + AC + complex suites per size:
MODEL=tabpfn SIZES="small medium large" bash scripts/benchmark_run_tests.sh
# strict OMP {1,2,4} sweep for the best size, all 16 cells:
MODEL=tabpfn bash scripts/recipe_multirun_gate.sh clean <best_size> <TECH> <suite>
```
Also run the **pretrained-TabPFN in-context device-level baseline** (scratchpad script, not committed): `PYTHONPATH=TabPFN-main/src`, install missing deps (pandas, einops, huggingface-hub, safetensors, lightgbm) via tuna mirror; fit 10–50k stratified context per (tech,dev), predict test-split id/qg/qd/qb; report next to the port's device metrics.

**Report**: `docs/V6.9.0-tabpfn-report.md` (MRE/R²/NRMSE/MaxErr per tech — Rule 13; gate matrix + strict basket vs DN 14/16 and BSIM-AR 15/16; AC; runtime), `docs/CHANGELOG.md` V6.9.0 entry, CLAUDE.md touch-ups (LEVEL=75 rows). Commit per-phase on the branch.

## Phase 8 (optional, if clean results + time warrant)

Corridor curriculum `corroft` at the best size (warm-start from clean base, `{tech}_corro_{dev}.npz` data) — the recipe that won V6.8.0 for BSIM-AR. Supported without aux losses (class-weights are data-side).

## Verification

- Smoke suite above (Phase 5) before any long training.
- Ground truth is always NGSPICE BSIM-CMG via the existing gates — never simplified references.
- Byte-identical behavior for DN/TF paths: trainer sidecar guard change is a no-op for them; driver edits add a case; parser edits add a level. Quick regression: run one DN gate (`MODEL=direct SIZE=large gate_matrix_iso` single cell) after wiring to confirm nothing drifted.

## Top risks

1. **CPU gate cost of ICL reads** (K keys × icl layers per NR eval) → measured at smoke step 4; levers: cached ctx KV (built-in), batch_eval, reduce eval-time K (buffers are data), start gates at small/medium.
2. **EMA vs context buffers** → float32 pre-wrap fill + bit-exact audit (smoke 5).
3. **Solver enumerators missed** → explicit Phase 3 edits + real-inverter smoke before training.
4. **Model ignores context** (per-tech memorization makes ICL dead weight) → fine for accuracy; report a frozen-vs-resampled context ablation at device level for honesty.
5. **Jacobian roughness through attention** (the historical LEVEL=74 pain) → smooth C∞ components only, no dropout at eval, direct value head, FD-consistency smoke, existing gds-floor/Vds-correction safety net.

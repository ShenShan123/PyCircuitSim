# V6.8.0 — BSIMAR Transformer (LEVEL=74) recipe-parity campaign

**Goal.** Un-park the decode-only BSIMAR Transformer: ship every DirectNet
training method/recipe to it, train the scale × recipe matrix per tech, gate it
on the same NGSPICE BSIM-CMG ground truth as DirectNet, and report accuracy
side-by-side with the production DirectNet numbers (V6.6.6 report).

**Status: LIVE** (updated as executed — Rule: keep-plan-file-updated).

---

## 1. Starting point (scanned 2026-07-05)

- BSIMAR = `bsimar/models/transformer.py` — causal encoder, teacher forcing in
  train, AR at inference; 3 grouped context tokens (V/geom/tech-emb) + 8 AR
  target tokens (BSIMAR order `qg qb qd qs id gm gds gmb`) + parallel 5-cap
  head. Simulator side `pycircuitsim/models/mosfet_bsimar.py` via shared
  `_MOSFETNNBase` (autograd Jacobian through the AR loop) — **works** (smoke
  verified: LEVEL=74 DC sweep runs; ~88 ms/point @small, 1 CPU thread).
- Trainer already shared (`_train_loop(is_transformer=…)`): same data,
  normalizer (asinh), LDS-MAE, EMA/SWA, apply-filter, class-weights, seed.
- Datasets: NOT pruned — `bsimar/data/datasets/{tech}_{dev}.npz` (~2.0M rows
  each, 14 sample classes incl. `inv_trip`/`traj_corridor`) + corridor variants
  `{tech}_{cor,corr,corro}_{dev}.npz` all on disk. No regeneration needed.
- Params: tf small 0.67M / medium 1.94M / large 5.02M / xl 14.8M
  (DN: 0.07/0.3/0.9/2.1M — the tf tiers sit one class above DN tiers).

### Gaps closed by this campaign's port (all committed in V6.8.0)

1. `transformer.py` hardcoded `unknown_code_id=17` → **Rule 16 violation** for
   per-tech local vocabs (TSMC5 vocab=5 → CUDA assert on p_unknown dropout).
   Now derived `num_tech_codes - 1` (param), wired from the trainer.
2. `train_transformer` lacked `init_from` → curriculum (`*ft`/`crit*`) recipes
   impossible. Ported (strict state-dict match, same rules as DN).
3. Aux losses (`--sobolev/--subthresh/--charge-sobolev`) were guarded
   DirectNet-only. The math is model-agnostic; ported by permuting the
   output-side norm stats to BSIMAR column order inside `_train_loop`.
   Caveat recorded: under teacher forcing the supervised slope conditions on
   the TRUE prefix — exact for `qg` (first AR token; what the AC pole reads),
   approximate for later tokens (`id` sits at AR position 5).
4. No transformer `xl` preset → added (384×8L×ff1536, ~14.8M).
5. `--amp` (bf16 autocast) added for wall-clock (guarded incompatible with the
   double-backward aux losses). Final test metrics stay fp32.
6. Parser LEVEL=74 cascade had no per-tech preempt → added
   `tsmc{X}_tf_{large,medium,small,xl}_{dev}` (large-first, mirror of L73) +
   `_tf_` stem scope detection for the Rule-16 local-vocab remap.
7. Harness hook `PYCIRCUITSIM_NN_FORCE_LEVEL=74`: retargets every LEVEL=73
   model card at parse time so the ENTIRE complex/sweep/AC harness runs BSIMAR
   without parallel decks (loud `[NN-resolver] FORCE LEVEL` log per card).
   Verified: LEVEL=73 deck instantiates `NMOS_BSIMAR` under the env.
8. `scripts/recipe_train.sh` + `scripts/gate_matrix_iso.sh` parameterized with
   `MODEL={direct,transformer}` (TAG dn/tf; default = byte-identical legacy).
   recipe_train.sh: `clean` now trains the production slot (no --exp-name);
   TF curriculum init-from = own clean tf base (no v660clean special case —
   that is a DN-only artifact of the crit30f promotion).
9. `tests/verify_nn_dc_tran.py` `_cascade_handles_stem` extended to `_tf_`
   stems (the bsimar arm then per-tech-resolves instead of pinning one tech).

## 2. Training matrix

All jobs: `--apply-filter off --swa-mode ema --seed 42` (the clean-recipe
base), per-tech scope, 3× RTX 4090 via `MODEL=transformer scripts/recipe_train.sh`.

- **Phase A — scales (clean):** {small, medium, large} × 4 techs × 2 devs =
  24 ckpts on production stems `tsmc{X}_tf_{size}_{dev}`. (xl deferred to
  Phase C — train only if the capacity curve is still rising at large.)
- **Phase B — recipes @large** (the DN study's proven axes):
  - `csob` — charge-Sobolev (cap/AC axis; DN's best all-rounder alternate).
    300 ep, no AMP (double-backward).
  - `invtrip` — `--class-weights inv_trip=2.0` from scratch.
  - `corroft` — corridor w3.0 curriculum fine-tune from clean tf large
    (120 ep, lr 3e-4, corro data).
  - `crit30` — corridor w3.0 + inv_trip 2.0 curriculum (the DN production
    recipe, V6.6.4).
  - `crit15m` — corridor 1.5 + inv_trip 3.0 (DN promote-candidate) — stretch.
- **Phase C — data-driven round 2:** picked from A+B results (candidates: xl
  tier, seed probes s7/s17, sob, csobcrit; or transformer-specific ideas).

Cost calibration (measured): ~2.0M rows → ~1.6k steps/epoch @batch 1024;
per-epoch timing fp32 vs `--amp` measured before launching (see §5 log).

## 3. Gating (ground truth = NGSPICE BSIM-CMG, CPU-pinned)

1. **Device suites** per tech: `verify_nn_dc_tran.py --tech TSMC{X}` with
   `PYCIRCUITSIM_NN_CHECKPOINT_TF_{NMOS,PMOS}` per-tech pins — reports
   MRE/R²/NRMSE/MaxErr (Rule 13) for the bsimar arm next to DN.
2. **Complex 16-gate matrix**: `MODEL=transformer SIZE={s,m,l}
   RECIPES="clean csob …" scripts/gate_matrix_iso.sh` (TF pins +
   FORCE_LEVEL=74 wired in). Runtime risk: BSIMAR eval is ~30–100× DN on CPU
   — mitigations: high PAR across the 192-core box, possibly OMP>1 per cell
   (fixed-thread determinism recorded), gate the full matrix only for clean
   tiers + the best recipes.
3. **AC**: `verify_nn_ac.py` (TF pins) + `verify_complex_opamp_ac.py`
   (FORCE_LEVEL) for the promising recipes.
4. **OMP determinism sweep** for the final winner only.

## 4. Report

`docs/V6.8.0-bsimar-transformer-report.md`, mirroring the V6.6.6 structure:
capacity curve, recipe table, 16-gate matrix, device metrics per tech
(vs DN clean@large + crit30f@large baselines), AC, runtime cost of AR
inference, verdict + production recommendation. CHANGELOG V6.8.0 entry
(dead-ends recorded per repo rule).

## 5. Execution log

- 2026-07-05: scan done; port implemented (items 1–9 above); train+inference
  smokes PASS (tiny tsmc5 tf-small trains, saves, loads via LEVEL=74 and
  FORCE_LEVEL; resolver logs correct). Datasets confirmed on disk (no regen;
  a mistakenly-launched regen was killed before any write). Timing runs for
  fp32-vs-AMP @large in flight.

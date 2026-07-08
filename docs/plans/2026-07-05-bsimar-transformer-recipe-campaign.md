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

## 1b. Baselines to beat / historical context (scout briefings, 2026-07-05)

- **DirectNet production (crit30f@large) = 14/16 strict** (banks tsmc5-opamp
  0.21% + tsmc12-opamp 6.25%; fails tsmc7/tsmc16-opamp). Clean@large = 13/16
  single / 12/16 strict; DN capacity curve 7→10→13→10 (S/M/L/XL).
- **DN device level (clean@large):** DC 24/24; per-tech NMOS NRMSE 0.83–3.99%,
  PMOS 0.02–2.20%; recipe×size device-mean-NRMSE: clean L 1.7, xl 2.17.
  **DN AC:** device CS-amp 4/12 @large, opamp AC 0/12 @large; AC peaks @small.
- **BSIMAR history (git b5f6fdd, 2026-04-13):** v4 universal TF device NRMSE
  NMOS 0.270% / PMOS 0.260% (R² .9937/.9969, 5.02M params) vs DirectNetV4
  0.043%/0.033% @908K — DN 6–8× better with 5.5× fewer params → parked
  (Rule 15). Realistic campaign framing: can per-tech BSIMAR + the modern
  recipe stack MATCH the DN gate matrix? Known BSIMAR risks: AR exposure
  bias (teacher-forced train vs AR inference → error cascade), phys-best
  median fix already in-tree, slow CPU inference (SRAM gates previously hit
  ~2700 s wall-clock with far cheaper models).
- **OMP strict-determinism probe** must set `PYCIRCUITSIM_TORCH_THREADS=N`
  alongside OMP/MKL (torch pins to 1 by default since V6.6.6).
- **Corridor dataset gotcha:** the `{tech}_cor*_{dev}_tech_variant_labels.npy`
  sidecars are the ONLY valid label source (bench geometry is off-grid for the
  fingerprint labeller) — never delete/regenerate them.
- **Verified (numerically) against a scout claim:** the aux losses' column-sum
  autograd trick IS valid for the Transformer — attention mixes token
  positions within a sample, never batch rows; max|colsum−perrow| grad diff
  1.3e-7 (float noise) on a random tf-small. The teacher-forcing caveat
  (§1 item 3) is the only real approximation.

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
  a mistakenly-launched regen was killed before any write).
- 2026-07-05: timing @large (1.6M train rows, batch 1024, 4090): fp32
  48.5 s/epoch, bf16 --amp 36.6 s/epoch (~1.3×, loader-bound). **Decision: no
  AMP in the campaign** — comparability with the DN clean recipe outranks 1.3×.
- 2026-07-05: **BLOCKER found+fixed** — fused SDPA has no double-backward;
  transformer + charge-sobolev/sobolev now force the MATH attention backend
  at forward (commit 8e72713). csob + crit-curriculum (init-from +
  class-weights on corro data) smokes PASS on the transformer.
- 2026-07-05: **LATENT L74 INTEGRATION BUG found+fixed (d4151bf)** — first
  real checkpoint (tsmc5 tf-small: trainer-side AVG NRMSE 0.38%, id 0.84%,
  R² all positive under AR rollout) scored 389% device-DC NRMSE through the
  netlist path: `_MOSFETNNBase._out_col` preferred `norm_stats.output_columns`
  (canonical order, describing the STATS arrays) over the BSIMAR model-output
  layout → qg denormed as id. Historical BSIMAR norm files predate the field —
  classic silent-green integration regression. Probe now matches trainer path.
  All pre-fix L74 gate results are void (device suite 5 FAIL, first AC run).
- 2026-07-05: thread-oversubscription fixed (75d3d07): TRAIN_OMP=4 pin —
  loadavg 400→49, GPUs 65-90%, small back to ~26 s/epoch. Phase A relaunched
  smalls-first. DN inverter regression PASS after batched-eval inclusion
  (8f08c23): VTC 1.04% / tran 0.79% — DN-neutral.
- 2026-07-05: **SMALL-TIER RESULTS (post-fix)** — tsmc5 device suite 11/11
  PASS (NMOS DC 1.45%, PMOS DC 0.48%, VTC 4.00%, inv-tran 1.03% NRMSE);
  AC 2/2 PASS (pmos f3db ratio 1.000, magNRMSE 0.53% — beats DN's historical
  AC record); **16-cell complex matrix 11/16** (SRAM 4/4, SC 3/4, opamp 2/4
  [tsmc12 8.3%/tsmc16 6.2%], ring 2/4 [tsmc12 1.9%/tsmc16 2.1%]; fails =
  tsmc5/7 ring+opamp, tsmc16 SC). **DN small = 6/16 on the same matrix; DN
  clean large = 13/16.** tsmc16 column re-run on completed weights by the
  follower → tsmc16 switchcap flips to PASS: **small-tier FINAL = 12/16**
  (SRAM 4/4, SC 4/4, opamp {12,16}, ring {12,16}; fails = tsmc5/7 ring+opamp
  — the classic frontier cells). BSIMAR-small(0.67M) = DN-clean-large−1 on
  the identical matrix; DN small = 6/16. Heavy cells ≈ 1 h each, 1 thread.
- 2026-07-05: **SMALL TIER COMPLETE** — devices 44/44 PASS (4 techs × 11;
  DC NRMSE N/P: tsmc5 1.45/0.48, tsmc7 3.37/0.56, tsmc12 0.20/0.21,
  tsmc16 0.24/0.46 — inside DN-clean-large's device range at 0.67M params);
  AC **7/8 PASS** (only tsmc12-pmos magNRMSE 13%>10%; DN@large = 4/12).
  Scoreboard: complex 12/16 · device 44/44 · AC 7/8.
- 2026-07-05: **MEDIUM-TIER MATRIX = 14/16 single-run** — ALL FOUR OPAMPS
  PASS incl. **tsmc7-opamp (gain err 9.83%)**, the cell that fails all 22 DN
  recipes × 2 tiers (previously reached only by the V6.5.9 T3 solver
  fine-tune), and tsmc16-opamp (6.74%) which DN production also fails.
  Fails: tsmc5 ring 5.55%, tsmc7 ring 7.41% (the corridor-curriculum
  targets). BSIMAR-medium(1.94M) ties DN production crit30f@large (14/16)
  single-run — OMP∈{1,2,4} strict probe IN FLIGHT for the 4 opamps + 2
  passing rings before banking (opamps are historically multistable).
- 2026-07-05: **OMP STRICT PROBE (medium)** — opamps tsmc5 detPASS
  (1.46/1.46/1.46), tsmc12 detPASS (2.18/4.76/4.76), tsmc16 detPASS
  (6.74/2.77/6.78); rings tsmc12/16 detPASS (identical across OMP).
  **tsmc7-opamp FLIPs** (9.83/9.53/11.39@OMP4 vs 10% gate) → medium strict
  = **13/16** (single-run 14/16). Medium device 44/44; AC 4/8 (peaks at
  small, mirroring DN's capacity/AC trade-off). Phase B thesis: corridor
  (rings tsmc5/7) + inv_trip (opamp margin; tsmc7 needs ~1.5%) → 16/16
  strict is plausibly reachable.
- 2026-07-05: **LARGE TIER + PHASE A COMPLETE** — large matrix **13/16**
  (tsmc5-opamp 2.98% PASS joins; tsmc7-opamp 12.78% FAIL regresses from
  medium's 9.83%; rings tsmc5 7.38%/tsmc7 11.19% FAIL). Device 44/44 (DC
  N/P: tsmc5 1.60/0.02, tsmc7 4.77/0.10, tsmc12 0.12/0.35, tsmc16
  0.09/0.15 — tsmc7-NMOS grows with capacity: 3.37→4.07→4.77). AC 4/8.
  **Clean capacity curve (complex, single-run): 12 → 14 → 13 (S/M/L) —
  peaks at MEDIUM**, the DN peak-then-decline shape one tier earlier.
  AC by tier: 7/8 → 4/8 → 4/8 (peaks at small, like DN). Phase B (crit30/
  corroft/crit15m/invtrip @large, 32 jobs) training in flight; medium-tier
  curricula queued as follow-up (medium = the peak; its 3 misses map
  exactly onto the corridor + inv_trip levers).
- 2026-07-06: **PHASE B BREAKTHROUGH — corroft & crit30 @large = 15/16
  single-run each** (beats DN production 14/16). Both open BOTH low-VDD
  rings (tsmc5 3.88/3.85%, tsmc7 2.31/2.32%) and hold tsmc5/12/16 opamps;
  ONLY tsmc7-opamp fails, and it RAILS (99.98% gain err — corridor drifts
  it from clean-large's 12.78% into full gain collapse, the documented
  corridor↔opamp tradeoff). corroft (corridor-only) ≡ crit30 (corridor+
  inv_trip) to <0.5% on every cell → **the inv_trip anchor buys nothing on
  the transformer** (unlike DN, where crit30>corroft). OMP strict probe of
  both @large in flight. NEXT: tsmc7-opamp is THE 16/16 blocker; clean
  MEDIUM already passes it single-run (9.83%) while failing the 2 rings —
  so crit30/corroft @MEDIUM (corridor opens rings, medium opamp basin more
  robust) is the 16/16 shot. Queue after crit15m/invtrip @large free the GPUs.
- 2026-07-06: OMP strict probe (crit30/corroft @large) — all passing opamps
  DETERMINISTIC (tsmc5/12/16 PASS at OMP 1/2/4, both recipes) → the 15/16 is
  a real strict result, not a coin-flip. tsmc7-opamp + the 2 opened rings
  still probing (slow AR transient cells). **MEDIUM curricula LAUNCHED**:
  corroft + corro15 @medium (16 jobs; dropped crit30@medium as corroft≡crit30
  on the transformer; corro15=gentler w1.5 hedge to preserve the tsmc7-opamp
  basin). Concurrent with invtrip@large tsmc16 tail (I unblocked the queue —
  invtrip's 300-ep from-scratch tsmc16 pair was needlessly gating the
  higher-value 16/16 shot). crit15m@large gate also in flight.
- 2026-07-07: **OMP ring fragility (the strict-vs-single split)** — the two
  opened rings are OMP-fragile at the ~5% period-edge: corroft banks
  tsmc7-ring det (1/2/4 PASS) but tsmc5-ring FLIPs (OMP1 FAIL); crit30 is the
  MIRROR (tsmc5-ring det, tsmc7-ring FLIPs OMP1). Same weight→basin
  non-monotonicity as DN. → single-run 15/16 (both, real, beats DN single
  13/16) but **strict ~14/16** each (one ring unbankable), tying DN prod
  crit30f strict. Opamps stay det (tsmc5/12/16 PASS; tsmc7 rail). Report will
  carry BOTH single-run and strict columns per DN-report convention. Still
  need: tsmc7-opamp OMP (expect rail), medium-curricula gate. Corollary: a
  16/16-strict needs a recipe banking BOTH rings det — the medium tier (robust
  opamp basin) or a corridor-weight between 1.5/3.0 is the remaining lever.
- 2026-07-07: **crit15m@large = 15/16** — THIRD large curriculum, identical
  profile (all opamps but tsmc7 [rail 99.98%], all 4 rings PASS). So corroft
  (w3.0), crit30 (w3.0+anchor2.0), crit15m (w1.5+anchor3.0) — 3 distinct
  corridor recipes across the weight/anchor grid — ALL converge to 15/16 and
  ALL rail tsmc7-opamp. Clean finding: **at large the corridor lever and the
  tsmc7-opamp basin are mutually exclusive, independent of corridor weight or
  inv_trip anchor.** → the medium-tier curricula (warm-start from the tsmc7-
  opamp-PASSING clean-medium basin, 9.83%) is the only remaining 16/16 route.
- 2026-07-07: **corroft@medium = 15/16 — 16/16 ROUTE CLOSED.** The corridor
  RAILS tsmc7-opamp even from the passing medium basin (124% gain err, worse
  than large). Definitive: on the transformer **tsmc7-ring and tsmc7-opamp are
  mutually exclusive under the corridor lever at EVERY tier** — the ring-open
  perturbation IS the opamp-kill perturbation. Even per-tech recipe mixing
  (each BSIMAR ckpt is per-tech, Rule 16) caps tsmc7 at 3/4 (clean=opamp∧¬ring,
  corridor=ring∧¬opamp) → **campaign ceiling = 15/16**. tsmc7-opamp is the
  universal hard cell that needed DN's SOLVER-level T3 fine-tune (V6.5.9), not
  a data recipe — and T3 isn't ported (out of scope; structural, not a recipe).
  CONCLUSION forming: BSIMAR **15/16 single-run** beats DN uniform-recipe best
  (13/16 single) and its failure set is genuinely BETTER than DN production
  crit30f (14/16 strict) — BSIMAR banks **tsmc16-opamp** (DN prod FAILS it) +
  both rings; only tsmc7-opamp missing vs DN's tsmc7-opamp+tsmc16-opamp. Strict
  count pending the corroft@medium ring OMP sweep (if medium banks BOTH rings
  det → 15/16 strict, clearly > DN).
- 2026-07-07: **★ CAMPAIGN RESULT — corroft@medium = 15/16 STRICT ★** OMP
  probe: ALL FOUR rings deterministic PASS (tsmc5/7/12/16 @ OMP1/2/4) + 3
  opamps det + SRAM/SC stable; only tsmc7-opamp rails. **Beats DN production
  14/16 strict.** Medium rings sit inside the gate; large recipes' extra
  capacity pushes one ring to the OMP edge (→ 14/16 strict). invtrip@large
  (no corridor) = 13-14/16, fails BOTH rings, keeps opamps → orthogonal-
  conflicting levers on tsmc7 (corridor↔ring vs clean↔opamp). Ceiling = 15/16;
  tsmc7-opamp = solver-level (T3), out of scope. Report:
  docs/V6.8.0-bsimar-transformer-report.md. **CAMPAIGN COMPLETE.**
- 2026-07-07: final datapoint — corro15@medium (w1.5) = 15/16, tsmc7-opamp
  FAIL at 24.83% (NOT railed). Proves the ceiling is CONTINUOUS: tsmc7-opamp
  degrades monotonically with corridor weight (9.83 clean → 24.83 w1.5 → 124
  w3.0) while the ring needs nonzero weight → no weight threads the needle.
  All 72 checkpoints trained + gated; report/CHANGELOG/CLAUDE.md/memory done.
- 2026-07-05: **Phase A LAUNCHED** — MODEL=transformer clean,
  SIZES="large medium small" × 4 techs × 2 devs (24 ckpts), NSTREAMS=6 on
  GPUs 0-2; ~2.5-3 min/epoch per job at 2 jobs/GPU → larges land in ~12 h.
  Side-job: tsmc5 tf-small pair trained ahead of queue to validate the whole
  gating pipeline early. Eval-side check: nn_sweep "baseline" = baseline
  *config* vs NGSPICE (not a stored-DN comparison) → BSIMAR-safe under
  FORCE_LEVEL; verify_nn_ac/complex_ac carry no own pin logic → parser env
  pins/cascade apply.

# V6.7.0 — Universal DirectNet (TSMC16/12/7) + TSMC5 Fine-Tune Transfer Study + Universal Recipe Comparison

**Version: V6.7.0** (campaign designation; all artifacts — scripts, checkpoints `u716_*`, `results/uni_bench/`, report — belong to this version).
**Status: COMPLETE 2026-07-05 (all phases incl. 1b).** Verdict: universal corroft@large = 10/12 strict 0-FLIP (per-tech parity); TSMC5 onboarding = plain fine-tune @1M rows → 4/4 strict; retention collapses without replay; xl arm banks tsmc16-opamp but trades ALL rings → ceiling 11/12 NOT reached, corroft@large stands. Full findings: `docs/V6.7.0-universal-transfer-report.md`. Revised 2026-07-03 against `results/recipe_bench/ACCURACY_REPORT.md` (V6.6.6 25-recipe large retest + 22-recipe xl retest + V6.6.7 round-1) — recipe set, OMP standard, env-pin semantics, and an optional xl arm updated; see §2b. This file is the live routing doc — update it on every phase change / lesson (workflow rule).

## 1. Goal & context

All production DirectNet checkpoints today are **per-tech** (V6.6.4 crit30f, 14/16 strict — reconfirmed by the V6.6.6 full retest; the V6.6.7 round-1 candidates csobcrit/crit30a1 both landed 13/16 and did not displace it). The universal-scope model was retired in V6.1 and has never been trained on the current data/recipe stack. This campaign answers three questions:

1. **Universal viability** — train ONE universal DirectNet (18-code embedding) on TSMC16+12+7 data only.
2. **Transfer efficiency** — fine-tune it on TSMC5 with varying sample counts N and measure the accuracy-vs-N curve (device metrics + complex gates), i.e. how cheaply a new tech can be onboarded.
3. **Best universal recipe** — repeat the recipe comparison (V6.6.1/V6.6.3 style) at universal scope and rank recipes on the complex-circuit gates.

**Hard constraint: no existing code file is modified.** Everything runs through existing CLI flags. Three NEW standalone scripts are added (concat / subsampler / sweep-runner) — zero edits to existing files. (Alternative if new scripts are vetoed: regenerate data via `generate_nn_data.py --universal`, ~1–2 days CPU, and it loses the inv_trip/corridor recipe classes for 7/12/16 — V5' gates inv_trip to TSMC5-only at generation time.)

## 2. Key facts from the read-only scan (all verified)

- `bsimar.cli.train --tech-scope universal` is fully supported (it is the default scope): vocab = 18 (TSMC codes 0–16 + UNKNOWN 17), no local remap, default data path `universal_{dev}.npz` (`cli/train.py:120,155-162`). **No universal npz or checkpoint exists on disk** (retired/deleted 2026-05-12).
- `--init-from` loads model weights only and **hard-fails across vocab sizes** (`trainer.py:660-674`) → the whole campaign (base + fine-tunes) stays at `--tech-scope universal` so the 18-row embedding always matches. TSMC5 rows carry universal codes 0–3; fine-tuning trains those previously-unused embedding rows.
- `--max-rows N` (`data/dataset.py:151-157`) is the existing seeded row cap — but it is uniform-random, so tiny tiers would nearly lose the rare classes (traj_corridor ≈0.4%, inv_trip ≈3%). Hence a **stratified** tier subsampler script.
- `load_and_split_bsimar` reads only `inputs/geometry/outputs/sample_class/meta_sample_class_names` (`dataset.py:101-123`); the tech-code sidecar `<stem>_tech_variant_labels.npy` is validated by row count only (`eval/loo_labels.py:133-139`) → **concatenating per-tech npz + sidecars is exactly equivalent to a native universal file**. Sidecars already carry universal codes (tsmc7: 4–6, tsmc12: 7–11, tsmc16: 12–16, tsmc5: 0–3).
- All 4 techs' base datasets contain `inv_trip` rows and the `_corro` variants add `traj_corridor` rows (verified by class histogram) → the full Core-4 recipe family works at universal scope via concat.
- Inference scope is decided by checkpoint **stem prefix** (`pycircuitsim/parser.py:229`): anything not starting with `tsmc{5,7,12,16}_dn_` → universal scope → correct universal tech_code per tech via `local_variant_code("universal", tech, vt)`. Env pin `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}` short-circuits the whole cascade and works for every gate/test. **Since the V6.6.6 test-infra audit an absent pinned stem raises `FileNotFoundError`** (`parser.py:103-116`) — the old "silently ignored" failure mode is gone — but still verify the `[NN-resolver] ... scope=universal` stdout line (it additionally confirms the tech_code mapping is universal, not per-tech).
- Since V6.6.6 the complex/AC gate infra **pins torch to 1 thread by default** (`tests/common/complex.py:46`, `tests/common/complex_ac.py:42`); `PYCIRCUITSIM_TORCH_THREADS` overrides. Consequence: single-point gates are OMP-deterministic by default, and the multistability probe must set `PYCIRCUITSIM_TORCH_THREADS` explicitly (see Phase 2).
- `_best.pt` alone is NOT proof of a completed training run (a killed run leaves a best-so-far file) — completed runs carry a `*_best.pt.complete` marker (convention from `scripts/recipe_train.sh:114-116,164`). The Phase 1/3 training wrappers must write the marker, and Phase 2/4 eval must gate on it.
- `pycircuitsim/models/mosfet_directnet.py:44-83` infers `num_tech_codes` from the state dict → 18-row checkpoints load with no code changes.
- Existing sweep scripts (`recipe_train.sh`, `gate_matrix_iso.sh`, `recipe_eval.sh`, `recipe_multirun_gate.sh`) hardcode `tsmc{X}_dn_` stems → unusable for universal checkpoints (symlinking to those names would corrupt tech_code for 3 of 4 techs). New runner required.
- Env: `conda run -n pycircuitsim` (env lives at `/data1/shenshan/.conda/envs/pycircuitsim`, NOT `/home/...`); 3× RTX 4090 currently ~100% busy with resident jobs (~17–20 GB free each); `/data2` has 6.2 TB free. Gates run CPU-pinned: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `NGSPICE_BIN=$PWD/tools/ngspice-45.2/bin/ngspice`.
- Dataset sizes on disk (rows, nmos/pmos): tsmc7 1.82M/2.19M, tsmc12 2.55M/2.53M, tsmc16 2.55M/2.54M, tsmc5 2.02M/2.02M → universal-716 ≈ 6.92M (nmos) / 7.26M (pmos) rows.

## 2b. What `results/recipe_bench/ACCURACY_REPORT.md` (V6.6.6/V6.6.7) changes for this plan

The full 25-recipe large retest + 22-recipe xl retest landed after this plan was drafted. Findings that reshape it:

1. **Weight→basin map is TIER-dependent.** At `large`, crit30/crit30f = 14/16 strict (banks tsmc5+tsmc12-opamp); at `xl` the *same* crit30 drops to 12/16 while corroft/crit10/crit15m@xl = 14/16 strict, banking **tsmc16-opamp (~6.5%)** which production fails. A universal model (~7M rows, 3 techs in one net) sits at yet another point on the capacity axis — do not assume the per-tech-large basin map transfers. → Phase 1b optional xl arm added.
2. **Curriculum RELOCATES basins, it does not compose.** csobcrit (csob base + crit30 curriculum) lost csob's tsmc16-opamp basin (1.28% detPASS → detFAIL) and landed 13/16; crit30a1 showed the anchor hop {tsmc16}→{tsmc5,tsmc12} is DISCONTINUOUS in inv_trip ∈ (1.0, 2.0). → no recipe-combo arm at universal scope; expect each recipe to pick a basin set, not a union.
3. **tsmc7-opamp fails for all 25 recipes at both tiers** (structural non-existence at this tier family). → treat as a known-fail cell in the ranking; exclude it from opamp means (report convention).
4. **Calibration bar for the 12 ranking gates (TSMC7/12/16):** per-tech clean, crit30f, csob, and corroft ALL score 10/12 strict — they differ only in *which* opamp cells they bank (crit30f: tsmc12; csob: tsmc12+16 but drops tsmc7-ring; corroft: tsmc16). A universal recipe ≥10/12 strict is per-tech-parity; 11/12 (everything but tsmc7-opamp) is the realistic ceiling.
5. **AC is a tie-breaker, not a target:** AC peaks at `small`, is 0/needs-luck at `xl`, and only csob@large passes tsmc16 AC in the per-tech matrix.
6. **TSMC5 full-data per-tech reference for the Phase 3/4 transfer curve:** crit30f@large tsmc5 = 4/4 strict (ring 4.04%, opamp 0.21% detPASS, SRAM 6.31%, SC 2.06%).

## 3. Naming scheme (load-bearing)

| Artifact | Stem pattern |
|---|---|
| Universal dataset | `uni716_{nmos,pmos}.npz` (+ `uni716_corro_*` variant) + `_tech_variant_labels.npy` sidecars |
| TSMC5 tier datasets | `tsmc5ft_n{N}_{dev}.npz` + sidecars (stratified from `tsmc5_corro_{dev}.npz`) |
| Universal base ckpts | `u716_dn_{clean,csob,corroft,crit30u}_{large[,xl]}_{dev}` (via `--exp-name`; xl only in optional Phase 1b) |
| TSMC5 fine-tunes | `u716f5_{plain,crit}_n{N}_large_{dev}` |

None start with `tsmc{X}_dn_` → parser scope = universal (correct tech_code for all techs). None match the resolver's hardcoded fallback names (`refac_dn_*`, `v4_*`) → **production per-tech resolution is untouched**; the new checkpoints are reachable only via explicit env pin. No promotion happens in this campaign.

## 4. Phase 0 — Datasets (CPU, ~30 min)

**New script `scripts/uni_concat_npz.py`** — concat per-tech npz files (order: tsmc7, tsmc12, tsmc16) for `inputs/geometry/outputs/sample_class`, copy `meta_sample_class_names` (identical across techs), write provenance metas; concat sidecars in the same row order.
- Build: `uni716_{dev}.npz` from base files and `uni716_corro_{dev}.npz` from the `_corro` files.
- Validation (in-script + manual): row sums match, sidecar codes ⊆ {4..16}, class histogram = sum of parts, 100 random rows bit-identical to their source file.

**New script `scripts/uni_subsample_npz.py`** — stratified-by-`sample_class` subsample of `tsmc5_corro_{dev}.npz` (proportional allocation, ≥1 row per non-empty class, seeded) → `tsmc5ft_n{N}_{dev}.npz` + aligned sidecar.
- Tiers: N ∈ {2 000, 10 000, 50 000, 200 000, 1 000 000}; the "full" tier uses `tsmc5_corro_{dev}.npz` directly (~2.02M rows).

## 5. Phase 1 — Universal base trainings (Core-4 recipes × 2 devices = 8 runs, GPU)

All runs: `conda run -n pycircuitsim python -u -m bsimar.cli.train --model direct --size large --device-type {dev} --tech-scope universal --cuda --overwrite --apply-filter off --swa-mode ema --seed 42` plus:

| Recipe | Extra flags | Data | Depends on |
|---|---|---|---|
| `clean` | — | `uni716_{dev}.npz` | — |
| `csob` | `--charge-sobolev` | `uni716_{dev}.npz` | — |
| `corroft` | `--class-weights traj_corridor=3.0 --lr 3e-4 --epochs 120 --patience 40 --init-from u716_dn_clean_large_{dev}` | `uni716_corro_{dev}.npz` | clean |
| `crit30u` | `--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40 --init-from u716_dn_clean_large_{dev}` | `uni716_corro_{dev}.npz` | clean |

`--exp-name u716_dn_{recipe}_large`; `--data` passed explicitly. Wave 1 = clean+csob (4 long jobs), wave 2 = the two curriculum fine-tunes (4 fast jobs). Warm-starting the curricula from the *clean* base (never from another curriculum checkpoint) mirrors the crit30f-stacking-trap fix in `scripts/recipe_train.sh:133` — the universal analog of the v660clean archive is `u716_dn_clean`, which this campaign trains itself, so no archive indirection is needed. Each training wrapper must `touch <ckpt>.complete` on normal exit (§2 marker convention).

**GPU policy (default, user can override):** pre-flight `nvidia-smi`; launch round-robin on GPUs 0/1/2 alongside the resident jobs (memory suffices; compute is time-sliced). ~3× per-tech-large epoch time on ~7M rows — budget 1–2 days wall for wave 1. Nothing gets killed without asking.

Recipe-set rationale (**Core-4, revised 2026-07-03 per ACCURACY_REPORT.md**): clean = no-curriculum baseline (per-tech 13/16 single-run); crit30u = analog of the production winner (crit30f, 14/16 strict @large); csob = charge-axis lever, best device/AC all-rounder and the only large recipe holding the tsmc16-opamp basin (1.28% detPASS); **corroft replaces invtripft** — corridor-only curriculum, 13/16 strict @large AND the top xl recipe (14/16 strict, banks tsmc16-opamp), i.e. the tier-robust member, whereas invtripft retested at 12/16 strict (below clean) and inv_trip was refuted as a ring lever back in V6.6.2. The four now span the design cleanly: no-curriculum / charge-axis / corridor-only / corridor+anchor. **Deliberately excluded:** csobcrit and crit30a1 (both tested 13/16 per-tech on 2026-07-03 — curriculum relocates rather than composes, §2b#2); ekv/sob/seeds (per-tech losers). Expandable later if the user wants Broader-7.

**Phase 1b (optional, decide AFTER Phase 2 results):** train the Phase 2 winner (or clean + winner if the winner is a curriculum) at `xl` on the same data → `u716_dn_{recipe}_xl_{dev}` (4 runs max). Justification: the weight→basin map is tier-dependent (§2b#1) and the universal net fits ~3× the per-tech data, so `large` may sit below the universal over-fit boundary; a single xl probe answers whether the per-tech capacity curve (peaks at large) or the xl-curriculum surprise (corroft/crit10/crit15m 14/16) is the right prior at universal scope. Skip if Phase 2 shows universal is not viable at all (≤8/12 strict everywhere).

## 6. Phase 2 — Universal recipe evaluation & ranking (CPU)

**New script `scripts/uni_gate_sweep.sh`** — for each recipe: export `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}=u716_dn_{recipe}_large_{dev}`, CPU pins + `NGSPICE_BIN` + isolated `PYCIRCUITSIM_COMPLEX_RESULTS`, then run:

- **12 ranking gates**: `tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py --tech {TSMC7,TSMC12,TSMC16}` (one gate per invocation, parallel across cells; verdict = exit code; grep the `[NN-resolver]` line to confirm the pin took).
- **TSMC5 zero-shot row** (diagnostic only, expected to fail — embedding rows 0–3 are untrained at this point): same 4 circuits at TSMC5, with timeout guard.
- **Device suites** per recipe: `tests/verify_nn_dc_tran.py --tech TSMC7,TSMC12,TSMC16` and `tests/verify_nn_ac.py --tech TSMC{7,12,16}` → MRE / NRMSE / R² / max-error per tech (Rule 13). Skip the baseline-gated parametric sweeps (`verify_nn_multi_tech_*`, `verify_complex_*_sweep`) — their sha256-pinned baselines encode per-tech production checkpoints and don't apply to universal ckpts.
- **OMP determinism sweep** on the opamp+ring cells (6 per recipe; SRAM/SC are deterministic gates) for ALL four recipes: `OMP_NUM_THREADS=N PYCIRCUITSIM_TORCH_THREADS=N` for N ∈ {1,2,4} — the current retest standard (ACCURACY_REPORT's strict columns use exactly this set). Since the V6.6.6 thread-pin the gate infra defaults torch to 1 thread, so the probe MUST set `PYCIRCUITSIM_TORCH_THREADS` explicitly or every N reruns the identical pinned config. Classify each cell detPASS / detFAIL / FLIP per the report convention.

Results → `results/uni_bench/{recipe}/...` + summary table. **Ranking rule:** strict pass count on the 12 gates (FLIP = fail — unbankable); tie-break = device NRMSE, then AC gate. Report per-cell basin identity, not just counts (recipes tie at 10/12 per-tech while banking *different* opamp cells, §2b#4). Aggregate opamp stats follow the report convention: FLIP cells excluded from detPASS, tsmc7-opamp excluded from the mean (known-fail everywhere). Calibration: ≥10/12 strict = per-tech parity; 11/12 = realistic ceiling.

## 7. Phase 3 — TSMC5 fine-tune tiers (24 runs, GPU, cheap)

From the **clean** universal base (isolates the sample-count effect; if Phase 2's winner ≠ clean, replicate the best-N tier from the winner as a cross-check):

```
... --tech-scope universal --size large --device-type {dev} \
    --apply-filter off --swa-mode ema --seed 42 \
    --lr 3e-4 --epochs 120 --patience 40 \
    --init-from u716_dn_clean_large_{dev} \
    --data tsmc5ft_n{N}_{dev}.npz \
    --exp-name u716f5_{ftrecipe}_n{N}_large --cuda --overwrite
```

- `plain` ft: flags as above. `crit` ft: + `--class-weights traj_corridor=3.0,inv_trip=2.0`.
- 6 tiers × 2 ft-recipes × 2 devices = 24 runs; small tiers take minutes (full tier is the long pole).
- **Known caveat (inherent to the pipeline, no code change):** each fine-tune re-fits its normalizer on the tier's train split (`dataset.py:206-210`), so tiny-N tiers see noisy normalization stats and a norm shift vs the universal pre-train — this is part of the measured few-shot cost and gets reported as such. Val split = 10% of N → early stopping is noisy at N=2k; the fixed 120-epoch budget bounds it.

## 8. Phase 4 — Fine-tune evaluation (CPU)

Per (tier × ft-recipe) checkpoint pair, env-pinned as in Phase 2:

- **TSMC5 target accuracy**: 4 complex gates (`--tech TSMC5`) + `verify_nn_dc_tran.py --tech TSMC5` + `verify_nn_ac.py --tech TSMC5` → gate passes and MRE/NRMSE/R²/max-err vs N. Full-data per-tech bar (§2b#6): crit30f@large tsmc5 = 4/4 strict — ring 4.04%, opamp 0.21% detPASS, SRAM 6.31%, SC 2.06%.
- **Retention/forgetting**: `verify_nn_dc_tran.py --tech TSMC7,TSMC12,TSMC16` for every tier (cheap); the 12 complex gates for the full tier + the best small tier.
- OMP sweep (`OMP_NUM_THREADS=N PYCIRCUITSIM_TORCH_THREADS=N`, N ∈ {1,2,4}, opamp+ring) on TSMC5 for the best (tier, recipe) — same standard as Phase 2.

Deliverable curve: accuracy vs N (per device, per ft-recipe), zero-shot → full-data, plus the retention curve.

## 9. Phase 5 — Report + bookkeeping

- Keep THIS file updated as the execution log (per-phase status, surprises, dead ends).
- `docs/V6.7.0-universal-transfer-report.md` — TL;DR, methodology, universal recipe ranking table (12 gates, strict OMP∈{1,2,4}, detPASS/FLIP classification and aggregate conventions identical to `results/recipe_bench/ACCURACY_REPORT.md` so the tables are directly comparable), TSMC5 sample-efficiency curves, retention analysis, device metric tables (Rule 13), **best recipe(s) recommendation**, dead ends.
- `docs/CHANGELOG.md` — new V6.7.0 entry (docs + new scripts + checkpoints; no production change, resolver untouched).
- Memory file for durable findings.

## 10. Verification

1. Phase 0 validation checks (row sums, bit-identity spot check, class histograms, sidecar code ranges).
2. Every gate/test run must show `[NN-resolver] ... -> u716..._best.pt (scope=universal, tech_code=N)`. A missing pinned file now raises `FileNotFoundError` at parse time (V6.6.6) rather than silently falling back — so a run that *completes* with a `tsmc{X}_dn_large` resolution means the pin variable itself wasn't exported (typo/env not inherited), not a missing checkpoint.
3. Before any Phase 2/4 eval, check the checkpoint's `*_best.pt.complete` marker — a bare `_best.pt` may be a killed run (§2).
4. Sanity anchor: run universal-clean through `verify_nn_dc_tran --tech TSMC16` BEFORE burning gate time; if device NRMSE ≫ the per-tech large's, flag capacity/recipe mismatch early.
5. Production untouched: `git status` clean except new scripts/docs; per-tech spot-check (`verify_complex_opamp.py --tech TSMC5` WITHOUT env pins) must still resolve `tsmc5_dn_large_*`.

## 11. Risks

- **Universal may simply lose to per-tech** (that's why V6.1 retired it) — a negative result is still the answer to "best universal recipe"; the report quantifies the gap vs the per-tech 14/16 production matrix (10/12 strict on the shared ranking gates, §2b#4).
- **Tier mismatch:** the weight→basin map is tier-dependent (§2b#1) and the universal net trains on ~3× per-tech data — the per-tech-large recipe ranking may not transfer at `large`. Mitigation: Phase 1b optional xl arm; don't over-conclude from a single tier.
- **Basin relocation, not composition** (§2b#2): each recipe should be read as selecting a basin set; a universal recipe that banks tsmc16-opamp will plausibly drop something else, as csobcrit did per-tech. Report per-cell basin identity, not just X/12.
- GPU contention with resident jobs stretches Phase 1 (worst case: run waves serially).
- One shared MLP across 3 techs may show the known mutual-exclusive-basin behavior at gate level (V6.6.1 lesson) — expected, reported per-gate.
- Sequencing: Phases 1→2 and 3→4 are strictly ordered; total elapsed estimate 3–5 days, mostly unattended training/eval.

## 12. Execution log

- **2026-07-04 (Phase 4 main sweep RESULTS — 72 cells, zero RESOLVER-MISS; curve in `results/uni_bench/transfer_curve.tsv`):**
  - **TSMC5 gate-level onboarding threshold is SHARP between N=200k and N=1M:** complex gates 0/4 for every tier ≤200k (both ft-recipes); plain@n1M = **4/4 single-run (incl. opamp!)**; nfull = 3/4 both (opamp fails); crit@n1M = 1/4 (SRAM/SC/opamp fail — crit ≠ better at gates despite better 200k device metrics).
  - **TSMC5 device-level curve (DC NRMSE, nmos):** n2k unusable (sweep N/A) → n10k DIVERGED (NRMSE ~1e69 — the §7 normalizer-refit caveat manifests as full-blown divergence at tiny N, not just noise) → n50k 35 % → n200k 10.1 % plain / 5.9 % crit → n1M 0.69/0.57 % → full 0.42/1.08 %. PMOS full 0.64 %; tran 0.52 %, inv-tran 1.27 % at full.
  - **Retention/forgetting (no-replay cost):** monotone-ish degradation with fine-tune N — TSMC12 DC 5.8 % (n2k) → 11 % (n50k) → 17-23 % (n200k+) FAIL; TSMC16 ~10 %, TSMC7 ~5.5 % marginal at full. One blow-up outlier: plain@n1M destroys TSMC12 (NRMSE 26424 %) while crit@n1M retains 5.9 % PASS — forgetting damage is high-variance across runs, not a smooth curve. AC TSMC5: PASS only at full tier (both recipes).
  - §8 follow-ups dispatched via direct `_cell` invocation (appends to SUMMARY.tsv — a fresh dispatcher run would TRUNCATE the phase-4 rows; gotcha documented): OMP {2,4} × {ring,opamp} × TSMC5 + 12 retention complex gates × {TSMC7,12,16}, for plain@n1M + plain@nfull (32 cells).
- **2026-07-05 (Phase 1b training DONE → xl eval dispatched):** clean@xl (early-stopped ~epoch 800-class run, best val 1.9e-4 both devices) + corroft@xl (warm-started, 120-cap) all 4 ckpts + markers. Eval `SIZE=xl RECIPES="clean corroft"` (64 cells) launched. **First cells: clean@xl tsmc16-opamp PASS + tsmc12-opamp PASS** — the tsmc16 rail at `large` is tier-local, as per-tech xl results predicted (§2b#1).
- **2026-07-05 (Phase 1b RESULTS → CAMPAIGN CLOSED):** the xl basin partition is a clean trade, 11/12 NOT reached:
  - **clean@xl = 8/12 strict + 1 FLIP** — banks BOTH tsmc12-opamp (5.55 %) AND tsmc16-opamp (6.41 %) det-PASS, but loses ALL rings (tsmc7 13.36 % / tsmc12 7.00 % det-FAIL; tsmc16 5.19 % FLIP — PASS only @OMP4).
  - **corroft@xl = 8/12 strict, 0 FLIPs** — corridor holds rings 3/3 (tsmc7 4.76 %) but ALL opamps rail (tsmc12: gain err 100 %, trip shift 146 mV — per-tech corroft@xl's tsmc16-opamp bank does NOT transfer to universal xl) and tsmc7-SC drops (charge err 2.15 % of VDD; droop fine). Curiosity: tsmc7 AC PASS — the campaign's only AC pass, at xl of all places.
  - Verdict: **corroft@large (10/12 strict, 0 FLIP) stands as best universal config.** The mutual-exclusive basin wall (V6.6.1) reappears at universal xl partitioned as opamps-XOR-rings. xl zeroshot SRAM cells TIMEOUT at 2700 s (xl CPU inference slower; diagnostic only, not a ranking gate).
  - Phase 5 executed: report §7/§8 finalized, CHANGELOG V6.7.0 finalized, memory written. Campaign total: 36 checkpoints (8 large + 4 xl bases, 24 fine-tunes), 264 eval cells, zero RESOLVER-MISS, ~1.5 days wall.
- **2026-07-04 (Phase 4 COMPLETE — extras in):** **plain@n1M TSMC5 = 4/4 STRICT** (ring+opamp det-PASS at OMP {1,2,4}) = ties the per-tech production bar (§2b#6) **with half the per-tech data**; plain@nfull opamp det-FAIL (n1M > nfull at gates — non-monotone in N; full-tier opamp basin loss is deterministic, not a flip). **Gate-level retention collapses**: n1M keeps only tsmc7-SRAM (1/12), nfull keeps 3/12 (tsmc7-SRAM/SC + tsmc16-SRAM) — fine-tuned ckpts are per-tech models in practice; retention needs replay (out of scope; report finding). Phase 4 CLOSED; Phase 1b xl still training (~epoch 200).

- **2026-07-03 (pre-start revision):** plan revised against `results/recipe_bench/ACCURACY_REPORT.md` (V6.6.6 large retest, xl retest, V6.6.7 round-1). Changes: §2 env-pin now raises on absent stem + torch thread-pin + `.complete` markers; new §2b (tier-dependent basin map, relocation-not-composition, tsmc7-opamp known-fail, 10/12 per-tech bar, tsmc5 transfer target); §5 Core-4 swaps invtripft → corroft + optional Phase 1b xl arm; §6/§8 OMP standard aligned to {1,2,4} with `PYCIRCUITSIM_TORCH_THREADS`; §10/§11 updated accordingly. No execution yet.
- **2026-07-03 (scripts authored):** the four new standalone scripts landed (`scripts/uni_concat_npz.py`, `uni_subsample_npz.py`, `uni_train.sh`, `uni_gate_sweep.sh`) — reviewed against the revised plan: Core-4 recipe args, clean-base warm-start guard (with `.complete` check), completion markers, resolver post-check (`scope=universal` on both device stems → RESOLVER-MISS verdict), OMP sweep sets `PYCIRCUITSIM_TORCH_THREADS`, direct env-python under `timeout` (no `conda run` orphan). No dataset/checkpoint artifacts yet at this point.
- **2026-07-04 (Phase 0 DONE):** all 28 source files verified on disk (base + `_corro` × {tsmc7,12,16} + `tsmc5_corro`, both devices, with sidecars); GPU pre-flight: GPUs 1/2 fully FREE, GPU 0 ~19% (better than the plan's all-busy assumption — wave 1 gets dedicated GPUs). Both builders PASS all in-script validations (row sums, sidecar code ranges, class histograms, 100/100 bit-identical spot checks): `uni716_nmos` 6,920,730 rows / `uni716_pmos` 7,256,992 (match §2's 6.92M/7.26M estimates), `uni716_corro_{nmos,pmos}` 6,947,421/7,283,683; all 10 `tsmc5ft_n{N}` tiers exact-size with rare classes preserved (n2000 keeps traj_corridor=9, inv_trip=67 — the stratification point). Note: a stale `uni716_corro_pmos.npz` from a killed Jul-3 partial run was overwritten by this build. Gotcha fixed: the four scripts lacked +x (`xargs` invokes `$SELF` directly → Permission denied on first dispatch); `chmod +x` applied.
- **2026-07-04 (version designated):** campaign officially named **V6.7.0** (user request). Plan title + CHANGELOG updated (in-progress V6.7.0 entry at top, to be finalized at close); the four scripts already carried "V6.7.0 universal-transfer campaign" headers — no script edits needed.
- **2026-07-04 (Phase 1 wave 1 LAUNCHED):** `RECIPES="clean csob" GPUS="1 2" NSTREAMS=4 bash scripts/uni_train.sh` detached via setsid — 4 jobs: clean+csob nmos on GPU1, clean+csob pmos on GPU2. Logs: `results/uni_bench/train_logs/u716_dn_{clean,csob}_large_{nmos,pmos}.log`. Pre-launch CLI verification: `--exp-name u716_dn_{recipe}_large` + `--device-type` → save_prefix `u716_dn_{recipe}_large_{dev}` (cli/train.py:123-125); `--init-from <stem>` resolves to `CHECKPOINT_DIR/<stem>_best.pt` (trainer.py:660-674) — wave-2 stems as passed by uni_train.sh are correct.
- **2026-07-04 (Phase 1 wave 1 DONE, ~6-7.5h/job — far under the 1-2 day budget):** all 4 ckpts + `.complete` markers + `_norm.npz` on disk. Best val: clean 2.55e-4/2.70e-4 (n/p), csob 2.39e-4/2.64e-4. **Universal device fidelity ≈ per-tech at first look** — test-set (aggregated over 3 techs): id NRMSE 0.006-0.012% R²=1.0000, gm ~0.013%, gds 0.16%(n)/0.31%(p) R²≥0.993, charges 0.002-0.004%; csob shows its charge-axis signature (qg/qd ~2× tighter than clean, id unchanged) — same pattern as per-tech. ~27-33s/epoch on ~5.5-5.8M train rows (split 80/10/10).
- **2026-07-04 (Phase 1 wave 2 LAUNCHED + anchor):** `RECIPES="corroft crit30u"` — 4 curriculum jobs on GPUs 1/2, warm-started from `u716_dn_clean_large_{dev}` on `uni716_corro_{dev}.npz`, 120-epoch cap. In parallel (CPU): §10.4 sanity anchor — `verify_nn_dc_tran.py --tech TSMC16` with clean env-pinned (`results/uni_bench/anchor/clean_tsmc16_dc_tran.log`).
- **2026-07-04 (§10.4 anchor PASS → Phase 2 partial start):** universal-clean TSMC16 anchor = **6/6 PASS** (DC 0.02 %/0.01 % n/p, VTC 1.02 %, inv-tran post-startup 0.86 %) with the resolver line confirming `scope=universal, tech_code=12` end-to-end — no capacity/recipe mismatch, universal is viable at the device level. Phase 2 dispatched EARLY for the two finished recipes (`RECIPES="clean csob"`, 64 cells, NPAR=6 on 192 cores) overlapping wave-2 GPU training; corroft+crit30u eval dispatches after wave 2. First cell in: clean tsmc12-opamp PASS.
- **2026-07-04 (Phase 2 clean+csob RESULTS — 64 cells, zero RESOLVER-MISS):**
  - **u716 clean = 9/12 strict, ZERO FLIPs** — SRAM 3/3, SC 3/3, ring {12,16} det-PASS / tsmc7-ring det-FAIL (period err 14.89 %, NRMSE 56 % — the per-tech under-drive class, relocated from tsmc5 to tsmc7), opamp: **tsmc12 det-PASS (gain 176.1 vs 188.4, err 6.54 %, NRMSE 1.29 %)**, tsmc16 det-FAIL (gain rails to 0.0 — the classic railed class), tsmc7 det-FAIL (known-fail everywhere). 1 below the 10/12 per-tech parity bar; shortfall cell = tsmc7-ring. **Full OMP determinism at large is NEW** — per-tech large had endemic opamp flips.
  - **u716 csob = 8/12 strict, 1 FLIP** — same cells as clean except tsmc12-opamp FLIPs (OMP1 FAIL / OMP2,4 PASS → unbankable). Per-tech csob's tsmc16-opamp basin does NOT survive the scope change (det-FAIL, railed) — basin relocation (§2b#2) applies to scope, not just curriculum/tier. csob < clean at universal scope.
  - AC 0/3 both (expected at `large`, §2b#5); dev suites PASS both; TSMC5 zero-shot 0/4 both as predicted (untrained embedding rows; ring/SC fail in seconds, opamp ~17 min, SRAM ~32 min).
- **2026-07-04 (Phase 2 COMPLETE — universal ranking; corroft = per-tech parity):**
  | recipe | strict/12 | FLIPs | basins |
  |---|---|---|---|
  | **corroft** | **10/12** | **0** | SRAM 3 + SC 3 + ring 3 (tsmc7-ring FIXED: 14.89 %→3.61 %) + tsmc12-opamp det-PASS 6.15 % |
  | clean | 9/12 | 0 | tsmc7-ring det-FAIL; tsmc12-opamp det-PASS 6.54 % |
  | crit30u | 9/12 | 1 | ring 3/3 but tsmc12-opamp PASS@1/FAIL@2,4 FLIP — the inv_trip anchor RELOCATES the opamp basin at universal scope (§2b#2 confirmed across scope) |
  | csob | 8/12 | 1 | tsmc7-ring det-FAIL + tsmc12-opamp FLIP; per-tech csob's tsmc16-opamp basin does NOT transfer |

  Zero RESOLVER-MISS across 128 cells. Corridor = ring lever CONFIRMED at universal scope (fixes tsmc7-ring in both curricula, the cell that opened the campaign 1 short of parity). tsmc16-opamp railed for ALL four recipes at `large` → 11/12 ceiling not reached at this tier. AC 0/12 (expected, §2b#5). **Ranking rule verdict: corroft = best universal recipe @large, ties the per-tech 10/12 calibration bar with full OMP determinism (which per-tech large never had).**
- **2026-07-04 (Phase 1b GO — xl arm LAUNCHED):** winner is a curriculum → per §5: clean@xl (2 jobs, running) then corroft@xl (2 jobs, chained; worker guards on clean-xl `.complete`). Rationale: per-tech corroft@xl banked exactly the tsmc16-opamp cell that rails universally at large.
- **2026-07-04 (Phase 3 DONE, 24/24 no failures → Phase 4 main sweep LAUNCHED):** all 24 `u716f5_*` ckpts + markers. Phase 4 = ONE dispatch per stem (72 cells) with `TECHS=TSMC5 DEV_TECHS=TSMC5,TSMC7,TSMC12,TSMC16 SECTIONS="gates dev ac"` — the 4-tech dev cell measures TSMC5 accuracy AND 7/12/16 retention in one run (avoids the dispatcher's per-dispatch SUMMARY.tsv truncation that two separate dispatches per stem would cause). OMP sweep + retention complex gates for full/best tiers deferred until the curve identifies the best (tier, recipe). n2000 gates fail in seconds (low-N curve point, expected).
- **2026-07-04 (Phase 1 wave 2 DONE → Phase 2 wave-2 eval + Phase 3 LAUNCHED in parallel):** corroft/crit30u all 4 ckpts complete (warm-start lines verified; 3 of 4 early-stopped ~epoch 42 = patience 40 from an early best — typical warm-started curriculum; best val 2.8-3.9e-4). Phase 2 eval dispatched for both (64 cells; first cell corroft tsmc12-opamp PASS). **New script `scripts/uni_ft_train.sh`** (Phase 3 runner, uni_train.sh conventions: worker/xargs dispatcher, skip-existing, `.complete` markers, clean-base init guard; tier `full` = `tsmc5_corro_{dev}.npz` directly, stem `u716f5_{ftrecipe}_nfull_*`) — all 24 fine-tune jobs dispatched on GPUs 1/2; n2000 tiers finish in seconds.

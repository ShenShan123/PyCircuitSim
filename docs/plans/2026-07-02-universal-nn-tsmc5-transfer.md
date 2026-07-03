# Universal DirectNet (TSMC16/12/7) + TSMC5 Fine-Tune Transfer Study + Universal Recipe Comparison

**Status: PLANNED (not started).** This file is the live routing doc — update it on every phase change / lesson (workflow rule).

## 1. Goal & context

All production DirectNet checkpoints today are **per-tech** (V6.6.4, crit30, 14/16 strict). The universal-scope model was retired in V6.1 and has never been trained on the current data/recipe stack. This campaign answers three questions:

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
- Inference scope is decided by checkpoint **stem prefix** (`pycircuitsim/parser.py:210-218`): anything not starting with `tsmc{5,7,12,16}_dn_` → universal scope → correct universal tech_code per tech via `local_variant_code("universal", tech, vt)`. Env pin `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}` short-circuits the whole cascade (`parser.py:84-107,129-130`) and works for every gate/test. The pin is **silently ignored if the file doesn't exist** — always verify the `[NN-resolver] ... scope=universal` stdout line.
- `pycircuitsim/models/mosfet_directnet.py:44-83` infers `num_tech_codes` from the state dict → 18-row checkpoints load with no code changes.
- Existing sweep scripts (`recipe_train.sh`, `gate_matrix_iso.sh`, `recipe_eval.sh`, `recipe_multirun_gate.sh`) hardcode `tsmc{X}_dn_` stems → unusable for universal checkpoints (symlinking to those names would corrupt tech_code for 3 of 4 techs). New runner required.
- Env: `conda run -n pycircuitsim` (env lives at `/data1/shenshan/.conda/envs/pycircuitsim`, NOT `/home/...`); 3× RTX 4090 currently ~100% busy with resident jobs (~17–20 GB free each); `/data2` has 6.2 TB free. Gates run CPU-pinned: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `NGSPICE_BIN=$PWD/tools/ngspice-45.2/bin/ngspice`.
- Dataset sizes on disk (rows, nmos/pmos): tsmc7 1.82M/2.19M, tsmc12 2.55M/2.53M, tsmc16 2.55M/2.54M, tsmc5 2.02M/2.02M → universal-716 ≈ 6.92M (nmos) / 7.26M (pmos) rows.

## 3. Naming scheme (load-bearing)

| Artifact | Stem pattern |
|---|---|
| Universal dataset | `uni716_{nmos,pmos}.npz` (+ `uni716_corro_*` variant) + `_tech_variant_labels.npy` sidecars |
| TSMC5 tier datasets | `tsmc5ft_n{N}_{dev}.npz` + sidecars (stratified from `tsmc5_corro_{dev}.npz`) |
| Universal base ckpts | `u716_dn_{clean,csob,invtripft,crit30u}_large_{dev}` (via `--exp-name`) |
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
| `invtripft` | `--class-weights inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40 --init-from u716_dn_clean_large_{dev}` | `uni716_{dev}.npz` | clean |
| `crit30u` | `--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40 --init-from u716_dn_clean_large_{dev}` | `uni716_corro_{dev}.npz` | clean |

`--exp-name u716_dn_{recipe}_large`; `--data` passed explicitly. Wave 1 = clean+csob (4 long jobs), wave 2 = the two curriculum fine-tunes (4 fast jobs).

**GPU policy (default, user can override):** pre-flight `nvidia-smi`; launch round-robin on GPUs 0/1/2 alongside the resident jobs (memory suffices; compute is time-sliced). ~3× per-tech-large epoch time on ~7M rows — budget 1–2 days wall for wave 1. Nothing gets killed without asking.

Recipe-set rationale (asked, no response → flagged default **Core-4**): clean = baseline (per-tech 13/16), crit30-analog = per-tech production winner (14/16), csob = best device/AC all-rounder, invtripft = single-lever control. ekv/sob/seeds were per-tech losers — skipped; expandable later if the user wants Broader-7.

## 6. Phase 2 — Universal recipe evaluation & ranking (CPU)

**New script `scripts/uni_gate_sweep.sh`** — for each recipe: export `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}=u716_dn_{recipe}_large_{dev}`, CPU pins + `NGSPICE_BIN` + isolated `PYCIRCUITSIM_COMPLEX_RESULTS`, then run:

- **12 ranking gates**: `tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py --tech {TSMC7,TSMC12,TSMC16}` (one gate per invocation, parallel across cells; verdict = exit code; grep the `[NN-resolver]` line to confirm the pin took).
- **TSMC5 zero-shot row** (diagnostic only, expected to fail — embedding rows 0–3 are untrained at this point): same 4 circuits at TSMC5, with timeout guard.
- **Device suites** per recipe: `tests/verify_nn_dc_tran.py --tech TSMC7,TSMC12,TSMC16` and `tests/verify_nn_ac.py --tech TSMC{7,12,16}` → MRE / NRMSE / R² / max-error per tech (Rule 13). Skip the baseline-gated parametric sweeps (`verify_nn_multi_tech_*`, `verify_complex_*_sweep`) — their sha256-pinned baselines encode per-tech production checkpoints and don't apply to universal ckpts.
- **OMP determinism sweep** `OMP_NUM_THREADS ∈ {1,2,3,4,8}` on the 12 gates for the top-2 recipes (V6.6.3 strictness standard).

Results → `results/uni_bench/{recipe}/...` + summary table. **Ranking rule:** strict pass count on the 12 gates across the OMP sweep; tie-break = device NRMSE, then AC gate.

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

- **TSMC5 target accuracy**: 4 complex gates (`--tech TSMC5`) + `verify_nn_dc_tran.py --tech TSMC5` + `verify_nn_ac.py --tech TSMC5` → gate passes and MRE/NRMSE/R²/max-err vs N.
- **Retention/forgetting**: `verify_nn_dc_tran.py --tech TSMC7,TSMC12,TSMC16` for every tier (cheap); the 12 complex gates for the full tier + the best small tier.
- OMP sweep {1,2,3,4,8} on TSMC5 gates for the best (tier, recipe).

Deliverable curve: accuracy vs N (per device, per ft-recipe), zero-shot → full-data, plus the retention curve.

## 9. Phase 5 — Report + bookkeeping

- Keep THIS file updated as the execution log (per-phase status, surprises, dead ends).
- `docs/V6.7.0-universal-transfer-report.md` — TL;DR, methodology, universal recipe ranking table (12 gates, strict OMP), TSMC5 sample-efficiency curves, retention analysis, device metric tables (Rule 13), **best recipe(s) recommendation**, dead ends.
- `docs/CHANGELOG.md` — new V6.7.0 entry (docs + new scripts + checkpoints; no production change, resolver untouched).
- Memory file for durable findings.

## 10. Verification

1. Phase 0 validation checks (row sums, bit-identity spot check, class histograms, sidecar code ranges).
2. Every gate/test run must show `[NN-resolver] ... -> u716..._best.pt (scope=universal, tech_code=N)` — never a `tsmc{X}_dn_large` fallback (that means the env pin was silently dropped: wrong stem or missing file).
3. Sanity anchor: run universal-clean through `verify_nn_dc_tran --tech TSMC16` BEFORE burning gate time; if device NRMSE ≫ the per-tech large's, flag capacity/recipe mismatch early.
4. Production untouched: `git status` clean except new scripts/docs; per-tech spot-check (`verify_complex_opamp.py --tech TSMC5` WITHOUT env pins) must still resolve `tsmc5_dn_large_*`.

## 11. Risks

- **Universal may simply lose to per-tech** (that's why V6.1 retired it) — a negative result is still the answer to "best universal recipe"; the report quantifies the gap vs the per-tech 14/16 production matrix.
- GPU contention with resident jobs stretches Phase 1 (worst case: run waves serially).
- One shared MLP across 3 techs may show the known mutual-exclusive-basin behavior at gate level (V6.6.1 lesson) — expected, reported per-gate.
- Sequencing: Phases 1→2 and 3→4 are strictly ordered; total elapsed estimate 3–5 days, mostly unattended training/eval.

## 12. Execution log

*(empty — fill per phase during execution)*

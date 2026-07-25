# V6.13.0 — A3 gds fix + TSMC6 retire + full re-gate — STATUS / HANDOFF

**Date:** 2026-07-24. **State:** code fixes shipped and committed; re-gate ~90 %
complete when the first session ended and the background jobs were killed. This file
is the resume point. Companion memory: `v6130-a3-fix-regate-campaign`.

## RESUMED 2026-07-24 20:10 — session 2 (this session)

The three unfinished compute blocks below were relaunched, and a second workstream
(the remaining systematic-audit bug fixes) runs alongside them.

- **Driver:** `/data2/shenshan/a3_gate_snap/a3_resume.sh`, launched detached
  (`setsid nohup`). It runs three pools concurrently — 33 missing matrix cells
  (PAR=33), 45 missing OMP runs (PAR=12), and `nn_ac_tf` — and folds the results
  back into `results/a3_regate/`. Progress: `results/a3_regate/_resume/`,
  `results/a3_regate/_resume_main.log`.
- **Frozen code snapshot.** The gates run from `/data2/shenshan/a3_gate_snap`, an
  rsync of the repo at `d2ea720` with `checkpoints/`, `tools/` and `PyCMG/`
  symlinked back. This decouples the campaign from the repo working tree, so
  bug-fix edits landing during the run cannot change the numerics half-way
  through. Every V6.13.0 number is therefore measured at exactly `d2ea720`.
- **Per-OMP resume.** `scripts/a3_omp_one.sh` (new) runs ONE OMP value and writes a
  sidecar `<log>.omp<n>`, instead of `recipe_multirun_gate.sh`'s all-of-{1,2,4}.
  The killed sweeps had already banked 1–2 of the 3 runs per cell; re-running only
  the missing values saves ~40 BSIM-AR gate-hours. Sidecars are folded into the
  cell log in OMP order when the pool finishes (concurrent appends to one log
  interleave, hence the sidecar).
- **Strict collector.** `scripts/a3_regate_omp_collect.py` (new) turns the OMP logs
  into per-group strict verdicts → `results/a3_regate/OMP_REPORT.md`.
- **Bug-fix worktree.** `/data2/shenshan/pcs-fixes` on branch `audit-fixes`, so the
  audit fixes never contaminate the V6.13.0 measurement commit on `main`. The
  triage of the 43 remaining audit findings and their two-wave ordering live in
  **`docs/plans/2026-07-24-audit-fix-waves.md`** — wave 2 is gate-affecting and is
  blocked on this campaign's results being committed.

## What is DONE and committed (branch `main`, NOT pushed)

| commit | what |
|---|---|
| `8ed35bd` | **gds fix** — negate `gds` (not just gm/gmb) + replace two-sided floor `max(gds,\|id\|·0.5)` with **guard F** (negatives→`\|id\|/50V`, positives bit-identical). `_floor_gds`→`_guard_gds`. `PYCIRCUITSIM_GDS_FLOOR_K` now rejected loudly; new knob `PYCIRCUITSIM_GDS_GUARD_K` (default 0.02). |
| `38c47d8` (+ PyCMG submodule `23b0ace`) | **retire TSMC6** — deleted 22 ckpts, datasets, `results/tsmc6_gate`, registry/driver/test entries. Tail codes 22-24 → nothing renumbered (`NUM_TOTAL_CODES` 25→22). Added `bsimar.config.assert_tech_is_distinct()`. Raw vendor PDK kept (IP), unreferenced. |
| `0126c44` | **sobolev gds guard** — `SobolevIdLoss` raises if it would supervise the reverse-corridor-corrupted gds column (audit A3-data). Latent; `--sobolev` default-off. |
| `88db8f3` | **fix `uni_gate_sweep.sh` clobber** — was truncating `SUMMARY.tsv` every dispatch. `UNI_OUT` now overridable. |
| `d26f321`, `1559583` | docs: retire TSMC6 sections in all 3 accuracy reports (kept each report's methodological lesson), CLAUDE.md/README → 5 techs, `scripts/a3_regate_collect.py` + `a3_regate_uni_collect.py`, collector baseline fix (dn_large slot carries crit30 weights). |

Sanity re-run after edits: `bsimcmg_op/dc/tran`, `subckt` 8/8 all PASS.

## Re-gate results IN HAND (frozen in `results/a3_regate/REPORT.md`)

**All single-run OMP=1 unless "strict" stated. Directly comparable to the
pre-fix single-run columns in the accuracy reports at `a96112a`.**

### Complex matrix — 25/28 groups complete (per-tech + universal)

DirectNet (all 10 done):

| group | pre | post | |
|---|---|---|---|
| dn/clean/large (= production crit30 weights) | 14 | **15** | **+1, and strict 15/16 zero-flip (OMP-confirmed)** |
| dn/crit30f/large (provenance copy) | 14 | **15** | independent re-measure agrees |
| dn/clean/small | 7 | **10** | +3 |
| dn/clean/medium | 10 | 10 | 0 |
| dn/clean/xl | 10 | **12** | +2 |
| dn/corroft/xl | 14 | **15** | +1 |
| dn/crit10/xl | 14 | 14 | 0 |
| dn/crit15m/xl | 14 | **16** | **+2 — 16/16 STRICT, zero flips (opamp+ring 8/8 at OMP 1/2/4; sram+swcap 8/8 det). FIRST uniform-recipe DirectNet full sweep; tsmc7-opamp passes strict.** |
| dn/csob/large | 12 | **11** | −1 (only regression) |
| dn/v660clean/large (genuine clean) | 13 | 13 | 0; strict 5/8 multistable, zero-flip |

PFN (all 3 done): small 11/16 (0, strict-confirmed flip-free), medium 10→**11**,
large 8→**9**.

BSIM-AR (done): clean small 12→**14**, medium 14/14, large 13→**14**, xl 13→**14**;
corro15 medium **16/16**, xl 15→**16**; corroft medium 15→**16**; invtrip large 13→**14**.
Ceiling moved **opamp→ring** (tsmc5-ring + tsmc7-ring, period 6.8–8.6 %).

Universal DN (8 stems, strict-vs-strict, `results/a3_regate_uni/REPORT.md`):
**net +3 cells, all 3 pre-existing OMP FLIPs eliminated (flip-free on all 8).**

### Device / parametric / L72 suites (`results/a3_regate/suites/`)

| suite | result |
|---|---|
| nn_ac_dn | **8/8** device AC (was railed) |
| nn_ac_pfn | **8/8** device AC (was 5/8) |
| nn_dc_tran | 24/24 |
| nn_lifted_source_dc | 12/12 (Rule 2 canary) |
| nn_multi_tech_dc | 54/55 — the one fail `TSMC12_pmos_nfin_10` is **bit-identical** to pre-fix → **DC exactly invariant confirmed at scale** |
| nn_multi_tech_tran | 64/64 |
| ac_l72 / multi_tech_{dc,tran}_l72 / bsimcmg_{dc,tran}_comprehensive | all PASS (L72 control). `multi_tech_dc_l72` 1 ERROR = `TSMC5_lvt_inv_l_24nm` NR non-convergence in the **pure BSIM-CMG path**, not NN — pre-existing, unrelated. |
| complex_opamp_ac | 0/4 — opamp open-loop AC still fails the full GBW/PM/mag gate (OP un-rails but criteria not all met); was 0/12 pre-fix. Not a regression. |

**HEADLINE: `tsmc7-opamp` is NO LONGER the universal ceiling** — passes for
BSIM-AR at every size and DirectNet at small/xl. The gds floor was masking a
railed OP (audit A3-measured). This contradicts CLAUDE.md §Overview and several
memories; must be retracted in the doc pass.

## What is NOT yet done (resume here)

### 1. Finish the 3 BSIM-AR groups that were still running when killed
`results/a3_regate/`: `tf_crit15m_xl` (8/16), `tf_csob_xl` (8/16),
`tf_corroft_xl` (9/16), and finish `tf_corroft_large` (15/16), `tf_crit15m_large`
(14/16), `tf_crit30_large` (14/16), `tf_crit30_xl` (10/16). These are the
8–12 h AR-opamp cells. Re-run with (per group, `tag=tf`, `MODEL=transformer`):
```
GATE_OUT="$PWD/results/a3_regate/<group>" GATE_SCRATCH=<scratch>/<group> \
  RECIPES="<recipe>" TECHS="TSMC5 TSMC7 TSMC12 TSMC16" \
  CIRCS="ring_osc opamp sram_snm switchcap" SIZE="<size>" MODEL=transformer \
  PAR=6 NN_PY=/data1/shenshan/.conda/envs/pycircuitsim/bin/python \
  bash scripts/gate_matrix_iso.sh
```
The driver writes per-cell `.cell_*` and rebuilds SUMMARY.txt; it has **no
per-cell timeout** (cells only died because the driving process was killed).
Launcher template: `scratchpad/redo_groups.sh` (session-scoped path — recreate).

### 2. Finish the strict-OMP sweeps for the headliners
`results/a3_regate/omp/`. DONE: dn/clean/large (15/16 strict, 0 flip),
dn/v660clean/large (5/8 multistable), pfn/clean/small; **dn xl set complete —
crit15m/xl 8/8 multistable strict = 16/16 STRICT overall (0 flip, LANDMARK),
corroft/xl 7/8, crit10/xl 6/8, clean/xl 4/8**. STILL PARTIAL (were running when
killed): all **tf** headliners (clean/large, corro15/xl, corroft/medium) +
dn_crit10_xl_TSMC7_ring (one straggler line). Driver per cell:
`MODEL=<m> bash scripts/recipe_multirun_gate.sh <recipe> <size> <TECH> verify_complex_{opamp,ring_osc}`
→ writes 3 `OMP=n -> PASS/FAIL` lines. A cell is strict-PASS iff all 3 pass.
Only opamp+ring are multistable; sram+switchcap are deterministic (take from
single-run). Collector snippet is inline in the session; or grep the logs.

### 3. nn_ac_tf (BSIM-AR device AC) — was still RUNNING
`results/a3_regate/suites/nn_ac_tf.log`. Expect 8/8 by analogy with DN+PFN.
Re-run: `PYCIRCUITSIM_NN_FORCE_LEVEL=74 python tests/verify_nn_ac.py` (CPU-pinned).

### 4. THEN the documentation pass (single sweep)
- `docs/accuracy/DirectNet-L73-accuracy.md`: §1 headline (production 15/16
  strict flip-free, banks tsmc16-opamp; tsmc7-opamp sole open cell), §11
  cross-family, §12.1 open gates, **§12.2 rewrite** — fix is SHIPPED now (the
  "still floors k=0.5 / not shipped" text is FALSE), fold in the confirmed
  matrix/AC/DC numbers. Add the crit15m/xl 16/16 result (pending strict) to the
  xl narrative. §12.4 denominator now clean /16.
- `docs/accuracy/BSIM-AR-L74-accuracy.md`: headline + §9 — opamp un-rails,
  ceiling now ring; corro15 & corroft full 16/16 sweeps.
- `docs/accuracy/PFN-L75-accuracy.md`: device AC 8/8, matrix +1/+1/0.
- `CLAUDE.md`: production line (DN 15/16), the "tsmc7-opamp universal ceiling /
  only T3 reached it" claims in §Overview + BSIM-AR bullet — RETRACT.
- `docs/CHANGELOG.md`: **new V6.13.0 entry** (draft in memory). Note V6.12.1's
  "gds fix is NOT shipped" line is now superseded.
- Memories: mark `nn-gds-sign-bug-open` RESOLVED; correct the tsmc7-opamp
  "unreachable" assertions in `v680-bsimar-transformer-15of16-strict`,
  `v659-t3-solver-lands-opamp-16of16`, and the DN production memories.

### 5. Commit the results snapshot + plan, then (only if the user asks) push.

## Gotchas learned this session
- `gate_matrix_iso.sh` and `uni_gate_sweep.sh` gates need CPU pinning
  (`CUDA_VISIBLE_DEVICES="" OMP=MKL=1`), repo ngspice; already baked into the drivers.
- Never drive these long gates through Agent/workflow subagents — the agent's
  Bash call ends and orphans/kills the gate (that is why 16 groups had to be
  re-run). Launch them as plain background bash with the driver's own concurrency.
- Cluster load was ~1400–1700 from other users all session → AR cells much
  slower than nominal.

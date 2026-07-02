# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology.

---

## V6.6.4 — crit30f PROMOTED to production (branch `V6.6`, 2026-07-02)

**Production `large` now carries the uniform crit30 recipe.** All 8 `tsmc{X}_dn_large_{nmos,pmos}`
slots were replaced with bit-identical copies of the V6.6.3-validated `tsmc{X}_dn_crit30f_large_*`
checkpoints (clean base + one identical curriculum fine-tune `--class-weights
traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40 --init-from` own clean large,
on the ring-only `corro` corridor data). The V6.6.0 clean-large originals are archived on disk as
`tsmc{X}_dn_v660clean_large_*` (re-gateable as recipe `v660clean`); `small/medium/xl` remain the
clean recipe. Checkpoints are gitignored — the promotion is a disk operation; this entry + CLAUDE.md
are the record.

**Post-promotion verification (default resolver path, no env pins):** 16-cell complex matrix +
opamp/ring OMP∈{1,2,4} sweep on the production stems reproduce the crit30f record — **14/16
deterministic** (tsmc5-ring 4.04 %, tsmc5-opamp 0.21 %, tsmc12-opamp 6.25 %, all SRAM/switchcap
PASS; tsmc7-opamp + tsmc16-opamp remain the two structural/basin FAILs). Production baseline moves
13/16 (fragile: 12 strict) → **14/16 strict**.

---

## V6.6.3 — Full-recipe re-test: crit30 supersedes crit15 at 14/16 STRICT (branch `V6.6`, 2026-07-02)

**Headline: re-testing ALL 22 on-disk uniform recipes under one uniform discipline (16-cell
matrix, OMP∈{1,2,4} determinism sweep for every recipe, opamp-AC matrix, finalist device
suites + lifted-source canary) shows the best uniform recipe is `crit30`
(`traj_corridor=3.0,inv_trip=2.0` curriculum) at 14/16 STRICT all-OMP — clean+2 — not V6.6.2's
crit15 (13 strict).** Validated by `crit30f`: the original crit30 training had been killed at
heterogeneous epochs (30–92), so all 8 checkpoints were retrained to the full spec — the honest
rerun reproduces the artifact cell-for-cell (tsmc5-opamp 0.21 %, strict 14/16). Full verdicts:
plan §15 (`docs/plans/2026-07-01-uniform-recipe-beyond-13of16.md`); tables
`results/recipe_bench/{RETEST_ACCURACY.md,DEVICE_RETEST.md,retest_data.json}`.

**What crit30 banks deterministically (OMP∈{1,2,3,4,8}, uncontended, isolated):** all 4 rings
(tsmc5 12.66→4.04 %, tsmc7 4.82→2.40 %, tsmc12 2.68 %, tsmc16 2.90 %), all 4 SRAM, all 4
switchcap, tsmc12-opamp 6.25 %, **and tsmc5-opamp 0.21 %** — the cell that is a coin-flip in
clean (2.1/100/0.7 across OMP) and detFAIL in crit15. Device level: ≥ clean everywhere
(device-DC mean NRMSE 1.64→1.46 %, device-AC 4/8→6/8, canary all-PASS). Within-gate continuous
regressions, reported honestly: SC droop max 13→32 % of allowance; tsmc12 opamp-AC gain err
5.1→9.8 dB (both sides FAIL the ≤3 dB AC gate regardless).

**V6.6.2 corrections (single-run coin-flip artifacts, per its own meta-lesson):** "crit20/crit30
collapse opamps" is wrong (crit20 = 13 strict; crit30 = 14); crit15's single-run tsmc16-opamp
PASS was the 7.1/100/100 flip. The corridor-weight → tsmc5-opamp-basin map is **non-monotone**
(w1.0 FLIP, w1.5/w2.0 detFAIL, w3.0 detPASS) — the inv_trip anchor makes w3.0 safe where corroft
(w3.0 alone) railed it. csob re-scoped: the only tsmc16-opamp detPASS + only opamp-AC PASS +
best tsmc12 device fit, but its tsmc12-opamp flips at OMP=8 and 2 rings detFAIL → 12 strict
(stays the AC/device alternative). tsmc7-opamp: 100 % across all 23 artifacts × all OMP —
structural wall unchanged (EKV+T3).

**Production: UNCHANGED** (clean large). **crit30/crit30f is the promote candidate** (user
decision). New infra: `scripts/{recipe_retest_collect.py,device_retest_collect.py,
device_matrix_iso.sh}`; env-overridable results dirs in the device suites
(`PYCIRCUITSIM_NN_RESULTS`) and gate harness (`GATE_OUT`/`OPDEF_OUT`/`OMPS`); lifted-source
canary `--techs` filter. Artifacts: `tsmc*_dn_crit30f_large_*` (8), prior §14 gate runs archived
as `results/recipe_bench/{gate_iso,opamp_def}_v662_prior`.

---

## V6.6.2 — The cross-wall combo breaks 13/16: crit15 = clean+1 (branch `V6.6`, 2026-07-02)

**Headline: the V6.6.1/§13 "13/16 is the uniform ceiling" verdict is REFUTED. A single
uniform recipe — `crit15` — nets 14/16 (single-run) / 13/16 (strict all-OMP) = clean+1,
and the +1 is the *deterministic* tsmc5-ring opening that the plan declared reachable only
by a per-case special or a structural fix.** Full re-verification + the new result live in
`docs/plans/2026-07-01-uniform-recipe-beyond-13of16.md` §14.

**Re-verification.** Independent isolated-dir CPU-pinned re-gate of every on-disk recipe
(`scripts/gate_matrix_iso.sh`, per-cell `PYCIRCUITSIM_COMPLEX_RESULTS` isolation) reproduced
the §13 single-run net counts EXACTLY: clean 13, invtripft 12, invtrip 11, cor 11, corft 9,
corrft 12, corroft 13, corro15 13, csob 12.

**The lever.** §12 tested the trajectory corridor (class 12, opens tsmc5-ring, drifts an
opamp) and inv_trip (class 7, opamp-margin holder, inert on the ring) only SEPARATELY —
always on opposite sides of the shared-MLP wall. The `corro` (ring-only corridor) dataset
carries BOTH classes, so one uniform `--class-weights traj_corridor=1.5,inv_trip=2.0` weights
them together. **crit15** = that combo, curriculum warm-start (`--lr 3e-4 --epochs 120
--patience 40 --init-from {tech}_dn_large_{dev} --data {tech}_corro_{dev}.npz`), same flags
for all 32 checkpoints — inside the honest-uniform contract.

**Deterministic (bankable) results — identical across OMP∈{1,2,4}:** tsmc5-ring 12.66%→4.0%
PASS (the headline +1), tsmc7-ring 4.82%(0.18%-margin)→2.4% (de-fragilized), tsmc12-ring 3.2,
tsmc16-ring 2.8, tsmc12-opamp 6.3% (robust, held), all 4 SRAM + all 4 switchcap PASS. crit15
Pareto-dominates clean robustly (every cell clean deterministically passes, crit15 also passes,
plus the ring).

**The opamp gate is a multistable coin-flip (v648/v659 confirmed live).** tsmc5-opamp and
tsmc16-opamp DC-gain flip between ~0-8% and 100% across OMP thread count *even uncontended*, in
BOTH clean and crit15. clean's single-run "tsmc5-opamp PASS" (2.1/100/0.7) and crit15's
"tsmc16-opamp PASS" (7.1/100/100) are each one coin-flip — they cancel; the deterministic ring
is the real differentiator. `scripts/opamp_sweep_def.sh` is the OMP∈{1,2,4} multistability probe.

**Round-2 (15/16 attempt) — NEGATIVE.** crit15m (inv_trip=3.0), crit15h (inv_trip=4.0),
crit10 (corridor=1.0): tsmc12-opamp was already robust in crit15, so more inv_trip added nothing
and KILLED tsmc16-opamp's O1-pass; crit20 (corridor=2.0) collapsed all four opamps. crit15's
w1.5/inv2.0 is the sweet spot; no crit variant reached a robust 15/16. tsmc7-opamp = 100% across
all 6 crit recipes × all 3 OMP (structural non-existence, unchanged — EKV+T3 territory).

**Dead-ends recorded:** crit15m/crit15h/crit10 (stronger anchor / gentler corridor — break
tsmc16-opamp), crit20/crit30 (higher corridor weight — collapse opamps; crit30 under-trained,
stragglers killed to prioritize the promising arm). All artifacts on disk
(`tsmc*_dn_{crit15,crit15m,crit15h,crit10,crit20,crit30}_large_*`), clean production untouched.

**Production:** UNCHANGED (clean `large`, resolver large-first). crit15 robustly dominates clean
and is the recommended promotion candidate (banks the deterministic tsmc5-ring), but shipping it
changes the model — deferred to the user. New harnesses: `scripts/gate_matrix_iso.sh`,
`scripts/opamp_sweep_def.sh`, `scripts/gate_grid.py`; `recipe_train.sh` extended with `crit*`.

---

## V6.6.1 — Uniform-recipe comparison sweep (branch `V6.6`, 2026-07-01)

**Headline: swept a family of uniform training recipes (charge-Sobolev, Sobolev,
EKV core, a uniform-seed sweep, and two combos) across all four capacity scales to
answer "is there a better uniform recipe than V6.6.0 clean?" — the answer is NO at
the production `large` tier.** Full `large` scoreboard vs NGSPICE BSIM-CMG: **clean
13/16** > csob 12 > cs7 / s7 / s17 11 > csobekv / ekv / s123 10 > sob 5. Report:
`docs/V6.6.1-recipe-accuracy-report.md`; auto-tables `results/recipe_bench/RECIPE_REPORT.md`.

**Recipes (uniform addenda on the clean base `--apply-filter off --swa-mode ema
--seed 42`):** `csob` (`--charge-sobolev`), `sob` (`--sobolev --sobolev-corridor-only`),
`ekv` (`--ekv-core`), `s7/s17/s123` (`--seed N`), combos `csobekv`
(`--charge-sobolev --ekv-core`), `cs7` (`--charge-sobolev --seed 7`). Checkpoints
saved `tsmc{X}_dn_{recipe}_{size}_{dev}` (clean production ckpts never clobbered).
Infra: `scripts/recipe_{train,eval}.sh` + `scripts/recipe_collect.py`.

**Findings:**
- **No uniform recipe/seed/combo beats clean's 13/16 at `large`.** The ceiling is a
  **mutual exclusivity of the value-surface basins** (opamp high-gain, ring
  switching-edge, SRAM butterfly): each recipe lands a *different* subset, and
  winning one loses another. Seed-42 (clean) lands the largest compatible set. The
  uniform-seed sweep confirms it: s7/s17 = 11, s123 = 10 — and crucially **tsmc12
  opamp passes only on seed-42** (every seed change breaks it).
- **Combo hypothesis refuted, instructively.** `csobekv` is the only checkpoint to
  pass tsmc7-ring **and** tsmc16-opamp together, and pushes tsmc5-ring to 5.80 %
  (closest large ckpt to the 5 % gate) — but the **EKV analytic core breaks
  tsmc12+tsmc16 SRAM and tsmc12 opamp**, netting 10/16. Basin-stacking is zero-sum.
  `csobekv` also converged to val-loss 10–40× worse than clean (EKV + charge-Sobolev
  fight on the shared `id` head). `cs7` = 11 (seed-7 lands tsmc5-opamp, breaks
  tsmc12-opamp).
- **`csob` is the best all-rounder** (charge-Sobolev, which supervises the autograd
  ∂q/∂V the AC/transient solvers consume): best device NRMSE @ `large` (1.50 vs
  1.70), better AC (5/12 vs 4/12; `csobekv` best at 6/12), more total gate-passes
  summed over scales (41 vs 40), lands tsmc16 opamp — for −1 complex gate at `large`.
  **Refutes the V6.5.x "charge-Sobolev dead on arrival" verdict** (that was measured
  only at `medium`; at `large`/`xl` it is a measurable device+AC win).
- **Also confirmed:** `sob` (Sobolev) is a KILL (5/16 — "deriv-fidelity ⟂ opamp"
  reproduced); the capacity curve (peak at `large`, over-fit at `xl`) holds under
  every recipe.

**Default:** production default UNCHANGED — **clean `large` remains optimal at the
production-tier headline metric.** `csob` is promoted to a first-class documented
alternative (pin via `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}=tsmc{X}_dn_csob_large_{dev}`).
All recipe checkpoints retained on disk for the study. _(The slow EKV-core combo
medium/small/xl tiers were not completed — `csobekv` is a confirmed non-winner at
`large`, so its remaining tiers cannot change the decision; the four core recipes
already cover the full small→xl curve.)_

---

## V6.6.0 — House-clean + uniform-recipe reset (branch `V6.6`, 2026-06-29)

**Headline: a deliberate reset from the V6.5.9 hand-tuned 16/16 to a clean,
reproducible baseline that reflects the GENUINE fidelity of the NN compact model.**
No more per-case special recipes — every DirectNet checkpoint is trained from
scratch on **one identical recipe**, and production = the uniform **`large`** tier
(the best uniform capacity, **13/16 complex gates**), not a cherry-picked
best-config-per-tech mix. Capacity curve: **small 7 → medium 10 → large 13 → xl 10
/ 16** (peaks at `large`, over-fits at `xl` — the classic DirectNet boundary). Full
accuracy matrix in `docs/V6.6.0-accuracy-report.md`.

**Why:** V6.5.x reached 16/16 only via bespoke per-tech interventions (tsmc5 ring
corridor + seed-7, tsmc16 seed-17, tsmc7 T3 differentiable-DC-solver + EKV core).
Those answer "can a tuned checkpoint pass this gate?", not "how faithful is the NN
model under one honest recipe?". V6.6.0 measures the latter.

**House-clean (all on branch `V6.6`):**
- **Datasets:** purged the stale `_v2/_cor/_v2cor` variants + `datasets_v2_backup/`
  (16 GB → 4.5 GB); kept the 8 clean `datasets/{tech}_{nmos,pmos}.npz` benchmark
  inputs.
- **Artifacts:** removed all version-pinned `results/*`, `training_logs/`,
  `tests/*_results/`, `__pycache__`, and the regenerable example-sim outputs.
- **Plans/reports:** removed `docs/plans/*` (executed V6.5.x forward plans) and
  `docs/test-infra-bug-report-2026-06-28.md` (fixes already landed in `b1ae08a`).
- **Scripts:** removed the version-pinned one-offs `scripts/v6_5_{4,5,8}_*`
  (corridor / KCL / T3 / seed-sweep campaign drivers); kept the durable
  `benchmark_*` pipeline + `train_per_tech_8cells.sh`.
- **Tests:** removed 11 gate-specific routing diagnostics (`diag_opamp_*`,
  `diag_g3_cdd_match`, `diag_nn_ring_trajectory`, `diag_nn_switchcap_trajectory`,
  `diag_passgate_*`, `diag_tg_conduction_nn_vs_l72`); kept the 4 reusable controls
  (`diag_l72_complex_control`, `diag_l72_switchcap_{control,uic_control}`,
  `diag_nn_jacobian_consistency`). All 23 `verify_*` gates retained.
- **Core code:** removed the 2 genuinely-dead `pycircuitsim/config.py` helpers
  (`verify_osdi_binary`, `get_modelcard_path`, 0 callers). Load-bearing
  recoverable features (monotonic / EKV / Sobolev paths, LEVEL=74 BSIMAR) and the
  public package exports were kept (conservative dead-code analysis).
- **Checkpoints:** cleared the checkpoints dir (123 MB → 0) for a clean-slate
  retrain. The V6.5.x hand-tuned specials (`tsmc5_dn_corringL_s7`, `tsmc7_dn_t3`,
  `tsmc7_dn_ekvhr`, `tsmc16_dn_lgs17`) were **archived off-repo** to
  `/data2/shenshan/v6.5.9_production_specials.tar.gz` (27 MB) — not regenerable
  cheaply, kept as rollback insurance.
- **Tooling:** `scripts/benchmark_train_sml.sh` gained a `GPUS` env override
  (default-preserving) so a run can dodge a busy/shared GPU.
- **Resolver:** `pycircuitsim/parser.py` now prefers the per-tech `large` slot
  first (was `medium`-first), so the no-override production default is `large`
  (the best uniform tier). All 32 real checkpoints stay on disk; the benchmark
  still pins each tier explicitly via `PYCIRCUITSIM_NN_CHECKPOINT_DN_*`.

**Retrain + result:** `benchmark_train_sml.sh` rebuilt all **32** checkpoints
(4 techs × nmos/pmos × small/medium/large/xl) from scratch on the control recipe
`--apply-filter off --swa-mode ema --seed 42` (no loss preset / EKV / Sobolev /
corridor / T3), reusing the 8 clean datasets. **Result: 13/16 complex gates at
`large`** (device 24/24, inverter 16/16 at every size, lifted-source canary 12/12;
L72 ground-truth OP/DC/TRAN/AC all PASS — reference intact post-clean). The 3 open
gates (tsmc5 ring, tsmc7/tsmc16 opamp) are the value-surface / fixed-point gaps the
retired V6.5.x specials used bespoke recipes to force closed; under one honest
recipe they mark the true fidelity frontier. AC: device CS-amp gain0 < 1.5 dB
everywhere (gm/gds autograd faithful), the Cgd RHP-zero phase + opamp DC-gain level
are the genuine limits. Full matrix + device-fidelity metrics:
`docs/V6.6.0-accuracy-report.md`.

**Rollback:** `git checkout V6.5.4` restores the pre-clean tree;
`tar xzf /data2/shenshan/v6.5.9_production_specials.tar.gz -C <checkpoints>` +
repointing the `tsmc{5,7,16}_dn_medium` symlinks restores the 16/16 specials.

---

## V6.5.9 — ★ 16/16: T3 differentiable-DC-solver lands the tsmc7 opamp (branch `V6.5.4`, 2026-06-29)

First-ever tsmc7 opamp PASS (DirectNet gain 178.0 vs NGSPICE 163.4, 8.92%) → production 16/16. Put the DC solve **inside the loss**: a differentiable unrolled Newton solver supervises the emergent transfer curve `Vout(Vin; θ)` against L72, so r_o is shaped by the gain target instead of the residual-minimisation shortcut that over-flattens it. This broke the V6.5.8 "gain stuck ~370" wall — the gain-163 root DOES exist on the ekvhr substrate; "370" was the gate's continuation landing on a different over-flattened branch. Findings: the gate gain is **bimodal + sampling-noisy** (peak |dVout/dVin| bounces 147–187 even on a 3%-NRMSE-faithful curve); the `--lam-lo-override` r_o cap is fine WITH curve supervision (it rails without it); **preservation (ring/switchcap), not existence, was the binding work** — the faithful good-curve root and a passing ring are mutually exclusive, so the shipped candidate sits on the ring-compatible shifted root. Installed `tsmc7_dn_t3` via the `tsmc7_dn_medium` symlink (retired in V6.6.0). Infra `scripts/v6_5_8_{harvest_opamp_topology,t3_solver_finetune,gate_t3,install_t3}`. Memory `[[v659-t3-solver-lands-opamp-16of16]]`.

---

## Test-infrastructure correctness sprint — 11 bugs fixed (branch `V6.5.4`, 2026-06-28)

Audit-driven fix of the verify harness; production pass-rates unchanged (NN device 24/24, complex 15/16), every fix re-checked vs NGSPICE. Durable fixes still in code: **B1 (CRITICAL)** — the per-tech device gates in `verify_nn_dc_tran.py` were pinning **tsmc5's** net (at UNKNOWN tech-code) for ALL techs; routed the 5 `model_path` sites through `_cascade_handles_stem` so each tech resolves its own checkpoint. **B3/B5** — SRAM scored PASS when every corner errored (`all([])==True`) and never compared to ground truth; now ANDs point-by-point NGSPICE-NRMSE tracking (≤10%), with `force_ic` reconciled as a printed **diagnostic** (not a gate; rails on TSMC7/12 only). **B4** — a diverged inverter transient could false-PASS (`_nr_partial` set-but-unread); now auto-FAIL. B2/B6–B11: sweep↔gate `uic` equivalence canary, ASAP7 skip (Rule 14), real-deck canary (32/32), `partial` gating, honest additive exit codes.

---

## V6.5.8 — EKV high-r_o core + vout-weighted KCL breaks the tsmc7-opamp rail (branch `V6.5.4`, 2026-06-28)

First non-railed tsmc7 opamp in the whole campaign — a high-r_o EKV **structural** core + a vout-weighted KCL existence fine-tune produced a real amplifying transfer curve (gain ~350–381), where every prior V6.4.x–V6.5.7 attempt railed to 0. **REFUTES the V6.5.6/7 "only T3 can create a reachable high-gain OP" verdict.** BUT gain ⟺ existence are **coupled through the output-stage r_o**: the reachable OP is over-gained (~2.2× the L72 target 163) and every calibration lever is a binary rail↔370 switch — vout-weight, lam-kcl, and (decisively) `--freeze-core` / `lam_lo`-cap all RAIL rather than lower the gain (the over-flattened r_o is *required* for reachability). ±10% gate not passed; nothing installed (15/16). Routes to T3 (V6.5.9), which sits on this EKV+KCL substrate. `_EKVCore` redesigned to floor-scaled physical `id_core + sqrt(id_core²+(κ·id_s)²)·α·tanh(trunk)` + exposed `--ekv-lam-lo`. Memory `[[v658-ekv-core-breaks-opamp-rail]]`.

---

## V6.5.7 — panel-review correction of the V6.5.6 opamp verdict (branch `V6.5.4`, 2026-06-27)

A 5-agent adversarial review found V6.5.6's "no high-gain zero exists / probe-closed / only-T3" **over-strong**. The real bind is full-system STABLE EXISTENCE with **`vout` the never-supervised node** (T1 pinned only the stage-1 balance node `vo1i`; the solver-conditioning probe was 20 *cold* multistarts along a 1-D line — reachability, not an existence proof; the "vout residual at V*_L72" non-existence argument is falsified by the PASSING tsmc12 at F_rel≈0.19). The cheap vout-prioritized existence retrain was then RUN & **KILLED**: `vout` F_rel floors 0.062 (wrecks the base surface +492%) → 0.13 (preservation-safe) vs the ~0.006 a high-gain zero needs; 0 high-gain solutions on all candidates; opamp gain 0.0 FAIL ⇒ the soft-wall is near-hard for the KCL-loss family. **fetlim also dead** (the L72 control lands gain 163 on the same fetlim-less path — voltage-limiting is not what discards the NN's step). Routes to EKV/T3. `finetune_kcl.py --vout-weight/--vout-target`. Memory `[[v657-vout-existence-retrain-kill]]`.

---

## V6.5.6 — 3-operator Phase-0 routing + T1 KCL-residual lever (branch `V6.5.4`, 2026-06-26)

**Organizing frame — the 3-operator taxonomy (durable):** DirectNet emits ONE surface but the solver reads it through THREE operators, each owning a different gap + a structurally different fix-class — id-VALUES→KCL→NR fixed point (G1: opamp gain, ring period); autograd dQ/dV→pole (G3: f3db); off-diagonal cgd→RHP zero (G4: HF phase). Charge-head retrains are DC-safe; id-surface retrains are DC-unsafe; the recurring ledger failure is applying the wrong fix-class. Phase-0 (four zero-GPU diagnostics) routed the gaps: **D1 EXISTENCE** — at the L72 high-gain OP the net signed NN current is NOT a residual zero (tsmc7 vo1i F_rel 0.128 vs passing tsmc12 0.002); **G3 dead on arrival** — autograd ∂qd/∂Vd already == cdd head == OSDI ~0.1%, so f3db is OP-drift/value-surface owned; **Jacobian-blend closed analytically** — the fixed-point LOCATION is a pure function of `id` VALUES (gm/gds set only the Newton path). **T1 net-node KCL-residual loss SOLVED existence** (vo1i 0.128→0.007 — the corridor never did) but the OP is an **unstable Newton fixed point (contraction)**, and preservation is binding (any λ strong enough to move existence regresses the ring; N2 Sobolev blocks existence on the shared id head). Track B: ring-anchor WORKS (ring 6.44%→2.29% PASS). Not installed (15/16). ⚠ Its "only-T3 / no-zero-exists" conclusion was **corrected by V6.5.7**. Memory `[[v656-t1-existence-to-contraction]]`, `[[nn-accuracy-3operator-taxonomy]]`.

---

## V6.5.5 — diagnostic-routed corridor retrain → 15/16 (branch `V6.5.4`, 2026-06-24/25)

Three zero-risk diagnostics localized the V6.5.4 open gates and routed a targeted corridor retrain. **tsmc5 ring = NMOS-conduction-owned** (the pull-down under-drives id ~23% at the switching edge; the charge-ON transient reproduces the 12.66% gate, ~66% of the period error) → lifted 3/4→4/4 via `large`+ring-corridor+seed7 (`tsmc5_dn_corringL_s7`); **capacity was the bind** — medium trades ring↔opamp, large+seed threads both. **tsmc7 opamp = value-surface-owned** — seeding the NN sweep from the L72 ground-truth OP at every point STILL rails to gain 0 (the high-gain OP is unstable on the NN surface; PTC/homotopy/OP-seed cannot fix it), confirmed unrecoverable by corridor exhaustively. Net 14→15/16. Memory `[[v655-diagnostic-routing-verdicts]]`, `[[v655-corridor-retrain-15of16]]`.

---

## V6.5.4 — fresh full retrain + best-config-per-tech → 14/16 (branch `V6.5.4`, 2026-06-23/24)

Retrained the entire capacity matrix from scratch on freshly regenerated data (32 models, one clean recipe `--apply-filter off --swa-mode ema --seed 42`), best config per tech → 14/16 (matching the V6.4.7 ship but clean, no stale/specialized checkpoints). **Native-L72 control (decisive new diagnostic, `tests/diag_l72_complex_control.py`):** running the exact gate circuits through PyCircuitSim's own solver with the ground-truth OSDI model (no NN) matches NGSPICE at ring 0.00% / opamp 0.00–0.10% ⇒ both remaining gaps are **genuinely NN-value-surface-owned**, not solver/harness (and the 2ps timestep is adequate). Residual 2 gates (tsmc5 ring, tsmc7 opamp) resist every clean-data size AND seed — the value-surface limit V6.4.7 only cleared with corridor-augmented data. Memory `[[v653-l72-control-ring-opamp-model-owned]]`.

---

## V6.5.3 — ★ the switchcap gap was a HARNESS CLOCK BUG, not solver/NN-owned (branch `V6.5.2`, 2026-06-23)

**Overturns V6.5.2.** The tsmc5 switchcap "11.84% over-charge" chased across the ENTIRE V6.4.x–V6.5.2 campaign (XL capacity, µA-band loss, charge-Sobolev, TG-corridor, EKV backbone, the "switchcap-is-solver-owned" verdict) was **two harness bugs**: (1) `render_directnet_netlist` rescaled `Vdd`/`=0.80` to the tech VDD but MISSED the space-delimited PULSE clock rail, so the DirectNet clock over-drove the tsmc5 pass gates to 0.80 V while the NGSPICE deck clocked to 0.65 V — exactly explaining the tech pattern (tsmc5 +0.15 over-drive → 11.8% FAIL; tsmc12/16 0.80 → no over-drive → PASS) → **11.84% FAIL → 1.56% PASS, switchcap 4/4**; (2) the "14.65% L72 floor" (V6.5.2) was a control DC-op with no `uic` pinning (the hold node seeded at the off-transistor leakage equilibrium). **LESSON (load-bearing): when an NN gate fails vs NGSPICE, FIRST diff the rendered NN netlist against the NGSPICE deck token-by-token — clock amplitude, supply rails, bias, sweep, geometry — BEFORE blaming the model or solver.** `uic` was also made first-class in the product path (`.tran ... uic` parsing + pinning in `run_transient`, default-off → non-uic decks byte-identical). Memory `[[v652-switchcap-is-harness-clock-bug]]`.

---

## V6.5.2 — charge-derivative levers + the (later-refuted) switchcap-is-SOLVER-owned finding (branch `feat/ac-analysis`, 2026-06-22)

> SUPERSEDED by V6.5.3 — the "switchcap is solver-owned / 14.65% L72 floor / not NN-fixable" conclusion was two harness bugs. The cap-fidelity sub-findings remain valid reference.

Both candidate switchcap levers KILLED (correctly — there was no model gap): **charge-Sobolev** (`--charge-sobolev`, couples autograd dQ/dV to the supervised cgg/cgd/cdg/cdd) left the switchcap unmoved (11.84→11.32%) and did NOT move AC f3db (⇒ f3db is OP-drift / value-surface owned, not cap-under-prediction); **TG-corridor data-aug** fixed PMOS cdd 62%→5% yet the switchcap charge didn't move (the tell it was never cap/model-owned). Valid reference: the NN autograd caps match OSDI ~0.3–2.5%; per-channel sign map `+cgg,−cgd,−cdg,+cdd` (OSDI off-diagonals SPICE-negated). Kept default-off recoverable. Memory `[[v67-switchcap-is-solver-owned]]` (⚠ refuted).

---

## V6.5.1 — XL capacity tier + µA-band loss lever (KILLED) (branch `feat/ac-analysis`, 2026-06-22)

**XL tier (512×8, 2.13M p) = the over-fit boundary:** complex-gate pass-rate PEAKS at `large` then DECLINES — **6→9→12→9/16 (S→M→L→XL)**; XL fits the device surface ~10× tighter (val 2e-4) yet has the WORST off-nominal NRMSE and **loses every value-surface-fragile gate `large` won** (tsmc5/tsmc12 opamp, tsmc7 ring flip PASS→FAIL) — the cleanest confirmation of V6.4.8-S1 ("capacity is not the bind"; more capacity over-fits and collapses the high-gain NR basins). **µA-band loss de-compression KILL (refutes the V6.4.8 roadmap):** retuning the default-off `SubthresholdIdLoss` to the µA band moved the tsmc5 switchcap <0.2% — the over-charge is sample-and-hold charge/transient behaviour, not µA-band-DC-current owned. Also fixed the `xargs -L1` trailing-blank silent job-collapse bug. Memory `[[v66-xl-overfit-and-uA-lever-kill]]`.

---

## V6.5 — AC small-signal accuracy of the NN models (branch `feat/ac-analysis`, 2026-06-22)

First time NN AC fidelity was gated against ground truth (NGSPICE `.ac` on the identical BSIM-CMG OSDI), across 24 DirectNet checkpoints + the opamp — the NN's small-signal caps are autograd dQ/dV of its predicted charges, the direct probe of the charge-surface derivatives no prior gate measured. AC **gain** excellent everywhere (24/24 gain0 err <1.5 dB ⇒ autograd gm/gds accurate); the cap-driven **pole** is good but tech-variable (device gate 13/24); the **Cgd-feedforward RHP-zero HF phase is NOT reproduced** (a transcapacitance-sign limitation, diagnostic); the **opamp AC inherits the DC value-surface fragility (0/12)** but where the OP lands well (tsmc12-large) GBW 0.97× / PM 1.3° — dynamics right, DC-gain level is the miss. **No retrain warranted** (a dQ/dV deficiency would show bad gain AND pole everywhere — the opposite). Harness `tests/common/complex_ac.py`, `verify_nn_ac.py`, `verify_complex_opamp_ac.py`. Memory `[[v65-nn-ac-accuracy]]`.

---

## AC analysis — small-signal frequency-domain (branch `feat/ac-analysis`, 2026-06-21)

Brought `.ac` from a ~60%-scaffolded, dead-on-arrival state to a working, NGSPICE-validated feature (`ACSolver` solves complex `Y = G + jωC` per swept frequency about the DC OP). Fatal fixes: `run_ac_sweep` imported an absent `pandas` (→ stdlib `csv`); `_stamp_mosfet_ac` skipped the MOSFET caps → added the **transcapacitance stamp** from the source-referenced 2-port `M=[[cgg,−cgd],[−cdg,cdd]]` embedded in the nodal 3×3 (charge-conserving, vanishing at ω→0). Validation `tests/verify_ac.py` 2/2 (passive RC 0.0000% NRMSE; BSIM-CMG NMOS CS-amp vs NGSPICE on the identical OSDI, gain err 5.4e-6 dB — transcapacitance sign confirmed). Gotcha: ngspice `wrdata vp()` emits radians (dump complex `v()`, compute phase in Python). Memory `[[ac-analysis-feature]]`.

---

## V6.4.9 — DirectNet small/medium/large capacity benchmark (branch `feat/v6.4.8`, 2026-06-21)

Clean single-recipe capacity study (24 ckpts, S 128×3 / M 256×5 / L 384×6). Circuit pass-rate rises with capacity **6→9→12/16 (S→M→L)** — but the composition is the finding: device Id-Vgs + inverter accuracy is excellent at EVERY size (NOT the bind; large slightly over-fits the device surface). The **opamp is the value-surface-fragile gate** (gain≈0 at S/M all techs; recovers only at `large`, and only tsmc5/tsmc12). Switchcap needs capacity (0→3/4; tsmc5 never); ring-osc tsmc12/16 every size + tsmc7 at large; SRAM butterfly 4/4 every size. More capacity does NOT close the recipe-sensitive gaps. Harness `scripts/benchmark_*`. Memory `[[v649-sml-capacity-benchmark]]`.

---

## V6.4.8+ — complex-circuit parametric sweep harness + TSMC7 broad retrain (KILL) (branch `feat/v6.4.8`, 2026-06-20)

Built the complex-circuit parametric sweep harness (`tests/common/complex_sweep.py` + 4 `verify_complex_*_sweep.py`, per-circuit stimulus, baseline-gated, sha256-pinned). **TSMC7 broad retrain = KILL (confirms S1):** retraining `medium` on broad `tsmc7_v2` data (to widen the swept envelope) drove opamp **gain→0** — breadth fits the value surface but COLLAPSES the offset-dominated opamp; reverted to the specialized pivcor (8.63% PASS, 15/16 protected). Sweep envelope: the opamp holds gain under *load* perturbations but collapses under almost ANY OP change (VT/NFIN/VDD/vcm); ring/switchcap robust. Also fixed an L cache-key bug + a single-point switchcap clock-render bug (the same class V6.5.3 later traced as the real switchcap failure). Memory `[[v648-broad-retrain-collapses-opamp]]`.

---

## V6.4.8 — value-surface accuracy campaign; ship the S2 win, 14 → 15/16 conditional (branch `feat/v6.4.8`, 2026-06-17→20)

**Methodology locked: all gates run CPU** (the fragile opamp lands a different NR basin on CUDA — `[[v648-gate-cpu-vs-cuda-basin]]`). **S0 floor-k KILL** — gain is wildly non-monotone in the `_floor_gds` coeff (it hops NR basins, not a gain∝1/k lever; gds cancels at the fixed point; the k=2.0 "PASS" is a false-pass) `[[v648-gds-floor-inert-on-opamp-gain]]`. **S1 `--size large` KILL** — the larger net fits the value surface BETTER yet collapses the opamp; capacity is not the bind `[[v648-s1-capacity-not-the-bind]]`. **S2 continuation-first DC sweep KEEP (the sole win, load-bearing)** — `run_dc_sweep` solves warm-started NN points from the neighbour with source-stepping OFF (GMIN retry as fallback; gated on `has_nn` so BSIM-CMG is byte-identical); tsmc7 opamp 10.78% FAIL → 8.63% PASS deterministically; the win is **path-preservation**, not basin-de-fragilization `[[v648-s2-continuation-first-opamp]]`. **S3 EKV analytic backbone KILL** — the additive-in-asinh residual overwhelms the offset-dominated µA band, neutral on switchcap `[[v648-s3-ekv-backbone-kill]]`.

---

## V6.4.7 — serialized accuracy campaign; SHIP at 14/16 + force_ic 8/8 (2026-06-10→16)

A strict serial S1–S19 chain (every lever committed-or-rewound before the next). Start = V6.4.4 canonical 8/16. **Durable behavioral changes still in the code:** S2 — the NMOS source-frame fix (NN Rule 2; `_raw_voltages` shifted only PMOS, so lifted-source NMOS saw phantom Vgs/Vds; new permanent canary `verify_nn_lifted_source_dc.py`, was 10–64% NRMSE, flipped tsmc12 opamp); S7 — the reverse-Vds clamp relaxation in `_apply_vds_correction` (C¹ taper, Id(Vds=0)=0 exact; shipped window 0.20/0.30·VDD_train — the wider 0.30/0.40 corridor was KILLED: tsmc5 opamp veto + force_ic collapse). **Key dead-ends / findings:** S6 — simulator EXONERATED (native-L72 ring-osc control = NGSPICE ratio 1.000); S9b — regen-v2 data + 2 load-bearing data-gen bug fixes: the `NN_DC_SOLVE_TOL` floor (legacy 1e-9 returned exact-0 for |id|<1e-9 → the zero-row artifact; 1e-12 for generation) + an atomic-write fix for a parallel modelcard-cache write race (partial card → degenerate, physically-wrong rows); S10 — Sobolev id-derivative KILL, MAJOR finding **derivative fidelity is ANTI-correlated with the opamp** (the Jacobian guides NR convergence but cancels at the fixed point; opamp gain / RO period are value-surface / NR-fixed-point owned) `[[v647-s10-deriv-fidelity-vs-opamp]]`; S12 — trajectory-corridor KEEP (11→14/16; harvest the bias tubes the transistors visit along the ground-truth trajectory, OSDI-label, append with `--class-weights` — the V6.5.5 corridor descends from this); S11 — subthreshold-id KILL (moved force_ic the wrong way); **S17c — force_ic 0/8→8/8 was a HARNESS BUG** (the 6T netlist pinned wordline=VDD with both bitlines forced = a non-physical read-disturb that exact OSDI physics ALSO fails; wordline-OFF retention → both NN and ground truth rail 8/8) — LESSON: run the native-L72 control before blaming the NN `[[v647-s11-subthreshold-vs-forceic]]`; S19 — replication discipline caught a bistable false-pass; trust the `verify_complex` gate, not the scorer proxy `[[v647-s19-scorer-vs-gate-opamp-replication]]`. Open at ship: tsmc5 switchcap 12.14% (later a clock-bug, V6.5.3), tsmc7 opamp 10.78%.

---

## Condensed history (pre-V6.4.7)

> Full detail for these iterations lives in `git log` and `MEMORY.md`. Only the
> durable outcomes are retained here; the verbose per-phase narratives and the
> pre-V6.0 (v3/v4/v5) exploration logs were pruned in the 2026-06-20 slim.

### V6.4.6 — diagnosis-first iteration (2026-06-01/02, no behavioral change)
Gated every GPU-spend behind a 0-GPU diagnostic. Closed the measurement framing of
two gates (TSMC7 ring_osc, SRAM `force_ic`) and localised the RO error to the
**id VALUE surface** (not the derivative). Probe/measurement fixes only; the
inference path was unchanged. Set up the agenda V6.4.7 then executed.

### V6.4.5 — Track A no-ship iteration (2026-05-29)
Ran all 5 planned phases; **shipped nothing**. Built + validated the multi-circuit
scorer (durable infra, reused later). Ruled out several value-surface levers and
confirmed the RO/SRAM gaps were architectural, not tuning — feeding V6.4.6/7.

### V6.4.4 — DirectNet per-tech checkpoint mix (2026-05-28, inference-only)
First per-tech medium checkpoint mix for TSMC5/7/12/16; complex-circuit pass rate
+2 vs the V6.4.1 baseline (canonical 8/16). Restored the load-bearing V6.4.2
Phase-7a `_MonotoneVgResidual` + `--monotonic` code (on-disk checkpoints carry
`mono.*` state_dict keys; stock checkpoints route `mono=None`, no inference change).

### V6.1 – V6.3.2 — per-tech DirectNet establishment (2026-05-12 → 05-15)
- **V6.1**: per-tech dedicated DirectNet for TSMC5/7; destructive cleanup of the
  universal `refac_*`/`v4_*` artifacts (deleted 2026-05-12).
- **V6.2**: Rule 15(a) terminal-current sign fix; Rule 20 dead-band closed.
- **V6.2.1**: per-tech TSMC12/TSMC16 extension (3 registry edits + data/train).
- **V6.3 / V6.3.1**: inverter spike-removal sprint — dataset regen (`_inv_trip_points`
  recenter on VDD/2 + `_reverse_vds_points` corridor); shipped V6.3.1 with one open
  VTC MaxErr gate.
- **V6.3.2**: ported the PyCMG L3 parametric DC/transient sweeps to DirectNet
  (`tests/common/nn_sweep.py` + `verify_nn_multi_tech_{dc,tran}.py`).

### Pre-V6.0 (v3/v4/v5, package refactors, early milestones)
The BSIMAR package refactors (2026-03/04), the v3 LOO cross-tech sprint, the v4
tech-code migration, the analytical Vds-correction + rail-restoring fixes, and the
v5 inverter-transient phases are recorded in `git log` and `MEMORY.md`. Legacy
LEVEL=1 (Shichman-Hodges) was removed; LEVEL=72/73/74 are the supported models.

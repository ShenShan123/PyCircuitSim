# Plan — uniform DirectNet recipe beyond 13/16 (V6.6.2 → V6.6.4)

**Date:** 2026-07-01→02 · **Branch:** `V6.6` · **Model:** DirectNet (LEVEL=73)
**Status:** ✅ **CLOSED at V6.6.4** — production `tsmc*_dn_large_*` = uniform **crit30** recipe,
**14/16 complex gates deterministic** across OMP∈{1,2,3,4,8}. **§5 holds the live post-close
frontier routing** (4-agent panel, 2026-07-02).

> Restructured 2026-07-02 for concision. Full result tables live under `results/recipe_bench/`;
> executed history in `docs/CHANGELOG.md` (V6.6.2–V6.6.4). Section map from the pre-restructure
> doc (cited by CHANGELOG/memories): old §2→§2, §12–§13→§3.1, §14→§3.2, §15→§3.3, §9/§10→§4,
> §16→§5.

---

## 1. The question & the answer

**Question (2026-07-01).** V6.6.0 fixed production at the honest uniform 13/16; V6.6.1 proved no
*single-lever* uniform recipe beats it. Is there a recipe **combo** that benefits all complex
testcases — i.e. nets above 13/16 with one identical recipe for all 32 checkpoints, no per-case
cherry-picking?

**Answer (2026-07-02).** **YES — `crit30`:** clean base (`--apply-filter off --swa-mode ema
--seed 42`) + one identical curriculum fine-tune, `--class-weights traj_corridor=3.0,inv_trip=2.0
--lr 3e-4 --epochs 120 --patience 40 --init-from {tech}_dn_large_{dev} --data
{tech}_corro_{dev}.npz` (`RECIPES="crit30f" bash scripts/recipe_train.sh`). **14/16 strict
all-OMP** vs clean's 12 strict / 13 single-run. The gain = the two levers §2 only ever tested
*separately* — the trajectory corridor (ring lever) and inv_trip (opamp-margin anchor) —
**compose** on the corro dataset; the inv_trip anchor is what makes corridor weight 3.0 safe.
Promoted to production as V6.6.4. **15/16 uniform was NOT reached by any recipe** — tsmc7-opamp
is structural non-existence, tsmc16-opamp is basin-anti-correlated (§5).

---

## 2. Diagnosis (pre-campaign inputs; confirmed by the campaign)

The 3 open gates at the clean-13/16 baseline:

| Gate | Headline | Owner (3-operator taxonomy) | Verdict after campaign |
|---|---|---|---|
| tsmc5 ring-osc | period_err 12.66 % (≤5) | id-value→KCL (66 %) + dQ/dV (34 %): NMOS under-drives at the 0.65 V switching edge | **OPENED** by the corridor (→4.04 % det.) |
| tsmc7 opamp | gain→0 (≤10 %) | The L72 high-gain OP is **not a stable fixed point** of the NN id-surface (vout F_rel 0.128 vs 0.002 needed; seeding from the exact L72 OP still collapses) | **UNCHANGED** — 100 % across all 23 recipes × all OMP; structural (§5) |
| tsmc16 opamp | gain→0 | **Basin-selection** miss (csob lands 1.28 %, corroft 7.34, s17 6.2) | **UNCHANGED under crit30** — every opener trades ≥2 strict cells (§5) |

**The wall:** one shared-weight MLP with mutually-exclusive value-surface basins. Value-surface
tightening (corridor) helps the value/charge-owned gates (rings, switchcap) and hurts the
derivative/fixed-point-owned ones (opamp gain, SRAM butterfly); inv_trip sits on the opposite
side. From-scratch re-rolls (seed/loss/EKV/capacity) are zero-sum — seed-42 clean is already the
largest compatible basin set. **The only net-positive lever class is local:** `--init-from`
warm-start (basin-preserving locality) + LR-neutral `--class-weights` (targeted reshaping;
LDS-renormalized to unit mean, `trainer.py:262-264`).

---

## 3. Campaign record (condensed — data links in §6)

### 3.1 Round 1 — single levers: the (false) "13/16 ceiling"

S0 reproduced the clean control 13/16 exactly (snapshot `results/recipe_v662/s0/`). Then every
single-lever uniform recipe opened some gates and traded others:

| Recipe | tsmc5-ring | Trades | Net (single-run) |
|---|---|---|---|
| clean (control) | 12.66 FAIL | — | 13/16 |
| invtripft (inv_trip=2.0 curriculum) | 12.46 FAIL (inert) | tsmc7-ring | 12 |
| invtrip (from-scratch) | 13.49 (worse) | tsmc5-opamp, tsmc7-ring | 11 |
| cor / corft (full corridor) | ~5 / **4.73 PASS** | opamps + tsmc16-SRAM | 11 / 9 |
| corrft (ring+SC corridor) | **4.61 PASS** | tsmc7-SC, tsmc12-SRAM | 12 |
| corroft / corro15 (ring-only, w3.0/w1.5) | **4.7 / 4.03 PASS** | tsmc5-opamp / tsmc12-opamp | 13 / 13 |

Round-1 verdicts (durable): **inv_trip is an opamp-margin lever, NOT a ring lever** (refuted the
plan's original bet); **the corridor IS the ring lever** (vindicated; reconstructed in-tree as
`scripts/v6_4_7_s12_{harvest,append}_corridors.py` + `recipe_{train,eval}.sh`); every
single-lever recipe trades a fragile gate → "13/16 ceiling" declared… prematurely.

### 3.2 Round 2 — the cross-wall combo: crit15 = clean+1

The corro dataset carries BOTH classes (traj_corridor 12 + inv_trip 7), so one uniform
`--class-weights traj_corridor=1.5,inv_trip=2.0` curriculum weights both sides of the wall at
once — the untested recipe. **crit15 = 14 single-run / 13 strict**, the +1 being the
*deterministic* tsmc5-ring opening. Round-2 escalations (crit15m/h, crit10, crit20) all negative.
**Meta-lesson that reshaped the yardstick:** the tsmc5/tsmc16-opamp cells are multistable OMP
coin-flips (0-8 % ↔ 100 % across OMP_NUM_THREADS, both in clean and crit) — single-run passes
near the gate are unbankable; only all-OMP-deterministic cells count (→ §4 discipline #3;
root-caused in §5-F1).

### 3.3 Round 3 — full 22-recipe re-test → crit30 = 14/16 STRICT → PROMOTED

Full uniform re-test (352 isolated gates + 528 OMP-determinism runs + opamp-AC matrix + finalist
device suites): **`results/recipe_bench/ACCURACY_REPORT.md`** (RETEST section; per-circuit
continuous metrics, determinism classes, AC — formerly RETEST_ACCURACY.md, merged 2026-07-03) +
**`DEVICE_RETEST.md`** (device DC / inverter / device-AC). Every prior
count reproduced. Strict all-OMP scoreboard:

| strict | recipes |
|---|---|
| **14/16** | **crit30, crit30f** |
| 13/16 | crit10, crit15, crit15h, crit20, corroft |
| 12/16 | clean, invtripft, corro15, crit15m, csob |
| ≤11 | cs7/s7/s17 11 · invtrip/cor/corrft/s123 10 · corft/csobekv/ekv 9 · sob 5 |

- **crit30 banks deterministically:** all 4 rings (4.04/2.40/2.68/2.90 %), all 4 SRAM, all 4
  switchcap, tsmc12-opamp 6.25 %, **tsmc5-opamp 0.21 %** (a coin-flip under clean) = clean+2
  strict, containing everything clean deterministically passes. Device level ≥ clean everywhere
  (DC mean NRMSE 1.64→1.46 %, device-AC 4/8→6/8, canary all-PASS); honest within-gate
  regressions: SC droop max 13→32 % of allowance, tsmc12 opamp-AC 5.1→9.8 dB (both FAIL ≤3 dB
  anyway). Ring waveform-NRMSE is phase-drift-dominated (R²<0) — period_err is the ring metric.
- **Round-2 corrections:** "crit20/30 collapse opamps" was a coin-flip artifact (crit20 = 13
  strict); the corridor-weight→tsmc5-opamp-basin map is **non-monotone** (w1.0 FLIP / w1.5, w2.0
  detFAIL / **w3.0 detPASS**) — the inv_trip anchor makes w3.0 safe where corroft (w3.0 alone)
  railed. The killed-early crit30 artifact was validated by a full-length rerun (**crit30f**) —
  cell-for-cell identical.
- **csob re-scoped:** the only tsmc16-opamp detPASS + only opamp-AC PASS + best tsmc12 device fit
  (0.43 % NRMSE, R² 0.994), but 2 rings detFAIL + tsmc12-opamp flips @OMP8 → 12 strict. Stays the
  documented AC/device alternative.

**→ PROMOTED (V6.6.4, user-approved):** `tsmc*_dn_crit30f_large_*` copied bit-identical into the
production `tsmc*_dn_large_*` slots; clean originals archived as `tsmc*_dn_v660clean_large_*`.
Post-promotion default-path verification reproduced 14/16 deterministic.

---

## 4. Durable discipline (applies to every future stage)

**Honest-uniform contract:** one recipe (same flags/values) generates all 32 checkpoints; the
reported matrix is that uniform run's. Warm-start is honest only if each tech inits from its own
checkpoint by a mechanical rule. Per-tech flags, hand-picked checkpoints, or runtime tech→recipe
routing are per-case specials (archived pattern, not shipped). Single-tech *diagnostic probes*
are fine; only shipped recipes must be uniform.

**Gating discipline (all six, learned the hard way):**
1. Authoritative `verify_complex_*` gates only — never the scorer proxy (v647-S19).
2. CPU-pinned: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, repo ngspice
   (CUDA lands different NR basins, v648).
3. Multi-run opamp/ring across OMP∈{1,2,4}; bank only deterministic cells (coin-flip lesson,
   §3.2; root cause §5-F1).
4. Full-matrix re-gate every experiment (16 gates + device + AC + lifted-source canary) — openers
   are breakers.
5. Before blaming the model: diff the rendered netlist token-by-token + run the native-L72
   `diag_l72_*` control (v652, v647).
6. Accuracy-first reporting: continuous before→after→Δ metrics (period_err, gain_err, charge_err,
   lobe-NRMSE, device NRMSE/MRE/R², AC gain0/f3db/GBW/PM); the X/16 count is derived, never
   reported alone; flag passes within ~1 % of a gate as fragile.

---

## 5. POST-CLOSE FRONTIER — can NN accuracy go further? (4-agent panel, 2026-07-02)

Panel: structural-existence / recipe-combinator / solver-side analysts + red-team, over the §3.3
retest ledger, solver/trainer source, and the V6.5.x record. **Routing only — nothing executed.**

### 5.1 Residual ledger after 14/16

| # | Open item | Owner | Evidence |
|---|---|---|---|
| 1 | tsmc7-opamp 100 % | **Structural non-existence** on every plain-MLP surface | all 23 recipes × all OMP; v657 vout F_rel floor 0.062–0.13 vs ~0.006 needed |
| 2 | tsmc16-opamp detFAIL under crit30 | **Basin** — exists on other surfaces (csob 1.28 % detPASS) but every opener trades ≥2 strict cells | RETEST determinism tables |
| 3 | Opamp open-loop AC fails ~everywhere | **Downstream of #1/#2** (OP-MISBIAS); non-railed tsmc12 cells (GBW 0.97, PM 1–2°, dc_gain 5–10 dB) are **gds-fidelity-owned** — the only lever (id-Sobolev) collapses the DC gate (S10 tension) → **no safe recipe lever** | RETEST AC matrix; sole PASS = csob-tsmc16, exactly where csob fixes the DC OP |
| 4 | OMP FLIP multistability | Harness numerics — **root-caused** (F1) | RETEST FLIP cells; v648 CPU↔CUDA |
| 5 | tsmc12 device 17/18 corner (R² 0.72); device-AC tsmc12/16 1/2 | Under-fit corner / charge head; csob fixes both — **gates nothing** | DEVICE_RETEST.md |

### 5.2 Findings

**F1 — OMP coin-flips root-caused:** torch intra-op GEMM reduction order in the NN
forward/backward (6×`Linear(384)` + 3 autograd sweeps per NR iteration). Smoking gun: the sweep
harnesses already pin `torch.set_num_threads(1)` (`tests/common/complex_sweep.py:35`,
`nn_sweep.py:37`); the single-point ship gates don't. Fix = pin in `tests/common/complex.py`
(harness, NOT an import-time library global). Byte-neutral on the banked 14; zero L72 risk
(OSDI does no torch work). **Keep the OMP sweep as a fragility diagnostic** — the knife-edge is
physically real (v648 CPU↔CUDA; v659 gain 150↔178); a pass surviving only under the pin is
"PASS (fragile)".

**F2 — the one untested recipe point: crit30-on-csob-base (`csobcrit`).** csob holds the
complementary basin set (tsmc16-opamp 1.28 % detPASS, only opamp-AC PASS, tsmc12 device R²
0.994) but detFAILs tsmc5/7 rings; the crit curriculum has only ever run on the clean base.
Mechanically ready (csob large ckpts on disk, arch-compatible, ~3-line `recipe_train.sh` change,
~6 GPU-h, distinct exp-name → zero production risk). **Red-team EV: ~25–35 % first-ever 15/16,
~50 % lands ~13** — the corridor ft is the same operator that killed fragile opamps (corft), so
the tsmc16 cell it means to inherit is the one it most risks; the inv_trip anchor is unverified
on csob's basin. Contract: fine as a probe; promotion = declaring csob the canonical base. **Must
be judged on the full suite (16 gates + OMP fragility + AC + device + canary)** — csob's value is
AC/device fidelity, which the corridor ft is known to erode (SC droop 13→32 %, tsmc12 opamp-AC
5.1→9.8 dB on clean→crit30).

**F3 — tsmc16-opamp existence-vs-reachability UNRESOLVED on the crit30 surface.** detFAIL at
every OMP = either the corridor ate the OP (non-existence-on-surface) or forward continuation
never reaches it. **Probe (~0 GPU):** thread-pinned forward / reverse / pseudo-transient
(`_pseudo_transient_dc` scaffolding in `simulation.py`) on BOTH crit30 and csob surfaces (csob =
positive control). Ship ruling: reverse/PTC are *diagnostics*; a shippable fix must be a UNIFORM
better-forward continuation (has_nn-gated OK, tech-routing not) landing the SAME OP NGSPICE lands
— v648-S2 is the precedent; direction-shopping-to-pass is Goodhart. RELTOL /
`use_deterministic_algorithms` rejected (don't address basin selection).

**F4 — tsmc7-opamp: plain-MLP T3 is genuinely untested; EKV can never warm-start onto
production.** Verified: crit30f is a plain MLP (no `core.*` keys) and `--init-from` raises on
arch mismatch (`trainer.py:660-674`) → hard fork: T3-on-crit30f is necessarily plain-MLP T3;
EKV+T3 needs a from-scratch substrate. The v656/v657 existence wall was hit by the *static* KCL
family; V6.5.9's winning T3 (unrolled differentiable-DC-solver curve supervision) ran only on
ekvhr — **T3's mechanism has never been tried on a plain MLP** (rail-prior ~50–65 %, not
"certain"). Cheapest decisive probe: tsmc7-only **T3-as-curriculum fund-or-kill** on crit30f
(~1–3 GPU-h + ~1 day integration; recover `v6_5_8_t3_solver_finetune.py` +
`v6_5_8_harvest_opamp_topology.py` from commit `7112f2c`; `--init-mode teacher --lr 5e-5
--ring-weight ≥1.0`; gate via `PYCIRCUITSIM_NN_CHECKPOINT_DN_*`). If it rails → do NOT iterate;
route to the EKV track.

**F5 — refuted/killed idea classes (don't re-propose):** opamp-AC recipe lever (none exists);
tsmc16-targeted corridor analog (wrong fix-class — corft proved opamp corridors collapse the
opamp); csob+crit30 distillation (one surface must host both mutually-exclusive basins — §2 wall);
KCL-residual auxiliary loss (v657: can't reach vout ~0.006); bidirectional/steeper-branch pick
(Goodhart + breaks SRAM bistability); OMP-flip as training regularizer (collapses into T3). The
tsmc12 device corner needs no action (csob documents the fix; gates nothing).

### 5.3 Routed next campaign

| Phase | What | Effort | Decision rule |
|---|---|---|---|
| **P0** (infra, first) | Thread pin in `tests/common/complex.py`; OMP sweep re-labeled fragility diagnostic | ~15 min | byte-neutral (verify crit30 matrix unchanged) |
| **P1** (probe) | tsmc16-opamp existence probe: fwd/rev/PTC on crit30 AND csob surfaces | ~1–2 h | high-gain OP found on crit30 → uniform forward-continuation track (15/16 solver-side possible); all rail → model track only |
| **P2a** (parallel) | **csobcrit** ft, 8 ckpts, distinct exp-name | ~6 GPU-h | promote-candidate ONLY if ≥15 strict, or 14 strict *containing* tsmc16-opamp with no crit30-detPASS cell lost — judged on the FULL suite |
| **P2b** (parallel) | tsmc7 plain-MLP-T3 fund-or-kill (F4); base-independent → transfers even if P2a wins | ~1–3 GPU-h + ~1 day | root at low collateral → scale uniformly (on-contract 16/16 route); rails → kill plain-MLP T3 permanently |
| **P3** | Consolidate after P2a; if P2b rails AND 16/16 mandated → `--ekv-core --subthresh` SRAM-safety go/no-go (~16 GPU-h) → uniform T3-on-EKV (~1–2 wk) only if that clears | — | protect the banked deterministic 14 at every step (§4) |

**Null hypothesis (guard against over-promotion):** crit30 (14/16 robust) + csob (documented
AC/device alternative) may already BE the honest Pareto frontier of this architecture. If P2a
confirms the trade and P1 shows non-existence, the correct verdict is keep production, declare
15/16-uniform out of recipe reach, and let 16/16 wait for the structural track — not destabilize
a deterministic 14 chasing a fragile +1.

---

## 6. Artifacts & data

- **Consolidated tables (authoritative):** `results/recipe_bench/ACCURACY_REPORT.md` — ONE file
  since 2026-07-03: RETEST section (formerly RETEST_ACCURACY.md; all 23 recipes at large + 22 at
  xl, per-circuit continuous metrics, OMP determinism, opamp-AC) + MATRIX section (formerly
  RECIPE_REPORT.md; the V6.6.5 size matrix). · `results/recipe_bench/DEVICE_RETEST.md`
  (finalist device DC/inverter/device-AC) · machine-readable `results/recipe_bench/retest_data.json`.
  **2026-07-03: xl-tier re-test appended** (sections suffixed `(xl)`): the identical isolated
  methodology re-run at `xl` for the 13 size-matrix recipes (`gate_iso_xl/`, `opamp_def_xl/`;
  `SIZE` env now parameterizes `gate_matrix_iso.sh` / `opamp_sweep_def.sh`). Reproduces the
  V6.6.5 counts cell-for-cell; NEW: xl basins are OMP-deterministic (strict ≈ single-run, sole
  FLIP = corft tsmc5-ring), best base-recipe xl = 12/16 strict (invtrip, s17); cor/corft xl
  tsmc7/12 ckpts structurally diverge (sinh overflow → NR fail, honest FAILs); opamp-AC 0 at xl.
  **Same day: the 9 curriculum recipes trained at xl** (72 ckpts, warm-start clean xl, corr/corro
  data) and gated identically → **corroft / crit10 / crit15m @xl = 14/16 STRICT all-OMP, tying
  production crit30f@large**, and they bank **tsmc16-opamp deterministically (~6.5–6.7%) — the
  cell production FAILS** (crit15m also banks tsmc5-opamp 3.4%; corroft/crit10 bank tsmc12 6.2%;
  all 4 rings detPASS across the crit family). tsmc7-opamp fails everywhere (wall unchanged).
  Weight→basin map is TIER-dependent (crit30: 14 at large, 12 at xl; crit10/crit15m peak at xl).
  §5 routing input: the tsmc16-opamp high-gain basin EXISTS and is trainable-to at xl (P1
  existence probe answered affirmatively at the xl tier).
- **Raw runs:** `results/recipe_bench/{gate_iso*,opamp_def*,device_iso,canary}/` (per-recipe ×
  per-cell logs — the evidence the collectors re-parse). The V6.6.7 house-clean pruned the
  superseded archives (`*_v662_prior`, `*_prod`, quarantine, campaign logs, `V662_CRIT_SUMMARY.md`,
  `recipe_v662/s0`) — their verdicts live in the CHANGELOG/plan narrative.
- **Prior studies:** `docs/V6.6.0-accuracy-report.md`, `docs/V6.6.1-recipe-accuracy-report.md`
  (+ auto-tables now in `ACCURACY_REPORT.md`'s MATRIX section, `recipe_data.json`).
  **V6.6.5 (2026-07-03): the MATRIX section is the FULL 13-recipe × 4-size matrix** (the 27
  large-only combos trained+gated at small/medium/xl; 864 cells, zero blanks). Size-axis lessons
  for §5 routing: corridor recipes INVERT the capacity curve (cor: 11/12/11/5 across s/m/l/xl —
  best sub-large cells, xl collapse); xl is basin-shuffled not uniformly over-fit (invtrip & s17
  12/16 at xl > clean's 10; sob 5/16@large → 10/16@xl); AC pass-rate peaks at SMALL for nearly
  every recipe; clean@large 13/16 still unbeaten in-matrix; tsmc7-opamp 0 for all 13×4. Campaign
  driver `results/recipe_bench/fill_campaign.sh`; CHANGELOG V6.6.5.
- **Checkpoints** (`external_compact_models/bsimar/checkpoints/`): production
  `tsmc*_dn_large_*` (=crit30f), archives `tsmc*_dn_v660clean_large_*`, all experiment stems
  `tsmc*_dn_{recipe}_large_*` kept as documented dead-ends/alternatives (csob = the AC/device
  alternative).
- **Harnesses:** `scripts/recipe_train.sh` (all recipes incl. `crit*`), `gate_matrix_iso.sh`,
  `opamp_sweep_def.sh`, `device_matrix_iso.sh`, collectors
  `scripts/{recipe_retest_collect,device_retest_collect,recipe_collect}.py`, grid renderer
  `gate_grid.py`; corridor pipeline `scripts/v6_4_7_s12_{harvest,append}_corridors.py`.
- **Env note:** `conda run` intermittently receives SIGSTKFLT here — gate via
  `/data1/shenshan/.conda/envs/pycircuitsim/bin/python` directly (as `recipe_eval.sh` does).

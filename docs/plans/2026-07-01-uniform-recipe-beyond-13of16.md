# Plan — a uniform DirectNet recipe that benefits all complex testcases (push past 13/16)

**Date:** 2026-07-01 · **Branch:** `V6.6` · **Model:** DirectNet (LEVEL=73) · **Status:** ✅ EXECUTED — **UPDATED 2026-07-02 (§15): the FULL re-test supersedes §14's crit15 verdict — `crit30` (corridor w3.0 + inv_trip 2.0) = 14/16 STRICT all-OMP, validated by a clean full-length retrain (`crit30f`), and is the promote candidate.** — **UPDATED 2026-07-02 (§14): §13's "13/16 ceiling" is BROKEN.** The two levers §12 only ever tested *separately* — the trajectory corridor (opens tsmc5-ring, drifts an opamp) and inv_trip (opamp-margin holder, inert on the ring) — COMPOSE when applied together on the corro dataset (which carries both class 12 and class 7). The uniform recipe **crit15** (`traj_corridor=1.5,inv_trip=2.0`, curriculum warm-start) = **14/16 single-run / 13/16 strict-all-OMP = clean+1**, the +1 being the *deterministic* tsmc5-ring opening §13 said needed a per-case special. It Pareto-dominates clean robustly. 16/16 still blocked by tsmc7-opamp (deterministic 100% under every recipe × every OMP — structural non-existence). Production decision (promote crit15?) deferred to user. **The original §13 verdict below is left intact as the pre-§14 record.**

---

## 12. EXECUTION LOG / VERDICTS (live — updated per "keep-plan-updated" workflow)

### S0 — baseline reproduced ✅ (2026-07-01)
Re-gate of on-disk `tsmc*_dn_large_*` clean checkpoints on current data = **13/16**, exactly the control.
Snapshot: `results/recipe_v662/s0/`. Open gates & fragile holds confirmed:
- **Open:** tsmc5-ring `period_err=12.66%`, tsmc7-opamp `gain_err=99.99%`, tsmc16-opamp `gain_err=100%`.
- **Fragile holds:** tsmc7-**ring** `4.82%` (only **0.18%** margin), tsmc5-opamp `2.10%`, tsmc12-opamp `6.25%` (seed-42 basin).

### S1 curriculum arm (invtripft) — ❌ REJECT: 12/16 (net −1) — the plan's central bet is REFUTED
Recipe: `--init-from {tech}_dn_large_{dev} --class-weights inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40`
(warm-start from own clean large; LR-neutral inv_trip upweight — renormalized post-product mean ≈1.0007, confirmed).
All 8 checkpoints trained (`tsmc*_dn_invtripft_large_*`). Full-matrix CPU-pinned gate + OMP∈{1,2,4} multi-run:

| Gate | clean → invtripft | Δ | verdict |
|---|---|---|---|
| **tsmc5-ring** (TARGET) | 12.66% → **12.46%** FAIL | −0.2pp (inert) | **lever failed** — inv_trip does NOT open the ring |
| **tsmc7-ring** (fragile hold) | 4.82% PASS → **7.32% FAIL** | +2.5pp | **BROKE** — deterministic (OMP 1/2/4 all 7.32%) |
| tsmc5-opamp | 2.10% → **0.67%** | −1.4pp | improved margin (no flip) |
| tsmc12-opamp | 6.25% → **0.35%** | −5.9pp | **strongly improved** (fragile→robust; no flip) |
| tsmc16-opamp | 100% → 100% | 0 | unchanged (basin miss; warm-start can't jump) |

**Conclusion — inv_trip is an OPAMP-robustness lever, NOT a ring lever.** It tightens the high-gain
moderate-inversion opamp surface (tsmc12-opamp 6.25→0.35) but leaves tsmc5-ring's switching-edge under-drive
untouched and *perturbs* tsmc7-ring past its 0.18%-margin edge. This **refutes §1/§4/§7's premise** that
`inv_trip` (the zero-reconstruction proxy) opens tsmc5-ring. The ring's demonstrated lever is `traj_corridor`
(V6.5.5, class 12) — absent from the data, needs a multi-script harvest reconstruction (recover
`v6_4_7_s12_{harvest,append}_corridors.py` + dep `v6_4_7_s6_l72_ro_control.py`, adapt `{tech}_v2_{dev}`→`{tech}_{dev}`
naming, re-run native-L72 trajectory harvest). §5 accuracy-first rule → **REJECT** (target didn't improve with
margin AND a passing gate flipped). Checkpoints kept on disk as documented dead-end artifacts; production stays clean.

### S1 from-scratch arm (invtrip) — ❌ REJECT: 11/16 (net −2, worst arm)
Recipe: `--class-weights inv_trip=2.0` from scratch, 800ep, seed42 (no init-from). Full-matrix gate:
- **tsmc5-ring 12.66 → 13.49%** (WORSE — from-scratch inv_trip is anti-corrective on the ring).
- **tsmc5-opamp 2.10% PASS → 100% FAIL** (BROKE — from-scratch re-roll flipped the basin).
- **tsmc7-ring 4.82% PASS → 10.40% FAIL** (BROKE, worse than curriculum's 7.32%).
- tsmc16-opamp 100 → 100% (did NOT open — the basin re-roll did not land it). tsmc12-opamp held (5.71%).

**Confirms RedTeam:** a from-scratch reweighted run re-rolls the basin lottery destructively (locality is what
preserves the 13; from-scratch has none). **Both inv_trip arms REJECTED.** inv_trip is fully refuted as a path
past 13/16 — it opens neither the ring nor tsmc16-opamp, and breaks gates. Its only value is opamp-margin
robustness (curriculum arm, warm-start-preserved). → the corridor arm is the sole remaining lever.

### S-corridor (traj_corridor) — reconstructed; FULL corridor CONFIRMS the operator-conflict wall
Recovered + adapted the V6.4.7-S12 harvest/append pipeline (all imports resolve; harvest self-contained). Built 8
`{tech}_cor_{dev}.npz` (class-12 traj_corridor, ~0.8–1.2% of rows, harvested along each tech's OWN native-L72
ring/opamp/SC/SRAM trajectories). Trained `cor` (from-scratch seed42) + `corft` (curriculum warm-start),
`--class-weights traj_corridor=3.0`.

**The corridor WORKS on the rings** (the plan's whole premise for tsmc5-ring — VINDICATED, unlike inv_trip):
- **corft tsmc5-ring 12.66 → 4.73% PASS (deterministic, all OMP)** — the target gate opened. tsmc7-ring 4.82→2.87,
  tsmc16-ring held, ALL switchcaps improved (0.89–2.54%). The corridor is uniform-safe on the value-owned gates.

**But it BREAKS the derivative-owned gates → corft = 9/16, cor lateral.** The `traj_corridor=3.0` supervision of
the *opamp* + *SRAM* trajectories tightens the value surface at their delicate operating points and collapses them
(the S10 derivative-fidelity⟂opamp-gain tension, now seen on SRAM too): corft tsmc5-opamp 2.10→100%,
tsmc12-opamp 6.25→16.10%, **tsmc16-SRAM butterfly 222→890 mV** (distorted). Even warm-start can't hold them.

**The wall, stated cleanly:** value-surface tightening is *good* for the rings/SC (value/charge-owned) and *bad*
for the opamp gain / SRAM butterfly (derivative/fixed-point-owned). inv_trip sat on the opamp side of this wall
(helped opamps, broke rings); the full corridor sits on the ring side (helps rings, breaks opamps/SRAM). Neither
uniform full recipe nets >13/16 — **CONFIRMS §2's mutually-exclusive-basins thesis at the operator level.**

### S-corridor-ring+SC (corrft) — 12/16: the opamp is FIXED, collateral moves to SC/SRAM
Ring+SC corridor (`--circuits ring_osc,switchcap`, curriculum warm-start). The opamp-exclusion WORKED — the
tsmc5 checkpoint went **4/4**: tsmc5-ring 12.66→**4.61 PASS**, tsmc5-opamp **2.24 PASS** (held!), tsmc5-SRAM PASS,
tsmc5-SC PASS. Full corrft matrix:

| Tech | ring | opamp | sram | switchcap |
|---|---|---|---|---|
| tsmc5 | **4.61 PASS** | **2.24 PASS** | PASS | 0.89 PASS |
| tsmc7 | 2.87 PASS | 91.5 FAIL (non-exist) | PASS | **1.76%/droop194% FAIL** ← broke |
| tsmc12 | 3.84 PASS | **4.33 PASS** (held!) | **lobe<0 FAIL** ← broke | 2.54 PASS |
| tsmc16 | 4.00 PASS | FAIL (basin) | PASS | 2.01 PASS |

**Net 12/16** (gained tsmc5-ring; collateral-lost tsmc7-SC + tsmc12-SRAM). The corridor's collateral **moved**
(opamp→SC/SRAM) but didn't vanish: the ring/SC supervision at weight 3.0 still perturbs the shared surface enough
to flip a fragile gate on a tech that DIDN'T need the corridor (tsmc7/12 rings already pass). tsmc12-SRAM broke
even though the SRAM trajectory was excluded — shared-MLP coupling, not trajectory overlap.

### S-corridor-ring-only (corroft, w3.0) — 13/16: SC/SRAM recover, collateral moves to tsmc5-opamp
`--circuits ring_osc` (exclude switchcap too). tsmc7-SC and tsmc12-SRAM both **RECOVERED** (PASS) — so those breaks
were caused by the *switchcap* corridor rows, NOT shared-weight coupling. BUT tsmc5-opamp now **broke → 100% FAIL**
(deterministic OMP∈{1,2,4}). Removing the SC rows broke tsmc5-opamp — the SC-trajectory rows had been *holding*
the opamp OP in-basin. Net ≈ 13/16 (trades tsmc5-opamp for tsmc5-ring).

### S-corridor-ring-only HALF-weight (corro15, w1.5) — 13/16: fixes tsmc5-opamp, breaks tsmc12-opamp
Hypothesis: the gentler weight-1.5 perturbation opens the ring (which has margin) without railing the delicate
opamp. On **tsmc5 it threaded perfectly**: ring 12.66→**4.03 PASS** + opamp **0.81 PASS** (railed at w3.0, held at
w1.5). But applied uniformly, **tsmc12-opamp broke → 100% FAIL** (deterministic OMP∈{1,2,4}) — the same "no-SC-rows
→ opamp drifts" effect, now landing on tsmc12 instead of tsmc5. tsmc7-SC PASS, tsmc12-SRAM PASS, tsmc7-ring 2.41,
tsmc16-ring held. **Net 13/16** (trades tsmc12-opamp for tsmc5-ring).

---

## 13. FINAL VERDICT — 13/16 is the confirmed uniform ceiling (campaign EXECUTED)

Every uniform recipe tested opens some gates but trades away others through the **one shared-weight MLP surface**.
The complete lever map (all CPU-pinned, authoritative `verify_complex_*` gates; opamp/ring trips OMP∈{1,2,4}):

| Recipe (uniform, honest-contract) | tsmc5-ring | What it TRADES | Net |
|---|---|---|---|
| clean (production control) | 12.66 FAIL | — | **13/16** |
| inv_trip curriculum (invtripft) | 12.46 FAIL (inert) | breaks tsmc7-ring | 12 |
| inv_trip from-scratch (invtrip) | 13.49 FAIL (worse) | breaks tsmc5-opamp + tsmc7-ring | 11 |
| full corridor from-scratch (cor) | ~5 (borderline) | breaks tsmc5-opamp | 11 |
| full corridor curriculum (corft) | **4.73 PASS** | breaks tsmc5+12-opamp, tsmc16-SRAM | 9 |
| ring+SC corridor (corrft) | **4.61 PASS** (tsmc5 4/4) | breaks tsmc7-SC, tsmc12-SRAM | 12 |
| ring-only w3.0 (corroft) | **4.7 PASS** | breaks tsmc5-opamp | 13 |
| ring-only w1.5 (corro15) | **4.03 PASS** | breaks tsmc12-opamp | 13 |

**What this campaign PROVED (the durable knowledge):**
1. **The corridor VINDICATES the ring lever** — refuting the fear that tsmc5-ring was unreachable. It opens
   tsmc5-ring deterministically (12.66→4.0–4.7 %), and on the tsmc5 checkpoint *alone* achieves **4/4** (corrft).
   inv_trip (V6.6.2's original plan bet) was the WRONG lever — it's an opamp-margin lever, not a ring lever.
2. **The wall is an operator conflict on a shared surface.** Value-surface tightening (corridor) helps the
   value/charge-owned gates (rings, switchcap) and *hurts* the derivative/fixed-point-owned gates (opamp gain,
   SRAM butterfly). inv_trip and the corridor sit on **opposite sides of the same wall**.
3. **The fragile gates are mutually anti-correlated.** The switchcap corridor rows *hold the opamp OPs in-basin*
   but break SC-droop/SRAM; removing them (ring-only) recovers SC/SRAM but drifts an opamp (tsmc5 at w3.0,
   tsmc12 at w1.5). No uniform (circuit-set × weight) threads all four fragile gates at once — because opening
   tsmc5-ring perturbs the shared MLP, and the techs whose rings *already pass* (tsmc7/12/16) can only be hurt.
4. **The uniformity constraint is what binds.** Each corridor recipe makes the tech that NEEDS it (tsmc5) better,
   at the cost of a tech that DIDN'T (tsmc7/12). A per-tech corridor (tsmc5-only) would bank 14/16 — but that is a
   per-case special, forbidden by the V6.6.0 honest-uniform contract (§10).
5. **tsmc7-opamp (non-existence) and tsmc16-opamp (basin) are untouched** by any recipe here — they remain the
   structural frontier (T3 / EKV, §8), as the pre-campaign diagnosis (§2) predicted.

**Outcome / actions taken:**
- **Production stays clean `large` = 13/16** (all 8 production checkpoints intact, dated 2026-06-29; every
  experiment used a distinct `--exp-name`, so production was never touched).
- All experiment checkpoints (`tsmc*_dn_{invtrip,invtripft,cor,corft,corrft,corroft,corro15}_large_*`) remain on
  disk as documented dead-ends (same policy as the V6.6.1 recipe-study artifacts).
- **New durable asset:** the V6.4.7-S12 trajectory-corridor pipeline is RECONSTRUCTED in-tree and adapted to the
  current dataset naming — `scripts/v6_4_7_s12_{harvest,append}_corridors.py` (+ dep `v6_4_7_s6_l72_ro_control.py`),
  with `--circuits`/`--frag-tag`/`--out-tag` for ring-only variants, and `recipe_{train,eval}.sh` +
  `recipe_multirun_gate.sh` + `corridor_gate_direct.sh` wired for corridor recipes. Available for future per-case
  work if the honest-uniform contract is ever relaxed, or for a uniform-T3 build (§8).
- Harness note: under this environment `conda run` intermittently receives SIGSTKFLT (empty logs); gate via the
  env python directly (`/data1/shenshan/.conda/envs/pycircuitsim/bin/python`). `recipe_eval.sh` was switched to it.

**Next (deferred, per §7-§8):** 14/16 uniform is not reachable by a recipe — it needs either (a) a per-case
corridor (breaks the contract) or (b) a structural change that lets tsmc5-ring open WITHOUT perturbing the other
techs' fragile gates (e.g. a region-gated corridor head, or a uniform-T3 that supervises Vout(Vin) so the opamp
gain is preserved by construction). 16/16 additionally needs tsmc7-opamp existence (EKV high-r_o + T3).

---

## 14. §13 REFUTED — the cross-wall COMBO breaks 13/16 → crit15 = clean+1 [EXECUTED 2026-07-02]

**Re-verification first (independent, isolated-dir CPU-pinned re-gate of every on-disk recipe,
`scripts/gate_matrix_iso.sh` — per-cell `PYCIRCUITSIM_COMPLEX_RESULTS` isolation → full 16-cell
concurrency with zero scratch collision).** Reproduced §13 EXACTLY at the single-run OMP=1 yardstick:
clean **13**, invtripft 12, invtrip 11, cor 11, corft 9, corrft 12, corroft 13, corro15 13, csob 12.
The §13 table is reproducible; tsmc7-opamp never passes under any recipe.

**The gap §12 never closed: the two levers were only ever tested SEPARATELY.** The trajectory corridor
(class 12) opens tsmc5-ring but drifts an opamp; inv_trip (class 7) is the opamp-margin holder but is
inert on the ring. §12 ran corridor-only (corft/corrft/corroft/corro15) and inv_trip-only
(invtrip/invtripft) — always one side of the wall. **The corro dataset carries BOTH classes**, so a
single uniform `--class-weights traj_corridor=W,inv_trip=2.0` weights both at once. That was the untested
recipe.

**crit15 = `--class-weights traj_corridor=1.5,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40
--init-from {tech}_dn_large_{dev} --data {tech}_corro_{dev}.npz`** (same flags all 32 checkpoints, each
warm-started from its OWN clean large — mechanical, inside the §10 contract):

| Yardstick | clean | crit15 | Δ |
|---|---|---|---|
| single-run OMP=1 (the §12/§13 yardstick) | 13/16 | **14/16** | **+1** |
| strict all-OMP∈{1,2,4} (§9 discipline #3, honest) | 12/16 | **13/16** | **+1** |

**What is DETERMINISTIC (bankable, identical across OMP∈{1,2,4}, uncontended):**
- **tsmc5-ring 12.66% FAIL → 4.0% PASS** — the exact gate §13 called "not reachable by a uniform recipe;
  needs a per-case corridor." crit15 opens it uniformly. This is the real +1.
- **tsmc7-ring 4.82% (0.18% margin) → 2.4%** — the fragile hold is de-fragilized.
- tsmc12-ring 3.2, tsmc16-ring 2.8; **tsmc12-opamp 6.3% robust** (held, same basin clean holds);
  all 4 sram + all 4 switchcap deterministic PASS. crit15 holds every robust cell clean holds.

**What is a COIN-FLIP (multistable, unbankable — the v648/v659 knife-edge OP, confirmed live):** the
tsmc5-opamp and tsmc16-opamp DC-gain cells flip between ~0-8% (pass) and 100% (fail) across OMP thread
count *even uncontended* — in BOTH clean and crit15. clean's single-run "tsmc5-opamp PASS" (2.1/**100**/0.7)
and crit15's single-run "tsmc16-opamp PASS" (7.1/**100**/100) are each ONE such coin-flip. They cancel in
the comparison → the honest differentiator is the deterministic ring, not the opamps. (Meta-lesson: parts
of §12's opamp trade-map are coin-flip artifacts; the net COUNTS still reproduce, but rings/tsmc12-opamp/
sram/switchcap are the only reliable per-cell signals.)

**crit15 Pareto-dominates clean robustly:** every cell clean deterministically passes, crit15 also
deterministically passes, PLUS crit15 deterministically adds tsmc5-ring. No robust regression.

**Round-2 (15/16 attempt) — NEGATIVE.** Hypothesis: a stronger inv_trip anchor stabilizes tsmc12-opamp's
(existing) high-gain basin. Trained crit15m (inv_trip=3.0), crit15h (inv_trip=4.0), crit10 (corridor=1.0).
Result: tsmc12-opamp was ALREADY robust in crit15 (6.3%), so more inv_trip added nothing there — and it
**killed tsmc16-opamp's O1-pass** (crit15m/crit15h/crit10 all → tsmc16-opamp 100% on every OMP). crit20
(corridor=2.0) collapsed all four opamps (higher corridor weight = worse opamps, per §12). **crit15's
w1.5/inv2.0 is the sweet spot; no crit variant reached a robust 15/16.**

**16/16 still blocked structurally.** tsmc7-opamp = 100% across ALL 6 crit recipes × ALL 3 OMP = confirmed
non-existence (§2/§8 EKV+T3 territory, unchanged).

**Artifacts (all on disk, clean production UNTOUCHED — every experiment used a distinct `--exp-name`):**
`tsmc*_dn_{crit15,crit15m,crit15h,crit10,crit20,crit30}_large_{nmos,pmos}`. New harnesses:
`scripts/gate_matrix_iso.sh` (isolated full-matrix gate), `scripts/opamp_sweep_def.sh` (OMP∈{1,2,4}
multistability probe), `scripts/gate_grid.py` (grid renderer); `recipe_train.sh` extended with the
`crit*` recipe family. **Production decision (promote crit15 → the 4 tsmc*_dn_large slots, banking the
deterministic tsmc5-ring +1) is deferred to the user** — crit15 robustly dominates clean, but shipping it
changes the model.


## 15. FULL RE-TEST (all 22 recipes, uniform discipline) — crit30 SUPERSEDES crit15: 14/16 STRICT [EXECUTED 2026-07-02]

**What ran (the complete uniform ledger, per user request "re-test all recipes and recipe combos,
evaluate through detailed accuracy metrics"):**
- **16-cell matrix @ OMP=1 for ALL 22 on-disk `large` recipes** (clean, invtrip/invtripft, the 5
  corridor variants, the 6 crit combos, and the V6.6.1 study set csob/cs7/csobekv/ekv/sob/s7/s17/s123)
  — 352 isolated CPU-pinned gates.
- **OMP∈{1,2,4} determinism sweep of opamp+ring for ALL 22** (528 runs — §14 had only swept
  clean+crit*; this closes the gap) + an **OMP∈{3,8} probe** and full 16-cell OMP∈{2,4} matrices for
  the finalists (crit30/clean/csob/crit30f).
- **Opamp open-loop AC matrix for all recipes** (88 runs), **device DC/tran/device-AC suites** and
  the **lifted-source canary** for the finalists (clean, corro15, crit10, crit15, crit30, crit30f, csob).
- Collectors: `scripts/recipe_retest_collect.py` (RETEST_ACCURACY.md + retest_data.json),
  `scripts/device_retest_collect.py` (DEVICE_RETEST.md), driver `scripts/device_matrix_iso.sh`;
  results under `results/recipe_bench/` (prior §14 runs archived as `*_v662_prior`).

**Reproduction:** every §13/§14 OMP=1 count reproduced exactly (clean 13, invtripft 12, invtrip 11,
cor 11, corft 9, corrft 12, corroft 13, corro15 13, csob 12, crit15 14...). The record is stable.

**THE NEW RESULT — the strict all-OMP scoreboard (the §9-honest yardstick, now uniform):**

| strict | recipes |
|---|---|
| **14/16** | **crit30, crit30f** |
| 13/16 | crit10, crit15, crit15h, crit20, corroft |
| 12/16 | clean, invtripft, corro15, crit15m, csob |
| ≤11 | cs7/s7/s17 11, invtrip/cor/corrft/s123 10, corft/csobekv/ekv 9, sob 5 |

**crit30** (`--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40
--init-from {tech}_dn_large_{dev} --data {tech}_corro_{dev}.npz`) deterministically banks, across
OMP∈{1,2,3,4,8}: all 4 rings (4.04/2.40/2.68/2.90 %), all 4 SRAM, all 4 switchcap, **tsmc5-opamp
gain_err 0.21 %** (clean: FLIP 2.1/100/0.7 — fails at OMP=2) and tsmc12-opamp 6.25 %. That is
**clean+2 strict**, and it contains everything clean deterministically passes.

**§14 corrections (both were single-run/coin-flip artifacts):**
1. *"crit20/crit30 collapse opamps; crit30 under-trained"* — WRONG on both counts. crit20 = 13
   strict (tsmc12-opamp detPASS). crit30's training HAD been killed at heterogeneous epochs
   (30–92), but a full-length uniform rerun (**crit30f**, all 8 checkpoints, early-stop per spec)
   reproduces the killed artifact **cell-for-cell** (tsmc5-opamp 0.21 %, locus NRMSE 27.85 vs
   27.84; strict 14/16). The recipe, not the accident, delivers 14/16. Both artifact sets kept.
2. *"crit15 = the sweet spot"* — crit15 is 13 strict: its tsmc5-opamp is detFAIL and its
   single-run tsmc16-opamp "PASS" was the 7.1/100/100 flip. The corridor-weight → opamp-basin
   map is **non-monotone** (w1.0 FLIP / w1.5 detFAIL / w2.0 detFAIL / w3.0 detPASS on
   tsmc5-opamp): w3.0+anchor lands the good basin where corroft (w3.0, NO inv_trip) railed it —
   the inv_trip anchor is what makes the high corridor weight safe (§14's compose thesis holds;
   only the weight pick changes).

**Accuracy ledger (crit30 vs clean, continuous, deterministic cells):** device DC mean NRMSE
1.64→1.46 % (tsmc5 2.60→1.91, MRE 8.45→6.28), inverter sweeps equal-or-better, device-AC 4/8→6/8
(tsmc5 2/2 at 0.37 dB), lifted-source canary all-PASS, SC charge_err mean 3.35→2.93 %,
SRAM lobe-NRMSE within noise (max +0.4 pp tsmc7). Honest regressions, all within gates: SC droop
13→32 % of allowance (max, tsmc16; gate ≤100 %), tsmc12 opamp-AC dc_gain_err 5.1→9.8 dB (both FAIL
the ≤3 dB AC gate regardless). Ring waveform-NRMSE is phase-drift-dominated (R²<0 for every recipe)
and not a valid cross-recipe fidelity signal; period_err is the ring metric.

**csob (the V6.6.1 alternative), re-scoped:** the ONLY recipe with tsmc16-opamp detPASS
(1.28–7.82 % over OMP{1,2,3,4,8}) and the only opamp-AC gate PASS (tsmc16, 2.1 dB), best tsmc12
device-DC (0.43 % NRMSE, R² 0.994) — but its tsmc12-opamp FLIPS at OMP=8, and tsmc5/tsmc7 rings
detFAIL (10.3/5.1 %) → 12 strict. It benefits a *different* (smaller) gate set than crit30; it
remains the AC/device-fidelity alternative, not the all-circuit recipe.

**Unchanged walls:** tsmc7-opamp = 100 % across all 23 recipe artifacts × all OMP (structural
non-existence → EKV+T3, §8). tsmc16-opamp under crit30 = detFAIL (its only uniform openers —
csob/corroft/s17 — each trade ≥2 strict cells elsewhere; basin anti-correlation confirmed).
15/16 uniform remains out of recipe reach.

**Recommendation:** crit30 (as validated by crit30f) replaces crit15 as THE promote candidate —
one uniform recipe, honest contract, **14/16 deterministic** vs production clean's 12 strict /
13 single-run.

**→ PROMOTED (2026-07-02, V6.6.4, user-approved):** the 8 `tsmc*_dn_crit30f_large_*` checkpoints
were copied bit-identical into the production `tsmc*_dn_large_*` slots; the V6.6.0 clean-large
originals are archived as `tsmc*_dn_v660clean_large_*`. Post-promotion default-path verification
(no env pins) reproduced 14/16 deterministic. This plan is now fully CLOSED — the remaining
frontier (tsmc7-opamp existence, tsmc16-opamp basin) is structural (§8: EKV + T3).

---

> **The question.** V6.6.0 shipped the honest uniform-recipe number **13/16 complex gates**
> at production `large`; V6.6.1 swept a family of *training* recipes on *frozen* data and
> proved **no uniform recipe/seed/combo beats 13/16** — the ceiling is *mutually-exclusive
> value-surface basins*. This plan asks the follow-up the user posed: **is there a recipe
> combo (or single) that benefits all complex testcases** — i.e. that nets *above* 13/16
> without per-case cherry-picking? It was written after a 4-agent analysis (lever-map +
> gate-diagnosis + advocate + red-team). See `docs/V6.6.0-accuracy-report.md`,
> `docs/V6.6.1-recipe-accuracy-report.md`.

---

## 1. TL;DR — the recommendation

- **The honest uniform ceiling is 14/16 solid, 15/16 credible stretch, 16/16 out of scope
  for a recipe-only campaign.** Of the 3 open gates, exactly **one — tsmc5 ring-osc — is
  reachable by a uniform lever.** The two open opamps are not: **tsmc7-opamp** is a
  *non-existence* failure (the high-gain fixed point is not on the stock surface at all —
  even seeding from the exact L72 OP gives gain 0) and **tsmc16-opamp** is *anti-correlated
  with tsmc12-opamp* (which passes **only** on seed-42), so any uniform re-roll that opens
  tsmc16 closes tsmc12 → net zero.
- **The recommended recipe is a two-phase, fully-uniform curriculum:** keep the clean
  from-scratch phase unchanged, then a **short, low-LR `--init-from` fine-tune with a single
  LR-neutral `--class-weights inv_trip=2.0`**, applied *identically* to all 32 checkpoints
  (each warm-started from its **own** clean `large` checkpoint). This is the only lever that
  is simultaneously (a) *basin-preserving* (locality of a warm-start keeps the 13 you have),
  (b) *targeted* at the exact failure locus (the `inv_trip` Vth-band class is already in the
  data and lands on tsmc5-ring's 0.65 V switching edge), and (c) *genuinely uniform* (same
  flags everywhere; the warm-start source is mechanically determined, not hand-picked).
- **Predicted outcome: 13 → 14/16** (opens tsmc5-ring, holds the other 13). tsmc16-opamp is
  a fragile stretch (Stage 3); tsmc7-opamp is explicitly **deferred** to a future structural
  T3 build.
- **Reporting is accuracy-first, not pass/fail-first (§5).** Every stage emits the *continuous*
  simulation-accuracy numbers each gate already computes — ring **period-err %** + waveform
  NRMSE, opamp **gain value + gain-err %** + trip-shift + Vout(Vin) locus NRMSE, switchcap
  **charge-err % + droop mV** + Vsamp NRMSE, SRAM **butterfly-lobe NRMSE** — plus device
  NRMSE/MRE/R²/MaxErr (Rule 13) and AC gain0/f3db/GBW/PM, as a **before(clean)→after** delta
  table per tech. The X/16 gate count is a *derived summary line*, so we can see a stage that
  improved accuracy everywhere even where no gate flipped — and catch silent accuracy erosion a
  pass/fail count hides.

---

## 2. Established diagnosis (inputs to this plan — treat as ground truth)

**The 3 open gates at production `large` (uniform clean recipe):**

| Gate | Headline | Solver operator that owns it | What's wrong | Reachable uniformly? |
|---|---|---|---|---|
| **tsmc5 ring-osc** | period err 12.66 % (≤5 %) | **A** id-value→KCL (66 %) + **B** dQ/dV pole (34 %) | NMOS pull-down under-drives `id` ~23 %, `gm` ~16–21 % at the steep 0.65 V moderate-inversion switching edge → stages switch too slowly → period too long | **YES** (closest gate; corridor once reached ~1 %) |
| **tsmc7 opamp** | gain → 0 (DC-gain-err ≤10 %) | **A** only (peak \|dVout/dVin\| = fixed point of the `id` surface) | The L72 high-gain OP is **not a stable fixed point** on the NN surface (residual F_rel 0.128 at node vo1i vs 0.002 for passing tsmc12; even seeding from the exact L72 OP → gain 0) | **NO** — non-existence; needs a structural high-r_o backbone + T3 gain-shaping |
| **tsmc16 opamp** | gain → 0 | **A** only, milder (0.8 V node) | **Basin-selection** miss, not non-existence: csob lands it (1.3 %), seed-17 (6.2 %). seed-42 puts its OP in the wrong basin | **NO uniformly** — coupled: opening it closes tsmc12-opamp (seed-42-only) |

**The central obstacle — one shared-weight MLP, mutually-exclusive basins.** Measured trades
(the reason 13 is the ceiling):

| Lever | Opens | Breaks | Net |
|---|---|---|---|
| EKV analytic core (whole-column, **not** region-gated) | tsmc7-ring, tsmc16-opamp, tsmc5-ring→5.8 % | tsmc12+tsmc16 **SRAM** (subthreshold butterfly), tsmc12-opamp | 10/16 |
| id-Sobolev | (deriv fidelity) | opamp gain, all seeds | 5/16 KILL |
| charge-Sobolev (csob) | tsmc16-opamp, device NRMSE, AC | tsmc5-opamp, tsmc7-ring | 12/16 |
| seed change (7/17/123) | a *different* 2-of-4 opamp subset | tsmc12-opamp (only survives on seed-42) | ≤11/16 |
| xl capacity | (tighter device fit) | tsmc5/tsmc12 opamp basins | 10/16 |
| csobekv combo | tsmc7-ring **and** tsmc16-opamp together | SRAM (EKV) | 10/16 — **basin-stacking is zero-sum** |

**Why the recommended lever class is the only net-positive one.** Any recipe that
**re-rolls the basin lottery from scratch** (new seed, new loss preset, denser data, EKV) is
zero-sum on this surface — seed-42/clean/`large` is already the *largest simultaneously-
compatible* basin set. The only class that can add a basin *without* perturbing the others
perturbs the seed-42 basin **locally**:
- **`--init-from` warm-start** — SGD stays in the basin neighbourhood (the genuinely
  basin-preserving lever; RedTeam's key correction below).
- **`--class-weights` (LDS-renormalized)** — LR-neutral landscape reshaping.

**RedTeam's load-bearing correction:** `--class-weights` renormalizes the LDS×weight product
to unit *mean* per target (`trainer.py:262-264`), so it is **effective-LR-neutral but NOT
basin-preserving** — reweighting raises gradient *variance*, and a *from-scratch* reweighted
run still lands a different basin. **Locality (init-from) is what preserves the 13, not the
renormalization.** Hence the primary recipe pairs the two: warm-start (locality) + a mild
`inv_trip` class-weight (targeting), at low LR.

---

## 3. Verified facts this plan depends on (checked in-tree, 2026-07-01)

- `--class-weights name=w` folds into the per-target LDS then renormalizes to unit mean →
  LR-neutral. Requires a `sample_class`-tagged `.npz`. `trainer.py:238-264`,
  `train.py:98-112,321`.
- **`inv_trip` = sample_class index 7 is ALREADY present** in every production dataset
  (`tsmc{5,7,12,16}_{nmos,pmos}.npz`, ~67.5 k rows) — it's the per-tech Vth-centered trip
  band (`--enable-inv-trip`, on in `benchmark_gen_data.sh`). Class-weighting it is **zero
  reconstruction**. `hot` (class 5, ~216 k rows) is also present. `traj_corridor` (12) and
  `tg_corridor` (13) are **absent** (0 rows; append drivers were house-cleaned).
- **All 8 production clean `large` checkpoints and all 8 datasets ARE on disk** — `--init-from`
  curriculum and re-gating are ready today; **no data-regen is required** (corrects the
  advocate's "datasets empty" premise).
- `--init-from <stem>` loads with `strict=False` but **raises on any missing/unexpected key**
  (`trainer.py:660-674`) → the warm-start architecture must match exactly, so a curriculum
  fine-tune **cannot** simultaneously switch on `--ekv-core`/`--monotonic` (different keys).
- **EKV core is NOT region-gated** — it replaces the whole `id` column on every row
  (`direct_net.py:397-404`); that is *why* it breaks SRAM subthreshold. `--ekv-lam-lo` is the
  saturation/max-gain knob but is **bimodal (rail ↔ ~370)**, not a continuous dial to gain-163.
- **`--ekv-core --subthresh` is composable and correct** (no CLI mutual-exclusion;
  `--subthresh` supervises the EKV-*composed* `id` in the 1e-12–1e-6 A band exactly where the
  SRAM butterfly lives) — the one **untested** basin-opener+preserver pairing.
- **No uniform T3 exists** — it was a per-case differentiable-DC-solver *script* (V6.5.9),
  house-cleaned to `/data2/shenshan/v6.5.9_production_specials.tar.gz`; only `--init-from`
  survives as a re-host. A uniform T3 is net-new code.
- **No inference-side blend/ensemble exists** — the parser resolves to exactly one checkpoint
  per (level, polarity) (`parser.py:164`); a clean+ekv blend needs new model-loading code and
  is, in effect, per-case routing.

---

## 4. The recommended recipe (exact flags)

**Uniform, two-phase.** Phase 1 = the unchanged clean recipe (already on disk). Phase 2 =
one short curriculum fine-tune, *identical flags for every (tech, dev)*:

```bash
# Phase 2 — for EACH tech in {tsmc5,tsmc7,tsmc12,tsmc16}, EACH dev in {nmos,pmos}:
conda run -n pycircuitsim python -u -m bsimar.cli.train \
  --model direct --size large --device-type {dev} --tech-scope {tech} \
  --apply-filter off --swa-mode ema --seed 42 \
  --init-from {tech}_dn_large_{dev} \          # warm-start from its OWN clean ckpt (locality)
  --class-weights inv_trip=2.0 \               # LR-neutral targeting of the trip band
  --lr 3e-4 --epochs 120 --patience 40 \       # short, low-LR local fine-tune
  --exp-name {tech}_dn_invtripft_large --cuda --overwrite
```

- **Why `inv_trip`:** tsmc5-ring is 66 % conduction (Op A) + 34 % cap (Op B); `inv_trip` is a
  full OSDI eval carrying *both* `id` and charge targets, so upweighting it tightens the
  id-value **and** dQ/dV surface at the exact under-drive band in one shot. It is per-tech
  Vth-centered → genuinely uniform: corrective where the ring under-drives (tsmc5),
  near-neutral where the trip band is already fit (tsmc7/12/16 rings pass).
- **Why init-from + low LR:** keeps every checkpoint in its seed-42 basin neighbourhood →
  preserves tsmc12-opamp (the seed-42-only gate) and the other 12, while nudging the ring
  conduction locally.
- **Predicted delta:** opens tsmc5-ring (12.66 % → ≤5 %), holds the 13 → **14/16**. Does not
  open tsmc7/tsmc16 opamps.

**Cheap parallel arm (bake-off, not primary):** the same `--class-weights inv_trip=2.0` but
**from scratch** (no `--init-from`, full `--epochs 800`). Higher variance → may re-roll a
basin (RedTeam). Keep only if it strictly beats the curriculum arm under the full gate.

---

## 5. Reporting: detailed simulation accuracy (the lead metric, not pass/fail)

The whole point of "benefit **all** complex testcases" is a *continuous* improvement in
fidelity — a gate that goes 12.66 % → 6 % has genuinely improved even though it hasn't flipped
to PASS, and a stage that holds 13/16 but quietly erodes a passing gate's margin from 6 % to
9.5 % is a regression a bare count would miss. So every stage reports the **continuous accuracy
numbers the gates already compute** (verified in `tests/common/complex.py` + each
`verify_complex_*.py`), as a **before(clean control) → after → Δ** table per tech. The X/16
count is a *derived summary line* at the bottom.

**Per-circuit accuracy quantities to report (all already emitted by the gates):**

| Circuit | Headline error | Full-waveform / locus fidelity | Extra reported |
|---|---|---|---|
| **ring-osc** (`verify_complex_ring_osc.py`) | NG period (ps), DN period (ps), **period_err_pct** (gate ≤5 %) | v(n5) **NRMSE % / MRE % / R² / MaxErr** (`full_metrics`) | partial-solve flag |
| **opamp** (`verify_complex_opamp.py`) | NG gain, DN gain, **gain_err_pct** (gate ≤10 %) | Vout(Vin) locus **NRMSE % / MRE % / R² / MaxErr** | **trip_shift_mV** (reported, not gated) |
| **switchcap** (`verify_complex_switchcap.py`) | **charge_err_pct** (% of VDD, gate ≤5 %); **droop \|dn−ng\| mV** + droop_pct_of_allowance | Vsamp **NRMSE % / MRE % / R² / MaxErr** | NG/DN charge (V), NG/DN droop (mV) |
| **SRAM SNM** (`verify_complex_sram_snm.py`) | butterfly-lobe **NRMSE %** point-by-point (the actual gate, ≤10 %) + all-lobes-positive | per-NFIN-corner lobe **NRMSE / MRE / R² / MaxErr** | snm_err scalar (reported, too geometry-sensitive to gate); force_ic probe (diagnostic) |

**Device-level accuracy (Rule 13) — `verify_nn_dc_tran`, `verify_nn_multi_tech_{dc,tran}`:**
per **tech × device × sweep-config** report **NRMSE % / MRE % / R² / MaxErr (µA)**, plus the
inverter **VTC trip %** and transient NRMSE, and the lifted-source canary NRMSE.

**AC accuracy — `verify_nn_ac`, `verify_complex_opamp_ac`:** device CS-amp **gain0 err (dB) /
f3db ratio / mag NRMSE %** (phase diagnostic); opamp **DC-gain err (dB) / GBW ratio / PM (°)**.

**Artifact & format.** Each stage writes `results/recipe_v662/<stage>/ACCURACY.md` in the shape
of the V6.6.0/V6.6.1 report tables (`docs/V6.6.{0,1}-accuracy-report.md` are the templates):
one block per circuit, rows = techs, columns = **clean / stage / Δ** for each metric above.
A collector (extend `scripts/benchmark_collect.py` / `scripts/recipe_collect.py`) parses the
CPU-pinned gate stdout — every gate already prints these numbers — into the table; no new metric
code is needed, only aggregation.

**Accuracy-first decision rule (supersedes the bare-count KEEP rule).** Keep a stage iff:
1. the **targeted** error improves materially *with margin* (e.g. tsmc5-ring period_err ≤4 %
   deterministic across OMP∈{1,2,4}, not a 4.9 % scatter pass); **and**
2. **no other continuous metric regresses beyond run-to-run noise** — not merely "no gate
   flips". Concretely: no passing gate's headline error grows by more than its measured
   run-to-run scatter (~±1 % for the VTC/opamp trips), no device NRMSE/MRE worsens by more
   than ~0.2 pp mean per tech, no SRAM/AC NRMSE worsens materially.

This makes the aggregate accuracy the ledger; the X/16 count rides on top of it.

---

## 6. Staged campaign (cheapest / highest-promise first)

Per CLAUDE.md: **git-commit the checkpoints before each stage; keep a stage only if it makes
progress, else `git reset`/revert.** Every stage emits the §5 `ACCURACY.md` before→after→Δ
table and is judged by the **§5 accuracy-first decision rule** (targeted error improves with
margin AND no other continuous metric regresses beyond noise); the X/16 counts in the "target"
column are the *derived summary*, not the criterion. GPU-hours are rough (4090s, tiny nets,
data-loader bound; `NSTREAMS` concurrent).

| Stage | What | Command sketch | Hypothesis (accuracy target) | KEEP iff (per §5 rule) | ~GPU-h |
|---|---|---|---|---|---|
| **S0** | Reproduce the control on the on-disk data/ckpts | re-gate existing `tsmc*_dn_large_*` | establish the **baseline accuracy table** (every metric in §5) + 13/16 on *this* data | matches 13/16 and the V6.6.0 device/AC numbers (this is the "before" column) | ~0 (gate only) |
| **S1 (primary)** | inv_trip curriculum fine-tune (§4) + cheap from-scratch arm | §4 block, both arms | tsmc5-ring **period_err 12.66 % → ≤5 %** + v(n5) NRMSE ↓; hold every other metric → **14/16** | ring period_err ≤4 % (det. across OMP∈{1,2,4}) **and** no other continuous metric regresses beyond noise (esp. tsmc5/tsmc12 opamp gain_err, all SRAM lobe-NRMSE, device NRMSE) | ~6 (init-from) + ~16 (scratch arm) |
| **S2** | escalate only if S1 ring lands 5–8 % | `inv_trip=3.0` **or** `inv_trip=2.0,hot=1.5` | drive ring period_err below 4 % without lifting opamp gain_err | ring period_err ≤4 %, no other metric regresses beyond noise | ~6–16 |
| **S3 (stretch → 15/16)** | tsmc16-opamp via *local* nudge | reconstruct `tg_corridor` (driver only; point-helper survives at `nn_generate.py:570`) → `--class-weights tg_corridor=2.0 --init-from tsmc16_dn_large_* --lr 2e-4` | tsmc16-opamp **gain_err → ≤10 %** (locus NRMSE ↓) while tsmc12-opamp gain_err stays ≤10 % | tsmc16-opamp gain_err ≤10 % **and** tsmc12/tsmc5 opamp gain_err not worsened beyond noise | ~1 day rebuild + ~6–10 |
| **S4 (diagnostic, off critical path)** | EKV+subthresh SRAM-rescue bake-off | `--ekv-core --subthresh` from scratch, all 32 | EKV opens tsmc7-ring + both opamps; `--subthresh` holds SRAM lobe-NRMSE ≤10 % | strictly beats the current best on the **full accuracy table** (note: EKV still locks opamp gain ~370 → tsmc7-opamp gain_err ≫10 %, stays FAIL without T3) | ~16 |
| **S5 (deferred)** | uniform T3 build (only if 16/16 mandated) | net-new differentiable-DC-solver loss term (see §8) | tsmc7-opamp gain_err → ≤10 % | — | multi-week |

**Ordering rationale:** S1 is the bankable uniform win (cheap, zero-reconstruction,
LR-neutral, local, hits both operator components of the one closeable gate). S3/S4 are
genuinely uncertain (the tsmc12 coupling and the EKV val-loss/SRAM priors are bad). S5 is a
project, not a flag.

---

## 7. Honest ceiling & scope

- **14/16 uniform — REALISTIC (S1).** The bankable win.
- **15/16 uniform — PLAUSIBLE, NOT guaranteed (S3).** tsmc16-opamp is basin-selection, but the
  tsmc12-seed-42 coupling makes it fragile; it may only be reachable by giving up tsmc12
  (net zero). Report honestly either way.
- **16/16 uniform WITHOUT per-case specials — NOT realistic as a recipe campaign.**
  tsmc7-opamp needs BOTH (a) a structural EKV-class high-r_o backbone so the high-gain OP
  *exists* (uniform EKV breaks SRAM/tsmc12-opamp unless `--subthresh` rescues them — untested,
  S4) AND (b) T3 gain-shaping to pull EKV's locked ~370 down to ≤10 %. That is S5.
- **tsmc7-opamp: EXPLICITLY DEFER.** Document it as the known structural frontier. Chasing it
  with recipe flags only burns the 13 (every EKV/seed attempt to date netted ≤12/16).

---

## 8. Deferred structural tracks (with effort)

- **Uniform T3 curve-supervision — DEFER.** Port the V6.5.9 per-case differentiable-DC-solver
  (unrolled-Newton `Vout(Vin;θ)` supervised inside the loss) into the trainer as a uniform
  loss term, stable across all 4 techs at once (the per-case version only ever ran one tech).
  Est. 1–2 weeks. Justified only once 14 (ideally 15) is banked and 16/16 is a hard requirement.
  **Honesty caveat:** T3 bakes a *circuit* into *device* training — for a uniform recipe that
  means distorting techs whose opamp already passes; keep it explicitly labeled.
- **Inference-side checkpoint blend — DEFER + scope flag.** Architecturally absent; and it does
  not compose physically (the fixed point of an averaged id-field ≠ the average of the fixed
  points; opamp gain is nonlinear in the surface; a blended dQ/dV is a new pole structure). A
  circuit-routed "ekv for tsmc7, clean elsewhere" blend **is** the retired per-case-specials
  pattern with a runtime wrapper. Only defensible as a *learned* input-space partition (EKV
  core smoothly gated by operating region) — a research track, not this campaign.

---

## 9. Mandatory gating discipline (avoid a false 14/16)

Every stage must obey all six (from RedTeam + the memory ledger):

1. **Authoritative `verify_complex_*` gates only — never the scorer proxy** (v647-S19: a scorer
   PASS failed authoritative replication, correcting 14→13).
2. **CPU-pinned & deterministic:** `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
   MKL_NUM_THREADS=1`, repo `tools/ngspice-45.2`. CUDA lands a *different* NR basin for the
   fragile gates (v648: 10.78 % CPU vs 47 % CUDA, same ckpt) — CUDA gate results are meaningless
   here.
3. **Multi-run the opamp/ring trip gates** (≥3 runs or OMP∈{1,2,4}); require ALL pass. VTC-trip
   has ~±1 % scatter and the opamp is multistable/bimodal (v659: 150 ↔ 178). A single pass near
   the ±10 %/≤5 % edge is a coin flip — bank only margin.
4. **Full-matrix re-gate every experiment** (16 complex gates + SRAM diagnostic + `verify_nn_dc_tran`
   device + `verify_nn_ac` + lifted-source canary). Never bank a +1 without confirming no −1 —
   the conflict map says openers are breakers.
5. **Before blaming the model for any newly-failing gate:** diff the rendered NN netlist vs the
   NGSPICE deck token-by-token, and run the native-L72 control (`diag_l72_*`) (v652: a 5-campaign
   "switchcap over-charge" was a PULSE-clock render bug; v647 force_ic was a harness bug).
6. **Report accuracy, not pass/fail** (see §5) — emit the full before→after→Δ continuous-metric
   table every run (period_err %, gain value + gain_err %, charge_err %, lobe-NRMSE %, device
   NRMSE/MRE/R²/MaxErr, AC gain0/f3db/GBW/PM); log each metric's distance to its threshold and
   flag any pass within ~1 % of the gate as fragile. The X/16 count is derived from this table,
   never reported alone.

---

## 10. The honest-uniform contract (keep it honest)

V6.6.0 retired per-case specials for **one identical recipe** across every (tech×device×size).
This plan stays inside that contract:

- **Every candidate must be ONE recipe (same flags/values) that generates the whole 32-checkpoint
  matrix, and the reported gate matrix must be THAT uniform run's.** No per-tech flag divergence,
  no per-tech checkpoint hand-selection, no runtime tech→recipe routing.
- **init-from curriculum is honest** only if the *same* Phase-2 flags hit all 32 and each tech is
  warm-started from its *own* clean checkpoint (a mechanical rule, not a hand-pick). Init-ing
  only tsmc5 with a ring weight would be a per-case special.
- **inference blend / uniform-T3 / per-tech `ekv-lam-lo`** are per-case by construction — if ever
  shipped, they go in an explicitly-labeled *non-uniform appendix*, exactly as V6.6.0 archived the
  V6.5.9 specials.

---

## 11. Immediate next actions

1. **S0:** re-gate the on-disk `tsmc*_dn_large_*` production checkpoints CPU-pinned; confirm the
   13/16 control on the current data **and capture the full §5 baseline accuracy table** (the
   "before" column for every downstream Δ). (No retrain, no regen.)
2. **S1:** run the §4 curriculum arm (8 fine-tunes, ~6 GPU-h) + the cheap from-scratch arm in
   parallel; full-matrix re-gate both; emit `results/recipe_v662/s1/ACCURACY.md`
   (before→after→Δ); keep the winner iff it clears the §5 accuracy-first rule and §9 discipline.
   Target **14/16** with the tsmc5-ring period_err improvement shown as a continuous number.
3. Wire the accuracy aggregation into `scripts/recipe_collect.py` (parse the CPU-pinned gate
   stdout — all metrics are already printed — into the §5 table shape); no new metric code.
4. Record the outcome (win *or* dead-end) in `docs/CHANGELOG.md` as V6.6.2, **publish the
   accuracy report** (`docs/V6.6.2-accuracy-report.md`, same shape as V6.6.{0,1}), and update
   this plan file with the verdict (per the "keep the plan file updated" workflow).

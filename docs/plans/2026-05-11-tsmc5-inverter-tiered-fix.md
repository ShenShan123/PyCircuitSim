# V6 follow-up — TSMC5 inverter accuracy: tiered fix plan

**Date:** 2026-05-11
**Branch:** `feat/v6` (stay on it — no new branches, no checkouts).
**Scope:** DirectNet (LEVEL=73) only. BSIMAR Transformer treated as control.
ASAP7 excluded.
**Predecessors:**
- `2026-05-09-v6-accuracy-plan.md` — V6 shipping set (Tier 1A clamp,
  Tier 1.5 env override, Tier 2 asinh-zscore DN small probe).
- `2026-05-10-v6-inverter-coverage-and-selection.md` — pre-session
  3-tier proposal (now superseded — Tier 1 cone overlay is in the
  "do not retry" list below).
- `2026-05-11-v6-tsmc5-followup-session.md` — H1/H2/H3/H4 falsified
  for TSMC5 (geometry, per-tech VT, per-tech asinh s_k, hot-region
  densification on Vgs×Vds).

## Starting state — V6 shipping set on `feat/v6` tip (`8c1cd38`)

| Cell | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|
| DN single-dev DC NRMSE % | 0.98 | 3.22 | 0.18 | 0.19 |
| DN inverter VTC | **NR_FAIL (59 286 % bounded)** | 9.97 PASS | 6.50 PASS | 5.40 PASS |
| DN inverter tran | **17.76 FAIL** | 10.88 PASS | 3.79 PASS | 8.90 PASS |
| BSIMAR inverter VTC | **72.66 FAIL** | 11.85 PASS | 10.44 PASS | 48.19 FAIL |
| BSIMAR inverter tran | **20.43 FAIL** | 10.43 PASS | 10.40 PASS | 14.18 PASS |

14/16 DN cells PASS. The 2 DN failures are both at TSMC5.

## Diagnostic ground (three parallel agents, 2026-05-11)

### Code-path agent — *what specifically diverges*
- `pycircuitsim/models/mosfet_nn.py:170` — `_vdd_estimate` (rule 19a's
  `VDD_train`) is computed **universal-wide** from `input_min/max` of
  the Vd dimension. Single per-model, not per-tech. At TSMC5 VDD =
  0.65 V, the rail-restoring step (a) sits right at the threshold.
- `pycircuitsim/solver.py:556-563` — Tier-1A trust-region clamp pins
  per-iteration `|ΔV| ≤ max_vs_voltage ≈ VDD`. NR can oscillate at the
  ceiling.
- `pycircuitsim/solver.py:581-590` — only "stuck" detector is
  `improvement_ratio > 0.9` for 2 consecutive iters → damping ×= 0.8.
  **No "no global progress over N iters" watchdog.** No hard escape.
- `pycircuitsim/solver.py:670-688` — after ~150 NR iters,
  oscillation-detection averages the last 3 voltage snapshots if
  variance < 10 × tolerance and **accepts** as the solution. This is
  why TSMC5 DN VTC returns a smoothed 59 286 % bounded number instead
  of a clean NR_FAIL.
- `pycircuitsim/simulation.py:306,320` — DC sweep already uses warm
  starts (`prev_solution` as initial guess at each Vin step). The
  catastrophic VTC is *not* a missing-warm-start issue.

### Dataset agent — TSMC5 is NOT under-sampled (off Vgs×Vds)
Over `universal_nmos.npz`, 12.33 M rows, TSMC-only:
- Row counts within 1.1 × of mean; TSMC7 is the actual minority tech.
- Identical 3-point T grid across techs.
- Narrower Vbs / L / NFIN absolute ranges are mechanically tied to
  PDK constraints (TSMC5 has smaller VDD, smaller NFIN_max=12, etc.),
  not a sampling defect.
- **Zero starved cells** in the joint `(L, NFIN, T, Vbs-sign)` 4-tuple
  for any tech.
- **Conclusion: this is not a data-coverage problem.** Whatever drives
  TSMC5's poor circuit-level metrics is a fixed-point / Jacobian
  smoothness problem at TSMC5's specific operating point.

### Web-research agent — relevant prior art (post-dead-end filter)
- **MSH arclength continuation (DATE 2024)** — for HiZ nodes (the
  inverter trip-point output node is exactly this in the failure
  regime — autograd gds ≈ 0 in saturation), path-following replaces
  "step Vin → NR from previous" so the solver cannot stall at a
  turning point. ~200 LOC in `solver.py`, no retrain.
- **ATANSH (Roychowdhury TCAD 2006)** — per-device homotopy
  `Id_homo = λ·Id_NN + (1−λ)·Id_linear`. Solver-side, orthogonal to
  GMIN.
- **Levenberg-Marquardt damping** — adaptive μ on the MNA diagonal,
  μ reduced on residual decrease, raised otherwise. Pairs with Armijo.
- **AutoPINN structural Vds transform (Micromachines 2023)** —
  `Vds_new = sign(Vds)·(√(Vds²+γ²)−γ)` enforces `Id(Vds=0)=0`
  *structurally* at the input. Their derivative loss uses *finite
  differences from training data*, sidestepping the asinh chain-rule
  mismatch that killed N4/B2.
- **KAN paper (Novkin 2025)** — NR convergence is dominated by
  **second-derivative smoothness**, not Id NRMSE. Confirms our
  empirical V6 finding: 56 K-param probe beats 1.5 M-param zscore on
  circuit cells.

## Plan: three orthogonal levers, ordered by simplicity × impact

Each tier is one commit on `feat/v6` with revert-on-fail. No tier
needs a new branch.

### Tier S1 — Solver-side: progress watchdog + bounded-numeric kill

**No retrain.** Pure solver patch.

Three hunks in `pycircuitsim/solver.py`:
1. **Progress watchdog.** Track `min_residual_norm` over the last
   N = 20 iters. If the running min hasn't decreased by
   ≥ 1e-3 × max_vs_voltage over N iters, declare *stuck-without-
   progress* (distinct from "stuck improving slowly"). Return
   `_last_solve_converged = False` so the simulator's
   `_solve_dc_with_retry` escalates GMIN / source-stepping.
2. **Bounded-numeric kill.** In the oscillation-acceptance path
   (`solver.py:670-688`), additionally check
   `|V_node| ≤ 5 × max_vs_voltage` for every node. If any node is
   bounded but absurd (the 59 286 % case is V_out ≈ 1.1 V at
   Vin = 0.325 V getting reported as a converged DC point), reject
   the average and return `_last_solve_converged = False`.
3. **GMIN-retry escalation.** Add one extra escalation step
   (source-stepping with 50 steps instead of 20) at the end of
   `_solve_dc_with_retry` for NN circuits.

**Expected outcome.** TSMC5 DN VTC: 59 286 % → clean NR_FAIL (worst
case) or PASS (best case). All 14 PASS cells: byte-identical numerics
(watchdog never fires when the existing path converges).

**Gate.**
- L1 BSIM-CMG byte-identical.
- 6 currently-PASSing DN inverter cells: regression < 0.5 pp.
- TSMC12 DN VTC stays at 6.50 ± 0.05 pp.

**Revert if:** any PASS cell regresses past 0.5 pp.

---

### Tier S2 — Vin-homotopy bisection fallback in DC sweep

**No retrain.** Triggered only on per-step solve failure for an NN
circuit. Touches only the DC sweep loop in `pycircuitsim/simulation.py`.

Patch. When the per-step solve fails for an NN circuit at Vin = Vk,
bisect: retry at Vin = (Vk−1 + Vk) / 2 with `prev_solution` as the
guess, then re-solve at Vk from that intermediate. Allow up to 4
bisections. Cheapest "real continuation" we can deploy without writing
arclength.

**Expected outcome.** Cleanly traces VTC across the trip cone even
when one Vin step alone is too aggressive a jump. Targets TSMC5 VTC
where current 5 mV steps + trust-region clamp cannot accommodate the
trip-region Δ.

**Gate.** Same as S1.

**Revert if:** any PASS cell regresses past 0.5 pp, OR S2 fails to
convert any TSMC5 cell that was clean-NR_FAIL after S1.

---

### Tier M1 — B0-medium asinh DN retrain (520 K params, 200 epochs)

**Retrain.** No code changes — CLI only. Single env-var-overridden
verify against existing test infrastructure.

```bash
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size medium --loss-preset e2 \
    --device-type nmos --exclude-techs asap7 --num-tech-codes 18 \
    --epochs 200 --cuda --exp-name v6_dn_medium_e2_asinh --overwrite
# (same for pmos)
```

Verify with `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}` set to the
new exp-name.

**Gate.**
- TSMC7 NMOS DC ≤ 2.5 % (currently 3.22 % on probe).
- No PASS-cell regression > 0.5 pp.
- TSMC5 inv-tran ≤ 14 % (currently 17.76, threshold 15).

**Revert if:** any TSMC12/16/7 inverter cell regresses past 0.5 pp.

---

### Tier M2 — TSMC5-only residual head (frozen backbone + LoRA)

**Retrain (scoped).** Add ~2 K-param residual MLP head conditioned on
`tech_code ∈ {0,1,2,3}` (TSMC5's four variants). Frozen backbone.
Trained for 30 min on TSMC5 rows only. Inference: residual added to
backbone output before denormalisation. Autograd flows through both
heads.

**Risk.** Per the H1/H4 dead-end log, anything that touches per-tech
output behaviour can shift Jacobian shape. Must be A/B'd at *circuit
level*, not training-NRMSE level (which has a 29× train→inverter gap
per V5 E3 evidence).

**Gate.** TSMC5 inverter PASS. Zero regression > 0.3 pp on
TSMC7/12/16. STOP here per user instruction.

**Revert if:** any non-TSMC5 cell regresses.

---

## Explicitly NOT in this plan (already-falsified)

- Per-tech VT shift in `_apply_vds_correction` (H2 catastrophic).
- Per-tech asinh `s_k` (H3 magnitude spread too small).
- Trip-cone / hot-region data overlays (V5 `inv_trip`, V5 D1 E4/E5,
  V6p — all regress other techs; the 2026-05-10 plan's Tier 1
  cone-overlay was a similar shape).
- Per-tech checkpoints split.
- SignConsistencyLoss / BoundaryLoss / id_gate / SlopeMatchLoss /
  ChargeConsistencyLoss (all deleted in 2026-05-03 trim).
- AR-finetune phase.

## Speculative follow-ups (out of M2 scope per user)

- **AutoPINN structural Vds input transform** — replaces rule 19's
  inference-time correction with a structural input transform.
  Retrain-from-scratch.
- **MSH arclength continuation** — full solver-side path-following.

## Execution protocol (every tier)

1. Confirm `git rev-parse --abbrev-ref HEAD == feat/v6` before any
   change.
2. Commit current state.
3. Apply tier.
4. Commit with `feat(v6): <tier> — <short name>`.
5. Run 16-cell verify with
   `PYCIRCUITSIM_NN_CHECKPOINT_DN_*` env vars (Tier 1.5 from V6 plan).
6. Pass → keep. Fail → `git reset --hard HEAD~1`, fill Outcomes.
7. Move to next tier. Stop after M2 and report.

## Outcomes

(Filled by the executing agents.)

| Step | Commit | Verify | Verdict |
|---|---|---|---|
| Plan | _this commit_ | n/a | n/a |
| Tier S1 | _reverted_ | L1 BSIM-CMG byte-identical (0.0124 / 0.0089 / 0.19 % — bit-identical to V6 baseline). TSMC5 DN VTC = 68.22 % NRMSE (V(out) locked at 0.80-0.91 V across the full Vin sweep, above VDD=0.65 V). Verify wall-time exploded (>10 min/tech) so the 16-cell run was killed after only TSMC5 + TSMC7 VTC completed; TSMC12/16 + all inv-tran untested. | **FAIL** — TSMC5 DN VTC does not meet "<50 % NRMSE OR clean NR_FAIL OR PASS"; per-point 50-step source-stepping escalation made the verify uncompletable. Reverted with `git restore pycircuitsim/solver.py pycircuitsim/simulation.py`. |
| Tier S2 | _reverted_ | TSMC5 59,286% (no fix — wrong attractor exists at every Vin, bisection cannot escape its basin); TSMC7 9.97→5.39% (real improvement on recoverable cells); TSMC12 6.50→4.55%; wall-time blew the 40-min cap (TSMC16 VTC + all 8 transients unmeasured) because the 5s/step bisection budget × 130 sweep points on TSMC5 = ~10 min/cell. Reverted. | REVERT |
| Tier M1 | _reverted_ | Trained v6_dn_medium_e2_asinh (340K params, 200 epochs cosine, MAE+per-target LDS, asinh-zscore, e2 4-output head). 10/16 PASS vs V6's 11/16. **TSMC5 DN VTC 59,286%→50.48%** (catastrophic→graceful, still FAIL); **TSMC7 DN VTC 9.97→2.85** (huge win); **TSMC16 DN VTC 5.40 PASS→16.10 FAIL** (−10.7 pp, hard regression); TSMC5 DN tran 17.76→16.38 (still FAIL); TSMC7 DN tran 10.88→14.41 (−3.5 pp regression); TSMC12 cells ±0.1 pp; TSMC16 DN tran 8.90→8.99 ✓. Capacity bump alone shifts the wrong-attractor problem across techs rather than removing it. Summary at `tests/v6_logs/m1b_summary.csv`. | REVERT |
| Tier M2 | _reverted_ | Built a 1,844-param TSMC5-only residual MLP head (4-output: id, qg, qd, qb) on the frozen V6 small-probe backbone (55,748 params). Trained NMOS+PMOS for 30 epochs on TSMC5-only rows (1.9M/2.0M). Training-space val improvement: 0.17%/0.31% — essentially flat (backbone already fits TSMC5 in-distribution). **TSMC5 DN VTC: 59,286→59,266%** (catastrophic blow-up unchanged); **TSMC5 DN tran: 17.76→15.69%** (+2.07 pp improvement, still FAIL just above 15% gate); TSMC7 DN VTC 9.97→5.42% (unexpected improvement — gating-by-tech-code has minor leakage); TSMC7 DN tran 10.88→14.38% (−3.5 pp regression, violates 0.3 pp gate); TSMC12/16 byte-identical ✓. 11/16 PASS, same as V6 baseline. Code left as untracked `external_compact_models/bsimar/models/tsmc5_residual.py` + `training/tsmc5_residual_train.py` for reproducibility. Summary at `tests/v6_logs/m2_summary.csv`. | REVERT |

### Tier M2 diagnostic — backbone wrong-attractor is structural

A residual head trained only on in-distribution data cannot fix the
wrong attractor at V_out ≈ 1.3×VDD because:

1. **Training data doesn't cover the wrong attractor.** The dataset
   samples currents at physically-valid (Vgs, Vds) — V_out > VDD is
   out-of-distribution at training time. The residual head has no
   gradient signal pointing toward "make Id non-zero at V_out > VDD
   to restore the rail."
2. **In-distribution fit is already saturated.** V6 small probe's
   training val loss is 0.00238; the residual moves it to 0.00237 —
   a 0.4% relative reduction. Per-tech training-space NRMSE is
   already at the 0.04-0.08% floor. The wrong-attractor failure is
   *NOT* a training-NRMSE problem.
3. **Tran improves marginally because mid-rail transitions ARE in-
   distribution.** The 2 pp tran improvement is real but bounded by
   what in-distribution gradient can achieve.

**Implication for any future tiered work:** the TSMC5 VTC catastrophic
NR_FAIL requires either (a) a structural input transform that
*forbids* the wrong attractor at V_out > VDD (e.g. AutoPINN-style
sign(Vds)·(√(Vds²+γ²)−γ), or learning rule 19a's rail-restoring
constants as model parameters), or (b) **training data that includes
out-of-distribution rail-overshoot rows with the correct restoring-
current label**. Both are bigger work than the 2026-05-09 plan
anticipated; this plan should not chase them without a fresh
diagnostic budget.

## Final session summary

| Tier | Outcome | TSMC5 DN VTC | TSMC5 DN tran | Other 14 cells |
|---|---|---:|---:|---|
| V6 baseline | shipping | 59,286% FAIL | 17.76 FAIL | 14/16 PASS |
| S1 | reverted | 68% wrong fixed point | n/a (wall-time) | n/a (wall-time) |
| S2 | reverted | 59,286% (no fix) | n/a (wall-time) | TSMC7 9.97→5.39, TSMC12 6.50→4.55 (real wins, but TSMC16 + transients unmeasured) |
| M1 | reverted | 50.48% (gross improvement, still FAIL) | 16.38 (still FAIL) | TSMC7 VTC 9.97→2.85 win; **TSMC16 VTC 5.40 PASS→16.10 FAIL** (hard regression) |
| M2 | reverted | 59,266% (unchanged) | 15.69 (still FAIL, +2 pp) | TSMC7 VTC +4.55 win (gating leak); TSMC7 tran −3.5 regression |

**Net for `feat/v6`:** zero shipping commits beyond the saved plan.
The 4 attempts converged on a structural finding: **TSMC5 inverter
VTC's catastrophic failure is not fixable by solver tweaks, capacity
bumps, or in-distribution residual heads.** The wrong attractor lives
in extrapolated territory that no in-distribution training signal
touches. The right next class of fix is structural — AutoPINN-style
input transform or rail-overshoot data labels — and is out of this
plan's scope.

**On-disk artefacts left after revert:**
- `tests/v6_logs/m1b_summary.csv` — M1 16-cell numbers.
- `tests/v6_logs/m2_summary.csv` — M2 16-cell numbers.
- `tests/v6_logs/m1b_nmos.log`, `m1b_pmos.log`, `m1b_verify.log`,
  `m2_train_nmos.log`, `m2_train_pmos.log`, `m2_verify.log` — training
  + verify transcripts for forensic re-runs.
- `external_compact_models/bsimar/models/tsmc5_residual.py`,
  `external_compact_models/bsimar/training/tsmc5_residual_train.py`
  — M2 implementation, untracked, kept for reproducibility.

V6 shipping set remains:
`v6_dn_small_e2_asinh_{nmos,pmos}_best.pt` (Tier 2 from
`2026-05-09-v6-accuracy-plan.md`). No code on `feat/v6` changed
during this session.

### Tier M1 diagnostic — capacity does not remove the wrong attractor

The medium retrain (6× the V6 small probe's params, same recipe) genuinely
fixed TSMC5 VTC's catastrophic 59,286% blow-up (down to 50.48% — graceful
failure now), and gave a huge win on TSMC7 VTC (9.97→2.85%). But it
shifted the failure pattern: TSMC16 VTC went from 5.40 PASS to 16.10 FAIL.
This is consistent with KAN paper's finding that NR convergence is
governed by *second-derivative smoothness*, not Id NRMSE: the medium
model has lower training NRMSE but a different Jacobian shape that puts
TSMC16's trip cone into a wrong-attractor basin.

**Implication for any future capacity bump:** B0-large or further capacity
work is NOT a free win. It needs structural Jacobian regularization
(AutoPINN-style derivative loss, or per-tech LoRA after the backbone
is fixed — see Tier M2) to prevent the failure pattern from
shape-shifting between techs.

### Tier S2 diagnostic (worth keeping in mind for any future solver work)

The bounded-Vout watchdog with `factor = 1.2 × max_vs_voltage` is the
right detector. The bisection retry is the wrong recovery action *for
TSMC5 specifically* because the wrong attractor (V_out ≈ 0.80–0.91 V at
VDD = 0.65 V) exists at every Vin in the sweep — there is no
"known-good prev_solution" to bootstrap from. For TSMC7/12 it works
because their wrong attractor is local to the trip cone. If a future
solver tier ships, the wall-time budget per Vin step should be ≤ 1 s
and the retry should fall back to clean NR_FAIL on TSMC5-class
failures rather than burning the full budget.

### Tier S1 failure notes (for the next attempt)

1. **L1 BSIM-CMG was untouched.** All three legacy tests passed bit-identical (`_has_nn_device(circuit)` guards on the new code paths held).
2. **Watchdog + bounded-numeric kill alone are not enough.** With those firing on every failing TSMC5 sweep point, the retry cascade (fast → GMIN → 50-step source-stepping) was repeatedly entered, each call burning through a 50-step source-stepping pass that itself didn't converge. The verify took >10 min per tech (was <1 min before).
3. **TSMC5 DN VTC actually got worse-shaped, not better.** Instead of the prior 59 286 % bounded artifact (smoothed oscillation accepted by `voltage_history` averaging), the patched solver returned a *consistent* V(out) ≈ 0.80–0.91 V across the full Vin sweep — a real fixed point at ~1.3× VDD. The 5× VDD bounded-numeric kill threshold (3.25 V at TSMC5) was too loose to catch this. Per-step retry escalations could not escape that basin.
4. **Plausible next attempt.**
   - Tighten the bounded-numeric kill to `|V_node| ≤ 1.5 × max_vs_voltage` (or `≤ VDD + 0.2 V`).
   - Cap the 50-step source-stepping escalation to a single point-level attempt with a hard iteration budget, not a fresh retry of the full DC sweep.
   - Consider running Tier S2 (Vin-homotopy bisection) directly, since the issue is per-step jump size at the trip-cone, not solver-level homotopy depth.
   - The watchdog itself is correct in concept; the failure is that the retry path it triggers does not have enough escape velocity. One variant: when the watchdog fires, return *the most-recent low-residual snapshot* (not the average) so the sweep keeps a sane warm-start.

# Accuracy frontier across the test circuits: the 3-operator taxonomy + Phase-0 zero-GPU routing

Date: 2026-06-26 · Branch `V6.5.4` (→ next) · Author: campaign notes

> **Thesis.** DirectNet (LEVEL=73) emits ONE surface (`id` + charges `qg/qd`), but the
> solver reads it through THREE different operators, and each open gap needs a structurally
> DIFFERENT fix-class. The single most-repeated failure in the negative ledger is applying
> the wrong fix-class (e.g. supervising a *derivative* to move a *value-surface fixed point*).
> Before any GPU spend, four zero-GPU diagnostics ROUTE the entire (expensive) decision —
> honoring the project's #1 lesson: localize before you retrain. Produced by a 32-agent
> analysis workflow (6 grounded finders → 5 design lenses → adversarial verify of 20 levers,
> 9 killed as re-treads → synthesis).

## 0. Current state (settled)

Production = best-config-per-tech (V6.5.5). **DC complex gates 15/16.** Four accuracy gaps:

- **G1 — tsmc7 opamp DC gain→0** (the lone open DC gate). PROVEN value-surface / fixed-point
  *stability* limit by `tests/diag_opamp_basin_seed.py` (1c): seeding the NN sweep from the
  L72 ground-truth OP at EVERY point still rails to gain 0, even though the NN small-signal
  gain there is ~142 (0.58× of L72=163). The high-gain mid-rail OP is UNSTABLE as a DC fixed
  point of KCL-with-NN-currents. Corridor *coverage* exhausted (medium+large × seeds
  {7,17,42,31} × W{3,8} all gain→0).
- **G2 — opamp open-loop AC = 0/12.** Inherits G1; but tsmc12-large reproduces GBW 0.97× /
  PM 1.3° ⇒ the dynamics are right, only the DC-gain *level* (G1) is the miss. No independent
  G2 lever.
- **G3 — device CS-amp pole f3db = 13/24.** Gain 24/24 excellent; pole tech-variable
  (tsmc5-NMOS, tsmc12/16-PMOS under-predict output cap, ratio 1.1–1.6).
- **G4 — Cgd-feedforward RHP-zero phase NOT reproduced** (30–80° off by the −3 dB corner).
  **Diagnostic-only — NOT a gate** (`verify_nn_ac.py` passband-masks phase).

## 1. The organizing insight — one surface, three operators

The NN emits `id` + `qg/qd` once per forward (`mosfet_nn.py:366-371`). The solver reads it
three ways, and each owns different gates:

| Operator | reads | owns | fix-class |
| --- | --- | --- | --- |
| `id` **values** → KCL → NR fixed point (`solver.py:303-309`) | id/gm/gds | **G1 opamp DC gain**, ring period | **A: value-surface / fixed-point** |
| autograd **`dQ/dV`** (diagonal `cdd`) → pole (`mosfet_nn.py:353-362` → `_stamp_cap_ac:2570-2573`) | charges | **G3 f3db**, **G2 opamp-AC dynamics** | **B: charge-derivative value** |
| off-diagonal **`cgd=∂qg/∂Vd`** sign+mag → RHP zero (`mosfet_nn.py:355` → `solver.py:2584,2586`) | charges | **G4 HF phase** | **C: transcap sign** |

**The blast-radius corollary (verified in code).** The DC stamp (`_stamp_mosfet_dc:254-256`)
consumes only `id/gm/gds/gmb` — never a cap or charge — and `_apply_vds_correction`
(`mosfet_nn.py:372,568-662`) rewrites `id/gm/gds/gmb` and never touches a charge key.
Therefore:
- **Any charge-head retrain (B/C, G3/G4) is first-order DC-safe** — cannot move the 15 DC
  gates or the G1 opamp.
- **Any `id`-surface retrain (A, G1) is DC-unsafe** — must clear the full 16-gate matrix +
  lifted-source canary.

**Why the wrong-class trap recurs.** id-Sobolev (S10) supervised a *derivative* (B-fix) to
move opamp gain (an A-gap) and COLLAPSED it 4/4 seeds — the Jacobian cancels at the converged
FP (`solver.py:304`: the `−g·v` terms reconstruct the local linearization; only the
`i_leaving` *value* is balanced). Class A is only movable by a quantity that SURVIVES
convergence: a coupled *current value* (net-node KCL residual) or a *contraction* property —
never a single device's slope.

## 2. Per-gap ranked levers (honest odds; all G1 levers <30%)

### G1 — tsmc7 opamp DC (Class A)
A subtlety 1c never separated: it runs the full damped DCSolver, so it conflates
(i) the L72 OP is **not a residual zero** of the NN-current map (*existence* failure) vs
(ii) it **is** a zero but the Newton map **repels** (*contraction* failure). **The Phase-0
G1 probe resolves which, and routes the entire G1 effort.**

1. **T1 — grouped net-node KCL-residual loss** (~25-30%, *only if* existence failure).
   Penalize `(Σ NN current into each free node {vtail,n1,vo1i,vout})²` at each harvested OP.
   The one quantity the corridor never pinned: it supervised each device's *absolute* `id`,
   never the *difference* `i_Mn2 − i_Mp4` at `vo1i`. Distinct (value not derivative; native-µA
   frame, so the asinh `s_id≈2.6e-5` compression that kills absolute-id levers doesn't apply).
   DC-unsafe → full-matrix gate + re-run 1c (gain must HOLD >0 seeded) mandatory.
2. **N2/T3 — contraction penalty at labeled OPs** (~15-25%, *only if* contraction failure).
   N2 = one-shot Jacobian-norm penalty (cheap); T3 = unrolled K-step Newton (heavy). Caveat:
   gradient flows through the same `d(gm,gds)/dweights` channel that collapsed the opamp in
   S10 — guard hard, gate on 1c-holds-gain>0.
3. **T2 Jacobian blend — DEMOTE to a 5-min falsifier, not a fix.** 1c is fatal (a Jacobian
   edit can't manufacture a residual zero). Run env-gated, expect inert, close the family.

### G2 — opamp AC (Class A + working pole)
**No independent lever.** Gated entirely by G1; any G1 win carries G2 free. Track as a
downstream readout of G1; do not spend directly.

### G3 — device f3db (Class B — but evidence says OP-drift)
The "asinh-compressed `dQ/dV`" story is CONTRADICTED by a recorded experiment
(`CHANGELOG:185-197`): `--charge-sobolev` did NOT move f3db (1.585→1.778); the TG-corridor
fixed PMOS `cdd` 62%→5% yet f3db DIDN'T budge. Recorded verdict: a cap-derivative deficiency
would degrade gain AND pole everywhere; gain is 24/24 perfect ⇒ f3db is **OP-drift /
value-surface owned** (partly hostage to the G1 bind). Phase-0 probe confirms before any
retrain. Ceiling ~13 → maybe 15/24.

### G4 — RHP-zero phase (Class C, diagnostic-only — moves NO gate)
The harness is structurally blind to G4 (`phase_maxerr_inband_deg` only measures where the NN
already matches <7°). **Add a beyond-corner HF-phase metric first** or nothing is
interpretable. Admissible lever: scoped off-diagonal charge-Sobolev pinning `(cgd,cdg)` toward
their *separate* (accurate) supervised columns — magnitude fix preserving true asymmetry.
Fidelity-only work.

## 3. Phase 0 — four zero-GPU diagnostics (byte-identical to the 15 gates; route everything)

All CPU-pinned, L72-in-PyCircuitSim reference where applicable, NEW test/diag files only.

- **P0-1 (decisive). `tests/diag_opamp_kcl_residual.py` — G1 existence vs contraction.**
  Load the *production* tsmc7 checkpoint; harvest the L72 opamp OP (reuse the
  `scripts/v6_5_5_harvest_corridor.py` L72-control builder); `_eval` each opamp device at the
  L72 node voltages; assemble `F(node)=Σ signed NN terminal id` into each free node
  {vtail,n1,vo1i,vout}. **Decision D1:** `|F(vo1i)|,|F(vout)|` large O(µA) → EXISTENCE → fund
  T1. `F≈0` → CONTRACTION → fund N2/T3, T1 inert.
- **P0-2. `tests/diag_g3_cdd_match.py` — G3 OP-drift confirmation.** Dump autograd `cdd` vs the
  supervised `cdd` output column on the tsmc12-PMOS CS-amp grid. **Decision D2:** already match
  → f3db is OP-drift, the charge lever is dead on arrival; mismatch → A3 has a (low) shot.
- **P0-3. T2 Jacobian-family closure (env-gated, 5 min).** Wire the env-gated blend
  `g=(1−w)g_autograd+w·g_head` (default w=0, byte-identical, mirror the `_GDS_FLOOR_K`/
  `_REV_TAPER` env pattern); sweep `w∈{.25,.5,.75}` through `diag_opamp_basin_seed.py`; judge
  on converged-locus-NRMSE monotonicity (NOT gain/frac_each — an inconsistent blend basin-hops,
  the D5 E3-false-pass). Expected inert → record, close the Jacobian side for G1.
- **P0-4. G4 visibility.** Add a beyond-corner HF-phase metric to `tests/common/complex_ac.py`
  / `verify_nn_ac.py` (the existing passband mask is blind to G4). No experiment is
  interpretable without it.

## 4. Phase 1 — the one GPU bet that survives Phase 0 (conditional)

- D1=existence → build the T1 harness (widen `eval_single_point` to keep `ig/is/ib` — dropped
  at `sweep.py:37-38`; `_harvest_traj` emits device→node topology+sign+OP-group id — discarded
  at `harvest:90-107`; grouped collate — `dataset.py:44` is per-row i.i.d.; `GroupedKCLLoss`).
  Retrain tsmc7 N/P, opamp-masked, small λ. Ship gate: 1c-holds-gain>0 AND full 16-gate matrix
  + device DC/AC + lifted-source canary unregressed.
- D1=contraction → N2 (one-shot Jacobian-norm contraction penalty) before the heavy T3.
- DC-safe & parallel: if D2 left A3 alive, charge-Sobolev on `cdd` for one G3 cell; G4
  off-diagonal charge-Sobolev — both gated on the full AC matrix (a prior global charge-Sobolev
  regressed a PMOS AC cell), both knowing they move no headline DC count.

## 5. Realistic ceiling
- **DC: 16/16 ≈ 1-in-4**, contingent on D1=existence AND T1 surviving the full matrix. More
  likely outcome routes to N2/T3 at ~15-25% with real collapse risk. **Plan for 15/16 stable.**
- **AC device gate: ~13 → maybe 15-16/24** (most hostage to OP-drift).
- **AC opamp (G2): 0/12 until G1 moves.** G4: improvable but un-scored.

## 6. Newly-recorded dead-ends (from this analysis)
1. **Charge-Sobolev / s_q-refit as an f3db fix** — re-tread of the V6.5.2 KILL; f3db is
   OP-drift-owned (constructively disproved by the TG-corridor cdd 62%→5%, f3db unchanged).
2. **AC transcap symmetrization (`NN_SYMMETRIC_CAPS`-in-AC) as a G4 fix** — physically wrong;
   transcaps are non-reciprocal by construction (OSDI `|cgd|≈0.7e-16` vs `|cdg|≈0.9-2.5e-16`);
   the non-reciprocity IS what generates the zero; mean-symmetrization blows the better-fit
   `cgd` error to 69-183%.
3. **`I−D⁻¹G` spectral penalty** — solver runs full Newton, not Jacobi; at a true FP the
   Newton spectral radius is 0 regardless of weights. Only N2's *value*-form survives.
4. **Hard diff-pair antisymmetry head** — wrong invariant; the gate measures the opamp in the
   *imbalanced* regime across two *separate* checkpoints; an intra-surface reflection can't
   express a cross-device-type difference.
5. **Mirror-ratio relational loss** — the ratio half is a finite-difference gds (cancels at
   FP); the only surviving (tail-balance) half IS a T1 row.
6. **LTE sub-stepping / LDS-quantile for G3** — wrong gap (G3 is AC, no time integration;
   LDS reweights output columns never read at inference).

## 7. Decision points
- **D1** (P0-1): existence → T1; contraction → N2/T3.
- **D2** (P0-2): cdd-match → G3 charge lever dead; mismatch → A3 low shot.
- T2 (P0-3) expected inert → Jacobian side closed for G1 permanently.
- G4 only after P0-4 makes it visible; fidelity-only (moves no gate).

---

## 8. RESULTS (executed 2026-06-26) — Phase 0 routed, Phase 1 (T1) converts existence→contraction

**Phase 0 — all four diagnostics ran; every decision resolved.**
- **D1 = EXISTENCE.** P0-1 `tests/diag_opamp_kcl_residual.py`: tsmc7 stage-1 balance
  **vo1i F_rel = 0.128** at the L72 high-gain OP vs the passing **tsmc12 control 0.002**;
  L72 self-check F_L72 = 2.5e-12 A (sign assembly valid). The high-gain OP is NOT a
  residual zero of the NN current map ⇒ funded **T1** (net-node KCL-residual loss);
  N2/T3 inert when the FP doesn't exist. vo1i is the only predictive node (tsmc12 has
  large vout/vtail residuals yet passes).
- **D2 = MATCH.** P0-2 `tests/diag_g3_cdd_match.py`: autograd cdd ≈ supervised cdd ≈
  OSDI to ~0.1% on the tsmc12-PMOS grid ⇒ the G3 charge-Sobolev lever is dead on
  arrival; f3db is OP-drift/value-surface owned. A3 closed.
- **P0-3 closed analytically (subsumed by D1).** F(V*)=Σ i_leaving(V*)=0 (id VALUES)
  defines the DC solution; gm/gds (autograd or any blended head) is only the
  Newton-step Jacobian — it sets the convergence PATH, not the fixed-point LOCATION.
  No Jacobian edit can manufacture the missing residual zero. The Jacobian side of G1
  is closed without running T2.
- **P0-4 done.** Beyond-corner HF-phase metric added (`tests/common/complex_ac.py` +
  `verify_nn_ac.py`, additive/diagnostic-only); tsmc12 beyond-corner phase err max
  140–152° vs passband 35–41° — G4 now measurable. Moves no gate.

**Phase 1 — T1 net-node KCL-residual lever: built, validated, KEY RESULT.**
- Harness: `scripts/v6_5_5_harvest_kcl.py` (59 opamp OP-groups, ±0.06 V band, L72
  self-check 2.5e-12) → `scripts/v6_5_5_finetune_kcl.py` (joint N+P fine-tune from
  production `large`; KCL at vo1i couples Mn2[N]/Mp4[P]; native-µA asinh denorm;
  base-data LDS-MAE anchor) → `scripts/v6_5_5_gate_kcl.sh`.
- **T1 SOLVED EXISTENCE → converted it to CONTRACTION.** Re-running P0-1 on the
  T1-trained checkpoint flips its own verdict: tsmc7 **vo1i F_rel 0.128 → 0.007**
  (k2_c, λ=50) — the L72 OP is now a genuine residual zero of the NN current map
  (the corridor never achieved this). 1c: seeding from the L72 OP recovers 0% gain
  — the OP exists but is an UNSTABLE Newton fixed point.
- **Did NOT pass the opamp gate / NOT installed.** (a) CONTRACTION: the existent OP
  repels (autograd gm·ro too flat per 1b), cold continuation rails. (b) PRESERVATION
  binding: any λ that moves existence regresses the ring (k2_a λ=5: 5.97%, k2_c λ=50:
  6.44%, vs 5%) + device DC/tran — opamp & ring share the NMOS bias region. ⇒
  production unchanged 15/16; k2_* left on disk (gitignored) as the existence-fixed
  start for the contraction campaign.
- **Dead-end recorded:** unbalanced KCL (computed every step on the same 59 groups,
  ~1068× over-applied) thrashes/wrecks the surface (l05: anchor +2142%, ring 15.9%,
  switchcap droop 2260%); the fix is balanced per-step scaling + grad-clip + frozen
  embed + min-drift selection.

## 9. NEXT — solver lever PROBE-CLOSED; retrain (Track B) is the only path
**Track A (DC-safe solver lever) is DEAD.** `tests/diag_opamp_solver_conditioning.py`
multi-started the opamp DC solve at vin* from a grid of mid-rail seeds × {stock
damped+LM, GMIN homotopy}, on BOTH k2_c (T1) and production: **all 20 converged
solutions RAIL (vout=0.000); zero high-gain solutions.** The 0.7% k2_c residual is
a small-residual SHELF, not an exact zero — an exact high-gain DC fixed point needs
F≈0 at ALL free nodes (vout's residual stays large), so no seed/GMIN/trust-region
lever has anything to converge to. The solver side of G1 is closed.

**Track B (retrain) — the remaining contraction + preservation campaign.** Sequence:
1. **Ring-region anchor first** — add the tsmc7 ring trajectory corridor (the
   V6.5.5 ring harvest, retargeted to tsmc7) to the fine-tune ANCHOR so the shared
   NMOS bias region is explicitly pinned. Prerequisite to any tsmc7 id-surface edit.
2. **Push existence HARDER (F→~0 at ALL free nodes, not just vo1i) + localized N2.**
   The probe shows vo1i alone isn't enough — vout/n1 residuals must also reach ~0 so
   an exact high-gain zero appears. Add localized Jacobian/Sobolev supervision at the
   59 opamp OPs (pull autograd ∂id/∂V toward the accurate predicted gm/gds columns so
   the OP is Newton-attracting ≈ L72), KCL-anchored so it can't move the FP (avoids
   the S10 broad-collapse). Gate on 1c-holds-gain>0 AND the full tsmc7 matrix + ring
   + canary unregressed.
3. **Realistic ceiling unchanged:** 16/16 ≈ 1-in-4; plan for 15/16 stable. The win
   from this session: existence is PROVEN solvable, the solver shortcut is closed,
   and the residual is a narrower, characterized harder-existence+contraction target
   bounded by the ring-preservation constraint.

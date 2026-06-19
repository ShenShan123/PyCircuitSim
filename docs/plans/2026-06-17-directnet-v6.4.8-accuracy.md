# DirectNet V6.4.8 — Closing the value-surface gaps in complex-circuit accuracy

**Date:** 2026-06-17 · **Branch (proposed):** `feat/v6.4.8` (from `feat/v6.4.7` @ `d9c3d6b`).
**Authoring:** plan-mode synthesis + a 4-agent design review (3 independent design tracks — physics-structured architecture, inference/solver, capacity/ensemble/data — adversarially reconciled by a 4th). The review's single most important act was **falsifying its own cheapest proposal** (the "gds-floor coefficient" opamp fix), recorded below so it is never re-litigated.

**Predecessor:** `docs/plans/2026-06-10-directnet-v6.4.7-accuracy.md` (V6.4.7, shipped 14/16 + force_ic 8/8).

---

## Starting point (V6.4.7 ship)

Headline **14/16** complex-circuit gates + **force_ic 8/8** (the force_ic 0/8 was a wordline-ON read-disturb HARNESS bug; the retention gate now rails 8/8 for both NN and native LEVEL=72). Ship mix: tsmc7 = `pivcor_w2_s7`, tsmc16 = `s12cor_w3_s17`, tsmc5 + tsmc12 = V6.4.4 baseline.

The **four open accuracy gaps** (gate tolerances in `tests/verify_complex_*.py`):

| gap | gate | current | established owner |
|---|---|---|---|
| **tsmc7 opamp** | DC gain ±10% | **10.78%** (+0.78 pp) | systematic over-gain — **id-VALUE-surface** (gain = peak\|dVout/dVin\| of the converged DC locus) |
| **tsmc16 opamp** | DC gain ±10% | s17 5.14% PASS but **fragile** | **NR-fixed-point basin selection** — seeds land on gain≈197 (PASS) / ≈383 (+104%) / ≈0 (collapse) |
| **tsmc5 switchcap** | charge ±5% of VDD | **12.14%** | **strong-inversion saturation-edge forward over-conduction** (id-value, not subthreshold, not fixed-point) |
| **inverter VTC MaxErr** | ≤25 mV | 29.7–62 mV | high-gain trip-point value-surface fidelity (not a board gate cell) |

**The V6.4.7 verdict that frames this plan:** opamp gain, RO period, and SRAM bistability are **value-surface / NR-fixed-point owned**. Cheap data/loss/derivative levers are exhausted and several *actively harm* these gates (Sobolev derivative supervision collapsed the opamp 4/4 seeds, S10; subthreshold loss does not close force_ic and removes the asymmetry railing needs, S11). Only **structural** levers remain. This plan picks the structural levers that survive the established dead-ends.

---

## Design-review findings (new, code-grounded)

Four findings reorder the obvious candidates. The starred ones were verified against current code during the review.

1. **★ The "gds-floor coefficient" opamp fix is a MIRAGE — it is the recorded dead-end "gds floor tweaks, inert at fixed points" rediscovered.**
   A direct measurement found that at the opamp output-stage op point the autograd `gds` is wrong-sign/sub-floor and the *stamped* `gds` is synthesized 100% by `_floor_gds = max(gds, max(|id|·0.5, 1e-12))` (`mosfet_nn.py:505-507`, Rule 5). The tempting inference — "raise the `0.5` coefficient `k`, gain ∝ 1/k, close the 10.78% gap for ~5 LOC" — is **false for the gate's observable**. The opamp gate computes `gain = max|dVout/dVin|` by finite difference across **converged DC sweep points** (`tests/verify_complex_opamp.py:55-63`). At a converged fixed point the Newton step → 0, so the matrix `+g_ds·v_d` exactly cancels the `−g_ds·v_ds` baked into the Norton `i_eq` (`solver.py:296-309`): **`g_ds` drops out of the converged KCL; the fixed point is set entirely by the predicted current `id(V)`.** The `1/k` scaling is real only for the *AC small-signal* gain of the floored linearization, which the gate never evaluates. Three independent code/doc statements agree: Rule 5 (`CLAUDE.md:246` "floor only affects the NR Jacobian, not the converged solution"), the LM-damping docstring (`solver.py:95`), and the V6.4.7 dead-end ledger (`plan:160` "gds floor tweaks — Jacobian-only, inert at fixed points").
   **Corollary — the original Rule-1 break is a trap:** "stamp the predicted-gds head instead of autograd" would *lower* output gds (the predicted head measures *below* the floor at the op point) and make over-gain **worse**, not better.

2. **★ The production model is `medium` = 256-wide × 6 layers ≈ 343k params. `--size large` (384×6, ≈520k) was NEVER trained — there is no 384-wide checkpoint on disk.** This is genuine, untested capacity headroom, and the inference loader (`mosfet_directnet.py:_build_from_state`, lines 44-72) infers width/depth from state-dict shapes, so a `large` checkpoint loads with **zero inference-code change**.

3. **★ tsmc5 switchcap is the worst tech for a concrete, tech-specific reason.** Lowest VDD (0.65 V vs 0.75–0.80) and an SVT pass device → least gate overdrive; the over-conduction is localized to the **saturation-edge knee** (Vgs 0.24–0.40, Vds 0.09–0.30: +18% to +51% per the S5 dump `results/v6_4_7/s5_sc_dump_TSMC5.csv`), while deep-linear under-conducts identically across all techs. tsmc5's smaller asinh `s_id` (1.27e-5 vs tsmc7 1.92e-5) puts its µA sample currents near the asinh log-knee, compressing the loss gradient exactly in the linear-region currents that set the SC charge.

4. **tsmc16 opamp is a basin-selection problem, not a magnitude problem.** Seeds land deterministically on three discrete branches (197/383/0). This is the regime for an **asymmetric DC continuation** (anchor the sweep from a railed corner; preserve the warm start the per-point source-stepping currently resets) and same-basin weight averaging — *not* a value-surface or capacity lever.

**Net reordering:** the opamp over-gain is value-surface owned → only the structural backbone (or capacity) can move it; floor-k and predicted-gds are off the table; the cheapest untested shot at SC + inverter is `--size large`; tsmc16 fragility is a solver-continuation/ensemble problem.

---

## Ranked, serialized campaign

Project discipline (V6.4.7 §Sequencing): strictly serial, one lever at a time; each step starts from a committed clean state, runs its gates, then **commit-on-progress / `git reset` + dead-end record on kill**. Every retrain arm runs **≥4 seeds**, A/Bs against a frozen control, and any opamp flip is **re-verified on the authoritative `verify_complex_*.py` gate across `OMP_NUM_THREADS ∈ {1,2,4}`** before counting (the S19 lesson: the scorer proxy and the gate disagree on bistable cells).

**Protected-cell veto set (all steps):** the 14 passing complex cells + inverter 8/8 (`verify_nn_dc_tran.py --inverter-only`) + DC 55/55 + tran 64/64 + lifted-source 12/12. Watch especially the 3 passing ROs (gds-sensitive) and the passing opamps tsmc12 (4.97%) / tsmc5 (2.49%).

### S0 — Floor-k settling diagnostic (KILL-FIRST, ~0.5 GPU-h, ~5 LOC, read-only)

Env-gate the `0.5` in `_floor_gds` (`mosfet_nn.py:507`); sweep `k ∈ {0.3, 0.5, 0.6, 1.0, 2.0}` on the shipped tsmc7 opamp gate. **Purpose: adjudicate, not fix.**
- **Pre-registered prediction:** `gain_err_pct` moves ≪0.1 pp across the sweep (the fixed point is k-invariant per finding 1).
- **Kill / resolve:** flat in k → floor-k confirmed inert, Track-B headline closed, **no ship change**. If gain *does* move, it is via changed NR basin/partial-convergence (inspect iteration counts + accepted residual, `solver.py:847-883`); any such flip is presumed a **gate-loosening (E3)** and rejected unless proven a convergence correction.
- **Rules:** touches Rule 5's constant for the diagnostic only; nothing ships.

This guards the campaign against spending effort on an inert knob and against an E3 false-PASS. It costs under a GPU-hour and reorders everything onto S1.

### S1 — Train + evaluate `--size large` (384×6) per-tech checkpoints (~30–50 GPU-h, ~0 inference LOC)

`SIZE_PRESETS[("direct","large")]` (`cli/train.py:64-73`) is wired but never trained. Retrain per-tech (≥4 seeds × {nmos,pmos}, control-v2 recipe, EMA on per S9) and evaluate on the full board. Optionally carry the Sobolev id-derivative term at a **protective λ** (small/corridor-weighted — the S10 lesson is that an aggressive λ collapses the opamp; default off unless it demonstrably helps the inverter without touching the opamp).
- **Targets:** tsmc5 SC 12.14% (most addressable), inverter VTC MaxErr, and — decisively — **a test of whether the S12 "corridor fixes RO but collapses the passing opamp" tension is capacity-bound.** If `large` lets a tech carry the value-corridor without collapsing its opamp, that reframes every later lever.
- **Kill gate:** SC not ≤5% **and** inverter MaxErr not ≤25 mV **and** no opamp/SC NRMSE gain beyond the documented ±1% run-to-run band → rewind, record "capacity is not the bind."
- **Compat:** loader infers shape → no inference change. Checkpoint is a plain `net.*`/`tech_embedding.*` state dict.
- **Rules:** Rule 17 (keep `--tech-scope` ∈ tsmc{5,7,12,16}); Rule 18 (stays LEVEL=73). E2 stands — do **not** trim the 13 output heads (4-output regressed the inverter).

### S2 — Asymmetric branch-selection DC continuation for the tsmc16 opamp (+ same-basin SWA rider) (~30–60 LOC solver, low GPU)

The tsmc16 opamp passes only via a lucky seed on the 197 basin. Make the basin selection **deterministic**: anchor the DC sweep from a railed Vin corner (where the branch is unambiguous) and **preserve the warm start across sweep points** — today the per-point source-stepping rescales all sources from 0 and discards `prev_solution` (`solver.py:622-642`), which is the path-dependence S19 observed. Carry a **same-basin SWA** weight-average over the 197-basin seeds as a variance-reduction rider (`--swa-mode swa` exists, never used as a promotion recipe; averaging across *different* basins is meaningless, so cluster-then-average).
- **Targets:** de-fragilize the tsmc16 opamp (deterministic PASS); a robustness win, not a magnitude win.
- **Kill gate / E3 guard:** the tracked branch must match the **NGSPICE BSIM-CMG locus** (`verify_complex_opamp.py:108-112`), not merely be the passing one; if continuation still lands on the 383/0 basin → record dead-end (sibling to the killed symmetric P0-A homotopy).
- **Scope:** opamp only — force_ic is already MET 8/8, so the SRAM ambition here is redundant.
- **Distinct from dead-ends:** the killed P0-A was a *symmetric conductance* homotopy that folds at g*≈1e-5 S; this is an *asymmetric sweep-parameter / railed-corner* continuation. D3 warm-starts were dead because they *added* a source; this *stops the existing reset that destroys* the warm start.
- **Rules:** none broken (solver-orchestration; Rule 1/5 untouched). Must not change the converged solution of any *monostable* opamp (tsmc7/12/5) — verify identical converged Vout there.

### S3 — Analytic charge-based (EKV-like) backbone + bounded NN residual on the `id` column (~120–180 LOC model + ~30 CLI, ~30–50 GPU-h)

The one **S10-trap-proof** attack on the value-surface opamp/SC bias. Compose the `id` output as `Id(V) = Id_core(Vgs,Vds,Vbs; θ) + tanh-bounded · r_NN(V)`, where `Id_core` is a differentiable closed form (EKV/charge-sheet style) whose Vds/Vgs functional dependence is fixed-physical (monotone `Id(Vgs)`; finite, positive, rolling-off saturation `gds` via a `(1+λ·Vds)` CLM term with NN-predicted **positive** coefficients), and `r_NN` is the existing trunk passed through a bounded gate (±20–50%). The 12 other heads stay (E2 smoothness prior).
- **Why it escapes the S10 trap:** S10 supervised the *derivative via a loss*, which competed with the value head for capacity and flattened the high-leverage op-point gds → gain collapse. Here the slope is **wired into the functional form** — autograd through the differentiable core yields a structurally-correct, sign-/magnitude-bounded `gds` with **zero added loss terms**, so there is no value-vs-derivative competition. Rule 1 is *preserved* (and made more physical): gm/gds/gmb still come from `torch.autograd.grad(id, V)`.
- **Targets:** tsmc7 opamp +0.78 pp and any residual SC over-conduction (a charge-sheet triode law self-limits low-Vds conduction — the SC's exact failure mode).
- **Pre-condition:** fund only **after S1** shows capacity alone is insufficient (else S3 is redundant for SC).
- **Kill gate:** opamp gain err not <±10% with the inverter held, **or** the backbone collapses a passing opamp the way S12's corridor did → drop the residual bound or the core, record dead-end. Keep λ in the FinFET band 0.3–1.2 V⁻¹ (a physical constraint, not a tunable knob).
- **Compat (corrected):** `_build_from_state` (`mosfet_directnet.py:44-72`) auto-detects `mono.*` **only** — a `core.*` key prefix needs a **new detection branch** (small, ~10 LOC, but real; not the "zero-change like mono.*" the design draft claimed). A stock checkpoint with no `core.*` keys still loads unchanged.
- **Rules:** Rule 1 preserved (verify with `tests/diag_nn_jacobian_consistency.py`); Rule 18 (LEVEL=73). Checkpoint format gains an optional `core.*` block.

### Composition + promotion (S4)

Compose the surviving arms into the per-tech mix; re-run the **full authoritative-gate board** (every cell, not the scorer) + perturbed-circuit blind holdouts (RO stage count, opamp Cload, off-default-Vin SC); promote the honest count; any opamp flip re-verified on the gate across OMP∈{1,2,4}.

---

## Explicitly NOT funded (and why)

- **Floor-k as an accuracy fix** — inert at the converged fixed point (finding 1); recorded dead-end. Survives only as the S0 diagnostic.
- **"Stamp predicted gds" (the Rule-1 break the user invited us to consider)** — investigated and **rejected on evidence**: the predicted-gds head sits *below* the active floor at the op point, so consuming it makes over-gain worse. A `max(floor, gds_pred)` *robustness* floor is acceptable as a convergence rider but is not an accuracy arm; the fully-consistent "integrate predicted conductances to reconstruct id" variant biases the op point and is out of scope unless S2/S3 stall.
- **SWA as a tsmc7 opamp lever** — reduces variance, not tsmc7's *systematic* +0.78 pp bias; kept only as the S2 tsmc16 de-fragilization rider.
- **Sobolev / subthreshold / symmetric homotopy / seed lotteries / head-trimming / full-corridor** — all recorded dead ends (S10, S11, P0-A, D5, E2, S7). Sobolev and subthreshold stay default-off recoverable infra.

## Which "Design Rules" this plan reconsiders

The user noted some design rules might not hold. After the review, the resolution is that **the rules mostly hold**, and the one genuinely reconsidered was tested and kept:
- **Rule 1 (autograd Jacobian consistency)** — the candidate break ("stamp predicted gds") was the headline temptation; the evidence killed it. S3 *preserves* Rule 1 by construction. Rule 1 stands.
- **Rule 5 (gds floor)** — its "inert at fixed points" claim is correct and is in fact the load-bearing fact that kills floor-k. No change.
- **Rule 17 (exclude ASAP7)** and **Rule 18 (DirectNet-only / BSIMAR parked)** — unchanged; all levers stay on per-tech LEVEL=73. (Revisiting LEVEL=74 BSIMAR or ASAP7 would be a separate, larger campaign and is not justified by these four gaps.)

## Realistic outcome

**Best case 15/16**; **16/16 contingent** on the S3 backbone landing *both* the tsmc7 opamp and the last SC margin without an S12-style collapse — a real but unproven ceiling. The honest framing: the V6.4.7 campaign already harvested the cheap wins, so expected net gain is fractional (≈0.5–1 cell). The campaign's value is concentrated in **S1** (a nearly-free test of the untrained capacity tier and the capacity-tension hypothesis) and **S3** (the only principled, S10-trap-proof structural attack on the value-surface bias). S0 and S2 are cheap insurance: S0 prevents a wasted gate-loosening, S2 converts a fragile tsmc16 pass into a deterministic one.

## Verification

- **Gates:** `tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py` (the board) + `verify_nn_dc_tran.py` (inverter) + `verify_nn_multi_tech_{dc,tran}.py` (55/55, 64/64) + `verify_nn_lifted_source_dc.py` (12/12), all `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- **Authoritative-gate re-verification** of any opamp flip across `OMP_NUM_THREADS ∈ {1,2,4}` with deterministic reproduction required before counting.
- **Rule 16** quartet (MRE / R² / NRMSE / MaxErr) reported per tech for every arm.
- Gate files under `results/v6_4_8/`. Dead ends recorded with the numbers that killed them. CHANGELOG + CLAUDE.md updated before any commit.

---

## Progress log (execution)

**Branch:** `feat/v6.4.8` (from `feat/v6.4.7` @ `d9c3d6b`). **Methodology locked:**
all gates run **CPU** (`CUDA_VISIBLE_DEVICES=""`, `OMP=MKL=1`, repo
`tools/ngspice-45.2`) — reproduces S19's 10.78 % exactly; CUDA lands the fragile
opamp on a different NR basin (47 %). Shipping tsmc7/tsmc16 ckpts installed into
resolver slots `tsmc{7,16}_dn_medium_*`. ⚠ **tsmc5/tsmc12 V6.4.4 baselines are
absent on this machine** — must be installed from the S19 sha256 manifest for the
full S4 board.

### ✅ S0 — floor-k settling diagnostic — **KILL** (committed `ed6a1b9`)
Swept `k ∈ {0.3,0.5,0.6,1.0,2.0}` on the shipped tsmc7 opamp gate. k=0.5 → 10.78 %
(reproduced). Gain wildly non-monotone (15.0 → 10.78 → **0** at k=0.6 → 42.7 →
7.30): floor-k **hops NR basins**, not a controllable accuracy lever. k=2.0 "PASS"
= **E3 false-pass** (NRMSE still 31.6 %). Env-gate kept as default-off diagnostic
infra (`PYCIRCUITSIM_GDS_FLOOR_K`, Rule-5 preserved). Bonus finding: **tsmc7 opamp
is itself basin-fragile** → strengthens the S2 case. `S0_floork_diagnostic.md`.

### ✅ S1 — `--size large` (384×6, ≈0.9 M params) — **KILL: "capacity is not the bind"**
control-v2 recipe (`large`, `--apply-filter off`, EMA, v2 data), 4 seeds ×
{nmos,pmos}, pilots tsmc5 + tsmc7 (full 800 epochs, val MSE ~3e-4).
- **tsmc5 switchcap flat ~11.3 %** (no SC win); RO regressed 2.61 → 9.6–12.7 %;
  opamp 2/4 collapse; inverter NRMSE 1–2 % (no gain).
- **tsmc7 opamp 0/4 PASS, 3/4 collapse to gain 0** (vs 10.78 % baseline); RO
  regresses (only s42 passes); SC unchanged.
- **The larger net fits the value surface BETTER yet LOSES the NR-fixed-point
  properties** (opamp gain, RO period). The deriv-fidelity ⟂ opamp/RO tension is
  **capacity-independent and capacity-worsened**; the S12 corridor-vs-opamp tension
  is **not** capacity-bound. Confirms the plan thesis. `tsmc{5,7}_dn_lg_*` kept on
  disk (gitignored), **none promoted**. `S1_large_{tsmc5_pilot,tsmc7_verdict}.md`.

#### ✅ S1 RE-RUN (2026-06-19/20) — "does re-running change the result?" → **NO**
Two independent checks, both reproduce the KILL (`S1_rerun_verdict.md`):
1. **Re-eval** of the original `tsmc7_dn_lg_s*` under the NEW continuation-first
   solver: s7→gain 0, s17→361 (+121%) — still FAIL ⇒ the collapse is not a
   source-stepping artifact.
2. **Fresh re-train** (parallel ns `tsmc7_dn_lgB_s*`, same recipe/seeds, GPU 1,
   ~17 h wall under heavy CPU contention): opamp **4/4 FAIL** (s42→0, s7→0, s31→0
   collapse; s17→**361.4, byte-identical** to the original s17 basin) — an exact
   reproduction of S1's 0/4. The over-gain basin is a robust architecture+seed
   property, not training noise. **S1 stays KILLED; `large` not promoted.** The S2
   win is on the **medium** ckpt — continuation-first cannot rescue the large net
   (its value surface has no recoverable high-gain branch on most seeds).

### ✅ S2 — continuation-first DC sweep — **KEEP: tsmc7 opamp FLIPS 10.78%→8.63% PASS**
Implemented in `run_dc_sweep` (`simulation.py`): for warm-started points (`point>0`)
on NN circuits, the fast path now solves DIRECTLY from the neighbour with
source-stepping **disabled** (sources at full, NR from the warm start); the GMIN
retry restores the 5-step source-stepping homotopy as the fallback. Gated on
`has_nn` → BSIM-CMG path byte-identical. `S2_dc_continuation.md`.
- **A/B (shipping slots, CPU):** tsmc7 opamp **10.78% FAIL → 8.63% PASS**
  (reproduces the 10.78% baseline under the stash, then improves); **deterministic
  across OMP∈{1,2,4}** (8.63/8.63/8.63, byte-identical). tsmc16 opamp 5.14%→4.92%
  PASS and now **NG-locus-faithful** (NRMSE 69.5→17.0, trip −146→−10mV).
- **No regression** on the testable techs (tsmc7+tsmc16): ring_osc 2.86/3.99,
  sram_snm force_ic+all-pos, switchcap 1.02/2.01, inverter VTC 1.89/1.27, DC-55
  23/23. **Board for the testable techs 7/8 → 8/8.**
- **Plan hypothesis REFUTED (recorded dead-end):** continuation does NOT change
  basin selection — the `s12cor_w3` seed family stays 0/197/383/0 (s31 unchanged at
  383). The 197/383/0 split is **value-surface-owned**, not solver-path-owned. The
  win came from path-preservation (shaving ~2 pp off tsmc7 over-gain + recovering the
  tsmc16 faithful trip), not de-fragilization.
- **Caveats → S4:** tsmc7's pass is a gain-gate pass on a still-unfaithful locus
  (trip −144mV, NRMSE 68%) — S3 still motivated. **tsmc5/tsmc12 unverified** (absent
  baselines); must confirm their monostable opamps are unchanged (and run
  lifted-source 12/12) once installed. Headline **14/16 → 15/16 conditional**.

### ⏭ Next (resume here)
- **S3** — EKV-like analytic backbone + bounded NN residual. Now doubly motivated:
  S1 proved the value-surface bias is not capacity-curable, and S2 confirmed the
  tsmc7 over-gain is value-surface-rooted (continuation only shaved the magnitude,
  did not make the locus faithful). Only a structural functional-form fix remains.
- **S4** — compose + promote; **first** install the absent tsmc5/tsmc12 baselines
  from the S19 sha256 manifest, then re-verify the S2 change does not regress their
  opamps + run lifted-source 12/12.

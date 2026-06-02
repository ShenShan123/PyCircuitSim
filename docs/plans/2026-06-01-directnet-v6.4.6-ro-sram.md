# DirectNet V6.4.6 — Close TSMC7 ring-osc + SRAM `force_ic` (diagnosis-first, opamp-safe)

**Date:** 2026-06-01  •  **Status:** SHIPPED at **9/16** (no behavioral change) — 2026-06-02  •  **Branch:** `feat/v6.4.6` (cut from `feat/v6.4.4` @ `54c4759`)
**Authoring:** synthesised from a 28-agent team discussion (6 expert lenses × propose → adversarial critique → architect synthesis); see Appendix B.

> **FINAL OUTCOME (2026-06-02) — shipped 9/16, commit `6495c34`, V6.4.4 stays canonical.** Phase 0 (6/6) killed the RO Jacobian-distillation lever (P0-C) and unlocked an SRAM solver path (P0-A). **Phase 1 then found the SRAM path is a *measurement* fix, not a gate-close:** the `force_ic` early-return left `_last_solve_converged` stale (guaranteed 0/8 regardless of NR); hardening it + adding a KCL-residual gate + **tightening the rail band `VDD/4`→`0.1·VDD`** gives the honest result — the released NN 6T cell lands in the **documented inboard attractor on all 4 techs → genuine `force_ic` 0/8`**. An intermediate "4/8 → 11/16" was **retracted in adversarial review as a false-PASS** of that attractor (storage-"0" at 24–30 % VDD; byte-identical to V6.4.4). The plan's constraint-continuation **homotopy was built and KILLED** (the railed point P0-A found is *NR-unstable*; folds at g*≈1e-5 S). **Post-ship RO diagnostics P0-G + P0-H (2026-06-02)** then localised the RO gap precisely: integrator selection / finer tstep do **not** close it (P0-G: both Trap & BE converge to a ~50.4 ps continuum limit, ~8 % > NG; integration truncation is only ~0.4 ps of the 4.18 ps gap), and the **charge VALUES are exact** (P0-H: ≤2 aC) → the ~3.7 ps residual is owned by the **NMOS dynamic `id` VALUE** (~20 % peak pull-down under-prediction). **Net V6.4.6:** the only code shipped is the corrected SRAM probe; **RO and SRAM both deferred to V6.4.7** — RO with a *concrete, de-risked* lever (id-value correction, pending the **P0-I** causal id-injection swap), SRAM as a model-fidelity off-leakage gap (Phase-3 physics core, P0-D-caveated). The revised "10/16 (SRAM via Phase 1)" target below was NOT met — Phase 1 closed 0 gates; see CHANGELOG "V6.4.6".

> **One-sentence summary.** Climb out of V6.4.4's **9/16** by closing the two gates V6.4.5 *confirmed are architectural* — TSMC7 ring oscillator (8.97 % → ≤5 %) and SRAM `force_ic` rail-snap (0/8 → up to 8/8) — but treat them as **two independent problems with two independent root causes**, gate every GPU-spend behind a **zero-GPU diagnostic that can kill the lever before it is built**, and make the only retrain path **opamp-safe by construction** (frozen-base LoRA) so we never repeat V6.4.5's 13/16 opamp collapse.

> **Headline bet.** The SRAM gate is most likely a **solver bug, not a model bug** (`solver.py:938-952` throws away the correctly-railed constrained solve and re-solves *unconstrained* into the NN inboard attractor) — fixing it is **0 GPU** and worth up to **8 gates**, *if* a self-consistent railed DC fixed point exists (Phase-0 probe decides). The RO gate is a **slope/charge-Jacobian** error that pointwise MAE never sees; the highest-EV close is **frozen-base LoRA + analytic-OSDI Jacobian distillation**, gated behind a 0-GPU overlay that says *which* surface (gds vs caps vs `qs`/BDF-2) owns the 4.2 ps phase walk. The from-scratch split-head rewrite is **deferred and funded last** — it is the highest opamp risk and most likely exonerated by the Phase-0 cap ablation.

---

## 1. Status entering V6.4.6

V6.4.4 is canonical at **9/16** complex-circuit gates (re-baselined exactly in `results/v6_4_5/phase1_v6_4_4_rebaseline.md`):

| Benchmark   | Pass | Failing cells |
|-------------|:----:|---------------|
| ring_osc    | 3/4  | **TSMC7 8.97 %** (NG 46.64 ps, DN 50.82 ps; gate ≤5 % → need ≤48.97 ps; NRMSE 54 %, R²<0) |
| opamp       | 1/4  | TSMC7 30.67 %, TSMC12 10.94 %, TSMC16 100 % (flat). Only TSMC5 passes (2.64 %) — **out of scope, protected as no-worsening** |
| sram_snm    | 4/4 butterfly, **0/8 `force_ic`** | every tech lands at q≈0.87 / qb≈0.20 (and mirror) |
| switchcap   | 1/4  | only TSMC7 (out of scope, protected) |

**Closed gates we must not break (regression budget):** inverter VTC 8/8 (NRMSE 1.21/2.37/2.05/1.33 %), inverter transient 8/8 (1.62/1.09/1.41/1.45 %), TSMC5 opamp PASS, sram butterfly 4/4, switchcap TSMC7 PASS, extended harness DC 55/55 + transient 64/64.

### What V6.4.5 already ruled out (do **not** re-propose as-is)

| # | Dead end | Evidence |
|---|----------|----------|
| D1 | `NN_SYMMETRIC_CAPS=1` (force cgd=cdg) | TSMC7 RO period **bit-for-bit unchanged** → RO drift is not cap-*asymmetry* |
| D2 | RO `max_substeps=4` (LTE control) | 8.97 % → 8.04 % at 2× wall time, misses gate → not LTE-dominated |
| D3 | SRAM butterfly-lobe warm-start | still settles q≈0.7–0.8 → q≈0.18 is a **true NN attractor**, not a poor seed |
| D4 | Rule-15 `Ioff_rail = max(\|id_raw\|, k·NFIN·1nA)` added to id | doubles conducting current, inverter VTC 1.21 → 11.56 % at smallest k. Reverted |
| D5 | seed/recipe lottery (32 trainings) + `--monotonic` recipe | RO floor 8.96 % feasible; **13/16 retrains collapse the TSMC7 opamp to gain 0** |

**The two open gates are therefore confirmed *architectural / solver-path*, not seed/recipe/flag-addressable.** V6.4.6 is the architectural iteration.

---

## 2. The central reframe — *the slope NR consumes is never trained*

The single most important fact uncovered in this planning round, verified against the code:

- At **inference**, the conductances the Newton solver consumes are `gm = -∂id/∂Vg`, `gds = ∂id/∂Vd`, `gmb = -∂id/∂Vb` taken by **`torch.autograd.grad(id, V)`** (`mosfet_nn.py:362-370`, denorm chain-rule `:484-505`). The **predicted `gm/gds/gmb` output columns are discarded**. Capacitances are likewise `autograd` of the `qg/qd` charge columns.
- During **training**, the loss is `MAELoss × per-target LDS weights` on the 13 output columns **pointwise** (`losses/bni_mae.py`, `trainer.py:_epoch_train` — first-order, no `create_graph`). **The autograd Jacobian of `id` is never tied to the analytic OSDI `gm/gds/gmb` that already sit in dataset columns 1/2/3.** The slope of the trained `id`-surface — exactly what NR integrates — is unconstrained.
- The dataset (`external_compact_models/bsimar/data/datasets/tsmc{5,7,12,16}_{nmos,pmos}.npz`, ~2 M rows each) carries the analytic OSDI derivatives already: `outputs[:, k]` for `k∈{1,2,3}` = `gm/gds/gmb`, `k∈{8..12}` = `cgg/cgd/cgs/cdg/cdd`. **Jacobian/derivative matching needs no data regen.**

Both failing gates are slope/curvature problems, which is exactly what an unconstrained-slope fit gets wrong:

- **RO phase walk** = a `gds` + charge-Jacobian (Cgd/Cgg) shape error integrated ~12× over the BDF-2 oscillation cycle. The `id` *value* fit is already excellent (VTC 1.21–2.37 %); the *slope/cap shape* is not supervised.
- **SRAM non-rail attractor** = the off-state `∂id/∂Vg` (subthreshold slope) is too shallow / carries a spurious mid-rail `Id`, so the cross-coupled DC equilibrium of the NN inverter pair sits inboard. The `id` asinh normalisation (`asinh_scale_id ≈ 5.73e-5`) crushes the ~1 nA leakage band ~6 decades below the on-state, so the subthreshold roll-off carries near-zero loss gradient.

**Consequence for the plan:** the levers worth building are the ones that supervise the *slope* (Jacobian distillation) or *replace* the off-state slope with a closed-form exponential (physics core) or *bypass the model entirely* for SRAM (solver path). Seeds, flags, and pointwise-MAE retrains are already dead.

> **P0-C correction (2026-06-01) — the RO half of this reframe is FALSIFIED.** `gds` and the cap-derivatives are **Jacobian-only**: the transient companion injects the *resistive* current from the `id` **value** (`_stamp_mosfet_dc:304`, `i_eq = i_leaving − g_ds·v_ds − …`) and the *capacitive* current from the **charge** values `qg/qd` (`_stamp_mosfet_transient:1718` `i_g_cap = coeff·charges["qg"] − h_g`), while `gds`/`cgg`/`cgd`/… enter only the Jacobian matrix and the matching RHS offset (`:1757-1782`) and **cancel exactly at the converged NR fixed point**. So the "`gds` + charge-Jacobian shape error integrated over the BDF-2 cycle" bullet does **not** hold for the *RO period*: P0-C swapped in the exact OSDI gds/caps and the period moved ≤0.01 ps. The RO period is owned by the **id-VALUE + charge-VALUE (qg/qd) trajectories + the BE/Trap/BDF-2 truncation**, none of which a slope/cap Jacobian-distillation touches. The **SRAM** slope argument (off-state `∂id/∂Vg` shaping the cross-coupled equilibrium) is *not* affected — but it is closed by Phase 1 (solver path), not by distillation.

---

## 3. Target & hard constraints

**Target:** ≥ **11/16** (V6.4.4 9/16 + ring_osc TSMC7 + ≥1 SRAM state), realistic **11–12/16**, stretch **13/16** (RO + full SRAM 8/8). Inverter 8/8 held; extended harness 55/55 + 64/64 held; TSMC5 opamp PASS held.

> **Revised after Phase 0 (2026-06-01).** ring_osc TSMC7 is **removed from the V6.4.6 target** — P0-C proved the Phase-2 Jacobian-distillation lever is dead-on-arrival (the RO period is owned by the id-value/charge/BDF-2 integration, not the distillable gds/cap Jacobian; deferred to V6.4.7). **Revised target: 10/16 = V6.4.4 9/16 + ≥1 SRAM `force_ic` state via Phase 1** (stretch: full SRAM 8/8). Inverter 8/8, extended harness 55/55+64/64, TSMC5 opamp PASS still held.

**CLAUDE.md constraints (re-stated, with V6.4.6 relaxations):**

- **Rule 1 (mandatory, non-negotiable):** `gm/gds/gmb` stay `autograd(id, V)`; caps stay `autograd(q, V)`. Every architecture keeps `id` end-to-end differentiable. A LoRA adapter, a physics skeleton added to the `id` column, and a Jacobian-distillation loss all preserve this — the autograd graph still flows through the same `id` scalar.
- **Rule 10 (no new loss terms) — RELAXED for the architectural track.** V6.4.5's Track B already sanctioned an **analytic-OSDI** Jacobian-distillation term. The Phase-1b Sobolev dead end used *self-distilled autograd* targets; matching **analytic OSDI** derivatives is a different, well-posed experiment. The relaxation is scoped to derivative-matching against ground-truth OSDI columns and the physics-core residual ratio — **not** a return of `DirectLoss`/`SlopeMatchLoss`/sign-consistency.
- **Rule 15:** the inference-time Vds correction stays inference-only. The Phase-1 solver-path fix is a *solver* change, not a Vds correction; it does not couple to retraining.
- **Rule 3 (surgical):** per-tech, minimal-delta; **no full 8-cell from-scratch retrain** (that is the D5 collapse regime). Retrain-based phases use frozen-base LoRA exclusively.
- **Rule 16:** report MRE / R² / NRMSE / MaxErr at every gate, every diagnostic.
- **Rules 17–18:** ASAP7 excluded; LEVEL=74 BSIMAR parked. DirectNet only.
- Charge conservation `qs = -(qg+qd+qb)` always enforced analytically.

---

## 4. Phase 0 — Diagnostics + holdout/baseline freeze (0 GPU, ~1.5 days)

**The whole iteration's go/no-go lives here.** Six zero-GPU measurements, each retiring or redirecting a downstream phase. No model or solver *behaviour* changes — instrumentation only. Every result is recorded with the Rule-16 quartet, even when it kills a phase (a negative is a first-class deliverable). Pin `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` throughout (the ~20× VTC trip gain makes the inverter scatter ~±1 %).

| # | Diagnostic | Decides |
|---|-----------|---------|
| **P0-A** | **force_ic railed-fixed-point existence probe** (~2 min, *run first*). In `force_ic_probe` (`tests/verify_complex_sram_snm.py`), after the unconstrained re-solve, compute the unconstrained MNA/KCL residual `‖b − A·v‖` at the **rail-seeded** solution via the existing `_dc_residual_at` (`solver.py:908`). | Whether a self-consistent **railed DC fixed point exists** for the released NN 6T cell. Residual small at the rail → `force_ic` is a recoverable **solver** problem (Phase 1 viable, ≤8 gates at 0 GPU). Residual large → no railed equilibrium (D3 confirmed) → re-spec the gate (transient write-then-hold) or close it with an off-leakage model change (Phase 3). **The single most decision-critical unknown.** |
| **P0-B** | **RO-trip Jacobian overlay** (~1 hr). On the frozen V6.4.4 `tsmc7_nmos/pmos`, dump **post-Rule-15** autograd `gm/gds` and autograd `Cgd/Cgg/Cdg/Cdd` along the *actual failing RO trip trajectory* (one cycle from the existing sim), overlay vs analytic OSDI `gm/gds` (cols 1/2) and caps (cols 8–12) at the **same bias points**. Report MRE/R²/NRMSE/MaxErr per surface. | *Which surface owns the 4.2 ps phase walk.* If gds/caps already track OSDI within ~10 % post-correction → RO is **not** a slope/cap error (it lives in `qs=-(qg+qd+qb)` reconstruction or BDF-2 truncation) → the entire Jacobian-distillation RO lever (Phase 2) is **dead before any GPU**. If gds diverges → distill current-Jacobian. If caps diverge → distill charge-Jacobian. If both → distill both. |
| **P0-C** | **Cap-swap RO ablation** (~20 min). In a scratch TSMC7 RO transient, replace **only** the NN autograd caps with analytic OSDI caps (keep NN id/gds); re-measure period. Mirror: swap only gds, keep NN caps. | Whether caps vs gds own the phase walk. D1's bit-for-bit null gives a **strong prior that caps are exonerated** — if period stays ~50.8 ps under the cap swap, KILL the charge-Jacobian / Softplus-cap-head family and route RO to gds-distillation (or the BDF-2/`qs` investigation if P0-B says slope is already fine). |
| **P0-D** | **SRAM off-transistor attractor instrumentation** (~30 min). At the stuck q≈0.18 force_ic point on TSMC7, print the OFF transistor's `(Vgs, Vds)`, its **post-Rule-15** inference `id`/`gds`, the analytic OSDI `id`/`gm` at that exact bias, and `Vgs` vs the per-tech `VTH`. | (a) Is the off-leakage genuinely **over-modelled** (NN `id` ≫ OSDI `id` → distillation/skeleton can help) or already ~0 (OSDI `id`≈0 too → attractor is charge/homotopy-driven, the leak family is dead)? (b) Is the OFF device in **deep-off** (`Vgs≪VTH` → an off-floor gate is safe) or **moderate inversion** (`Vgs≈VTH` → any off-gate also perturbs the inverter trip = D4 territory)? Retires either the leak-magnitude family or the gate-S/skeleton family. |
| **P0-E** | **Subthreshold normalisation / sign-noise audit** (~30 min). On `tsmc7_nmos.npz` confirm `asinh_scale_id` crushes the leakage band, and quantify the fraction of `|id|<1e-7` rows that are **negative / literal-0** (preliminary probe: ~45 % negative, ~6 % literal 0 — PyCMG floor noise). Check the analytic OSDI `gm` in the `subthresh` `sample_class` (code 2, ~97 200 rows). | Whether a subthreshold log-reweight or log-derivative distillation is even **safe**. If the sub-1e-7 band is sign-random floor noise, dropping the asinh floor amplifies junk and corrupts the surface → use a clipped/winsorised signed-floor target (`|id|>1e-12`, Huber on ln-current) or abandon the reweight. |
| **P0-F** | **Baseline + holdout freeze** (~10 min). Run V6.4.4 on the **selection-blind** cells and pin a committed baseline JSON from *actual sims* (not the markdown): TSMC12 RO (PASS 3.01 %), TSMC16 RO (PASS 2.88 %), TSMC12 opamp (FAIL 10.94 %), TSMC5 opamp (PASS 2.64 %). | The Goodhart holdout set (Phase 4): TSMC12 RO + TSMC16 RO are valid **PASS/FAIL vetoes**; TSMC12 opamp can only be a **no-worsening numeric diff** (it already fails); plus a held-out SRAM NFIN corner never used in selection. |

**Gate file.** `results/v6_4_6/phase0_diagnostics.md` (overlay tables + the decision each result triggers).

**Decision tree out of Phase 0:**

```
P0-A railed residual small?  ── yes ─▶ Phase 1 continuation viable (SRAM, 0 GPU)
                             ── no  ─▶ Phase 1 → transient write-then-hold re-spec; SRAM model-fix → Phase 3
P0-B/C RO owner?  ── gds diverges ─────▶ Phase 2 distill current-Jacobian (gds)
                  ── caps diverge ─────▶ Phase 2 distill charge-Jacobian (caps)  [unlikely; D1 prior]
                  ── both within ~10% ─▶ Phase 2 RO lever DEAD → BDF-2/qs investigation (new scope) or RO stays open
P0-D SRAM leak?   ── NN id ≫ OSDI id, deep-off ─▶ Phase 3 skeleton/off-floor viable
                  ── OSDI id ≈ 0 / Vgs≈VTH ─────▶ Phase 3 skeleton DEAD → SRAM closable only via Phase 1
P0-E subthresh band clean? ── yes ─▶ log-reweight allowed as a Phase-2 aux;  ── no ─▶ clipped target only
```

### Phase 0 results (2026-06-01) — 6/6 done

Per-diagnostic gate files are written under `results/v6_4_6/phase0_{A,B,C,D,E,F}_*.md`
(+ `baseline_v6_4_4.json`) and consolidated in `results/v6_4_6/phase0_diagnostics.md`.
Each diagnostic ran with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
canonical V6.4.4 checkpoints — no retrain, no checkpoint mutation, tracked tree left
clean (P0-A's `solver.py` instrumentation was reverted; `git diff` over code is empty).

| # | Status | Headline result (Rule-16 where applicable) | Decision triggered |
|---|--------|--------------------------------------------|--------------------|
| **P0-A** | ✅ done | A **railed DC fixed point EXISTS**: unconstrained KCL residual at the rail seed = 8.5e-5 (TSMC5) / 1.26e-4 (TSMC7) vs threshold 6.5e-3 / 7.5e-3 (ratio 0.013–0.017, both states). The STUCK inboard point (q≈0.82 / qb≈0.23) has an equally small residual ⇒ the released NN 6T cell is **bistable**; `solver.py:938-952` selects the wrong basin. | **Phase 1 VIABLE (0 GPU, ≤8 SRAM gates).** Track the railed branch via constraint-continuation homotopy; harden the probe's KCL-residual accept first. Transient write-then-hold re-spec NOT triggered (held in reserve). |
| **P0-B** | ✅ done | Along the TSMC7 RO trip cycle (26 pts, stage-5 N+P), **gds is the dominant divergent surface**: NRMSE 22.9 % (NMOS) / 20.4 % (PMOS), R² 0.24/0.34, MaxErr 0.82/0.45 mS — far outside the ~10 % bar (gds-floor binds only 2–4/26 pts, so not a floor artifact). gm/id track in dynamic range (NRMSE ≤6 %); **caps largely exonerated** (cgg/cdg ≈0.5–1 %, R²≈1.0; only cgd/cdd-NMOS soft but ≤145 aF vs the 0.5 fF stage load). | **Phase 2 ALIVE → distill the CURRENT-Jacobian (gds), not caps.** Charge-Jacobian / Softplus-cap-head family provisionally exonerated (D1 prior), pending the P0-C causal swap. |
| **P0-C** | ✅ done | RO period ablation (TSMC7, 3×): baseline **50.82 ps**; **cap-swap 50.82 ps (bit-for-bit)**; **gds-swap 50.83 ps** (floored & no-floor) — vs NG 46.64 ps. Injecting the exact OSDI gds/caps moves the period **≤0.01 ps** (400× smaller than the 4.18 ps gap). | **Phase-2 RO Jacobian-distillation DEAD before any GPU.** P0-C overrides P0-B: the divergent gds is real but **causally inert** on the period — confirmed by the solver code (gds & cap-derivatives are Jacobian-only; they cancel at the converged NR fixed point — `_stamp_mosfet_dc:304`, `_stamp_mosfet_transient:1718-1782`). The period is set by the **id-VALUE + charge-VALUE (qg/qd)** trajectories + BE/Trap/BDF-2 truncation. **Kill gds-distill, charge-distill, and the deferred Softplus-cap-head for RO; defer the RO gate to a scoped V6.4.7 dynamic-id / qs / BDF-2 investigation** (plan §11 risk-1). |
| **P0-D** | ✅ done | At the TSMC7 stuck point (q=0.815, qb=0.226) the off-leak is **over-modelled**: the pinning pull-down Mnr sources −6.36 µA vs OSDI −0.84 µA (7.5×); the NN carries a ~5.66 µA subthreshold floor where OSDI rolls off ≥3–4 decades (Id–Vgs overlay NRMSE 21.5 %, R² 0.47). **But** Mnr sits at Vov=+45 mV ≈ VTH (Vgs 226 vs OSDI-VTH 181 mV) — weak inversion = D4 territory. | **Phase 3 skeleton KEEP-WITH-STRONG-CAVEAT, fallback only.** Leak-magnitude error is real (family alive) but the pinning device is near-VTH, so any off-gate risks the inverter trip; fund only if Phase 1 fails, and it must clear the VTC regression budget. |
| **P0-E** | ✅ done | `tsmc7_nmos.npz`: the `\|id\|<1e-7` band is sign-random PyCMG floor noise (45.0 % negative of band; 85.4 % negative of the non-zero tail; 6.00 % of all rows literal-0). asinh crushes the 1nA–100nA band to **0.0113 % of the normalised id range** ⇒ pointwise MAE sees ~nothing there. OSDI gm/gds in the subthresh class are **CLEAN** (0 % NaN, <1.7 % wrong-sign). | **Raw asinh-floor drop / log-reweight on the id value = UNSAFE.** Safe Phase-2 lever = clipped support (`\|id\|>1e-12`) + Huber-on-ln-current, distilling against the clean OSDI gm/gds (the *derivative* target, not the floor-noise id value). Dovetails with P0-B (gds-distillation). |
| **P0-F** | ✅ done | V6.4.4 baseline re-measured from actual sims, **matches the plan/CHANGELOG exactly**: TSMC12 RO 3.01 % PASS, TSMC16 RO 2.88 % PASS (blind vetoes); TSMC5 opamp 2.64 % PASS, TSMC12 opamp 10.94 % FAIL (protected/no-worsening); TSMC7 RO 8.97 % (NG 46.64 / DN 50.82 ps, target). Blind SRAM corner TSMC7 NFIN=3: butterfly positive, `force_ic` 0/2 (inboard attractor). | **Holdout set FROZEN** in `baseline_v6_4_4.json` (pinned to HEAD `54c4759` + 8 checkpoint sha256s). Usable as the Phase-2/3 Goodhart veto set. |

**Net Phase-0 read (6/6).** **One lever killed, one unlocked, one demoted:** **Phase 1 (SRAM) is UNLOCKED** (P0-A — a railed equilibrium provably exists; the headline 0-GPU bet), **Phase 2's RO Jacobian-distillation lever is DEAD** (P0-C — gds/caps are Jacobian-only and causally inert on the period; the RO walk is owned by the id-VALUE + charge-VALUE trajectories + BDF-2 truncation, deferred to V6.4.7), and **Phase 3 is demoted to a caveated SRAM fallback** (P0-D — over-modelled leak is real but the pinning device sits at ≈VTH). The from-scratch split-head (§10) stays **deferred and is now confirmed unnecessary for RO** — its cap-head fixes a Jacobian surface that does not set the period. The retrain loss-design lesson from P0-E (clipped Huber-on-ln-current distillation against clean OSDI gm/gds) carries forward to V6.4.7 if/when the dynamic-id route is scoped. **Revised V6.4.6 target: 10/16** — V6.4.4 9/16 + ≥1 SRAM `force_ic` state via Phase 1 (stretch: full SRAM 8/8); the RO gate moves to V6.4.7 (plan §11 risk-1).

---

## 5. Phase 1 — `force_ic` railed-solution recovery (SRAM, inference-only, 0 GPU, ~2 days)

**The guaranteed-shippable, zero-regression-surface win — *if* P0-A says a railed fixed point exists.**

**Root cause (verified):** `solver.py:938-952` — on the `force_ic` path the solver pins the `.ic` nodes with temporary voltage sources, converges (correctly railed), then **removes the temp sources, sets `force_ic=False`, and re-solves UNCONSTRAINED** using the railed result only as an initial guess. It returns that unconstrained solution at line 952. The NN's cross-coupled pair has a global basin toward q≈0.18, so the release walks straight there. Worse, the early `return` at 952 means **`_last_solve_converged` (set at line 988) is never updated on this path** — it is stale.

**Step 1 — harden the probe FIRST (≈30 LOC, mandatory before any continuation).** The current acceptance (`verify_complex_sram_snm.py:178-181`) is `_last_solve_converged AND |q−q0|<VDD/4 AND |qb−qb0|<VDD/4` — a **stale flag** plus rail-proximity, with **no unconstrained-KCL residual check**. Add a hard `_dc_residual_at` acceptance gate on the *released* solution and fix the stale flag, so a pinned-node artifact can never false-PASS. *(The current 0/8 is genuine — q≈0.87 is well outside VDD/4 of the rail — but without this gate any future "win" could be a phantom. This is the top Goodhart risk in the iteration.)*

**Step 2 — branch on P0-A:**

- **If a railed fixed point exists** → replace the one-shot release with a **constraint-continuation homotopy**: relax the IC temp-V-sources via `λ: 1 → 0` (e.g. soft pin = series conductance scaled by λ, or a sequence of shrinking trust-region re-solves), full NR per stage, each warm-started by the previous, supplies held at full VDD, source-stepping asserted OFF inside the homotopy. Track the railed branch instead of falling into the global basin.
- **If no railed fixed point exists** → re-spec the gate as a **transient write-then-hold** (drive q→VDD via wordline access, release, integrate `.tran`, assert no collapse within VDD/4 over a hold window) — the physically correct SRAM-retention test. Record that the **DC `force_ic` fixed point does not exist for the released NN cell** as a dead end with the residual numbers.

**Kill criteria.**
- P0-A residual large on TSMC5 **and** the continuation slides off at some λ\*<1 to q≈0.18 → KILL the continuation variant; pivot to transient-hold.
- If even the transient-hold collapses → SRAM `force_ic` is confirmed model-fidelity → route to Phase 3.

**Ships if.** The hardened probe (with the KCL-residual gate) PASSES on all 4 techs × 2 states **without per-tech λ tuning**, with the held-out SRAM NFIN corner blind. If only the transient-hold re-spec passes, ship it as a corrected gate definition plus the dead-end record. **0 GPU, zero model-regression surface** → this de-risks the whole iteration.

**Gate file.** `results/v6_4_6/phase1_force_ic_recovery.md`.

---

## 6. Phase 2 — Frozen-base LoRA + analytic-OSDI Jacobian distillation (ring_osc, TSMC7 first, ~8–16 GPU-hr, ~4 days)

> **❌ KILLED by Phase 0 (P0-C, 2026-06-01) — NOT funded.** The gate condition below resolved **NO**: P0-B found `gds` diverges 20–23 % NRMSE, but P0-C proved the divergence is **causally inert** — swapping the exact OSDI gds/caps into the live TSMC7 RO transient moves the period ≤0.01 ps, because gds and the cap-derivatives are Jacobian-only and cancel at the converged NR fixed point (code-confirmed; see the §2 P0-C callout). gds-distill, charge-distill, **and** the deferred Softplus-cap-head (§10) are all dead for the RO gate; the RO walk lives in the id-value/charge trajectory + BDF-2 truncation → scoped V6.4.7 investigation. The LoRA-on-frozen-base + clipped-Jacobian-distillation **machinery** described below is retained for the record and possible V6.4.7 reuse, but **no GPU is spent on it in V6.4.6**.

**Only funded if P0-B/C say the slope/caps are actually wrong.** Closes TSMC7 RO by bending **only the slope/cap shape NR integrates over the BDF-2 cycle** toward the analytic OSDI derivatives, *without moving the `id` value the inverter and TSMC5 opamp live on.*

**Why LoRA-on-frozen is the opamp-collapse antidote.** D5 collapsed the TSMC7 opamp to gain 0 in 13/16 from-scratch retrains. A LoRA adapter on the **frozen** V6.4.4 base is **zero at init → byte-identical V6.4.4 → opamp PASS by construction**, and it caps the regression surface to a single merged-checkpoint re-validation instead of a from-scratch seed lottery.

**Build (all NET-NEW code — no LoRA infra, no double-backward, no warm-start path exists today; budget honestly):**

1. **LoRA wrapper (~50 LOC):** rank ~4–8 adapters `W' = W + B·A` on the shared-trunk `nn.Linear`s of the frozen `tsmc7` base (`bsimar/models/direct_net.py`). Only adapter params train; base `requires_grad=False`.
2. **Jacobian-distillation loss (~60 LOC, double-backward):** in a new training path (`trainer.py:_epoch_train` is first-order — needs `create_graph=True`), add, per the P0-B verdict:
   - `λ_J · ‖autograd(id)→gds_OSDI (col 2)‖ (+ gm col 1)` and/or
   - `λ_C · ‖autograd(qg,qd)→caps cdg/cgd/cgg/cdd (cols 8–12)‖`
   - **distilled in PHYSICAL space after the denorm + Rule-15 chain**, so the supervised quantity is exactly what NR consumes; weight by `1/sqrt(s²+y²)` to undo the asinh small-Id down-weighting in the trip band. Sweep `λ_J ∈ {1e-3, 1e-2, 1e-1}`. On the subthreshold band, distill on clipped support (`|id|>1e-12`), Huber on ln-current (per P0-E).
3. **Merge-on-save + load test (~40 LOC):** `_build_from_state` (`mosfet_directnet.py:44-72`) detects only `net.*`/`mono.*` keys — split/LoRA keys would be **rejected at load**. Merge `W'=W+BA` at save into the stock `net.*` layout; add an explicit stock-checkpoint round-trip load test. **Re-run the FULL regression suite on the MERGED checkpoint** — merged weights are new weights; the freeze guarantee holds only at init.
4. **Per-epoch early-abort on the held-out opamp gain** (Phase 4 hook), so a candidate that starts flattening the opamp dies mid-train, not at ship.

**Order:** train **one** TSMC7 cell (N+P) and validate before any sibling-tech or seed sweep.

**Kill criteria.**
- P0-B shows post-Rule-15 autograd gds/caps already within ~10 % of OSDI → KILL distillation (the 4.2 ps is in `qs`/BDF-2; route to deferred). **Most likely outcome given the D1/D2 prior — accept it cheaply.**
- One TSMC7 LoRA cell still >48.97 ps (5 % gate) → KILL.
- Merged checkpoint regresses TSMC7 inverter VTC >5 mV MaxErr, or drops TSMC5/TSMC12 opamp gain below baseline, or trips a selection-blind veto → REJECT that candidate.

**Ships if.** Merged TSMC7 checkpoint takes ring_osc <5 % **AND** holds inverter 8/8 (+5 mV MaxErr), TSMC5 opamp PASS, TSMC5/TSMC12 opamp gain no-worsening, butterfly 4/4, switchcap TSMC7 PASS, DC 55/55, tran 64/64, **AND** the selection-blind TSMC12 RO + TSMC16 RO do not regress. If TSMC7 lands but a sibling tech is touched, ship **TSMC7-only** (per-tech checkpoint mix — the V6.4.4 precedent).

**Gate file.** `results/v6_4_6/phase2_lora_jacobian_distill.md`.

---

## 7. Phase 3 — Multi-region physics-anchored core (DEFERRED SRAM fallback, conditional, 0–10 GPU-hr, ~3–5 days)

**Only funded if Phase 1 cannot rail the latch AND P0-D confirms a genuine off-leakage-magnitude error on the `id` surface.** Closes SRAM `force_ic` by construction: a closed-form subthreshold exponential owns the OFF region (≥4–6-decade suppression → railed equilibrium), the **unchanged frozen MLP** owns moderate/strong inversion (inverter/opamp preserved by construction — the lowest-D5-risk structure).

**Form.** `Id = MLP_id(V)·S(V) + Id_subexp(V; θ)·(1−S(V))` with a smooth on-state gate `S`. Use a **multi-region** core (the single-region EKV is empirically falsified for FinFET — fitting pulls ideality `n→2.6`, ~1.6 vs ≥6 decades subthreshold) with subthreshold ideality **pinned `n ≤ 1.3` (SS ≤ ~80 mV/dec) as a HARD constraint, not fitted**. Autograd flows through both terms (Rule 1 ✓); the physical `Id_core` is pushed through a differentiable asinh+zscore inside `forward()` — **this crosses the phys/normalised boundary `mono.*` never did**, so budget ~150–250 LOC for the compose-at-inference variant, ~400 LOC if retrained.

**Cheapest variant first — compose-at-inference, 0 GPU:** frozen MLP-on + prefit `Id_sub`-off, no retrain. Ship the SRAM fix without retraining if it passes `force_ic`.

**Pre-GPU fit gate.** Least-squares-fit the multi-region core to the `tsmc7_nmos` subthresh class (code 2) requiring **both ≥4-decade suppression AND ≤5 % inv_trip** before any GPU is spent.

**Kill criteria.**
- P0-D shows OSDI `id`≈0 at the attractor (leak already correct) → KILL (attractor is charge/homotopy, skeleton useless).
- OFF device sits at `Vgs≈VTH` (moderate inversion → forcing `S→0` perturbs the inverter trip via ~20× gain = D4 territory) → KILL.
- The fit gate cannot hit ≥4-decade suppression + ≤5 % inv_trip → KILL (record `n`/SS/decade numbers as a dead end).

**Ships if.** Compose-at-inference (or gated frozen-base retrain) rails `force_ic` on all 4 techs (blind NFIN corner held) AND holds the full regression budget.

**Gate file.** `results/v6_4_6/phase3_physics_core.md`.

---

## 8. Phase 4 — Goodhart/holdout promotion protocol + dead-end record (infra, 0 GPU, ~1 day)

**The antidote to the D5 failure** (13/16 opamps to gain 0, caught only at ship). Wraps every retrain-based phase; near-zero cost, pays for itself on the first avoided bad ship.

- **Committed baseline JSON** from actual V6.4.4 sims (`OMP_NUM_THREADS=1`), pinned in Phase 0.
- **Selection-blind cells — NEVER used to pick adapters/seeds/λ:** TSMC12 RO + TSMC16 RO as **PASS/FAIL vetoes** (both verified PASS); TSMC12 opamp + TSMC5 opamp as **quartet no-worsening numeric diffs** (TSMC12 opamp already FAILS, so it cannot be a PASS/FAIL veto); plus a held-out SRAM NFIN corner.
- **Hard fail-loud opamp gate:** `gain < 10 → reject` (the recalibrated V6.4.5 flag); per-epoch early-abort on held-out opamp gain.
- **Reuse + extend the V6.4.5 scorer infra:** `scripts/eval_v6_4_5_candidate.py` + `scripts/v6_4_5_search.py` (BSIMAR_CHECKPOINT_DIR-isolated, parallel-safe, never mutates canonical slots). Add the SRAM-residual gate and the blind-cell vetoes.
- Every Phase-0…3 dead end recorded with the numbers that killed it. Update CLAUDE.md + CHANGELOG before commit.

**Gate file.** `results/v6_4_6/phase4_promotion_protocol.md` + the committed `baseline_v6_4_4.json`.

---

## 9. Sequencing & promotion

| Day | Work | Gate it unlocks |
|----:|------|-----------------|
| 1 | Phase 0 (all six diagnostics; P0-A first) + Phase 4 baseline/holdout freeze | the entire decision tree |
| 2 | Phase 1 probe-hardening + continuation **or** transient-hold re-spec | SRAM, 0 GPU |
| 3 | Phase 2 LoRA wrapper + distillation loss + merge/load test (TSMC7 N+P, one cell) | RO, gated on P0-B/C |
| 4 | Phase 2 validation on the merged checkpoint + blind holdout; sibling-tech only if clean | RO ship decision |
| 5 | Phase 3 compose-at-inference (only if Phase 1 left SRAM open AND P0-D positive) | SRAM fallback |
| 6–7 | Promotion / CHANGELOG / CLAUDE.md / commit | ship |

**Promotion rules.**
- Each phase commits its result independently under `results/v6_4_6/phaseN_*.md` (Rule-16 quartet per cell). `results/v6_4_6/` follows the gitignore convention; gate files live there.
- The shipping V6.4.6 commit packages whichever subset closed gates **without regressing V6.4.4 or any selection-blind veto**: the Phase-1 solver fix (always shippable if it passes), the Phase-2 merged TSMC7 checkpoint (per-tech mix), the Phase-3 compose-at-inference core (if it rails SRAM).
- **No phase ships if it regresses V6.4.4 or any blind holdout.** A passed kill criterion is necessary, not sufficient — the post-phase headline must be ≥ V6.4.4 and the blind vetoes must hold.
- Snapshot before any retrain: `cp -r external_compact_models/bsimar/checkpoints /tmp/v6_4_4_backup_$(date +%Y%m%d)/` + `manifest.sha256`; use `BSIMAR_CHECKPOINT_DIR` to keep canonical slots intact.

**Realistic outcomes:** 10/16 = Phase 1 SRAM alone; 11–12/16 = + RO-safe LoRA; 13/16 = + full SRAM 8/8 or the physics core. Every architectural family is killable on 0–1 GPU-hr of evidence.

---

## 10. What V6.4.6 is NOT doing (deferred)

- **From-scratch split-head rewrite** (spectral-norm id-head + Softplus cap-head, ~0.27 M params, ~400–600 LOC, loader rewrite). HIGH cost, HIGH opamp risk (spectral-norm caps loop-gain — wrong direction for the high-gain TSMC5/TSMC12 opamps), and the Cgd/Cgg shape it fixes is most likely **exonerated by the P0-C cap ablation** (D1 prior). Fund **last**, and only if Phase-2 LoRA-on-caps plateaus >5 % **and** P0-B/C prove caps (not gds/BDF-2) own the walk.
- **Single-region EKV core** — empirically retired (`n→2.6`, SS impossible for FinFET). Only the multi-region variant (Phase 3) survives behind the fit gate.
- **Subthreshold asinh-rescale to 1e-9** — amplifies the ~45 %-negative/6 %-zero sign-random PyCMG floor noise → corrupts the surface. Only as a clipped/winsorised signed-floor target if P0-E shows a clean tail.
- **D4 `Ioff_rail` floor (any form, incl. the never-tried `Ioff_extra=max(floor−|id_raw|,0)`)** — the safe nA regime is SRAM-impotent (needs ~µA to move a 300 mV attractor); the potent regime doubles conducting current and breaks VTC; and `I_floor` is Vds-only without a Vgs gate, so it cannot move `force_ic`. Architectural off-leakage (Phase 3) is the correct lever.
- **Test-time seed ensemble + rail-voting for `force_ic`** — the unconstrained re-solve basin is global toward q≈0.18 (D3), so all members vote inboard. A fast NEGATIVE, useful only as the harness carrying Phase-1's continuation member.
- **Full 8-cell from-scratch retrain of any kind** — the D5 opamp-collapse regime. V6.4.6 uses frozen-base LoRA exclusively for retrains.
- **Cap-reciprocity symmetrization as an RO *fix*** — a weighted mean of two non-integrable cross-derivatives; collapses to D1. Demoted to a Phase-0 diagnostic that sizes the autograd-vs-OSDI cap gap.
- **opamp / switchcap gates beyond the passing TSMC5 opamp / TSMC7 switchcap** — not targeted; protected as no-worsening only.
- **ASAP7, LEVEL=74 BSIMAR** (Rules 17–18).

---

## 11. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| **RO owned by neither gds nor caps but by `qs=-(qg+qd+qb)` reconstruction / BDF-2 truncation** (D1/D2 prior strongly predicts P0-B shows gds/caps already within ~10 % of OSDI) → the whole Jacobian-distillation RO lever dies at 0 GPU, RO closable only by a solver-integration change not yet scoped. | **RESOLVED (2026-06-02), and it is NEITHER `qs` nor BDF-2.** P0-C fired this risk (gds/caps causally inert). The follow-up P0-G + P0-H then localised it: **(a)** BDF-2 truncation is ~0.4 ps only (P0-G: both Trap & BE converge to ~50.4 ps as tstep→0; the BDF-2 switch never even fires); **(b)** `qs`/charge VALUES are *exact* (P0-H: ≤2 aC). The ~3.7 ps continuum residual is the **NMOS dynamic `id` VALUE** (~20 % peak pull-down under-prediction). V6.4.6 shipped SRAM-probe-only at **9/16**; V6.4.7 opens an **id-VALUE** investigation, NOT a BDF-2/`qs` one — pending the P0-I causal id-injection swap. |
| **No self-consistent railed DC fixed point** (P0-A residual large) → Phase 1 continuation fails honestly; the easy 8 gates evaporate, headline drops toward 10/16. | The transient write-then-hold re-spec is the physically-correct gate and still a shippable, honest close; record the no-fixed-point result. |
| **LoRA merge-at-save destroys the freeze guarantee** — `W'=W+BA` is new weights; a rank-8 adapter at the ~20×-gain trip locus *can* still flatten the TSMC5/TSMC12 opamp. | Per-epoch early-abort on held-out opamp gain + full regression re-validation on the **merged** checkpoint, never trusting init. Ship TSMC7-only if a sibling is touched. |
| **Goodhart on the SRAM probe** — the current accept (lines 178–181) uses a stale `_last_solve_converged` + rail-proximity with no KCL check → a pinned-node artifact false-PASSES. | Residual-gate hardening is the **first** action in Phase 1, before any continuation code. |
| **Net-new code under-budgeted** — trainer is first-order, `_build_from_state` detects only `net.*`/`mono.*`, no warm-start/LoRA/double-backward path exists; the "mirrors mono, free auto-detect" framing is false. | Explicit stock-checkpoint round-trip load test; re-budgeted LOC (Phase 2 ~150, Phase 3 up to ~400 with the normalisation-boundary plumbing). |
| **Subthreshold distillation target degeneracy** — at the SRAM off-point OSDI `|id|≈0` so `gm/|id|` is 0/0, and ~45 % of sub-1e-7 rows are sign-random. | Distill on clipped support (`|id|>1e-12`), Huber on ln-current, gate on the P0-E sign-noise audit. |
| **Checkpoint sprawl / canonical churn** | `BSIMAR_CHECKPOINT_DIR` isolation (V6.4.4-shipped) for all candidates; snapshot + `manifest.sha256` before Phase 2/3; promote only at Day 6–7. |

---

## 12. Open questions (carry into execution)

1. **Does a self-consistent railed DC fixed point exist for the released NN 6T cell?** (P0-A — the single most decision-critical unknown; decides whether SRAM is a 0-GPU solver win or an architectural off-leakage problem.)
2. ~~Along the TSMC7 RO trip, does **post-Rule-15 autograd gds/Cgd already track analytic OSDI within ~10 %**? If yes, the 4.2 ps is in `qs`/BDF-2 — *which* solver-side lever then?~~ **ANSWERED (P0-B/C/G/H, 2026-06-02):** gds diverges (P0-B) but is causally inert (P0-C); the 4.18 ps is NOT in `qs`/BDF-2 (P0-H: charge VALUES exact; P0-G: tstep→0 limit ≈50.4 ps, truncation ~0.4 ps). The owner is the **NMOS dynamic `id` VALUE** (P0-H: ~20 % peak pull-down under-prediction). The V6.4.7 lever was to be an **id-VALUE correction** (frozen-base LoRA distill against clean OSDI `id`, clipped-Huber per P0-E), gated by the **P0-I** causal id-injection swap. **P0-I RAN (2026-06-03, `results/v6_4_6/phase0I_id_injection.md`) and returned a THIRD outcome neither branch foresaw — non-separability, not inert-vs-closes-gap.** Injecting the exact OSDI `id` (NMOS-only AND symmetric N+P) moved the period **enormously the WRONG way**: a genuine full-rail uniform **~92 ps** oscillation (baseline 50.83 / N+P 92.30 / NMOS 92.74 ps), ~2× baseline and *further* from NG 46.64 — the *opposite* direction from swapping id+charge together (NGSPICE 46.64). Unlike the Jacobian (P0-C, inert/separable, ≤0.01 ps), the **`id` VALUE is NOT separable from the NN charge model**; the RO period is a joint (id, charge) property. So P0-I **cannot cleanly confirm/refute** "id owns the gap," and the **id-VALUE-only LoRA is no longer de-risked** — V6.4.7 must gate any id-only fix on the live RO period immediately and consider a **joint id+charge correction (or retrain)**. (Caveat: the injection bypasses Rule-15 + floors gds, so 92 ps is a proxy warning, not proof a real autograd-consistent LoRA fails.) P0-I was also numerically hard — the naive swap diverged (inconsistent-Jacobian artifact); rebuilt as a consistent exact-bias OSDI op-point (v2) that converges but is ~20–35× slow.
3. At q≈0.18, is the OFF transistor's OSDI `id` actually >0 (over-modelled leak, fixable) or ≈0 (leak correct, attractor is charge/homotopy)? And is its `Vgs` deep-off or ≈`VTH`?
4. Can a merged rank-4–8 LoRA adapter bend gds/caps enough to move the RO period >4 % **while** holding the TSMC5 opamp gain — i.e. is there enough DOF to fix the slope without touching the trip-locus `id` value? (Unprovable until one TSMC7 cell trains.)
5. Is the analytic OSDI `gm/gds` in the subthresh class clean enough to be a distillation target given the sign-random sub-1e-7 noise? (P0-E.)
6. If `force_ic` is re-spec'd to a transient write-then-hold, does that count toward the 16-gate headline, or does it change the denominator? (Scoring question for the final report.)
7. Does the multi-region core with `n≤1.3` fit FinFET `inv_trip` curvature ≤5 % **simultaneously** with ≥4-decade subthreshold, or does the strong-inversion curvature force `n` up again as it did for the single-region EKV? (Phase-3 fit gate.)

---

## 13. Definition of done

1. Phase 0 committed: six diagnostics with the Rule-16 quartet + the decision each triggered; baseline JSON + holdout set pinned.
2. Each funded phase has a committed `results/v6_4_6/phaseN_*.md` (Rule-16 quartet per cell), with kill-criterion logs for dropped levers.
3. V6.4.6 headline ≥ **11/16** (realistic 11–12, stretch 13), inverter 8/8, TSMC5 opamp PASS, extended harness 55/55 + 64/64, **and no selection-blind veto regressed**. If a gate could not be closed, the honest 10/16 ships with the dead-end record (failures are first-class — CLAUDE.md).
4. CHANGELOG V6.4.6 written; CLAUDE.md "Status" paragraph + promoted-artifact provenance + the new dead ends updated.
5. Every dropped lever logged with the empirical numbers that killed it (the D-protocol).

---

## Appendix A — verified code touchpoints

| Concern | Location |
|---------|----------|
| force_ic pin-then-release re-solve (the bug) | `pycircuitsim/solver.py:938-952` (returns at 952; `_last_solve_converged` set at 988 is never reached on this path) |
| KCL/MNA residual probe (reusable) | `pycircuitsim/solver.py:908` `_dc_residual_at` |
| SRAM probe acceptance (no KCL check, stale flag) | `tests/verify_complex_sram_snm.py:157-189` (accept at 178-181) |
| autograd `gm/gds/gmb = grad(id,V)` + negation | `pycircuitsim/models/mosfet_nn.py:322-327`, `:362-370` |
| denorm chain-rule (asinh) | `mosfet_nn.py:484-505`; gds floor `:502-505` |
| inference Vds correction (Rule 15) | `mosfet_nn.py:509-587` (`_apply_vds_correction`, rail ramp g_max=1mS) |
| checkpoint auto-detect (only `net.*`/`mono.*`) → merge LoRA at save | `pycircuitsim/models/mosfet_directnet.py:44-72` |
| model forward, mono residual on `id` col 0 | `bsimar/models/direct_net.py` (`DirectNet.forward`, `_MonotoneVgResidual`) |
| loss (MAE×LDS, no Jacobian term) + first-order train loop | `bsimar/losses/bni_mae.py`; `bsimar/training/trainer.py:_epoch_train`, `train_directnet`, `_train_loop` (`column_weights` at ~167) |
| dataset schema | `outputs (N,13)`: id@0 gm@1 gds@2 gmb@3 qg@4 qd@5 qs@6 qb@7 cgg@8 cgd@9 cgs@10 cdg@11 cdd@12; `sample_class` subthresh=code 2 |
| reusable candidate scorer (BSIMAR_CHECKPOINT_DIR-isolated) | `scripts/eval_v6_4_5_candidate.py`, `scripts/v6_4_5_search.py` |

## Appendix B — agent-team perspectives consulted

Synthesised from 28 agents: 6 propose lenses × adversarial critique × architect synthesis (21 proposals, 14 survived adversarial review). Highest-EV survivors and how they map to phases:

| Lens | Primary recommendation → phase | Key adversarial finding absorbed |
|------|--------------------------------|----------------------------------|
| Staff/pragmatist | force_ic-is-a-solver-bug → **Phase 1**; frozen-base LoRA → **Phase 2**; defer split-head | force_ic is the only 0-GPU/8-gate win; retrain must be opamp-safe by construction |
| Jacobian-objective | analytic-OSDI Jacobian distillation → **Phase 2** | the id *value* fit is already excellent; only the *slope* is wrong; use analytic (not self-distilled) targets |
| Physics grey-box | multi-region core + OFF-floor → **Phase 3** | single-region EKV falsified for FinFET (`n→2.6`); the OFF-gate `S(Vov)` can fire in the wrong region (D4) |
| Data-centric | the move is **re-scaling/derivative-supervising existing data**, not harvesting | both gates are *inside* the training box (not extrapolation) → "more data" is dead on arrival; sub-1e-7 band is sign-random noise |
| Solver/inference | force_ic continuation homotopy → **Phase 1** | floor-only-below leakage is SRAM-impotent at safe scale; ensemble votes unanimously inboard |
| Architecture surgeon | split-head (spectral-norm id + Softplus cap) → **deferred** | HIGH opamp risk + most likely exonerated by the P0-C cap ablation; fund last |

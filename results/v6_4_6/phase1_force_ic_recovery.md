# Phase 1 — `force_ic` railed-solution recovery (SRAM, inference-only, 0 GPU)

**Date:** 2026-06-02  •  **Branch:** `feat/v6.4.6`  •  **Status:** Done — probe-hardening SHIPPED; homotopy KILLED; **0 SRAM gates closed (9/16 held)**
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Implements:** plan §5 "Phase 1". Files changed (code): `pycircuitsim/solver.py`,
`tests/verify_complex_sram_snm.py` (no checkpoint mutation; `git diff` over code
is confined to these two files).

> **Headline (corrected after adversarial review).** **Phase 1 closes ZERO SRAM
> gates; the honest headline stays 9/16.** Two genuine deliverables — both
> *correctness*, not gate-closing:
> 1. **Probe-hardening (SHIPPED):** the original `0/8` was partly a **stale-flag
>    artifact** (the `force_ic` early-return never set `_last_solve_converged`,
>    so the old gate's first clause was *always* `False`). Fixing it + adding a
>    KCL-residual gate + **tightening the rail band from `VDD/4` to `0.1·VDD`**
>    makes the probe report the HONEST result: the released NN 6T cell lands in
>    the inboard attractor (q≈0.87 / qb≈0.20, the plan's *documented failure
>    mode*) on **all 4 techs → genuine `0/8`**.
> 2. **Constraint-continuation homotopy (BUILT + KILLED):** the railed branch
>    P0-A found exists as a *residual* fixed point but is **NR-unstable**; the
>    continuation folds into the symmetric metastable point on all 4 techs.
>    Reverted; recorded as a dead end.
>
> An intermediate version of this gate file claimed **4/8 → 11/16**. That was a
> **false-PASS** caught in review (see §4.1) and is retracted. RO stays out of
> V6.4.6 scope (P0-C). V6.4.4 remains canonical; V6.4.6 ships a corrected probe
> + dead-end records, no behavioral change.

---

## 1. Root cause (verified)

`pycircuitsim/solver.py` force_ic cleanup block. On the `force_ic` path the
solver pins the `.ic` nodes with temporary voltage sources, converges (correctly
railed), then removes the pins, sets `force_ic=False`, and re-solves
**UNCONSTRAINED**, warm-started only by the railed result, and returns that at
the early `return`. Two distinct defects in the *measurement*:

1. **Stale convergence flag.** The early `return` means `_last_solve_converged`
   (set at the bottom of `_solve_newton`) is **never reached on this path** → it
   retains its `__init__` value `False`. The SRAM probe's old acceptance was
   `_last_solve_converged AND |q−q0|<VDD/4 AND |qb−qb0|<VDD/4` → the first clause
   is *always* `False` on force_ic → a guaranteed `0/8` **regardless of where NR
   actually landed**.
2. **No KCL-residual gate** (the top Goodhart risk): a pinned-node artifact could
   false-PASS once the stale flag was fixed.

The unconstrained re-solve itself lands at the **asymmetric inboard attractor**
(matching P0-A's STUCK-final), the SAME point the plan §1 calls the *documented
failure mode*: TSMC5 q≈0.70/qb≈0.16, TSMC7 q≈0.82/qb≈0.23, TSMC12
q≈0.866/qb≈0.192, TSMC16 q≈0.865/qb≈0.199. **This is model+solver-determined and
byte-identical to V6.4.4** (all 8 checkpoint sha256s match the frozen
`baseline_v6_4_4.json`; no retrain happened in V6.4.6). The stale-flag bug had
simply been *under-reporting* V6.4.4's own `force_ic` behavior as `0/8`.

---

## 2. Step 1 — harden the probe (SHIPPED)

**Solver (`solver.py`).** On the force_ic release path, after the unconstrained
re-solve, compute the *released* solution's KCL residual via the existing
`_dc_residual_at`, set the acceptance threshold with the solver's own pattern
`max(_RESID_ABS_FLOOR, 100·reltol·rhs_scale)`, and set
`_last_solve_converged = (finite voltages) AND (residual ≤ threshold)` — no
longer stale. Expose `_last_dc_residual`, `_last_dc_resid_threshold`,
`_last_residual_ok` (None on every non-force_ic solve, so a reader can tell the
value is live). Isolated to the `if _ic_temp_sources:` branch; all non-force_ic
paths return before it and are byte-identical.

**Test (`verify_complex_sram_snm.py`).** Acceptance is now a HARD residual gate
**AND** a tightened rail-proximity band:

```
resid_ok = (_last_dc_residual <= _last_dc_resid_threshold)   # a TRUE fixed point
rail_ok  = (|q−q0| < 0.1·VDD) and (|qb−qb0| < 0.1·VDD)        # held NEAR the rail
PASS     = resid_ok and rail_ok
```

**Band is `0.1·VDD`, NOT `VDD/4`** — see §4.1 for why `VDD/4` was a false-PASS.

### Step-1 honesty check (run BEFORE the homotopy)

With only Step 1 applied and the force_ic body still the buggy one-shot release
(temporary scaffold; reverted), every tech landed at the **symmetric metastable**
point q≈qb (0.596 / 0.677 / 0.730 / 0.730), all with `resid_ok=True,
rail_ok=False`:

```
TSMC5  state1: q=0.596 qb=0.596  resid=6.39e-5 thr=6.50e-3 (resid_ok=True rail_ok=False) -> FAIL
TSMC7  state1: q=0.677 qb=0.677  resid=1.01e-4 thr=7.50e-3 (resid_ok=True rail_ok=False) -> FAIL
TSMC12 state1: q=0.730 qb=0.730  resid=7.74e-5 thr=8.00e-3 (resid_ok=True rail_ok=False) -> FAIL
TSMC16 state1: q=0.730 qb=0.730  resid=7.44e-5 thr=8.00e-3 (resid_ok=True rail_ok=False) -> FAIL
```

**Verified: the residual gate alone does NOT false-PASS a non-railed point** —
the metastable/inboard points are real small-residual fixed points
(`resid_ok=True`), so it is the **rail-proximity gate** that must (and now does)
reject them. (Log: `phase1_logs/step1_baseline_buggy.log`.)

---

## 3. Step 2 — constraint-continuation homotopy (BUILT, KILLED)

**Design (Norton soft-pin, the plan's recommended approach).** Replace the
one-shot release with a continuation that tracks the railed branch P0-A proved
exists. Each IC node gets a soft pull toward its IC value `V_ic` with
conductance `g`: a `Resistor` of value `1/g` from node→ground (conductance `g`
on the diagonal) **plus** a `CurrentSource([node,"0"], g·V_ic)` (Norton current
injection `+g·V_ic`). This is the Norton equivalent of `V_ic` behind conductance
`g` and keeps `matrix_size` constant. Sweep `g` large→0 on a **fixed schedule**
(same for all techs, no per-tech tuning); full NR per stage, each warm-started by
the previous; first stage warm-started at the railed constrained solve;
`force_ic=False`, `use_source_stepping=False`, supplies at full VDD throughout;
`g=0` → both temp terms vanish → fully unconstrained. try/finally restores
`force_ic`/`use_source_stepping`/`initial_guess` and removes all temp pins.

**Result — slide-off, identical 0/8 to the one-shot release.** On a 41-point
dense geometric schedule `geomspace(1.0, 1e-9, 40) + [0.0]`, the homotopy tracks
the railed branch down to **g*≈1e-5 S**, then the branch undergoes a **fold /
turning point** and the continuation slides into the symmetric metastable point
(q≈qb), 0/8. Per-stage trace (TSMC5 state1):

| g (S)  | q | qb | g (S)  | q | qb |
|-------:|---:|---:|-------:|---:|---:|
| 1e+0   | 0.650 | 0.000 | 1e-4   | 0.650 | 0.155 |
| 1e-2   | 0.650 | 0.006 | 7e-5   | 0.650 | 0.189 |
| 1e-3   | 0.650 | 0.057 | 4e-5   | 0.650 | 0.230 |
| 6e-4   | 0.650 | 0.078 | 2e-5   | 0.650 | 0.279 |
| 3e-4   | 0.650 | 0.101 | **1e-5** | **0.649** | **0.344** |
| 2e-4   | 0.650 | 0.126 | **8e-6** | **0.622** | **0.538**  ← fold |
|        |       |       | 0      | 0.596 | 0.596 |

**Why the branch folds — the railed point is NR-unstable.** P0-A measured the
railed point's *residual* `‖b−A·x‖∞ ≈ 8.5e-5` (≪ threshold) — it IS a fixed
point of the residual function. But it is **unstable under the full re-stamp NR
map** `x → A(x)⁻¹b(x)`. Independently reproduced in review: a single re-stamp
solve seeded **exactly** at the literal rail gives `qb: 0.0 → 0.159 V` in one
step (q stays 0.65). The OFF storage node has near-zero conductance to ground
(deep-subthreshold gds ≈ gmin), so the Newton step `Δqb = residual / g_qb`
*explodes* — the soft-pin's added conductance `g` is exactly what bounds it; as
`g→0` below g*≈1e-5, `g_qb` collapses and the iterate is thrown across the
separatrix. The **series-resistor fallback** is the Norton dual (R=1/g) and folds
at the same R*=1/g*, so it cannot rail the cell either.

**Kill outcome (plan §5).** The plan's literal kill criterion is a *conjunction*
("P0-A residual large on TSMC5 **AND** the continuation slides off"). P0-A found
the residual *small* (the fixed point exists), so the literal criterion did NOT
fire — only the slide-off did. This is a **stronger negative** than the plan
foresaw: *the continuation slides off the railed branch even though the railed
point provably exists*, because that point is NR-unstable, not absent. Neither a
finer schedule nor the (equivalent) series-R fallback helps. The homotopy was
**reverted**; only the Step-1 probe-hardening ships. The transient write-then-
hold re-spec was **NOT implemented** (held in reserve; see §7).

---

## 4. Results — Rule-16 table (final shipped solver + tightened band)

Released-solution KCL `‖b−A·x‖∞`; threshold `max(1e-6, 100·reltol·‖b‖∞)`. Band
`0.1·VDD`. (NFIN=2, the force_ic cell's fixed geometry.)

| Tech  | VDD  | State  | q landed | qb landed | residual | threshold | resid_ok | rail band | rail_ok | PASS |
|-------|-----:|--------|---------:|----------:|---------:|----------:|:--------:|----------:|:-------:|:----:|
| TSMC5 | 0.65 | state1 | 0.702 | 0.163 | 8.500e-05 | 6.500e-03 | yes | 0.065 | **no** | FAIL |
| TSMC5 | 0.65 | state0 | 0.163 | 0.702 | 8.500e-05 | 6.500e-03 | yes | 0.065 | **no** | FAIL |
| TSMC7 | 0.75 | state1 | 0.815 | 0.226 | 1.255e-04 | 7.500e-03 | yes | 0.075 | **no** | FAIL |
| TSMC7 | 0.75 | state0 | 0.226 | 0.815 | 1.255e-04 | 7.500e-03 | yes | 0.075 | **no** | FAIL |
| TSMC12| 0.80 | state1 | 0.866 | 0.192 | 1.005e-04 | 8.000e-03 | yes | 0.080 | **no** | FAIL |
| TSMC12| 0.80 | state0 | 0.192 | 0.866 | 1.005e-04 | 8.000e-03 | yes | 0.080 | **no** | FAIL |
| TSMC16| 0.80 | state1 | 0.865 | 0.199 | 9.690e-05 | 8.000e-03 | yes | 0.080 | **no** | FAIL |
| TSMC16| 0.80 | state0 | 0.199 | 0.865 | 9.690e-05 | 8.000e-03 | yes | 0.080 | **no** | FAIL |

**Genuine `0/8`.** Every state is a true small-residual fixed point
(`resid_ok=True`), but all 8 land in the inboard attractor — the storage-"0" node
parks at 0.16–0.23 V (24–30 % VDD) on every tech → all fail the `0.1·VDD` rail
band. This **confirms D3** (the inboard point is a true NN attractor) rather than
closing it.

### 4.1 Why `VDD/4` was a false-PASS (the retracted 4/8)

The intermediate gate file used the pre-existing `VDD/4` band and reported
TSMC12/16 as PASS. Adversarial review showed this is a **VDD-scaling artifact**,
not retention:

- The inboard attractor sits at qb ≈ 24–30 % VDD on every tech. `VDD/4` = 25 %
  VDD straddles it: for the 0.80 V techs qb≈0.19 < 0.20 sneaks *inside* the band;
  for the 0.65/0.75 V techs the same physical attractor falls *outside* the
  smaller band. **TSMC5 rails CLOSER to ground in absolute volts (qb=0.163) than
  the "passing" TSMC12 (qb=0.192), yet TSMC5 "failed".** The PASS/FAIL split was
  set purely by VDD-scaling the band, not by any tech genuinely railing.
- The released NN read-disturbed half-cell trips at **0.60/0.68/0.73/0.73 V** vs
  NGSPICE BSIM-CMG's physical **0.33/0.39/0.42/0.41 V**, and the inboard qb
  (0.16–0.23 V) is comparable to the *entire* true SNM (0.19–0.24 V). A "0" with
  ~1× SNM margin above ground is not a retained logic 0.

The `0.1·VDD` band sits well below the 24–30 % attractor on **every** tech, so it
reports the honest `0/8` while still accepting a genuinely railed solution if a
future model/solver produces one.

---

## 5. Blind NFIN corner (overfit guard)

The force_ic cell is fixed at `bt.nfin=2`; rebuilt at **NFIN=3** via
`dataclasses.replace(BENCH[tech], nfin=3)` (`phase1_logs/blind_nfin3_cell.log`).
Same inboard attractor; same `0/8` under the `0.1·VDD` band:

| Tech  | State  | q | qb | rail band | PASS |
|-------|--------|---:|---:|----------:|:----:|
| TSMC7 | state1 | 0.822 | 0.215 | 0.075 | FAIL |
| TSMC12| state1 | 0.876 | 0.180 | 0.080 | FAIL |
| TSMC16| state1 | 0.874 | 0.180 | 0.080 | FAIL |

(state0 mirrors.) The result is **structural** (the inboard attractor, not the
NFIN corner), confirming nothing is overfit — there is no homotopy schedule in
the shipped code to overfit.

---

## 6. Regression confirmation

- **Inverter gate 8/8 PASS** (`phase1_logs/inverter_regression.log`):
  VTC NRMSE TSMC5/7/12/16 = 1.21 / 2.37 / 2.05 / 1.33 %; inverter-tran
  = 1.62 / 1.09 / 1.41 / 1.45 %. Byte-identical to V6.4.4 — the change is
  isolated to the `if _ic_temp_sources:` branch (inverter runs `force_ic=False`).
- **Butterfly lobes 4/4 positive** (all four techs, Phase-1 full run) —
  **unchanged by the band by construction**: the `0.1·VDD` band is a one-line
  threshold *inside* `force_ic_probe`; the butterfly pass count is computed
  independently in `directnet_lobe`/`run_one` (a DC sweep, never `force_ic`), so
  it cannot be affected. The honest `0/8` `force_ic` was directly re-confirmed
  under the tightened band (`phase1_logs/`, and a standalone `force_ic_probe`
  check: TSMC5/7/12/16 all `rail_ok=False`).
- **Code diff confined to two files:** `pycircuitsim/solver.py`,
  `tests/verify_complex_sram_snm.py`. No checkpoint touched.

---

## 7. Decision / outcomes

- **SHIP Step 1 (probe-hardening).** It fixes a real correctness bug (stale flag)
  + closes the top Goodhart risk (KCL gate) + tightens the rail band so the
  documented inboard attractor can no longer false-PASS. The HONEST `force_ic` is
  **0/8** on all 4 techs.
- **KILL Step 2 (homotopy).** The constraint-continuation homotopy slides off the
  railed branch at a fold near g*≈1e-5 S on all 4 techs (0/8), because the railed
  point, though a residual fixed point (P0-A), is **NR-unstable**. Reverted;
  recorded as a dead end.
- **0 SRAM gates closed → headline 9/16 (unchanged).** Per the lead's decision,
  V6.4.6 ships as an honest no-behavioral-change iteration: a corrected probe +
  the homotopy dead-end record.
- **NOT implemented (per instruction):** the transient write-then-hold re-spec —
  low EV (the inboard point is a *stable* basin, so a transient hold would most
  likely also collapse to it; it mostly re-frames the gate). Held in reserve.

**P0-A refined, not contradicted.** A railed *residual* fixed point exists; what
does **not** exist is a railed point *stable under the unconstrained NR
iteration*, which is what a DC continuation needs. Closing TSMC5/7/12/16
`force_ic` needs either the transient re-spec or an architectural off-leakage
change (Phase 3, caveated by P0-D), not a solver continuation.

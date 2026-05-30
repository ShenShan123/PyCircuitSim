# Phase 2 — Zero-code solver probes

**Date:** 2026-05-28  •  **Branch:** `feat/v6.4.4` (HEAD `801ac6e`)  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Probe targets:** TSMC7 ring oscillator (≤ 5 % gate), SRAM `force_ic` rail-snap (4/4 gate).

## Headline

> **After Phase 2, complex-circuit pass rate moves from 9/16 → 9/16.**
> All three probes either KILLed by their stated criterion or DIAGNOSTIC-ONLY.
> No code shipped from Phase 2. Phase 4 (Rule-15 `Ioff_rail`) is now MANDATORY
> per Probe 3 finding: SRAM q ≈ 0.16–0.20 is a true NN attractor, not a poor
> warm start.

---

## Probe 1 — `NN_SYMMETRIC_CAPS=1` on RO + SC (zero code)

### Commands

```
NN_SYMMETRIC_CAPS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python tests/verify_complex_ring_osc.py \
  > results/v6_4_5/phase2_logs/ro_symcaps.log 2>&1

NN_SYMMETRIC_CAPS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python tests/verify_complex_switchcap.py \
  > results/v6_4_5/phase2_logs/sc_symcaps.log 2>&1
```

### Ring oscillator — symcaps vs Phase 1 baseline

| Tech   | Baseline PerErr % | symcaps PerErr % | Δ      | Status |
|--------|------------------:|-----------------:|-------:|:------:|
| TSMC5  |              2.98 |             2.98 |  0.00  | PASS   |
| TSMC7  |          **8.97** |         **8.97** |  0.00  | FAIL   |
| TSMC12 |              3.01 |             3.01 |  0.00  | PASS   |
| TSMC16 |              2.88 |             2.88 |  0.00  | PASS   |

NRMSE / R² also bitwise identical (TSMC5 25.65/0.6402, TSMC7 54.37/−0.6525,
TSMC12 33.83/0.3404, TSMC16 29.49/0.4917). The symmetric-cap stamping does
not perturb RO at all on these checkpoints — the cap-asymmetry hypothesis
from the V6.4.4 iter-2 diagnosis is **not** what is driving TSMC7's 8.97 %
period drift.

### Switched-cap — symcaps vs Phase 1 baseline

| Tech   | Baseline ChgErr % | symcaps ChgErr % | Status (both) |
|--------|------------------:|-----------------:|:-------------:|
| TSMC5  |             14.68 |            13.75 | FAIL → FAIL   |
| TSMC7  |              3.06 |             0.42 | PASS → PASS   |
| TSMC12 |              8.33 |             5.88 | FAIL → FAIL   |
| TSMC16 |             13.13 |             7.65 | FAIL → FAIL   |

SC charge error improves on every tech (TSMC7 0.42 % is a clear win), but no
new cell crosses the gate; pass count stays 1/4. NRMSE also drops on TSMC5/12/16.

### Kill criterion

> *"Drop (1) if it does not move TSMC7 RO closer than 7 % OR if it regresses
> any other RO/SC cell."*

TSMC7 RO unchanged at **8.97 % > 7 %**. **KILL.**

### Verdict — **KILL**

- Flag stays default OFF (no code change).
- SC numerics improve under symcaps but cross no gate; not worth the asymmetric
  RO non-effect.
- TSMC7 RO drift is **not** cap-asymmetry — diagnosis confirmed. Track B's B1
  cap-asymmetry probe is consistent with this null result; if its δ histogram
  is uniformly small, RO is model-fidelity (Phase 5 retrain territory).

### Kept / reverted

Nothing to revert (env-var only). No file change.

---

## Probe 2 — `max_substeps=4` for ring oscillator only

### Code edits applied (before measurement, reverted after kill)

1. `tests/common/complex.py` — `run_directnet_transient` gained an optional
   `max_substeps: int = 1` kwarg, threaded through to `TransientSolver(...)`.
   Default 1 keeps every other caller bitwise identical.
2. `tests/verify_complex_ring_osc.py::run_directnet_ro` — read `RO_MAX_SUBSTEPS`
   env var (default `"1"`), pass `max_substeps=int(...)` through.

### Command

```
RO_MAX_SUBSTEPS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python tests/verify_complex_ring_osc.py \
  > results/v6_4_5/phase2_logs/ro_substeps4.log 2>&1
```

Wall time: 22 min (Phase 1 baseline ~11 min) — ~2× cost, well under the 4×
allowance in the kill criterion.

### Ring oscillator — substeps=4 vs Phase 1 baseline

| Tech   | Baseline PerErr % | substeps4 PerErr % | Δ        | Status |
|--------|------------------:|-------------------:|---------:|:------:|
| TSMC5  |              2.98 |               2.62 |  −0.36   | PASS   |
| TSMC7  |          **8.97** |           **8.04** |  −0.93   | FAIL   |
| TSMC12 |              3.01 |               2.57 |  −0.44   | PASS   |
| TSMC16 |              2.88 |               2.50 |  −0.38   | PASS   |

NRMSE worsened on TSMC7/12/16 (54.37→57.11, 33.83→39.25, 29.49→33.73) even
though the period error dropped — LTE sub-stepping smooths the trip
trajectory enough to shave 0.4–0.9 % off the period drift but doesn't fix
the underlying TSMC7 cap-shape error along the trip. TSMC5/12/16 remain PASS.

### Kill criterion

> *"Drop (2) if it does not move TSMC7 RO to ≤ 5 % at ≤ 4× wall time.
> (Period stability ≠ LTE → revert.)"*

TSMC7 went 8.97 % → 8.04 %, **> 5 %**. **KILL.**

### Verdict — **KILL**

Wall-cost is within budget (2× < 4×) but the headline gate is not closed.
Per the kill criterion the harness edits are reverted; the env var stays
unshipped. Period stability is not LTE-dominated on TSMC7.

### Kept / reverted

```
git checkout -- tests/common/complex.py tests/verify_complex_ring_osc.py
```

Confirmed clean (see `git status` block at bottom). No inverter smoke test
was needed: the probe failed its own gate before reaching the regression
check.

---

## Probe 3 — SRAM butterfly-lobe warm-start (diagnostic-only)

### Code edit applied (then reverted)

`tests/verify_complex_sram_snm.py::force_ic_probe` — call `ngspice_lobe`
once per tech at `bt.nfin`, interpolate `qb` at `q=VDD` to get
`near_zero = float(np.interp(bt.vdd, lobe["q"], lobe["qb"]))`, seed
state-1 with `(VDD, near_zero)` and state-0 mirror `(near_zero, VDD)`.
Convergence threshold `bt.vdd/4` unchanged. Added diagnostic print of the
seed used.

### Command

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python tests/verify_complex_sram_snm.py \
  > results/v6_4_5/phase2_logs/sram_warmstart.log 2>&1
```

Wall time: ~23 min (baseline ~12 min) — extra `ngspice_lobe` per tech.

### `force_ic` results — warm-start seeds vs converged solution

| Tech   | `near_zero` seed (mV) | state1 (q, qb) — seed (VDD, near_zero) | state0 (qb, q) — seed (near_zero, VDD) | state1 | state0 |
|--------|----------------------:|----------------------------------------:|----------------------------------------:|:------:|:------:|
| TSMC5  |                  83.2 |                          (0.702, 0.163) |                          (0.163, 0.702) |  FAIL  |  FAIL  |
| TSMC7  |                 122.7 |                          (0.804, 0.243) |                          (0.243, 0.804) |  FAIL  |  FAIL  |
| TSMC12 |                 112.7 |                          (0.797, 0.448) |                          (0.448, 0.797) |  FAIL  |  FAIL  |
| TSMC16 |                 114.4 |                          (0.795, 0.471) |                          (0.471, 0.795) |  FAIL  |  FAIL  |

Comparison vs Phase 1 baseline (rail seeds `(VDD, 0)` / `(0, VDD)`):

| Tech   | Phase 1 state1 (q, qb) | Phase 2 state1 (q, qb) |
|--------|------------------------|------------------------|
| TSMC5  | (0.873, 0.196)         | (0.702, 0.163)         |
| TSMC7  | (0.867, 0.200)         | (0.804, 0.243)         |
| TSMC12 | (0.866, 0.192)         | (0.797, 0.448)         |
| TSMC16 | (0.865, 0.199)         | (0.795, 0.471)         |

Butterfly 4/4 unchanged: all 12 NFIN corners still positive across techs
(min(qb) ≥ 428 mV).

### Verdict — **DIAGNOSTIC, q ≈ 0.16–0.20 IS A TRUE NN ATTRACTOR**

Rail-seed and lobe-seed (123 mV closer to the rail) both converge to a
non-rail fixed point. The basin of attraction of the q ≈ 0.16–0.20
equilibrium is wide enough to swallow seeds at `(VDD, 83–123 mV)`. This
is not a poor warm start — the cross-coupled NN inverter pair has a
genuine inboard DC equilibrium because off-leak Id at `(Vgs=0, Vds=VDD)`
is under-modelled.

**Phase 4 priority: MANDATORY.** Per the V6.4.5 plan §Phase 2:
> *"if butterfly warm-start lands on rails, escalate Phase 4 priority; if
> it still lands on q ≈ 0.16, Phase 4 (Rule-15 Ioff_rail) becomes mandatory."*

### Kept / reverted

```
git checkout -- tests/verify_complex_sram_snm.py
```

**Reverted.** Rationale:
- 4/4 force_ic still FAIL — the probe edit changes nothing about the SRAM
  pass count.
- It nearly doubled the SRAM-test wall time (12 → 23 min) for no gate gain.
- Phase 4 work changes the NN's off-state Id, not the seed; the diagnostic
  is preserved in this gate file's tables.
- Butterfly 4/4 is preserved by the revert (the edit was scoped to
  `force_ic_probe` only — butterfly path was untouched).

---

## Decisions block

| Probe | Status        | Action                                                                                  |
|------:|:--------------|:----------------------------------------------------------------------------------------|
| 1     | KILL          | No code change. `NN_SYMMETRIC_CAPS` stays default OFF. SC improvement noted, not shipped. |
| 2     | KILL          | Harness edits reverted (`git checkout -- tests/common/complex.py tests/verify_complex_ring_osc.py`). |
| 3     | DIAGNOSTIC    | Edit reverted (`git checkout -- tests/verify_complex_sram_snm.py`). Phase 4 escalated to MANDATORY. |

### Post-revert `git status`

```
On branch feat/v6.4.4
Your branch is up to date with 'origin/feat/v6.4.4'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	docs/plans/2026-05-28-directnet-v6.4.5-ro-sram.md

nothing added to commit but untracked files present (use "git add" to track)
```

`git diff --stat` is empty (no tracked file modifications). The untracked
plan doc and the Phase 2 logs / this gate file are the only Phase 2
artifacts; per the plan, Phase 2 is **zero shipped code**.

---

## What Phase 2 settled

1. **TSMC7 RO drift is NOT cap-asymmetry.** `NN_SYMMETRIC_CAPS=1` has zero
   effect on RO numbers; Track B's B1 cap-asymmetry probe would corroborate
   this with a δ-histogram. Phase 4 / Phase 5 must look at model-fidelity
   levers (Id/gds at the trip and along the cycle).
2. **TSMC7 RO drift is NOT LTE.** `max_substeps=4` shaves 0.9 % off the
   period error (8.97 → 8.04 %) but cannot close to ≤ 5 %; periodic
   integration error is not the main contributor. Phase 5 retrain remains
   the candidate cure.
3. **SRAM `force_ic` q ≈ 0.16–0.20 is a true NN attractor**, not a poor
   warm start. Phase 4 (Rule-15 `Ioff_rail`) is mandatory; if Phase 4
   doesn't close ≥ 1/4 SRAM cells, only the Phase 6 split-head retrain
   (V6.4.6) can fix it.

## Headline (counted)

| Sub-result      | Phase 1 baseline | Phase 2 net (after reverts) |
|-----------------|-----------------:|----------------------------:|
| RO              |              3/4 |                         3/4 |
| Opamp           |              1/4 |                         1/4 |
| SRAM butterfly  |              4/4 |                         4/4 |
| SRAM force_ic   |              0/8 |              not counted in 16 |
| Switched-cap    |              1/4 |                         1/4 |
| **Total /16**   |          **9/16** |                    **9/16** |

(`force_ic` is the diagnostic probe inside the SRAM cell, not a separate
gate in the 16-cell count.)

Inverter 8/8 and extended harness (55 / 64) NOT re-verified in Phase 2 —
no shipped code change can perturb them.

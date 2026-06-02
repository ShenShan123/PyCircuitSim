# Phase 0 — Diagnostic P0-G: RO time-integration study (TSMC7)

**Date:** 2026-06-02  •  **Branch:** `feat/v6.4.6`  •  **Status:** DONE — RO is model-owned; no shippable integrator/tstep fix.
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Plan:** `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md` §2 (P0-C correction), §4 P0-C/P0-B, §11 risk-1, §12 Q2.
**Driver:** `scripts/v6_4_6_p0g_integration_study.py`
**Logs:** `results/v6_4_6/phase0_logs/p0g_step0_probe_tsmc7.log`,
`p0g_r1_integrator_select.log`, `p0g_r2_convergence.log` (Trap),
`p0g_r3_be_convergence.log` (BE).

INSTRUMENTATION-ONLY. The genuinely unexplored 0-GPU axis P0-C left open: the
**BE/Trap/BDF-2 time-integration truncation**. P0-C proved gds/caps (the NR
Jacobian surfaces) are causally inert on the RO period (swap → ≤0.01 ps). This
study isolates the integration side. The solver probe (`RO_INTEG_PROBE`) and
forced-integrator (`RO_INTEG_FORCE`) hooks are env-gated and **reverted after
the study** (precedent: the reverted `P0A_RESIDUAL`); `git diff` over
`pycircuitsim/` is empty at hand-off.

Ground truth is ALWAYS the OSDI/NGSPICE BSIM-CMG period (CLAUDE.md Validation
rule): **NG = 46.64 ps**, gate ≤5% ⇒ DirectNet must reach **≤48.97 ps**.
Baseline DN = **50.82 ps** (8.97% FAIL; matches P0-C/P0-F exactly).

---

## Step 0 — does the BDF-2 stiffness auto-switch ever fire? **NO.**

The hypothesis under test: the solver does BE(step1)→Trap(step2+)→**BDF-2
auto-switch on stiffness** (NR>20 iters, one-way `_stiff_switched`,
`solver.py:2046-2052` / `:2189-2194`). BDF-2 is numerically dissipative;
excess numerical damping *lengthens* an oscillator period — consistent with
DN (50.82) > NG (46.64). If the switch fires during the TSMC7 RO, forcing pure
Trapezoidal should shorten the period toward NG.

**Result (probe over the full 601-step baseline TSMC7 RO transient):**

```
[RO_INTEG_PROBE] transient start: num_steps=601 dt=2.000e-12 t_stop=1.200e-09 ...
```
…and **zero** `[RO_INTEG_PROBE] BDF-2 stiffness switch FIRED` lines for the
entire run (`grep -c FIRED → 0`). Baseline period 50.82 ps, 8.97% FAIL.

**⇒ BDF-2 NEVER fires.** The NR iteration count never exceeds 20 on any step,
so `_stiff_switched` stays `False` and the TSMC7 RO transient runs **pure
Trapezoidal** (BE on step 1 only, Trap for steps 2–600). The
"over-damping-via-BDF-2-switch lengthens the period" hypothesis is therefore
**moot for the baseline period** — the baseline is *already* the
non-dissipative integrator. R1 reduces to the sensitivity check (force BE-only
/ BDF-2-only), exactly as the plan's Step-0 contingency specifies.

---

## R1 — integrator selection (base tstep 2 ps)

`RO_INTEG_FORCE` pins one integrator for every step >1 (step 1 stays BE for
self-start). `force trap` doubles as an instrumentation sanity check: it must
reproduce the baseline 50.82 ps bit-for-bit (the baseline already runs Trap).

| variant | period (ps) | %err vs NG | NRMSE % | R² | verdict |
|---------|------------:|-----------:|--------:|----:|:-------:|
| baseline (BE1→Trap→BDF-2 auto) | 50.82 | 8.97 | 54.37 | −0.6525 | FAIL |
| **force Trapezoidal**          | 50.82 | 8.97 | 54.37 | −0.6525 | FAIL |
| **force Backward-Euler**       | **45.14** | **3.22** | **40.38** | **+0.0884** | **PASS** |
| **force BDF-2**                | 51.49 | 10.39 | 56.52 | −0.7855 | FAIL |

**Two findings, one surprising.**

1. **`force Trapezoidal` == baseline bit-for-bit** (50.82 ps, NRMSE 54.37 %,
   R² −0.6525 — identical to the printed precision). This *proves* the Step-0
   finding causally: the baseline TSMC7 RO is already pure Trapezoidal, and the
   forced-integrator instrumentation is faithful (forcing the integrator the
   baseline already uses reproduces it exactly).

2. **The dissipation ordering holds — but the textbook hypothesis is inverted
   for closing the gate.** Period rises monotonically with numerical
   dissipation: **BE 45.14 < Trap 50.82 < BDF-2 51.49 ps**. The plan's premise
   (BDF-2 over-damping *lengthens* the period; pure Trap should *shorten* it
   toward NG) is *directionally correct* (BDF-2 > Trap), but Trap is **already**
   the active integrator and sits 4.18 ps **above** NG — so removing damping
   cannot help. Instead the *most*-dissipative, *lowest-order* integrator
   (**Backward-Euler**) drags the period DOWN through NG to 45.14 ps and
   **PASSES** (3.22 %), and even the waveform shape improves (NRMSE 54.4→40.4 %,
   R² flips negative→positive). BDF-2 is slightly worse than Trap (51.49 ps).

**This makes integrator selection a *live* lever — not the moot sensitivity
check Step-0 implied.** But BE passing at the coarse 2 ps step raises the
decisive question the convergence study (R3) must answer: is BE's 45.14 ps a
genuine integrator effect, or a fortuitous cancellation of BE's large O(h)
truncation error against the model's id/charge VALUE error that would unwind as
tstep→0?

## R2 — uniform-tstep convergence study (baseline = Trapezoidal)

The base-tstep DirectNet RO runner formats the .tran step as
`f"{tstep*1e12:.0f}p"`, truncating any sub-ps step to `0p` (dt=0 → ValueError).
The driver bypasses that by writing the .tran line in seconds (scientific
notation) so the study can refine below 1 ps; the base-step result is
bit-identical to the gate (50.82 ps), confirming the bypass is faithful.

| tstep | period (ps) | %err vs NG | verdict | Δ vs coarser |
|-------|------------:|-----------:|:-------:|-------------:|
| base/1 = 2.000 ps   | 50.82 | 8.97 | FAIL | — |
| base/2 = 1.000 ps   | 50.49 | 8.25 | FAIL | −0.33 |
| base/4 = 0.500 ps   | 50.39 | 8.05 | FAIL | −0.10 |
| base/8 = 0.250 ps   | 50.37 | 8.00 | FAIL | −0.02 |
| **Richardson o2 (500/250 fs)** | **50.36** | **7.98** | FAIL | — |

Trapezoidal deltas (−0.33, −0.10, −0.02) shrink ~3.3–5× per halving — clean
order-2 convergence to a PLATEAU. **Richardson (order-2) ⇒ P0 ≈ 50.36 ps
(7.98 %)**, ~3.7 ps above the 48.97 ps gate. Under the baseline integrator a
finer fixed tstep does NOT close the gate — the residual is **model-owned**, not
truncation.

## R3 — BE convergence study (does BE's base-step pass survive tstep→0?)

R1 found `force be` PASSES at the base step (45.14 ps). BE is order-1 (O(h)):
if its pass is a truncation artifact, the period must climb back toward the
Trap plateau as tstep→0.

| tstep | period (ps) | %err vs NG | NRMSE % | verdict | Δ vs coarser |
|-------|------------:|-----------:|--------:|:-------:|-------------:|
| base/1 = 2.000 ps   | 45.14 | 3.22 | 40.38 | **PASS** | — |
| base/2 = 1.000 ps   | 47.94 | 2.78 | 45.54 | **PASS** | +2.80 |
| base/4 = 0.500 ps   | 49.20 | 5.48 | 58.42 | **FAIL** | +1.26 |
| base/8 = 0.250 ps   | 49.79 | 6.76 | 56.19 | **FAIL** | +0.59 |
| **Richardson o1 (500/250 fs)** | **50.39** | **8.03** | — | FAIL | — |

**BE rises monotonically toward the Trap plateau and CROSSES BACK above the
gate at 500 fs (49.20 ps, FAIL), continuing to 49.79 ps at 250 fs and
Richardson-extrapolating to ≈ 50.39 ps (8.03 %) — the SAME ~50.4 ps limit as
Trap.** BE's base-step pass (45.14 ps) is a fortuitous cancellation of its
large O(h) truncation error (shortens the period) against the model's id/charge
VALUE error (lengthens it). Both integrators are consistent discretisations of
the same ODE, so both converge to the same model-owned limit; BE merely passes
*through* the gate window at coarse steps (2 ps & 1 ps). The waveform NRMSE
WORSENS with refinement (40→46→58 %) — the better period at coarse BE is not a
better trajectory.

---

## Interpretation & DECISION

**The TSMC7 RO period is MODEL-OWNED, not solver/integration-owned. No
integrator or tstep choice closes it ≤5 % in a defensible way.**

The two consistent integrators both converge, as tstep→0, to the same limit:

| integrator | tstep→0 limit (Richardson) | %err vs NG | gate ≤5 %? |
|------------|---------------------------:|-----------:|:----------:|
| Trapezoidal (baseline) | ≈ 50.36 ps | 7.98 % | NO |
| Backward-Euler         | ≈ 50.39 ps | 8.03 % | NO |

That common ≈ **50.4 ps** continuum limit is the **true** DirectNet TSMC7 RO
period for this checkpoint — ~3.7 ps (≈ 8 %) above NG 46.64 ps, far outside the
48.97 ps gate. It is set by the **id-VALUE + charge-VALUE (qg/qd) trajectories
the companion model stamps** (`_stamp_mosfet_dc:304`, `_stamp_mosfet_transient:
1772`), exactly the P0-C bucket — confirming P0-C/§11-risk-1 from the
*complementary* (integration) side: even with truncation driven to zero, the
period stays at ~50.4 ps. (P0-H, the id/charge-VALUE overlay, completed
concurrently and localised this ~3.7 ps residual to the **NMOS dynamic `id`
VALUE** — charges are exact; see the reconciliation note in the Verdict and
`results/v6_4_6/phase0H_ro_value_overlay.md`.)

**Chain of evidence:**
1. **Step-0:** BDF-2 stiffness auto-switch NEVER fires on the TSMC7 RO
   (`grep -c FIRED = 0`). The baseline is already pure Trapezoidal — the
   "BDF-2 over-damping lengthens the period" hypothesis is moot for the period.
2. **R1:** `force Trap` == baseline bit-for-bit (50.82 ps) → instrumentation
   faithful. Dissipation ordering BE 45.14 < Trap 50.82 < BDF-2 51.49 holds, but
   Trap (the non-dissipative, default integrator) is already 4 ps ABOVE NG, so
   removing damping cannot help; only the lowest-order BE drops *through* NG.
3. **R2 (Trap convergence):** clean order-2 plateau at ≈ 50.36 ps; finer tstep
   does NOT close the gate under the default integrator.
4. **R3 (BE convergence):** BE's base-step pass (45.14 ps, 3.22 %) is a
   coarse-grid O(h) truncation artifact — it climbs 45.14→47.94→49.20→49.79 ps
   and crosses back above the gate at 500 fs, extrapolating to the same
   ≈ 50.4 ps model limit. BE's better period at coarse steps is NOT a better
   trajectory (waveform NRMSE rises 40→58 %).

This is consistent with the recorded RO dead ends and does NOT re-tread them:
D1 (caps, bit-for-bit null), D2 (LTE adaptive sub-stepping — a different axis),
P0-C (gds/cap Jacobian causally inert). This study closes the LAST untested
axis (time-integration *selection* + uniform-step *truncation*) and finds it
also inert: the gap is in the model's dynamic id/charge VALUES.

## Ship test

**Not warranted — there is no shippable candidate.** The task's ship-test
condition was: *if forcing Trapezoidal AND/OR a finer tstep brings TSMC7 RO
≤ 48.97 ps.* Both resolve NO:
- **Forcing Trapezoidal** → 50.82 ps (= baseline; Trap is already active). FAIL.
- **Finer fixed tstep (under the default Trap)** → 50.49 / 50.39 / 50.37 ps,
  plateau 50.36 ps. FAIL at every step.

The only configuration that passed the gate — `force Backward-Euler` at the
coarse 2 ps / 1 ps step — is **disqualified by its own convergence study (R3)**
as a truncation artifact, not a finer-step or better-integrator fix. Shipping it
would mean (a) swapping the whole solver to a **lower-order, more dissipative**
integrator globally (regressing transient fidelity on every NN and BSIM-CMG
circuit — the BSIM-CMG verify suites are byte-locked to the current BE→Trap→
BDF-2 cascade), AND (b) relying on the current coarse 2 ps default, since BE
itself FAILS the RO gate at ≤500 fs. Both are unsound. Running the full
regression surface (other 3 RO techs, inverter 8/8, switchcap TSMC7) is
therefore not justified — there is no candidate to regression-clear.

**Verdict.** RO stays open and **model-owned**, exactly as P0-C concluded.
V6.4.6 already shipped at **9/16** (Phase 1 found SRAM `force_ic` is 0/8 honest —
the probe-hardening was a measurement fix, not a gate-close), and the TSMC7 RO
gate is deferred to V6.4.7 as a **model-owned `id`-VALUE** problem (NOT a
solver-integration or Jacobian-distillation lever — both now empirically dead).

> **Reconciled with P0-H (ran concurrently).** The closing note below originally
> said P0-H "produced no numbers" — that was the stale log from before P0-H
> completed. **P0-H DID complete** and localised the ~3.7 ps continuum residual:
> the **charge VALUES (qg/qd/qs) are exact** (≤1.2 % NRMSE, ≤2 aC ≈ ≤2 % of the
> per-stage swing) → no charge-value lever; the **`id` VALUE** carries the
> residual (NMOS on-state NRMSE 9.6 %, ~20 % peak pull-down under-prediction,
> direction-consistent with the longer DN period). So the ~3.7 ps model residual
> this study isolated is owned by the **NMOS dynamic `id` VALUE**. The cheapest
> next 0-GPU step is the **P0-I causal id-VALUE swap** (inject exact OSDI `id`
> into the live RO, re-measure the period — the P0-C analogue) to confirm
> magnitude before funding any V6.4.7 id-value distillation.

---

## Reproduce

```bash
# Step-0 probe (BDF-2-fires check):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RO_INTEG_PROBE=1 \
  conda run -n pycircuitsim python -u tests/verify_complex_ring_osc.py --tech TSMC7
# R1 integrator selection + R2 Trap convergence:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python -u scripts/v6_4_6_p0g_integration_study.py --mode r1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python -u scripts/v6_4_6_p0g_integration_study.py --mode r2 --max-refine 4
# R3 BE convergence:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python -u scripts/v6_4_6_p0g_integration_study.py --mode r2 --force be --max-refine 4
```

Driver: `scripts/v6_4_6_p0g_integration_study.py` (deliverable). Solver hooks
`RO_INTEG_PROBE` / `RO_INTEG_FORCE` are env-gated instrumentation, **reverted**
after the study (`git diff` over `pycircuitsim/` empty at hand-off).

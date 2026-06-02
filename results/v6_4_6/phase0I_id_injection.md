# Phase 0 — Diagnostic P0-I: causal OSDI-`id`-VALUE injection (TSMC7 RO)

**Date:** 2026-06-02/03  •  **Branch:** `feat/v6.4.6`  •  **Status:** DONE — **INCONCLUSIVE as a clean id-isolation test + a V6.4.7 caution.** Injecting the exact OSDI `id` (NMOS-only AND symmetric N+P) produces a genuine, full-rail, uniform **~92 ps** oscillation — ~2× the 50.83 ps baseline and *further* from NG 46.64 ps. The `id` VALUE is **not separable** from the NN charge model the way the Jacobian is (P0-C). The planned V6.4.7 **id-VALUE-only LoRA is no longer de-risked.**
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Plan:** `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md` §2 (P0-C correction), §11 risk-1, §12 Q2 (the P0-I open question).
**Scripts:** `scripts/v6_4_6_p0i_id_injection.py` (v1, divergent), `scripts/v6_4_6_p0i_id_injection_v2.py` (v2, converging).
**Logs:** `results/v6_4_6/phase0_logs/p0i_id_injection.log` (v1), `p0i_v2_{analytic_smoke,np_full,waveform_diag}.log` (v2).

INSTRUMENTATION-ONLY — no shipped-behaviour change, no retrain, no checkpoint
mutation. The `_MOSFETNNBase._eval` monkeypatch is in-process and ALWAYS restored
(`run_variant` `finally`), so `git diff` over `pycircuitsim/` stays empty at
hand-off (verified). Ground truth is ALWAYS the OSDI binary via PyCMG / NGSPICE
BSIM-CMG (CLAUDE.md Validation rule): **NG = 46.64 ps**, gate ≤5% ⇒ DirectNet
must reach **≤48.97 ps**. Baseline DN = **50.82 ps** (8.97% FAIL; reproduced here
to **50.83 ps**, +0.01 ps — instrumentation faithful).

P0-I is the intended **causal** 0-GPU test for the open TSMC7 RO gate — the
analogue of P0-C, but for the `id` VALUE instead of the gds/cap Jacobian surfaces.
It is the gate that was to de-risk (or kill) the V6.4.7 id-VALUE LoRA lever before
any GPU.

## 1. Why the id VALUE is not a P0-C-style inert swap (mechanism)

P0-C swapped the exact OSDI **gds and cap-derivatives** into the live RO and the
period moved **≤0.01 ps** — those surfaces are **Jacobian-only**: they enter the
NR matrix *and* a matching RHS offset and **cancel at the converged fixed point**
(`_stamp_mosfet_dc:304`, `_stamp_mosfet_transient:1718-1782`). The **`id` VALUE
does NOT cancel**: the transient stamps the resistive companion current directly
from it (`i_eq = i_leaving − g_ds·v_ds − g_m·v_gs − …`). Changing the `id` VALUE
changes the converged solution and the whole trajectory — so it **must** move the
period if the id-VALUE owns the gap. Because the trajectory moves under injection,
OSDI `id` must be evaluated **live, at the evolving bias** inside the NR loop (it
cannot be pre-tabulated against the baseline trajectory the way P0-C's swap was).

This is exactly what makes P0-I far harder than P0-C — and, as it turned out,
numerically pathological (§3–§4).

## 2. The injection, and the three consistency schemes built to make it converge

The patch replaces, for the selected RO devices, the NN `id`/`gm`/`gds`/`gmb`
with an OSDI-derived operating point, **keeping the NN charges qg/qd/qs/qb and
caps** (P0-H proved the charge VALUES are exact ≤2 aC, so they need no injection).
gm/gds/gmb are Jacobian-only (cancel at the fixed point, P0-C), so any period
change is attributable to the injected **id VALUE**.

| Scheme | id VALUE source | Jacobian source | Outcome |
|--------|-----------------|-----------------|---------|
| **v1** (`…id_injection.py`) | OSDI `id`, 1 mV bias cache (piecewise-CONSTANT) | OSDI gds + **NN gm/gmb** (inconsistent) | **DIVERGES** (§3) |
| **v2-percell** | OSDI `id`, per-cell tangent plane | FD of OSDI `id` (NN convention) | converges, **~35× too slow** (§4) |
| **v2-bilinear** | OSDI `id`, bilinear over (Vd,Vg) grid | analytic gradient of the interpolant | converges, **~35× too slow** (§4) |
| **v2-analytic** (final) | OSDI `id` at the **exact live bias** | OSDI analytic gm/gds/gmb, signs probed vs FD | converges; **used for the causal run** (§5) |

OSDI→NN sign map (probed against finite differences of the OSDI `id`):
`OSDI gm = -∂id/∂Vg == NN gm`; `OSDI gmb = -∂id/∂Vb == NN gmb` (Vbs≡0 in the RO ⇒
inert); `OSDI gds = -∂id/∂Vd`, and the NN frame stamps `floor(+∂id/∂Vd)=|id|·0.5`
(the Rule-5 floor — reproduced for an apples-to-apples Jacobian; P0-C already
proved the gds value is period-inert).

## 3. v1 — the naive injection DIVERGES (instrumentation artifact, not a verdict)

| variant | result |
|---------|--------|
| baseline | 50.82 ps (faithful) |
| id-inject NMOS-only | **NR fails at t=2.6e-11 s** (max-delta stuck 1.37e-4 V) |
| id-inject N+P | **NR fails at t=6.0e-11 s** (max-delta stuck 6.6e-5 V) |

Root cause: the injected OSDI `id` was served piecewise-**constant** from the 1 mV
bias cache while the Jacobian (NN `gm` + injected `gds`) claimed a finite slope, so
NR limit-cycled **below the cache cell width** (stuck delta 0.14 mV < 1 mV). This
is an **instrumentation artifact**, not a causal result; v2 fixes it.

## 4. v2 — consistent injection CONVERGES but is computationally pathological

All three v2 schemes converge (`partial=False`, run to completion), proving the v1
divergence was the inconsistent-Jacobian artifact. But every scheme costs **~1 hr
per 0.25 ns** — ~20–35× the per-device baseline:

| scheme | window | wall | converged | note |
|--------|:------:|-----:|:---------:|------|
| v2-percell  (0.5 mV) | 0.20 ns | 4123 s | ✅ | period n/a (window short) |
| v2-bilinear (0.5 mV) | 0.25 ns | 4176 s | ✅ | period n/a (window short) |
| v2-analytic NMOS-only | 0.25 ns | 3377 s | ✅ | period 92.74 ps |

Cost mechanism (sub-stepping is disabled by default, `max_substeps=1`): the
hybrid **OSDI-`id` + NN-charge** device causes frequent NR **failures at the
normal timestep** → adaptive dt-halving → ~370 NR iterations per device-timestep.
The Jacobian-smoothness of the analytic scheme did not remove it; the stiffness is
intrinsic to the hybrid device. This forces a reduced measurement window for the
causal run (§5).

## 5. The causal run — injecting OSDI `id` moves the period the WRONG way

Final causal measurement (v2-analytic, exact-bias OSDI op-point):

| variant | window | period (ps) | %err vs NG | Δ vs baseline | NRMSE % | R² | wall |
|---------|:------:|------------:|-----------:|--------------:|--------:|---:|-----:|
| **baseline** | 0.60 ns | **50.83** | 8.96 | — | 43.46 | -0.055 | 385 s |
| id-inject **N+P** (symmetric) | 0.60 ns | **92.30** | 97.86 | **+41.48** | 61.94 | -1.144 | 10912 s |
| id-inject NMOS-only | 0.25 ns | 92.74 | 98.80 | +41.92 | 57.40 | -0.805 | 3377 s |

**The headline:** injecting the exact OSDI `id` — NMOS-only OR symmetric N+P —
drives the measured period to **~92 ps ≈ 2× the baseline**, i.e. *further* from
NG (46.64 ps), not toward it. Both injection variants agree at ~92 ps.

**The red flag:** NGSPICE with the **full** OSDI model oscillates at **46.64 ps**.
A *faithful* OSDI-`id` injection should therefore approach ~46 ps, not ~92 ps. The
2× value (92 ≈ 2×46.6) is the signature of either a **period-doubled / sub-harmonic
measurement artifact** (the rising-midpoint crossings would be ~2× apart while the
fast cycle is ~46 ps) **or** a genuine doubling driven by the hybrid device (NN
charges / floored gds dominating). These imply **opposite** V6.4.7 conclusions, so
§6 resolves it from the waveform before any verdict is drawn.

## 6. Artifact-vs-real resolution (waveform diagnostic) — GENUINE, not an artifact

Short baseline+NMOS run (0.20 ns, `P0I_SAVE_DIR` set; waveforms saved to
`phase0_logs/p0i_waveforms/wave_*.npz`). All midpoint (Vmid=0.375 V) crossings +
swing:

| variant | Vmin | Vmax | swing | n_rise | n_fall | half-periods (ps) | reading |
|---------|-----:|-----:|------:|:------:|:------:|-------------------|---------|
| baseline        | -0.030 | 0.780 | 0.810 V | 4 | 3 | `26,26,24,26,26,26` | uniform → clean **52 ps** |
| id-inject NMOS  | -0.017 | 0.782 | **0.799 V** | 2 | 2 | `44,48,46` | uniform → genuine **~92 ps** |

**Verdict: GENUINE ~92 ps oscillation, NOT a period-doubled artifact.** The
injected RO swings **full rail** (0.799 V ≈ baseline 0.810 V) with **uniform**
~46 ps half-periods (a sub-harmonic would show alternating tall/short half-periods,
e.g. `~23,~69,~23`). The N+P 0.60 ns run independently measured 92.30 ps with ≥3
clean rising crossings. (The NMOS row's `period=nan` is only because the short
0.17 ns window fits 2 rising crossings < the 3 the estimator needs; the half-period
data is unambiguous.)

## 7. Interpretation & DECISION — the `id` VALUE is NOT separable (P0-I inconclusive + V6.4.7 caution)

Three measured RO periods (all full-rail, uniform oscillations):

| device model | id | charge/caps | RO period | vs NG 46.64 |
|--------------|----|-----------|----------:|:-----------:|
| baseline (full NN)         | NN   | NN   | 50.83 ps | +9 % |
| **P0-I injection** (id-only swap) | **OSDI** | **NN** | **92.30 ps** | **+98 %** |
| NGSPICE (full OSDI, ground truth) | OSDI | OSDI | 46.64 ps | 0 |

**Swapping `id` alone (NN→OSDI) moves the period the OPPOSITE direction
(50.83→92.3, slower) from swapping id+charge together (NGSPICE 50.83→46.64,
faster).** So unlike the Jacobian — which P0-C proved is **inert and cleanly
separable** (gds/cap swap moved ≤0.01 ps because it cancels at the NR fixed
point) — the **`id` VALUE is strongly coupled to the charge model**: the RO period
is a joint (id, charge) property, and isolating the id-value produces an
inconsistent hybrid whose period is dominated by the id↔charge mismatch, not by
the id-value's faithful contribution.

This is a **third outcome** the plan's open-question-2 did not enumerate: not
"id moves the period ≤0.01 ps ⇒ inert" and not "id closes the gap ⇒ owns it", but
**"id moves the period enormously in a non-faithful direction ⇒ non-separable."**

**Consequences for V6.4.7:**

1. **P0-I cannot confirm or cleanly refute "the id VALUE owns the RO gap."** The
   live-injection P0-C analogue **does not transfer** to the id-value, because the
   id-value is not separable from the charge model the way the Jacobian is. The
   clean causal isolation P0-C achieved for the Jacobian is **ill-posed for id**.
2. **The standing P0-G/P0-H correlational evidence still localises the residual**
   to the NMOS dynamic `id` (charges exact ≤2 aC; integration ~0.4 ps; ~20 % peak
   pull-down under-prediction). P0-I does **not** overturn *where* the residual
   lives — only *whether a per-device id-only correction is the right lever*.
3. **The plan's frozen-base LoRA id-VALUE-ONLY distillation is NO LONGER a
   de-risked lever.** P0-I evidence is that an isolated id correction (keeping the
   trained charge heads) destabilises the RO period — and could move it the wrong
   way. V6.4.7 should: **(a)** gate any id-only correction against the live RO
   period *immediately* (not just pointwise id MAE); and **(b)** consider a
   **joint id+charge** correction (or a fuller retrain) rather than id-only.
4. **Caveat (honesty):** the injection proxy carries artifacts a real
   autograd-consistent LoRA would not — it bypasses the Rule-15 Vds correction and
   stamps the floored `|id|·0.5` gds instead of the model's autograd gds. So the
   92 ps is a **proxy warning, not proof** that a real id-LoRA fails. It shifts the
   burden of proof: an id-only RO fix must now be *demonstrated* on the RO period,
   not assumed from the P0-H localisation.

**Net:** RO stays open for V6.4.7, but the indicated lever is **re-scoped from
"id-VALUE-only LoRA" to "joint id+charge correction (or retrain), validated on the
RO period early."** No V6.4.6 behaviour changed (instrumentation only).

## 8. Reproduce

```bash
# v1 (divergent, for the record)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n pycircuitsim \
  python scripts/v6_4_6_p0i_id_injection.py
# v2 causal run (baseline + symmetric N+P)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 P0I_TSTOP_NS=0.6 P0I_SETTLE_NS=0.3 \
  P0I_VARIANTS=baseline,np conda run -n pycircuitsim \
  python scripts/v6_4_6_p0i_id_injection_v2.py
# waveform diagnostic (artifact-vs-real)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 P0I_TSTOP_NS=0.2 P0I_SETTLE_NS=0.03 \
  P0I_VARIANTS=baseline,nmos P0I_SAVE_DIR=results/v6_4_6/phase0_logs/p0i_waveforms \
  conda run -n pycircuitsim python scripts/v6_4_6_p0i_id_injection_v2.py
```

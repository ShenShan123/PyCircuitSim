# V7.4.2 — why BSIM-AR got worse with capacity, and the rebuild

**Status:** investigation complete, campaign pending.
**Retracts:** `BSIM-AR-L74-clean.md` §4 "capacity hurts monotonically" and the
matching DirectNet framing in `DirectNet-L73-clean.md`.

---

## 1. The claim under investigation

V7.4.0 measured BSIM-AR's strict complex score falling **18 → 17 → 15 → 13/20**
from `small` to `xl` and concluded that "additional capacity helps the
feed-forward MLP and hurts the autoregressive model." That conclusion is wrong.
Capacity does exactly what it should. What degrades is an *interpolation* the
training data never constrained, and the benchmark circuits sit inside it.

## 2. What was ruled out

**Exposure bias — refuted.** BSIM-AR trains and selects on a teacher-forced
loss but the simulator runs the free-running loop, so exposure bias was the
obvious suspect. It is not present: on 60 k held-out `tsmc5_nmos` rows the
AR/TF mean-error ratio is 1.31 at `small` and **1.02–1.05** at `large`/`xl`,
and per-column AR error tracks TF to three digits at every tier above `small`.

**Device-surface accuracy — refuted, in the opposite direction.** Bucketed by
`|id|` decade, *every* quantity the solver consumes improves monotonically with
capacity, at both median and p95, in every band. `id` median error over the
conducting decades goes 2.2 % → 0.8 % → 0.3 % → 0.3 %; `cgg` 1.0 → 0.4 → 0.2 →
0.2 %. There is no band where a larger tier is worse.

**The solver / harness — refuted.** The native-L72 control
(`tests/diag_l72_complex_control.py`) runs the byte-identical ring deck through
PyCircuitSim's own solver with the ground-truth OSDI model: TSMC5 and TSMC7
match NGSPICE to **0.00 %**, TSMC12/16 to 0.77 %/0.64 %. The gap is entirely
model-surface-owned.

## 3. The actual mechanism

The ring oscillator is not noisy — it is **biased, always in one direction, and
converging**. TSMC5 period vs a 73.23 ps reference:

| tier | BSIM-AR | DirectNet |
|---|---|---|
| small | 75.83 ps | 78.26 ps |
| medium | 77.20 ps | 77.83 ps |
| large | 78.31 ps | 82.26 ps |
| xl | 78.92 ps | 83.74 ps |

Every tier of every family at every technology is **too slow, never too fast**
(16/16 for DirectNet). Capacity is not adding error; it is removing the
underfitting noise that used to partly cancel a fixed bias.

### Root cause: L is sampled only at PDK bin corners

`pycmg/parser.py::scan_pdk_geometry_combos` emits, for each PDK length bin,
only `(lmin, nfinmin)` and `(lmin, nfinmax)`. Every training row therefore has
`L` equal to some bin's **lower corner**, and within-bin length dependence is
never supervised — while the bins themselves are wide:

| tech | shortest bins (nm) | sampled L | ring NMOS L=16 nm |
|---|---|---|---|
| TSMC5 | [6, 20] (3.3× span) | 6 | **off-grid**, 2.7× from the knot |
| TSMC7 | [8, 11], [11, 20] | 8, 11 | **off-grid**, 1.45× from the knot |
| TSMC6 | = TSMC7 data | 8, 11 | **off-grid** |
| TSMC12 | [16, 20] | 16 | **on-grid — exact** |
| TSMC16 | [16, 20] | 16 | **on-grid — exact** |

The complex-circuit benchmarks pin NMOS `L=16 nm` / PMOS `L=20 nm`
(`tests/common/complex.py`). PMOS 20 nm is a knot for all five techs; NMOS
16 nm is a knot for exactly two. `BenchTech`'s docstring claims the benchmark
geometry is "trained for NMOS L=16nm … so the model interpolates rather than
extrapolates" — that has not been true of these datasets.

### The measurement

NN vs BSIM-CMG (L72) `id`, NFIN=2, T=27 °C, Vg=VDD, Vd swept VDD→VDD/2 — the
corridor an inverter's falling edge integrates. Mean signed error %:

**TSMC7 NMOS ulvt**

| L (nm) | on grid | small | medium | large | xl |
|---|---|---|---|---|---|
| 8 | yes | +1.01 | +0.72 | +0.51 | **+0.33** |
| 11 | yes | +2.40 | +0.62 | +0.34 | **+0.19** |
| 13 | no | −1.49 | −1.54 | +0.88 | +4.68 |
| **16** | **no** | **−7.54** | **−8.47** | **−9.32** | **−13.19** |
| 18 | no | −8.76 | −9.35 | −9.71 | −10.88 |
| 20 | yes | +1.06 | +0.46 | +0.33 | **+0.21** |
| 36 | yes | +1.63 | +0.60 | +0.26 | **+0.15** |

**TSMC5 NMOS lvt** — same signature: on-grid 6/20/36 nm converge to
+0.27/+0.46/+0.13 %, while L=16 nm drifts +0.71 → +1.12 → −1.62 → **−3.29 %**
and L=28 nm drifts −4.25 → **−5.91 %**.

**DirectNet, TSMC5 NMOS lvt** — the same measurement on the *other*
architecture, and the sharpest statement of the effect:

| L (nm) | on grid | small | medium | large | xl |
|---|---|---|---|---|---|
| 6 | yes | +0.98 | +0.03 | −0.02 | **+0.07** |
| 10 | no | −5.66 | +5.83 | +2.34 | +4.84 |
| 13 | no | −6.98 | +0.41 | −10.97 | **−17.39** |
| **16** | **no** | −6.17 | −1.98 | −9.89 | **−11.11** |
| 20 | yes | −4.19 | +0.01 | −0.06 | **+0.02** |
| 28 | no | −2.97 | −6.41 | −5.57 | −10.64 |
| 36 | yes | +3.17 | −0.17 | +0.00 | **+0.03** |

At `xl` DirectNet is accurate to **±0.07 %** on every sampled knot and wrong
by **11–17 %** between them — a 300× ratio. The grid is being memorised, not
generalised. The mechanism is architecture-independent, exactly as the
family-independent ring bias predicted.

At the knots the fit converges to a fifth of a percent. Between the knots the
interpolant is unconstrained, and **more capacity lets it sag further**. A
9–13 % weak NMOS lengthens the falling edge, which is precisely the ring's
5.86 → 6.63 → 7.20 → 10.15 % period error at TSMC7.

### The prediction it makes, and the check

*The techs whose ring degrades with capacity should be exactly the techs whose
benchmark L is off-grid.* Observed: TSMC5, TSMC6, TSMC7 degrade (1/4, 1/4, 0/4
ring cells); TSMC12 and TSMC16 are flat (3/4, 4/4; period error 1.5–2.4 % at
every tier). **5/5.** The magnitude ordering matches the gap width too — TSMC5
(widest gap) and TSMC7 are the worst, TSMC12/16 (exact knot) are clean.

This also unifies the two families. BSIM-AR `small` (0.67 M) is already larger
than DirectNet `large`; the two families occupy different segments of one
curve, DirectNet still climbing the on-grid-accuracy side while BSIM-AR has
crossed into interpolation drift. DirectNet's TSMC5 ring degrades too — it was
simply already failing that cell at `small`, so the scoreboard did not show it.

## 4. Secondary finding — the solver stamps an unsupervised Jacobian

Independent of the above, and worth its own lever. Only **4 of 13** predicted
columns ever reach the solver (`id`, `qg`, `qd`, `qb`); `gm`/`gds`/`gmb` are the
autograd ∂id/∂V and the five caps are the autograd ∂q/∂V, taken through the
whole AR chain in which `id` is token 5 conditioned on four predicted charges.
The loss supervises **values only**, so the stamped derivative is unconstrained.
Median physical error over the conducting corridor, `tsmc5_nmos`:

| quantity | small head / jac | medium | large | xl |
|---|---|---|---|---|
| gm | 2.65 / **10.96** | 0.78 / 3.47 | 0.43 / 1.63 | 0.21 / **1.28** |
| gds | 3.73 / **118.62** | 1.37 / 31.65 | 0.49 / 16.85 | 0.36 / **11.33** |
| gmb | 1.49 / **73.27** | 0.50 / 22.33 | 0.19 / 11.80 | 0.12 / **7.80** |

The stamped `gds` is 11–119 % off where the supervised head column is 0.4–3.7 %.
This does **not** drive the capacity trend (it improves with capacity like
everything else), but it is a plausible driver of the bimodal opamp gain and of
NR basin fragility. `--sobolev` / `--charge-sobolev` already exist and are wired
for the Transformer; they target exactly this.

## 5. The fix

**Sample inside each PDK length bin.** Subdivide `[lmin, lmax]` geometrically
so no adjacent pair of L knots differs by more than `max_l_ratio` (default
**1.35**), and cross each interior L with that bin's NFIN corners. Mechanical,
uniform, and applied to every tech — no benchmark-specific value is inserted.

The rule deliberately leaves the benchmark's 16 nm **between** knots for all
five techs (TSMC5 lands 15.70 nm, TSMC7 14.83 nm), so the complex gates remain
a genuine interpolation test rather than a memorised point.

Cost: ~2.4× rows per (tech, device). Per-bin voltage sampling is left
**identical**, so the L axis is the only variable against V7.4.0.

Ships behind an explicit flag (`--max-l-ratio`), default off — legacy
regeneration stays byte-identical, per the project's perf/data discipline.

## 6. Campaign

| phase | work | where |
|---|---|---|
| P1 | `--max-l-ratio` in the PDK scan → `get_geometry_combos` → `enumerate_bins` → generator CLI | PyCMG |
| P2 | Coverage guard: every `BenchTech` geometry within `max_l_ratio` of a sampled knot, fails loud | `tests/verify_data_geometry_coverage.py` |
| P3 | Regenerate 10 datasets (5 techs × 2 polarities) at `--max-l-ratio 1.35`; old datasets preserved | 152 cores |
| P4 | Retrain 40 clean BSIM-AR checkpoints | GPUs 1–4 |
| P5 | Re-gate: complex ×20 strict-OMP, device DC/tran/AC, opamp AC | CPU-pinned |
| P6 | Rebuild `BSIM-AR-L74-clean.md`; retraction notes in `methodology.md`, `DirectNet-L73-clean.md`; CHANGELOG V7.4.2 | docs |

Held for a decision: whether DirectNet is retrained on the same data. The two
clean reports are only a like-for-like architecture comparison while both sit
on identical rows.

## 7. Reproducing the investigation

Diagnostics live in the session scratchpad; the load-bearing one is the
on-grid/off-grid L probe — build each tier from its `_config.npz`, evaluate
`id` against `pycmg.nn_generate.eval_single_point` at the ring's NFIN/VT/T over
Vg=VDD, Vd ∈ [VDD/2, VDD], and read the mean signed error by L.

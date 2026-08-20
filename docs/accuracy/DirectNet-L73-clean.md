# DirectNet (LEVEL=73) — the clean recipe

**What it is.** A feed-forward MLP compact model. Seven inputs (Vgs, Vds, Vbs,
NFIN, L, T, and a tech code through `nn.Embedding`), thirteen outputs.
`gm`/`gds`/`gmb` are the **autograd Jacobian** of the predicted `id`, and the
small-signal capacitances are the `dQ/dV` autograd of the predicted terminal
charges — nothing is fitted twice.

**What "clean" means.** One training run, no addendum:
`--apply-filter off --swa-mode ema --seed 42`. It is the control every recipe
in [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md) is measured against.

**Measurement provenance.** The datasets and S/M/L/XL checkpoints are the
preserved V7.4.0 clean rebuild. Every table below was remeasured in one
complete CPU-pinned pass on 2026-08-19 at commit `24c181a`, after the solver
began requiring convergence at the final physical homotopy step and before AC
linearization. The shared DirectNet/BSIM-AR archive contains 280 checkpoint
artifacts with manifest SHA-256
`8e4245f1ab563cd116a789cb02388e0f7b736186141694d3242ede2a7ed07868`.
The old V7.4.0 gate verdicts are historical; the exact comparison and
retractions are in
[`simple-circuits-recheck-2026-08-19.md`](simple-circuits-recheck-2026-08-19.md).

Gate definitions, the strict-OMP rule, the `gds`-fix code ladder and the
measured noise floor: [`methodology.md`](methodology.md). Read it before
comparing any two numbers here.

| tier | width × depth | params | CPU cost, 1 thread |
|---|---|---|---|
| `small` | 128 × 3 | ≈ 0.06 M | — |
| `medium` | 256 × 5 | ≈ 0.40 M | — |
| `large` | 384 × 6 | **≈ 0.92 M** | **1.5 ms/eval** |
| `xl` | 512 × 8 | ≈ 2.13 M | 3.4 ms/eval |

> **Denominators changed in V7.3.0.** TSMC6 is now folded into the headline, so
> complex totals are **/20**, device AC **/10** and opamp AC **/5**. Every
> earlier report scored /16, /8 and /4 over the four electrically distinct
> techs. **No total here is comparable to a pre-V7.3.0 total without
> rescaling.**
>
> **V7.4.0 strengthens `methodology.md` §7 from exemplary to exhaustive.** §7
> named five differing PDK keys and showed they were absent from the BSIM-CMG
> Verilog-A; this campaign enumerates *all* of them. Of 1748 shared modelcard
> parameters exactly **11 differ in value**, plus 74 keys unique to TSMC6 and
> 12 to TSMC7 — 97 in total — and **all 97 have zero occurrences anywhere in
> the Verilog-A sources**, against 333 of the 1737 identical keys that do
> appear. The 97 form one coherent family: TSMC's TMI layout-dependent-effect
> layer (LOD, ODX OD-to-OD spacing, isolated-CPODE). Not one core device-physics
> parameter differs — `vth0`, `u0`, `vsat`, `dvt0/1`, `eta0`, `phig`, `eot`,
> `toxp`, `hfin`, `cgso/cgdo` are bit-identical. That matches silicon: TSMC N6
> is a design-rule-compatible EUV derivative of N7, the same transistor with
> re-calibrated layout-stress models — and the stress layer is exactly what
> OpenVAF never compiles and this layout-free flow never stamps. Measured
> consequence, reproduced here on freshly generated data: every array of
> `tsmc6_{nmos,pmos}.npz` is bit-identical to `tsmc7_*` bar the tech-name
> string. TSMC6 is a **controlled repeat** whose only degree of freedom is
> training nondeterminism.

---

## 1. Headline — complex gates by tier

Strict: a cell passes only if it passes at OMP ∈ {1, 2, 4}.

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **8/20** | 2/5 | 0/5 | 5/5 | 1/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| medium | **11/20** | 2/5 | 0/5 | 5/5 | 4/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-switchcap |
| large | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |
| xl | **12/20** | 2/5 | 0/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp |

## 2. By testcase

The four circuits load different parts of the NN surface — the ring the
switching edge, the opamp the high-gain fixed point, the SRAM the bistable
latch, the switchcap the charge and off-state surface. A family can be
excellent on three and fail the fourth, so the per-testcase view is the one
that localizes a weakness.

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 6.87% | FAIL 6.19% | FAIL 5.49% | **PASS** 2.21% | **PASS** 1.59% |
| medium | FAIL 6.28% | FAIL 10.65% | FAIL 10.81% | **PASS** 2.15% | **PASS** 2.28% |
| large | FAIL 12.34% | FAIL 7.38% | FAIL 7.40% | **PASS** 2.14% | **PASS** 2.23% |
| xl | FAIL 14.36% | FAIL 11.83% | FAIL 10.49% | **PASS** 2.98% | **PASS** 2.77% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| medium | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| large | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |
| xl | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.00% | **PASS** 2.28% | **PASS** 1.96% | **PASS** 1.50% | **PASS** 2.21% |
| medium | **PASS** 7.96% | **PASS** 2.39% | **PASS** 2.22% | **PASS** 1.63% | **PASS** 2.57% |
| large | **PASS** 6.35% | **PASS** 1.80% | **PASS** 2.11% | **PASS** 1.23% | **PASS** 1.49% |
| xl | **PASS** 5.80% | **PASS** 1.86% | **PASS** 1.85% | **PASS** 1.69% | **PASS** 3.71% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 1.81% | FAIL 2.24%† | FAIL 2.00%† | FAIL 3.92%† | FAIL 2.57%† |
| medium | FAIL 1.92%† | **PASS** 2.73% | **PASS** 2.89% | **PASS** 4.10% | **PASS** 3.35% |
| large | **PASS** 3.71% | **PASS** 2.54% | **PASS** 2.55% | **PASS** 4.13% | **PASS** 3.28% |
| xl | **PASS** 3.88% | **PASS** 2.66% | **PASS** 2.65% | **PASS** 4.19% | **PASS** 3.38% |

† failed on **hold droop**, the half of this gate the headline number does not show — the metric above is inside its threshold.

## 3. By technology

| tech | ring\_osc | opamp | sram\_snm | switchcap | all cells |
|---|---|---|---|---|---|
| **TSMC5** | 0/4 | 0/4 | 4/4 | 3/4 | **7/16** |
| **TSMC6** | 0/4 | 0/4 | 4/4 | 3/4 | **7/16** |
| **TSMC7** | 0/4 | 0/4 | 4/4 | 3/4 | **7/16** |
| **TSMC12** | 4/4 | 0/4 | 4/4 | 3/4 | **11/16** |
| **TSMC16** | 4/4 | 0/4 | 4/4 | 3/4 | **11/16** |

The split that matters is **supply voltage, not vendor node**. The 0.65–0.75 V trio
(TSMC5/6/7) fail the ring at **every one of the four tiers**, and the 0.80 V
pair (TSMC12/16) pass it at every tier. Sixteen ring cells, zero exceptions
either way. Their transfer curves are steepest exactly where the NN
under-drives, and no amount of capacity has ever moved that.

**The TSMC6 column remains the controlled repeat.**
TSMC6 and TSMC7 train on bit-identical rows with an identical recipe, so every
disagreement between them is training-run luck and nothing else. Under the
current convergence contract they agree on **16/16 complex verdicts**. The
older 14/16 split came entirely from two opamp cells that had been accepted
without final-step convergence.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 2/4 | 1/4 | 1/4 | 2/4 | 2/4 | **8/20** |
| medium | 1/4 | 2/4 | 2/4 | 3/4 | 3/4 | **11/20** |
| large | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |
| xl | 2/4 | 2/4 | 2/4 | 3/4 | 3/4 | **12/20** |

The current strict curve is **8 → 11 → 12 → 12/20**. The ring column is frozen
at 2/5, SRAM at 5/5, and opamp at 0/5. All improvement therefore comes from
switchcap (1/5 → 5/5), mostly by `medium`; `large` and `xl` are tied. The old
11 → 12 → 14 → 15 capacity story included nonconverged opamp fixed points and
is retracted. Capacity buys the switchcap surface here, not a reliable opamp
operating point.

## 5. Device-level suites

Parametric DC is **`gds`-invariant**, so its numbers are comparable across the
whole campaign history; transient and AC are not (`methodology.md` §6).

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 2.17 | 2.61 | 1.48 | 1.01 | 0.80 | 69/69 |
| medium | 1.72 | 2.33 | 1.57 | 0.62 | 0.67 | 69/69 |
| large | 2.57 | 1.96 | 1.37 | 1.33 | 0.71 | 69/69 |
| xl | 3.02 | 2.91 | 2.06 | 1.34 (17/18) | 2.58 (12/14) | 66/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 3.58 | 1.57 | 1.58 | 1.76 | 1.75 | 80/80 |
| medium | 1.92 | 1.45 | 1.47 | 1.52 | 1.51 | 80/80 |
| large | 1.67 | 1.46 | 1.46 | 1.50 | 1.47 | 80/80 |
| xl | 1.67 | 1.47 | 1.45 | 1.52 | 1.47 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✗ f3db 1.58 / ✗ | ✗ gain 1.591 dB, mag 18.95 % / ✗ | ✗ gain 1.591 dB, mag 18.95 % / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| medium | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| large | ✗ f3db nan, mag 10.50 % / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| xl | ✗ f3db nan, mag 14.64 % / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 17.30 dB | FAIL 32.54 dB | FAIL 28.51 dB | FAIL 24.03 dB | FAIL 39.03 dB | **0/5** |
| medium | FAIL 13.33 dB | FAIL 32.03 dB | FAIL 31.60 dB | FAIL 29.51 dB | FAIL 34.20 dB | **0/5** |
| large | FAIL 11.57 dB | FAIL 31.75 dB | FAIL 31.66 dB | FAIL 28.98 dB | FAIL 35.40 dB | **0/5** |
| xl | FAIL 17.96 dB | FAIL 30.13 dB | FAIL 30.24 dB | FAIL 27.91 dB | FAIL 34.62 dB | **0/5** |

Reading the DC column, `medium` is the best device fit on TSMC5, TSMC12 and
TSMC16; `large` takes TSMC6 and TSMC7. Two results bind:

* **`xl` is the first tier ever to break parametric DC**: 66/69 configs, with
  TSMC12 at 17/18 and TSMC16 at 12/14. Every other tier is a clean 69/69.
  Extra capacity is now visibly damaging the device surface even though the
  circuit score improves through `large` — the sharpest evidence yet that the
  two are measuring different things.
* **Device AC is 0/10 at every tier because all DC operating points are
  nonconverged.** Some printed response-shape errors remain small, but they are
  diagnostics about a non-fixed state and cannot be promoted to gate passes.

The transient column, by contrast, is flat and excellent everywhere from
`medium` up (1.45–1.67 % on every tech) — it saturates early and carries no
capacity information at all.

## 6. What the AC gates actually diagnose

Both AC gates are now dominated by their prerequisite: **a converged DC fixed
point**. Device CS-amp AC is 0/10 at every tier, explicitly
`FAIL-NONCONVERGED`; opamp open-loop AC is 0/5 at every tier after its DC
bias sweep selects a railed branch. Printed gain/pole/phase metrics remain
diagnostics only. This pass therefore cannot distinguish a charge-derivative
defect from a DC-basin defect, and the old AC pass counts are retracted.

> **The opamp open-loop AC row is a lower bound, not a result.**
> `verify_complex_opamp_ac` picks its bias by `argmax |dVout/dVin|` on a 2 mV
> grid, but a two-stage Miller opamp with 33–48 dB of gain has a transition
> only 3–14 mV wide. On three of the four original techs the **NGSPICE
> reference's own** operating point at the chosen bias falls outside the
> gate's 15–85 %·VDD validity window, and that window is then applied to the NN
> alone — so a model that faithfully reproduces the reference is scored
> `OP-MISBIAS`. This is a gate-construction defect and is deliberately **not**
> fixed here: changing an accuracy gate changes the accuracy record, which is a
> separate decision.

## 7. What is open

| open | reading |
|---|---|
| **Low-VDD rings, every tech, every tier** | 0/4 on TSMC5, TSMC6 *and* TSMC7 — 12 cells, no exceptions, period error 5.5–14.4 %. Deterministic, not marginal, and *worsening* with capacity (TSMC5: 6.9 % at `small` → 14.4 % at `xl`). Closed only by the corridor curriculum — see the recipes report. |
| **Miller opamp DC** | 0/5 at every tier. The previous nine passes were accepted under the old sticky homotopy-convergence flag and do not survive the current final-step contract. |
| **`xl` parametric DC** | 66/69 configs — the first tier ever to break device DC. Watch it before promoting `xl` for anything device-level. |
| **`switchcap` at `small`** | 4/5 fail on **hold droop**, the half of the gate the headline metric does not show. The charge number is inside threshold on all four. |
| **Device and opamp AC** | 0/10 and 0/5 at every tier. Fix the DC operating-point convergence/basin before interpreting charge dynamics. |

Device DC (69/69 below `xl`), transient (80/80 everywhere), SRAM (20/20) and
switchcap above `small` remain strong. The unresolved fixed-point classes are
the low-VDD ring and every Miller opamp; AC must remain downstream of those.

## 8. GPU acceleration fidelity — separate from the CPU scoreboard

The V7.4 GPU axis ran the resolver-visible clean `large` checkpoints with all
perturbing acceleration levers enabled: batched transient commit, CUDA NN
evaluation, batched COO stamping and NATURAL MNA ordering. T3 covered the four
electrically distinct technologies × four circuits × OMP {1,2,4} (**48
runs**).

The historical GPU run had zero thread-count flips or runtime failures, and
SRAM/switchcap remain useful fidelity evidence. Its 12/16 accuracy claim is
not comparable to the current CPU contract, because the two opamp passes
predate honest final-step convergence. The current four-tech CPU `large`
basket is 10/16 (ring 2/4, opamp 0/4, SRAM 4/4, switchcap 4/4). Re-run T3
before treating GPU acceleration as current opamp evidence.

T4 then compared the complete `{commit,gpu,stamp,order}` path directly with
the flag-off reference on both 6T latch states × four technologies: **8/8
PASS, zero basin flips, zero errors**, worst max|ΔV| **0.1206 mV** and worst
q-NRMSE **0.0101% of VDD**. This clears the GPU fidelity gates; CUDA remains an
explicit opt-in because the CPU/flags-off path is still the scored contract,
not because a V7.4 mismatch remains.

## 9. Reproducing

```bash
# 1. datasets (10 = 5 techs x 2 polarities); BOTH flags are required
bash scripts/benchmark_gen_data.sh 12

# 2. train the clean control, all tiers, on GPU
MODEL=direct RECIPES=clean TECHS='tsmc5 tsmc6 tsmc7 tsmc12 tsmc16' \
  SIZES='small medium large xl' GPUS='1 1 1 0' NSTREAMS=12 TRAIN_OMP=4 \
  bash scripts/recipe_train.sh

# 3. gate one (tier, tech) cell, isolated and CPU-pinned
BSIMAR_CHECKPOINT_DIR=external_compact_models/neural_network/v740_archive/checkpoints \
NN_PY=$(command -v python) bash scripts/v710_regate.sh \
  _one dn xl TSMC12 verify_complex_ring_osc 1

# 4. the coverage map: what is measured, by which pass, and what is missing
BSIMAR_CHECKPOINT_DIR=external_compact_models/neural_network/v740_archive/checkpoints \
python scripts/v730_coverage.py --set clean --tag dn \
  --passes simple-recheck --require-complete

# 5. rebuild this file from the evidence
python scripts/v710_regate_collect.py --root results/simple_recheck_24c181a
python scripts/v730_docs_build.py
```

Checkpoints (gitignored): `external_compact_models/neural_network/v740_archive/checkpoints/`.
Raw runs: `results/simple_recheck_24c181a/`.
GPU evidence: `results/v720_gpu_regate/t3_gpu_bundle/` and
`results/v720_gpu_regate/t4_gpu_bundle/`.

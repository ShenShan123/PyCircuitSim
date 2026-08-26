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
preserved V7.4.0 clean rebuild. Every generated table below was remeasured in
the complete V7.5.17 CPU-pinned pass after the coverage-audit contracts were
enforced. Campaign manifest SHA-256
`b3fd59028cd5ec6961f329ba5b1d9205c4d835dded27a81c9683cd7cef06195d`
pins gate commit `db1b2958e17c72c6b6506fe43efe34e17cd97859` and all 280
checkpoint artifacts. Raw evidence is under `results/v7517_clean/`. The
V7.5.15 recheck remains the exact audit trail for the earlier convergence and
campaign retractions:
[`simple-circuits-recheck-2026-08-19.md`](simple-circuits-recheck-2026-08-19.md).

Gate definitions, strict-OMP scoring, comparability, and evidence rules are in
[`methodology.md`](methodology.md).

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

TSMC6 remains the controlled repeat defined in `methodology.md` §7.

---

## 1. Headline — complex gates by tier

Strict: a cell passes only if it passes at OMP ∈ {1, 2, 4}.

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| small | **5/20** | 2/5 | 0/5 | 2/5 | 1/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc12-sram_snm, tsmc16-sram_snm, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| medium | **7/20** | 2/5 | 0/5 | 1/5 | 4/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc7-sram_snm, tsmc16-sram_snm, tsmc5-switchcap |
| large | **9/20** | 2/5 | 0/5 | 2/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-sram_snm, tsmc6-sram_snm, tsmc12-sram_snm |
| xl | **10/20** | 2/5 | 0/5 | 3/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc12-sram_snm, tsmc16-sram_snm |

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
| xl | FAIL 14.36% | FAIL 11.82% | FAIL 10.49% | **PASS** 2.98% | **PASS** 2.77% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL | FAIL | FAIL | FAIL | FAIL |
| medium | FAIL | FAIL | FAIL | FAIL | FAIL |
| large | FAIL | FAIL | FAIL | FAIL | FAIL |
| xl | FAIL | FAIL | FAIL | FAIL | FAIL |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 5.83%† | **PASS** 2.28% | **PASS** 1.96% | FAIL 0.86%† | FAIL 1.31%† |
| medium | FAIL 5.93%† | FAIL 2.39%† | FAIL 2.22%† | **PASS** 1.63% | FAIL 1.15%† |
| large | FAIL 6.35%† | FAIL 1.30%† | **PASS** 2.11% | FAIL 1.11%† | **PASS** 1.49% |
| xl | **PASS** 5.80% | **PASS** 1.86% | **PASS** 1.85% | FAIL 1.40%† | FAIL 1.83%† |

† failed on **lobe positivity**, the half of this gate the headline number does not show — the metric above is inside its threshold.

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
| **TSMC5** | 0/4 | 0/4 | 1/4 | 3/4 | **4/16** |
| **TSMC6** | 0/4 | 0/4 | 2/4 | 3/4 | **5/16** |
| **TSMC7** | 0/4 | 0/4 | 3/4 | 3/4 | **6/16** |
| **TSMC12** | 4/4 | 0/4 | 1/4 | 3/4 | **8/16** |
| **TSMC16** | 4/4 | 0/4 | 1/4 | 3/4 | **8/16** |

The split that matters is **supply voltage, not vendor node**. The 0.65–0.75 V trio
(TSMC5/6/7) fail the ring at **every one of the four tiers**, and the 0.80 V
pair (TSMC12/16) pass it at every tier. Sixteen ring cells, zero exceptions
either way. Their transfer curves are steepest exactly where the NN
under-drives, and no amount of capacity has ever moved that.

**The TSMC6 column remains the controlled repeat.**
TSMC6 and TSMC7 train on bit-identical rows with an identical recipe, so every
disagreement between them is training-run luck and nothing else. In V7.5.17
they agree on **15/16 complex verdicts**; the sole split is the `large` SRAM
lobe-positivity gate. Both columns reject every opamp fixed point under the
current convergence contract.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 | **5/20** |
| medium | 0/4 | 1/4 | 1/4 | 3/4 | 2/4 | **7/20** |
| large | 1/4 | 1/4 | 2/4 | 2/4 | 3/4 | **9/20** |
| xl | 2/4 | 2/4 | 2/4 | 2/4 | 2/4 | **10/20** |

The V7.5.17 strict curve is **5 → 7 → 9 → 10/20**. The ring column stays at
2/5 and opamp at 0/5. Switchcap improves 1 → 4 → 5 → 5/5, while SRAM moves
2 → 1 → 2 → 3/5. The V7.5.16 8 → 11 → 12 → 12 curve is superseded by this
audited pass. Capacity helps the charge and latch surfaces here, but does not
produce a reliable opamp operating point.

## 5. Device-level suites

Parametric DC is **`gds`-invariant**, so its numbers are comparable across the
whole campaign history; transient and AC are not (`methodology.md` §6).

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean NRMSE % / mean MRE % / min R² / max error µA; config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 7.98 / 37.79 / -1.790 / 494 (20/26) | 10.10 / 38.96 / -3.560 / 484 (20/26) | 11.37 / 43.94 / -3.560 / 484 (15/21) | 4.40 / 13.70 / -2.105 / 198 (27/30) | 4.74 / 15.25 / -2.324 / 200 (23/26) | 105/129 |
| medium | 7.44 / 35.87 / -1.894 / 494 (20/26) | 10.06 / 38.13 / -3.144 / 484 (20/26) | 11.59 / 44.03 / -3.144 / 484 (15/21) | 4.09 / 12.79 / -2.194 / 198 (27/30) | 4.59 / 14.68 / -2.149 / 198 (23/26) | 105/129 |
| large | 8.26 / 37.89 / -1.656 / 494 (20/26) | 9.62 / 36.87 / -3.552 / 484 (20/26) | 11.19 / 43.13 / -3.552 / 484 (15/21) | 4.40 / 13.55 / -1.705 / 198 (27/30) | 4.61 / 14.42 / -2.151 / 198 (23/26) | 105/129 |
| xl | 8.57 / 38.73 / -1.617 / 494 (20/26) | 10.35 / 38.58 / -3.270 / 484 (20/26) | 11.76 / 44.23 / -3.270 / 484 (15/21) | 4.58 / 13.72 / -2.557 / 211 (26/30) | 5.62 / 16.56 / -2.151 / 236 (21/26) | 102/129 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE % / mean MRE % / min R² / max error mV; config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| small | 3.32 / 12.21 / 0.937 / 544 (19/20) | 1.49 / 8.35 / 0.991 / 249 (18/20) | 1.53 / 8.01 / 0.991 / 247 (17/20) | 1.37 / 7.16 / 0.991 / 222 (18/20) | 1.30 / 6.46 / 0.992 / 216 (16/20) | 88/100 |
| medium | 1.51 / 6.63 / 0.991 / 186 (16/20) | 1.37 / 8.32 / 0.991 / 250 (18/20) | 1.38 / 8.30 / 0.991 / 250 (18/20) | 1.35 / 8.32 / 0.992 / 220 (18/20) | 1.34 / 7.19 / 0.992 / 211 (17/20) | 87/100 |
| large | 1.39 / 7.38 / 0.991 / 185 (19/20) | 1.37 / 8.59 / 0.991 / 249 (19/20) | 1.39 / 7.99 / 0.991 / 249 (17/20) | 1.30 / 8.23 / 0.992 / 221 (19/20) | 1.31 / 7.05 / 0.992 / 213 (17/20) | 91/100 |
| xl | 1.39 / 7.27 / 0.991 / 185 (19/20) | 1.38 / 8.49 / 0.991 / 250 (19/20) | 1.37 / 8.52 / 0.991 / 250 (19/20) | 1.34 / 8.12 / 0.992 / 222 (18/20) | 1.29 / 7.33 / 0.992 / 213 (19/20) | 94/100 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✗ f3db 1.58 / ✗ | ✗ gain 1.591 dB, mag 18.95 % / ✗ | ✗ gain 1.591 dB, mag 18.95 % / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| medium | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| large | ✗ f3db nan, mag 10.50 % / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |
| xl | ✗ f3db nan, mag 14.64 % / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | ✗ / ✗ | **0/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, valid refined reference and converged NN OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |
| medium | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |
| large | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |
| xl | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | FAIL — dB | **0/5** |

The expanded audit matrix changes the device denominators, so these counts must
not be compared directly with the old `/69` DC or `/80` transient totals. Two
results bind:

* Parametric DC is **105/129** at `small`, `medium`, and `large`, then
  **102/129** at `xl`. The additional `xl` losses are in TSMC12 (26/30) and
  TSMC16 (21/26); extra capacity does not improve the device surface.
* **Device AC is 0/10 at every tier because all DC operating points are
  nonconverged.** Some printed response-shape errors remain small, but they are
  diagnostics about a non-fixed state and cannot be promoted to gate passes.

Parametric transient is **88 → 87 → 91 → 94/100**. Its numeric fits are tightly
clustered outside the TSMC5-`small` outlier, but the configuration verdicts
still improve non-monotonically with capacity and must remain visible.

## 6. What the AC gates actually diagnose

Both AC gates are dominated by their prerequisite: **a converged DC fixed
point**. Device CS-amp AC is 0/10 at every tier, explicitly
`FAIL-NONCONVERGED`; opamp open-loop AC is 0/5 at every tier with
`OP-NOT-CONVERGED`. Printed gain/pole/phase metrics remain diagnostics only.

V7.5.17 retains the corrected opamp-bias contract. Each simulator performs a
physical 0.1 mV refinement around its coarse maximum-gain point, and every
NGSPICE reference bias in this pass is inside the 15–85% VDD validity window.
The zero is therefore no longer a coarse-grid lower bound: it is a result about
the NN fixed point. It still cannot diagnose charge derivatives independently
until the operating-point gate converges.

## 7. What is open

| open | reading |
|---|---|
| **Low-VDD rings, every tech, every tier** | 0/4 on TSMC5, TSMC6 *and* TSMC7 — 12 cells, no exceptions, period error 5.5–14.4 %. Deterministic, not marginal, and *worsening* with capacity (TSMC5: 6.9 % at `small` → 14.4 % at `xl`). Closed only by the corridor curriculum — see the recipes report. |
| **Miller opamp DC** | 0/5 at every tier. The previous nine passes were accepted under the old sticky homotopy-convergence flag and do not survive the current final-step contract. |
| **`xl` parametric DC** | 102/129 configs, three fewer than every smaller tier. Watch the TSMC12/16 losses before promoting `xl` for device-level work. |
| **`switchcap` at `small`** | 4/5 fail on **hold droop**, the half of the gate the headline metric does not show. The charge number is inside threshold on all four. |
| **Device and opamp AC** | 0/10 and 0/5 at every tier. Fix the DC operating-point convergence/basin before interpreting charge dynamics. |

Switchcap is the strongest complex class at 15/20; SRAM is 8/20. The expanded
device gates retain every DC and transient failure in their `/129` and `/100`
denominators. The unresolved fixed-point classes are the low-VDD ring and every
Miller opamp; AC must remain downstream of those.

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
basket is 8/16 (ring 2/4, opamp 0/4, SRAM 2/4, switchcap 4/4). Re-run T3
before treating GPU acceleration as current opamp evidence.

T4 then compared the complete `{commit,gpu,stamp,order}` path directly with
the flag-off reference on both 6T latch states × four technologies: **8/8
PASS, zero basin flips, zero errors**, worst max|ΔV| **0.1206 mV** and worst
q-NRMSE **0.0101% of VDD**. This clears the GPU fidelity gates; CUDA remains an
explicit opt-in because the CPU/flags-off path is still the scored contract,
not because a V7.4 mismatch remains.

## 9. Reproduction

The published measurements belong to the exact V7.4.0 checkpoint population,
not to the clean recipe in the abstract. A different checkpoint digest is a
different experiment. Retraining the same seed is a new stochastic control,
not a reproduction of these tables.

Preserved checkpoints (gitignored):
`external_compact_models/neural_network/v740_archive/checkpoints/`. Raw runs:
`results/v7517_clean/`.
GPU evidence: `results/v720_gpu_regate/t3_gpu_bundle/` and
`results/v720_gpu_regate/t4_gpu_bundle/`.

The complete launch, coverage, report-build, and new-control commands are in
the [README](../../README.md#run-the-complete-clean-checkpoint-matrix).

# DirectNet (LEVEL=73) — the clean recipe

**What it is.** A feed-forward MLP compact model. Seven inputs (Vgs, Vds, Vbs,
NFIN, L, T, and a tech code through `nn.Embedding`), thirteen outputs.
`gm`/`gds`/`gmb` are the **autograd Jacobian** of the predicted `id`, and the
small-signal capacitances are the `dQ/dV` autograd of the predicted terminal
charges — nothing is fitted twice.

**What "clean" means.** One training run, no addendum:
`--apply-filter off --swa-mode ema --seed 42`. It is the control every recipe
in [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md) is measured against.

**Every number below is from the V7.4.0 rebuild.** Datasets, checkpoints and
gate verdicts were all regenerated from scratch on new hardware: the ten
per-tech datasets rebuilt with `--enable-inv-trip --enable-subvt-off`, then all
four tiers × five techs × both polarities trained clean on GPU straight into
the production slots, then gated CPU-pinned. So the clean `large` row is now
simply the checkpoint served by the resolver. The `v660clean_large` archive
detour and the `crit30f` production promotion belonged to the previous
hardware generation; neither is present in the V7.4.0 artifact set. Nothing
here is inherited from an earlier pass.

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
| small | **11/20** | 2/5 | 3/5 | 5/5 | 1/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc16-opamp, tsmc6-switchcap, tsmc7-switchcap, tsmc12-switchcap, tsmc16-switchcap |
| medium | **12/20** | 2/5 | 1/5 | 5/5 | 4/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc12-opamp, tsmc16-opamp, tsmc5-switchcap |
| large | **14/20** | 2/5 | 2/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc6-opamp, tsmc16-opamp |
| xl | **15/20** | 2/5 | 3/5 | 5/5 | 5/5 | 0 | tsmc5-ring_osc, tsmc6-ring_osc, tsmc7-ring_osc, tsmc5-opamp, tsmc16-opamp |

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
| large | FAIL 12.34% | FAIL 7.38% | FAIL 7.40% | **PASS** 2.14% | **PASS** 2.36% |
| xl | FAIL 14.36% | FAIL 11.97% | FAIL 10.49% | **PASS** 3.26% | **PASS** 2.90% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | FAIL 101.56% | **PASS** 7.23% | **PASS** 6.09% | **PASS** 5.03% | FAIL 100.00% |
| medium | FAIL 100.00% | FAIL 100.00% | **PASS** 5.86% | FAIL 100.00% | FAIL 100.00% |
| large | FAIL 11.61% | FAIL 100.00% | **PASS** 5.25% | **PASS** 5.97% | FAIL 100.00% |
| xl | FAIL 100.00% | **PASS** 4.11% | **PASS** 3.94% | **PASS** 5.78% | FAIL 100.00% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| small | **PASS** 6.00% | **PASS** 2.28% | **PASS** 1.96% | **PASS** 1.50% | **PASS** 2.21% |
| medium | **PASS** 7.96% | **PASS** 2.39% | **PASS** 2.22% | **PASS** 1.63% | **PASS** 2.58% |
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
| **TSMC6** | 0/4 | 2/4 | 4/4 | 3/4 | **9/16** |
| **TSMC7** | 0/4 | 4/4 | 4/4 | 3/4 | **11/16** |
| **TSMC12** | 4/4 | 3/4 | 4/4 | 3/4 | **14/16** |
| **TSMC16** | 4/4 | 0/4 | 4/4 | 3/4 | **11/16** |

The split that matters is **supply voltage, not vendor node**, and in V7.4.0 it
is no longer a tendency but a clean partition: the 0.65–0.75 V trio
(TSMC5/6/7) fail the ring at **every one of the four tiers**, and the 0.80 V
pair (TSMC12/16) pass it at every tier. Sixteen ring cells, zero exceptions
either way. Their transfer curves are steepest exactly where the NN
under-drives, and no amount of capacity has ever moved that.

**The TSMC6 column is the noise floor, and this campaign puts a number on it.**
TSMC6 and TSMC7 train on bit-identical rows with an identical recipe, so every
disagreement between them is training-run luck and nothing else. Across the
16 comparable cells they agree on **14** — all four rings, all four SRAMs, all
four switchcaps — and disagree on exactly two, both in the opamp column
(`medium` and `large`, where TSMC7 passes and TSMC6 rails). That single
column of luck is worth a **2-cell spread in the per-tech total** (TSMC7
11/16, TSMC6 9/16). Any tech-to-tech or recipe-to-recipe difference of two
cells or fewer in the opamp column is therefore indistinguishable from noise,
and must not be read as a result.

## 4. By scale

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | all |
|---|---|---|---|---|---|---|
| small | 2/4 | 2/4 | 2/4 | 3/4 | 2/4 | **11/20** |
| medium | 1/4 | 2/4 | 3/4 | 3/4 | 3/4 | **12/20** |
| large | 2/4 | 2/4 | 3/4 | 4/4 | 3/4 | **14/20** |
| xl | 2/4 | 3/4 | 3/4 | 4/4 | 3/4 | **15/20** |

> **RETRACTED in V7.4.0: "DirectNet peaks at `large`."** Every report through
> V7.3.0 carried that claim. On the rebuilt checkpoints the curve is
> **monotonically increasing** — 11 → 12 → 14 → **15/20** — and `xl` is the
> best clean tier, not a degraded one. The retraction is about the *shape of
> the capacity curve only*; `large` remains the production tier for cost
> reasons (`xl` is 2.3× the parameters and 2.3× the CPU cost per eval for one
> extra cell). The recipe report is historical V7.3.0 evidence and was not
> rebuilt or promoted over these clean V7.4.0 slots.

What actually moves across the tiers is narrow. **The ring column is frozen at
2/5 for all four tiers** and the SRAM column is saturated at 5/5, so neither
contributes any capacity signal at all. The entire 11 → 15 climb comes from
just two columns: switchcap (1/5 → 5/5, essentially all of it landing by
`medium`) and opamp, which is the noisy one. Strip out the opamp column — the
one §3 shows is worth ±2 cells of pure training luck — and the honest reading
is that **capacity buys the switchcap surface and nothing else**.

So the safe statement is the weak one: more capacity does not *hurt* the clean
recipe, and the old "over-fit boundary" story does not survive a retrain.
A 4-cell spread across a 35× parameter range, most of it in one saturating
column and one noisy column, is not a strong capacity effect.

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
| small | 3.58 | 1.57 | 1.59 | 1.76 | 1.75 | 80/80 |
| medium | 1.92 | 1.45 | 1.47 | 1.52 | 1.51 | 80/80 |
| large | 1.67 | 1.46 | 1.46 | 1.50 | 1.47 | 80/80 |
| xl | 1.67 | 1.47 | 1.45 | 1.50 | 1.47 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| small | ✗ f3db 1.58 / ✓ | ✗ gain 1.591 dB, mag 18.95 % / ✓ | ✗ gain 1.591 dB, mag 18.95 % / ✓ | ✓ / ✓ | ✓ / ✓ | **7/10** |
| medium | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |
| large | ✗ f3db nan, mag 10.50 % / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **9/10** |
| xl | ✗ f3db nan, mag 14.64 % / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **9/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| small | FAIL 298.70 dB | FAIL 170.02 dB | FAIL 44.25 dB | FAIL 95.47 dB | FAIL 25.53 dB | **0/5** |
| medium | FAIL 15.25 dB | FAIL 32.15 dB | FAIL 31.42 dB | FAIL 29.62 dB | FAIL 34.28 dB | **0/5** |
| large | FAIL 19.15 dB | FAIL 31.98 dB | **PASS** 1.26 dB | **PASS** 1.82 dB | FAIL 35.47 dB | **2/5** |
| xl | FAIL 17.98 dB | **PASS** 1.15 dB | **PASS** 1.77 dB | FAIL 16.30 dB | FAIL 34.73 dB | **2/5** |

Reading the DC column: **`medium` is the best device fit** (lowest Id-Vgs NRMSE
on TSMC5, TSMC12 and TSMC16; `large` takes TSMC6 and TSMC7), while **`xl` is
the worst device fit and simultaneously the best circuit tier**. The inversion
this project has always reported is therefore not just intact after the
rebuild — it is **wider than before**, because the tier that wins the circuit
gates is now the one furthest from the reference device surface. Circuit gates
measure the value surface at fixed points, not pointwise device accuracy.

Two things in these tables are new in V7.4.0 and neither is cosmetic:

* **`xl` is the first tier ever to break parametric DC**: 66/69 configs, with
  TSMC12 at 17/18 and TSMC16 at 12/14. Every other tier is a clean 69/69.
  Extra capacity is now visibly damaging the device surface even while the
  circuit score climbs — the sharpest evidence yet that the two are measuring
  different things.
* **TSMC5 device AC reports `f3db = nan`** at `large` and `xl` (with magNRMSE
  10.5 % / 14.6 %). A NaN corner frequency is a degenerate fit, not a large
  error, and it is the one device-level result here that deserves a look
  before `xl` is used for anything AC.

The transient column, by contrast, is flat and excellent everywhere from
`medium` up (1.45–1.67 % on every tech) — it saturates early and carries no
capacity information at all.

## 6. What the AC gates actually diagnose

At the **device** level the picture is the familiar one: CS-amp AC is 7/10 at
`small` and 9–10/10 from `medium` up, so the autograd `gm`/`gds` feeding the AC
stamp are accurate and the dominant cap-driven pole is faithful wherever the
cell is well fit.

The **opamp** open-loop AC is not that picture, and V7.4.0 makes it starker
than earlier campaigns did: **0/5 at `small`, 0/5 at `medium`**, 2/5 at `large`
and `xl`. The small-tier errors (25–299 dB) are railed-operating-point
signatures, not small-signal inaccuracy — the gate is reporting that the DC
solve landed in the wrong basin, which is the §7 opamp problem showing up in an
AC harness rather than an independent AC defect. Read this row together with
the caveat below before drawing any conclusion from it.

Three limits are real and specific:

1. The **Cgd-feedforward RHP zero** — high-frequency phase lag — is not
   reproduced. It is *reported* by `verify_nn_ac` as a diagnostic and
   deliberately not gated.
2. Some cells **under-predict the output capacitance** (f3db ratio 1.1–2.5) and
   miss the magnitude gate.
3. The opamp AC inherits the DC value-surface fragility: where the operating
   point lands in the good basin the NN nails GBW and phase margin, so the
   *dynamics* are right and only the DC-gain *level* misses.

These are value-surface and feedforward limits, **not** a charge-derivative
deficiency — that would corrupt gain *and* pole everywhere, and it does not.

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
| **`tsmc5-opamp`** | 0/4. Three tiers rail outright (100 %); `large` misses at 11.6 % against a 10 % gate. The worst single cell in the clean matrix. |
| **`tsmc16-opamp`** | 0/4, railed at every tier. New in V7.4.0 — it passed at `small`/`xl` in V7.3.0, so this is either a genuine regression or opamp-column luck; §3 says a 2-cell opamp swing is noise, and this is 2 cells. **Do not treat it as established until a second seed is run.** |
| **`xl` parametric DC** | 66/69 configs — the first tier ever to break device DC. Watch it before promoting `xl` for anything device-level. |
| **`switchcap` at `small`** | 4/5 fail on **hold droop**, the half of the gate the headline metric does not show. The charge number is inside threshold on all four. |
| **Opamp open-loop AC** | 0/5 at the two smallest tiers. Partly unreachable by construction; see the caveat above. |

**Retracted here:** the V7.3.0 open item *"`tsmc7-opamp` at `large`"*. On the
rebuilt checkpoints TSMC7 passes the opamp at **all four tiers** (3.9–6.1 %
gain error) and is the strongest opamp column in the matrix. Likewise the
*"`tsmc5-opamp` margin is the thinnest"* framing: it is no longer a thin margin
but an outright failure.

The rings, the opamps and the opamp AC are all **value-surface / fixed-point**
properties — open-loop gain, oscillation period at a sharp edge — not
device-current or charge-derivative fidelity gaps. Device DC (69/69 below
`xl`), transient (80/80 everywhere), SRAM (20/20) and switchcap above `small`
are all strong. The one genuinely new device-level crack is `xl`'s 66/69.

## 8. GPU acceleration fidelity — separate from the CPU scoreboard

The V7.4 GPU axis ran the resolver-visible clean `large` checkpoints with all
perturbing acceleration levers enabled: batched transient commit, CUDA NN
evaluation, batched COO stamping and NATURAL MNA ordering. T3 covered the four
electrically distinct technologies × four circuits × OMP {1,2,4} (**48
runs**).

**Binding verdict: PASS.** SRAM and switchcap are 24/24 across OMP, all eight
strict cells pass, Rule 2 is 15/15 on CUDA, and there are zero thread-count
flips or runtime failures. The full report-only basket is **12/16 strict** —
ring 2/4, opamp 2/4, SRAM 4/4, switchcap 4/4 — exactly the current V7.4 CPU
clean-`large` basket. Every printed metric matches the CPU evidence except
TSMC5 opamp OMP=1, which moves 11.61% → 11.59% without changing its FAIL
verdict.

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
NN_PY=$(command -v python) bash scripts/v710_regate.sh \
  _one dn xl TSMC12 verify_complex_ring_osc 1

# 4. the coverage map: what is measured, by which pass, and what is missing
python scripts/v730_coverage.py --set clean --tag dn --require-complete

# 5. rebuild this file from the evidence
python scripts/v710_regate_collect.py --root results/v740_regate
python scripts/v730_docs_build.py
```

Checkpoints (gitignored): `tsmc{5,6,7,12,16}_dn_{small,medium,large,xl}_{nmos,pmos}`
— clean at every tier, so no archive stem is needed.
Raw runs: `results/v740_regate/`.
GPU evidence: `results/v720_gpu_regate/t3_gpu_bundle/` and
`results/v720_gpu_regate/t4_gpu_bundle/`.

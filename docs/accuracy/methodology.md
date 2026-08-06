# Accuracy methodology — the contract every number in `docs/accuracy/` obeys

One file, so no report restates it and no two reports drift. If a number
anywhere in `docs/accuracy/` carries no explicit exception, it was produced
exactly as described here.

---

## 1. Ground truth

**NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI model** — repo build
`tools/ngspice-45.2/bin/ngspice` (honoured via `NGSPICE_BIN`), modelcards
resolved by `pycmg.tech`. Same netlist, same geometry, same supply; only the
MOSFET model is swapped. Never a simplified, hand-derived or self-defined
reference.

BSIM-CMG is therefore never *graded* in these reports — it is the yardstick.
Its own record against NGSPICE lives in `../CHANGELOG.md`. The three graded
families are the NN compact models: **DirectNet** (73), **BSIM-AR** (74),
**PFN** (75).

## 2. The complex-circuit gate matrix — 4 circuits × 5 techs = 20 cells

The authoritative accuracy number for a checkpoint set. Verdict = the
`verify_complex_*` script's **exit code**, never a metric read by eye.

| circuit | script | gate |
|---|---|---|
| **ring_osc** — 5-stage ring oscillator | `verify_complex_ring_osc.py` | period error ≤ **5 %** |
| **opamp** — two-stage Miller, DC sweep | `verify_complex_opamp.py` | open-loop DC gain error ≤ **10 %** (trip shift reported, not gated) |
| **sram_snm** — 6T read-SNM butterfly | `verify_complex_sram_snm.py` | all lobes positive **and** lobe NRMSE ≤ **10 %**, across the NFIN corners |
| **switchcap** — switched-cap unit cell | `verify_complex_switchcap.py` | charge-transfer error ≤ **5 % of VDD** **and** hold droop ≤ max(10 % of the NGSPICE droop, 0.1 % of VDD) |

Two gates have a **second criterion their headline metric cannot show** —
switchcap's droop and sram_snm's lobe positivity. A cell can therefore FAIL
with its reported number inside the threshold; the reports mark those with a
dagger rather than leaving them looking self-contradictory.

Techs: **TSMC5, TSMC6, TSMC7, TSMC12, TSMC16**.

> **The denominator changed in V7.3.0.** TSMC6 is now folded into the headline,
> so complex totals are **/20**, device AC **/10**, opamp AC **/5**. Every
> report before V7.3.0 scored /16, /8 and /4 over the four electrically
> distinct techs and quoted TSMC6 separately. **No total in the current reports
> is comparable to a total in an older one without rescaling** — a V7.3.0
> "16/20" and a V7.1.0 "13/16" can be the same measurement.
>
> This is a presentation decision, not a retraction: §7's finding that TSMC6 is
> TSMC7 relabelled is unchanged, and the duplication is still a reason to read
> a TSMC6-vs-TSMC7 difference as training-run luck.

The four circuits load different parts of the NN surface — the ring the
switching edge, the opamp the high-gain fixed point, the SRAM the bistable
latch, the switchcap the charge/off-state surface. A family can be excellent on
three and fail the fourth, which is why every report carries a per-testcase
view.

## 3. Strict determinism — OMP ∈ {1, 2, 4}

The opamp and ring cells sit on multistable fixed points, and GEMM thread count
perturbs the last bits enough to land a different Newton basin. So every such
cell is measured at

```
OMP_NUM_THREADS = MKL_NUM_THREADS = PYCIRCUITSIM_TORCH_THREADS ∈ {1, 2, 4}
```

* **strict PASS** = passes at all three.
* **FLIP** = passes some, fails others. Not bankable, counted as a fail; a
  single-run pass on a FLIP cell is an artifact.
* `sram_snm` and `switchcap` are deterministic under the thread pin and are
  taken from the single run.

`OMP_NUM_THREADS` alone stopped moving torch's GEMM threading at the V6.6.6
harness thread-pin — `PYCIRCUITSIM_TORCH_THREADS` is what makes the probe
exercise the multistability axis.

Since the V6.13.0 `gds` fix (§6) flips became rare rather than endemic; they
were a symptom of a wrong-signed Jacobian entry, not an intrinsic property of
high-gain circuits. They are not extinct — treat a nonzero flip count in any
report table as unbankable and re-measure.

## 4. Device-level suites

| suite | what it gates |
|---|---|
| `verify_nn_dc_tran` | single-point resolver-path DC + transient |
| `verify_nn_multi_tech_dc` | parametric Id-Vgs over the full L / NFIN / VT sweep — NRMSE < **10 %** per config |
| `verify_nn_multi_tech_tran` | parametric inverter transient, same sweep |
| `verify_nn_ac` | device CS-amp small-signal: gain0 err ≤ **1.5 dB**, f3db ratio ∈ **[0.7, 1.43]**, magnitude NRMSE ≤ **10 %** (in-band phase reported, **not** gated) |
| `verify_complex_opamp_ac` | two-stage Miller **open-loop AC**: DC-gain err ≤ **3 dB**, GBW ratio ∈ **[0.6, 1.67]**, PM err ≤ **15°**, non-railed OP |
| `verify_nn_lifted_source_dc` | Rule 2 canary — source-relative frame, NRMSE ≤ 10 % |

The AC suites read the NN's **autograd charge derivatives** (`cgd = ∂qg/∂Vd`, …)
rather than its current surface, so they move independently of the DC gates.

Metrics reported per Rule 13: **MRE %, R², NRMSE, Max error**, per tech.

## 5. Uniformity, isolation, hygiene

* **Uniformity.** A recipe is *one identical addendum applied to every
  (tech × device × size) checkpoint* — never a per-tech or per-gate special.
  Curriculum recipes warm-start from **their own tier's** clean checkpoint,
  which is mechanical, not per-case. The V6.5.x hand-tuned per-case
  interventions were retired in V6.6.0 because they answered "can a tuned
  checkpoint pass this gate?" instead of "how faithful is the model under one
  recipe?".
* **Isolation.** Every cell gets its own `PYCIRCUITSIM_COMPLEX_RESULTS` /
  `PYCIRCUITSIM_NN_RESULTS`. The harness scratch dirs are keyed by
  (circuit, tech) only, so parallel cells without isolation silently collide.
* **Scored CPU axis.** The headline reports pin `CUDA_VISIBLE_DEVICES=""`.
  This preserves the historical numerical contract and keeps acceleration
  changes out of the accuracy scoreboard.
* **Separate GPU fidelity axis.** CUDA is evaluated only when explicitly
  selected with `PYCIRCUITSIM_NN_DEVICE=cuda` and a pinned GPU. Its T3 bundle
  repeats the complex gates with every acceleration feature enabled; its T4
  gate probes the full latch basin. GPU verdicts are reported beside, never
  substituted for, CPU-pinned results because CUDA arithmetic can relocate a
  fragile Newton basin.
* **Checkpoint completion.** A bare `_best.pt` is *not* evidence of a finished
  run — the trainer writes it at every validation improvement. Completed runs
  carry `*_best.pt.complete`. V6.6.5 found and retrained 22 killed-run
  checkpoints this way.
* **Checkpoint selection.** Env pins
  `PYCIRCUITSIM_NN_CHECKPOINT_{DN,TF,PFN}_{NMOS,PMOS}` are read first and
  **raise** if the stem is absent — no silent fallback.
  `PYCIRCUITSIM_NN_FORCE_LEVEL={74,75}` retargets a LEVEL=73 deck at BSIM-AR or
  PFN, so the whole gate infrastructure runs any family with zero deck changes.
* **Thread-wait policy.** Gates export `OMP_WAIT_POLICY=passive` and
  `KMP_BLOCKTIME=0`. On an oversubscribed host an OMP>1 cell otherwise spends
  its time in busy-wait barriers — measured, a ring cell that takes 21 s at
  OMP=1 exceeded 10 min at OMP=4 while accumulating CPU at ~50 % of one core.
  This changes how threads wait, never how work is partitioned, so it is
  numerically neutral.

### V7.4 GPU-axis result

The complete perturbing bundle — batched transient commit, CUDA NN evaluation,
batched COO stamping and NATURAL MNA ordering — clears both binding gates on an
RTX PRO 6000 Blackwell. T3 runs 48 cells (four electrically distinct techs ×
four circuits × OMP {1,2,4}): SRAM + switchcap are **24/24**, Rule 2 is
**15/15**, there are zero flips/runtime failures, and the full report-only
basket is **12/16 strict**, exactly the V7.4 CPU clean-`large` basket. T4 is
**8/8**, zero basin flips/errors, worst max|ΔV| 0.1206 mV and worst q-NRMSE
0.0101% of VDD. Raw evidence is under `results/v720_gpu_regate/`.

## 6. Code-state ladder — which numbers compare to which

**Comparisons across states are only valid where stated.**

| state | commit | what it is |
|---|---|---|
| **pre-fix** | ≤ `a96112a` | everything up to V6.12.x, measured with the `gds` sign bug present |
| **V6.13.0** | `d2ea720` | the gds sign + guard fix; full complex re-gate of all on-disk groups |
| **V7.1.0** | — | device + AC + strict-OMP re-gate; V7.0.x perf flags default-off, so the numerics are V6.13.0's |
| **V7.3.0** | `73434d4` | five-tech report baseline: BSIM-AR, PFN `xl`, surviving recipes and PFN's first curriculum arm |
| **V7.4.0** | `c2cab3d` | new-hardware, from-scratch clean rebuild of all DirectNet and BSIM-AR tiers; 480 measured suite runs, CPU-pinned |
| **V7.4.0 GPU axis** | `5256d32` | opt-in CUDA T3 48/48 runs + T4 8/8 basin campaign; separate from the CPU scoreboard |

### The `gds` sign bug (fixed in `8ed35bd`, V6.13.0)

Inference negated `gm`/`gmb` but **not** `gds`. All three are derivatives of the
same signed `id`, so the sign comes from `id`'s convention, not from which
variable is differentiated. The two-sided floor `max(gds, |id|·0.5)` then
asserted an Early voltage ≤ 2 V — below the true median of every device — and
overrode the learned output conductance at **90.9 %** of amplifying points. The
floor was load-bearing **only because it masked the sign error**.

The fix is two changes that must ship together (sign alone is bit-identical;
the guard alone regresses device AC):

1. **sign** — negate `gds` with `gm`/`gmb`. OSDI `-d(id)/dVd` is positive at
   100.0000 % of conducting points over 111,630 evaluations across all ten
   production devices.
2. **guard F** (`_floor_gds` → `_guard_gds`) — positives pass through
   **bit-identical**; only negatives clamp, to `|id|/50 V`. 50 V is above the
   measured maximum true Early voltage (43.4 V), so the guard cannot bind on a
   physically correct value. The knob is `PYCIRCUITSIM_GDS_GUARD_K`;
   `PYCIRCUITSIM_GDS_FLOOR_K` is now rejected loudly.

**The structural signature: every cell the fix gained is an opamp.** Not one
ring, SRAM or switchcap cell moved, across all three families and every tier.
`gds` sets the small-signal output resistance — what a high-gain operating
point is made of — and cancels at the Newton fixed point everywhere else.

**Corollary — DC is exactly invariant**, so *pre-fix device-DC numbers remain
valid*. Pre-fix **AC**, **transient** and **opamp** numbers do not.

## 7. TSMC6 ≡ TSMC7 relabelled

**The finding is unchanged, and folding TSMC6 into the headline (§2) does not
soften it.** TSMC6 is not an independent technology under BSIM-CMG. Four
independent lines of evidence:

* `tsmc6_{nmos,pmos}.npz` are `array_equal` to `tsmc7_*` in `inputs`,
  `geometry`, `outputs` and `sample_class` over 1,816,830 / 2,187,292 rows —
  only `meta_tech_name` differs.
* The raw PDKs genuinely differ, but V7.4.0 exhaustively checked the complete
  modelcard delta: of 1748 shared parameters, **11 differ in value**, with 74
  keys unique to TSMC6 and 12 to TSMC7. All **97** differing/unique keys have
  zero occurrences in the BSIM-CMG Verilog-A, while 333 of the 1737 identical
  keys do occur. The entire delta is the TSMC TMI layout-stress layer (LOD,
  ODX spacing and isolated-CPODE); core device parameters such as `vth0`,
  `u0`, `vsat`, `eot`, `toxp`, `hfin` and `cgso/cgdo` are bit-identical.
* Two LEVEL=72 Id-Vgs sweeps at identical geometry match to the last digit.
* **V7.3.0 adds a fourth, at the corridor level:** the ring-only `corro`
  corridor, which did not previously exist for TSMC6, was harvested by
  *running the ring oscillator* under LEVEL=72 and came out `array_equal` to
  TSMC7's. Identical rows there mean the two techs' *circuits follow the same
  trajectory*, not merely that their datasets match.

**What this buys.** A duplicate technology is the one thing this project cannot
get any other way: **a controlled repeat.** Same data, same recipe, same code,
different training run. Read every TSMC6-vs-TSMC7 difference as training-run
luck plus Newton-basin luck — that is exactly what makes it the instrument for
§8.4's noise floor.

`bsimar.config.assert_tech_is_distinct()` still flags the collision;
`tsmc6`↔`tsmc7` is the sole entry in `ACKNOWLEDGED_DUPLICATE_TECHS`, so the
guard prints loudly and continues instead of raising. Nothing else is on that
list and nothing else should be, to silence a genuine onboarding mistake.
**Run the guard before onboarding a technology, not after gating one** — the
V6.9.0 onboarding gated TSMC6 9/9 DC and 14/14 transient, and passing those
told us nothing, because they were TSMC7's gates.

TSMC6 holds tail codes 22-24, so its presence renumbers nothing.

## 8. Standing measurement caveats

1. **The validation split cannot measure what the gates measure.**
   `dataset.py` takes a uniform random row permutation over a *dense per-bin
   lattice*, so every (variant, L, NFIN, T) bin appears in train, val **and**
   test. On tsmc5 NMOS: grid pitch 44.8 mV, median val→train nearest-neighbour
   distance **28.4 mV** — below the pitch. Device test-split metrics are
   optimistic relative to gate behaviour; a grouped / held-out-L split is the
   open follow-up.
2. **The SRAM `force_ic` residual half is a no-op.** The residual is built with
   the V-source branch-current slots left at zero, so it floors at the supply
   current and `resid_ok` is always True. The *rail* half of that probe is
   real; the residual half currently gates nothing.
3. **`verify_nn_ac` prints a DirectNet banner even under `FORCE_LEVEL=74/75`.**
   `nn_ac_tf.log` / `nn_ac_pfn.log` therefore *read* as DirectNet results while
   measuring BSIM-AR / PFN. Cosmetic, unfixed — trust the log's path, not its
   header.
4. **Single-cell rankings are inside the pipeline's own run-to-run noise —
   measured, not asserted.** The TSMC6 repeat retrains the same recipe on
   bit-identical rows and compares strict verdicts. Gate *counts* agreed at
   three of four tiers, but **which** cells passed swapped: `ring_osc` carries
   **±4 pp** of scatter across a **5 %** gate, and `opamp` is **bimodal** — a
   good basin (1.8–7.1 %) or a 100 % rail. `sram_snm` and `switchcap` reproduce
   to ≤0.3 pp and never flip.
   **So a recipe promoted on one ring or opamp cell is not a result; the same
   claim on a SRAM or switchcap cell is.** Family-level counts over many cells,
   and levers whose effect clears the floor (the corridor moves rings by
   ~8 pp), are unaffected.
   Reproducibility is also **family-dependent**. In the V7.4.0 clean rebuild,
   BSIM-AR reproduced 15/16 TSMC6-vs-TSMC7 verdicts (the split was the small
   ring) and DirectNet 14/16 (both splits were opamps). PFN's latest repeat is
   still V7.3.0, at 10/12.
5. **The opamp open-loop AC gate has a bias-resolution defect.**
   `verify_complex_opamp_ac` linearizes about `argmax |dVout/dVin|` found on a
   **2 mV** grid, but a two-stage Miller opamp with 33–48 dB of gain has a
   transition only **3–14 mV** wide. On three of four techs the **NGSPICE
   reference's own** output at the chosen bias falls outside the gate's
   15–85 %·VDD window — and that window is applied to the **NN only**, so a
   model faithfully reproducing the reference is scored
   `FAIL [OP-MISBIAS]`. **Treat the opamp-AC row as a lower bound.** The fix has
   two parts (refine the sweep near the transition; judge `op_valid` against
   the reference's operating point) and is deliberately **not** applied here:
   changing an accuracy gate changes the accuracy record, which is a separate
   decision.

**A gate result is a reproducible property of a checkpoint; it is not a
reproducible property of a recipe.** Re-gating the same weights is
deterministic — 223/223 complex cells agreed between two passes on different
days. Retraining the same recipe on the same rows is not.

## 9. Reproducing

```bash
# confirm the V7.4.0 clean checkpoint matrix is complete
python scripts/v730_coverage.py --tag dn --set clean --require-complete
python scripts/v730_coverage.py --tag tf --set clean --require-complete

# collect the completed new-hardware CPU gates
python scripts/v710_regate_collect.py --root results/v740_regate

# rebuild/check only the V7.4.0 reports backed by local raw evidence
python scripts/v730_docs_build.py --only dn,tf --recipes clean
python scripts/v730_docs_build.py --check

# GPU fidelity axis (opt-in; not part of the CPU scoreboard)
T3_AXIS=gpu GPU=1 NN_PY=python \
  bash scripts/v720_t3_flag_bundle.sh NATURAL
python tests/verify_latch_basin_gpu.py \
  --config commit+gpu+stamp+order --gpu 1 --ordering NATURAL
```

Local raw evidence: `results/v740_regate/` (V7.4.0 CPU clean) and
`results/v720_gpu_regate/` (V7.4.0 GPU axis). The V7.3.0 raw recipe/PFN trees
were not copied to the new machine; their rendered family reports are the
durable historical record. The builder renders only from a complete pinned
campaign pass. If that source is incomplete locally, it preserves the report
only after its committed SHA-256 matches, rather than mixing passes or writing
blank tables.

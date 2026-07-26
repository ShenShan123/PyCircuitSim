# Accuracy methodology — the contract every number in `docs/accuracy/` obeys

One file, so no report has to restate it and no two reports can drift. If a
number anywhere in `docs/accuracy/` is not accompanied by an explicit exception,
it was produced exactly as described here.

---

## 1. Ground truth

**NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI model**, repo build
`tools/ngspice-45.2/bin/ngspice` (honoured via `NGSPICE_BIN`), modelcards
resolved by `pycmg.tech`. Never a simplified, hand-derived or self-defined
reference — the same netlist topology, the same geometry, the same supply, with
only the MOSFET model swapped.

BSIM-CMG itself is therefore never *graded* in these reports; it is the yardstick.
Its own gate record (DC/tran/AC vs NGSPICE, five techs) lives in `../CHANGELOG.md`.

The three graded families are the NN compact models: **DirectNet** (LEVEL=73),
**BSIM-AR** (LEVEL=74), **PFN** (LEVEL=75).

## 2. The complex-circuit gate matrix — 4 circuits × 4 techs = 16 cells

The authoritative accuracy number for a checkpoint set. Verdict = the
`verify_complex_*` script's **exit code**, never a metric read by eye.

| circuit | script | gate |
|---|---|---|
| **ring_osc** — 5-stage ring oscillator | `verify_complex_ring_osc.py` | oscillation period error ≤ **5 %** |
| **opamp** — two-stage Miller, DC sweep | `verify_complex_opamp.py` | open-loop DC gain error ≤ **10 %** (trip shift reported, not gated) |
| **sram_snm** — 6T read-SNM butterfly | `verify_complex_sram_snm.py` | all lobes positive **and** lobe NRMSE ≤ **10 %**, across the NFIN corners |
| **switchcap** — switched-cap unit cell | `verify_complex_switchcap.py` | charge-transfer error ≤ **5 % of VDD** and hold droop ≤ max(10 % of the NGSPICE droop, 0.1 % of VDD) |

Techs: **TSMC5, TSMC7, TSMC12, TSMC16**. (TSMC6 was retired 2026-07-24 — §7.)

The four circuits are chosen to load different parts of the NN surface: the ring
loads the switching edge, the opamp the high-gain fixed point (output
conductance), the SRAM the bistable latch, the switchcap the charge/off-state
surface. A family can be excellent on three and fail the fourth; see
`by-tech.md` for which cells actually separate the families.

## 3. Strict determinism — OMP ∈ {1, 2, 4}

The opamp and ring cells sit on multistable fixed points, and GEMM thread count
perturbs the last bits enough to land a different Newton basin. So:

```bash
OMP_NUM_THREADS = MKL_NUM_THREADS = PYCIRCUITSIM_TORCH_THREADS ∈ {1, 2, 4}
```

* **strict PASS** = passes at all three thread counts.
* **FLIP** = passes some, fails others. A FLIP is **not bankable** and is
  counted as a fail; a single-run pass on a FLIP cell is an artifact.
* `sram_snm` and `switchcap` are deterministic under the thread pin and are
  taken from the single run.

Since the V6.6.6 harness thread-pin, `OMP_NUM_THREADS` alone no longer moves
torch's GEMM threading — `PYCIRCUITSIM_TORCH_THREADS` is what makes the probe
exercise the multistability axis. Drivers: `scripts/opamp_sweep_def.sh`,
`scripts/recipe_multirun_gate.sh`, `scripts/a3_omp_one.sh` (one OMP value per
invocation, resumable), `scripts/v710_regate.sh`.

> Since the V6.13.0 `gds` fix (§6) **every** re-measured group is flip-free.
> FLIPs were a symptom of a wrong-signed Jacobian entry, not an intrinsic
> property of high-gain circuits.

## 4. Device-level suites

| suite | what it gates |
|---|---|
| `verify_nn_dc_tran` | single-point resolver-path DC + transient (production checkpoints, 24 configs) |
| `verify_nn_multi_tech_dc` | parametric Id-Vgs over the full L / NFIN / VT sweep — NRMSE < **10 %** per config |
| `verify_nn_multi_tech_tran` | parametric inverter transient, same sweep |
| `verify_nn_ac` | device CS-amp small-signal: gain0 err ≤ **1.5 dB**, f3db ratio ∈ **[0.7, 1.43]**, magnitude NRMSE ≤ **10 %** (in-band phase reported, **not** gated) |
| `verify_complex_opamp_ac` | two-stage Miller **open-loop AC**: DC-gain err ≤ **3 dB**, GBW ratio ∈ **[0.6, 1.67]**, PM err ≤ **15°** (magNRMSE reported, **not** gated) |
| `verify_nn_lifted_source_dc` | Rule 2 canary — source-relative frame, NRMSE ≤ 10 % |

The AC suites are the ones that read the NN's **autograd charge derivatives**
(`cgd = ∂qg/∂Vd`, …) rather than its current surface, so they move independently
of the DC gates — a distinction that recurs throughout `by-scale.md`.

Metrics reported per Rule 13: **MRE %, R², NRMSE, Max error**, per tech.

## 5. Uniformity contract, isolation, hygiene

* **Uniformity.** A recipe is *one identical addendum applied to every
  (tech × device × size) checkpoint* — never a per-tech or per-gate special.
  Curriculum recipes warm-start from **their own tier's** clean checkpoint,
  which is mechanical, not per-case. The V6.5.x hand-tuned per-case
  interventions were retired in V6.6.0 for exactly this reason: they answered
  "can a tuned checkpoint pass this gate?" instead of "how faithful is the model
  under one recipe?".
* **Isolation.** Every cell gets its own `PYCIRCUITSIM_COMPLEX_RESULTS` /
  `PYCIRCUITSIM_NN_RESULTS`; the harness scratch dirs are keyed by
  (circuit, tech) only, so parallel cells without isolation silently collide.
* **CPU, never CUDA.** `CUDA_VISIBLE_DEVICES=""`. The fragile opamp fixed points
  land in different NR basins under CUDA float.
* **Checkpoint completion.** A bare `_best.pt` is *not* evidence of a finished
  run — completed runs carry `*_best.pt.complete`. V6.6.5 found and retrained 22
  killed-run checkpoints this way.
* **Checkpoint selection.** Env pins `PYCIRCUITSIM_NN_CHECKPOINT_{DN,TF,PFN}_{NMOS,PMOS}`
  are read first and **raise** if the stem is absent (no silent fallback, since
  V6.6.6). `PYCIRCUITSIM_NN_FORCE_LEVEL={74,75}` retargets a LEVEL=73 deck to
  BSIM-AR / PFN so the whole gate infrastructure runs any family with zero deck
  changes.
* **Denominators.** The authoritative complex matrix is **/16** and the device
  AC suite **/8**. Text from before the TSMC6 retire reports **/20** and **/10**
  (and AC as /12): those carried a TSMC6 column that duplicates TSMC7 (§7).

## 6. Code-state ladder — which numbers are comparable to which

Three code states produced the numbers in these reports. **Comparisons across
states are only valid where stated.**

| state | commit | what it is |
|---|---|---|
| **pre-fix** | ≤ `a96112a` | Everything up to V6.12.x. Measured with the **`gds` sign bug** present. |
| **V6.13.0 post-fix** | `d2ea720` | The gds sign + guard fix (`8ed35bd`). Full re-gate: complex matrix for all 28 on-disk checkpoint groups + 10 strict-OMP sweeps + the resolver-default device suites. |
| **V7.1.0** | HEAD | Post V7.0.x performance work (opt-in perf flags OFF, so numerics are the V6.13.0 numerics) + audit fix wave 1. Re-measures the **per-size / per-recipe device and AC suites** the V6.13.0 campaign did not reach, and extends the strict-OMP sweeps. |

### The `gds` sign bug (fixed in `8ed35bd`, V6.13.0)

Inference negated `gm`/`gmb` but **not** `gds`. All three are derivatives of the
same signed `id`, so the sign comes from `id`'s convention, not from which
variable is differentiated. The two-sided floor `max(gds, |id|·0.5)` then
asserted an Early voltage ≤ 2 V — below the true median of every device (OSDI
amplifying p50 3.3–9.8 V) — and overrode the learned output conductance at
**90.9 %** of amplifying points. The floor was load-bearing **only because it
masked the sign error**.

The fix is two changes that must ship together (measured: sign alone is
bit-identical; the guard alone regresses device AC 8/10 → 5/10):

1. **sign** — negate `gds` with `gm`/`gmb`. Autograd `d(id)/dVd` vs `-gds_head`
   = 0.12 rel err, vs `+gds_head` = 2.08 (2.0 being the arithmetic signature of a
   pure sign flip). OSDI `-d(id)/dVd` is positive at 100.0000 % of conducting
   points over 111,630 evals across all 10 production devices.
2. **guard F** (`_floor_gds` → `_guard_gds`) — positives pass through
   **bit-identical**; only negatives clamp, to `|id|/50 V`. 50 V is above the
   measured maximum true Early voltage (43.4 V), so the guard cannot bind on a
   physically correct value. `PYCIRCUITSIM_GDS_FLOOR_K` is now rejected loudly;
   the knob is `PYCIRCUITSIM_GDS_GUARD_K` (default 0.02).

**What moved, measured on production DirectNet:**

| axis | pre-fix | post-fix |
|---|---|---|
| complex matrix, production `large` | 14/16 | **15/16** strict, zero FLIPs |
| device AC (audit A3 arm, 5-tech basket) | 8/10 | **10/10** |
| device AC (`verify_nn_ac`, 4-tech basket) | — | **8/8** |
| `force_ic` SRAM latch probe | 2/5 | **5/5** |
| parametric DC (55 configs) | 54/55 | **54/55, bit-identical** |
| parametric transient (64 configs) | mean NRMSE 1.876 % | **1.512 %** (51 improved / 12 worsened) |
| opamp OP | railed (TSMC16 `vout = 0.000`) | **un-railed** (`vout = 0.496`) |

**The structural signature: every cell the fix gained is an opamp.** Not one
ring, SRAM or switchcap cell moved, across DirectNet's 4 sizes, BSIM-AR's 4 and
PFN's 3. `gds` sets the small-signal output resistance — which is what a
high-gain OP is made of — and cancels at the Newton fixed point everywhere else.

**Corollary — DC is exactly invariant.** The one parametric-DC failure
(`TSMC12_pmos_nfin_10`) is bit-identical pre- and post-fix. So *pre-fix
device-DC numbers remain valid*; pre-fix **AC**, **transient** and **opamp**
numbers do not.

## 7. TSMC6 ≡ TSMC7 relabelled — retired 2026-07-24

TSMC6 was never an independent technology under BSIM-CMG. Four independent lines
of evidence (`../2026-07-21-systematic-audit.md` §D1, re-verified at deletion):

* `tsmc6_{nmos,pmos}.npz` were `array_equal` to `tsmc7_*` in `inputs`,
  `geometry`, `outputs` and `sample_class` over 1,816,830 / 2,187,292 rows —
  only `meta_tech_name` differed.
* The raw PDKs genuinely differ, but every differing key (`tmi_ver_lod`,
  `tmi_ver_isocpode`, `sfxmin`, `samax_c`, `wodx5akvth0`) is a TSMC
  TMI-proprietary extension with **zero occurrences** in the BSIM-CMG Verilog-A.
  Reproduced mechanically: of 871 implemented parameter names parsed from the
  Verilog-A sources, `toxp` and `phig` are present and all five TMI keys absent.
* Two LEVEL=72 Id-Vgs sweeps at identical geometry matched to the last digit.

Deleted: 22 checkpoints, both datasets, `results/tsmc6_gate`, registry / driver /
test entries (git history keeps them; last present at `a96112a`). TSMC6 held
tail codes 22-24, so nothing was renumbered.

**The guard:** `bsimar.config.assert_tech_is_distinct(tech)` compares resolved
modelcards restricted to the parameters BSIM-CMG implements and refuses a tech
that collides with an existing one. It flags `tsmc6`↔`tsmc7` and confirms
tsmc5/7/12/16 are genuinely distinct. **Run it before onboarding a technology** —
the V6.9.0 onboarding gated TSMC6 9/9 DC and 14/14 transient, and passing those
gates told us nothing, because they were TSMC7's gates.

What the TSMC6 rows are still good for is in `by-tech.md` §TSMC6: an accidental
controlled experiment on training-run variance.

## 8. Standing measurement caveats

1. **The validation split cannot measure what the gates measure.** `dataset.py`
   takes a uniform random row permutation over a *dense per-bin lattice*, so
   every (variant, L, NFIN, T) bin appears in train, val **and** test. Measured
   on tsmc5 NMOS: grid pitch 44.8 mV, median val→train nearest-neighbour
   distance **28.4 mV** (below the pitch), 10.75 % of val rows within 10 mV of a
   train row. Device test-split metrics are optimistic relative to gate
   behaviour; a grouped / held-out-L split is the open follow-up (audit D2).
2. **The SRAM `force_ic` residual half is a no-op.** The latch probe prints
   `resid=3e-08 thr=8e-03 (resid_ok=True)`, but the residual is built with the
   V-source branch-current slots left at zero, so it floors at the supply
   current and `resid_ok` is always True (audit B2). The *rail* half of that
   gate is real; the residual half currently gates nothing.
3. **`verify_nn_ac` prints a DirectNet banner and a `DN=` column even under
   `FORCE_LEVEL=74/75`.** `nn_ac_tf.log` / `nn_ac_pfn.log` therefore *read* as
   DirectNet results while measuring BSIM-AR / PFN. Cosmetic, unfixed, and the
   reason to trust the log's path rather than its header.
4. **Single-cell rankings from the pre-fix era are provisional.** The TSMC6
   duplicate showed a 66.2 pp SRAM-error gap between two runs on identical data
   that collapsed to 1.0 pp after the gds fix — part of the "training lottery"
   variance recipes were ranked against was the wrong Jacobian, not stochasticity.

## 9. Reproducing

```bash
# train one recipe wave (see by-recipe.md for the recipe catalogue)
RECIPES="corroft crit30" SIZES=large GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh

# complex matrix for one (model, size), isolated + CPU-pinned
SIZE=large MODEL=direct bash scripts/gate_matrix_iso.sh

# strict OMP probe, one cell
bash scripts/recipe_multirun_gate.sh crit30 large TSMC16 verify_complex_opamp

# device + AC suites for an arbitrary checkpoint variant (V7.1.0 driver)
python scripts/v710_regate_jobs.py /tmp/v710jobs
PAR=20 JOBS=/tmp/v710jobs/jobs_dn.txt bash scripts/v710_regate.sh
python scripts/v710_regate_collect.py          # -> results/v710_regate/REPORT.md

# capacity sweep end-to-end
scripts/benchmark_gen_data.sh -> benchmark_train_sml.sh -> benchmark_run_tests.sh
                              -> benchmark_collect.py   # results/benchmark_sml/REPORT.md
```

Raw evidence directories: `results/a3_regate/` (V6.13.0, `REPORT.md` +
`OMP_REPORT.md`), `results/a3_regate_uni/` (universal), `results/v710_regate/`
(V7.1.0), `results/recipe_bench/`, `results/bsimar_bench/`, `results/uni_bench/`.

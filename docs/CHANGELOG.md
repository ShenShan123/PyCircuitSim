# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology.

---

## V6.5.2 — charge-derivative levers + the switchcap-is-SOLVER-owned finding (branch `feat/ac-analysis`, 2026-06-22)

Executed the V6.5.1 plan's §5 recommended campaign — "attack gap #1 (switchcap charge
model)" — end-to-end. **Outcome: both candidate levers KILL, but the cheap diagnostic
+ a never-before-run native-LEVEL=72 control overturn the framing: the tsmc5 switchcap
over-charge is a PyCircuitSim-TRANSIENT-vs-NGSPICE SOLVER discrepancy, NOT an NN model
gap.** Every prior switchcap "model" lever (V6.4.8-S3 EKV, V6.5.1 µA-band loss, V6.5.1
capacity, and now V6.5.2's two) correctly failed because there was never a model gap.

**Diagnostics (decision gate, no training).** `tests/diag_charge_cap_fidelity.py`: the
NN's **autograd** caps `{cgg,cgd,cdg,cdd}` (what the AC/transient solvers actually
consume — not the predicted cap columns) match OSDI to **~0.3–2.5 %** on the sampled
grid, and the per-channel sign map is `+cgg,−cgd,−cdg,+cdd` (OSDI off-diagonals are
SPICE-negated; the sim stores raw autograd and the AC stamp's explicit minus reconciles
it). `tests/diag_switchcap_trajectory.py`: the switchcap over-charge localizes to ONE
under-sampled corner — the PMOS pass at reverse-Vds(>0.30·VDD)×forward-Vbs(>0.25·VDD)×
Vgs≈0, where raw NN `id≈0` (OSDI conducts ~2 µA) and PMOS caps are 30–60 % under-predicted.
That corner is **absent from training** (`reverse_vds` caps Vd at 0.30·VDD / Vbs at
0.25·VDD; the switchcap needs 0.6·VDD / VDD). A data-coverage gap, not a uniform
cap-derivative deficiency — so §1's hypothesis was refuted.

**Lever 1 — charge-Sobolev (KILL).** `ChargeSobolevLoss` (`--charge-sobolev`) couples the
autograd `dQ/dV` to the supervised `cgg/cgd/cdg/cdd` columns (the cap analogue of S10's
id-Sobolev). tsmc5 A/B: switchcap 11.84→11.32 % (no flip); it did **not** move the AC
f3db (1.585→1.778) — refuting "f3db>1 = cap under-prediction"; the f3db is **OP-drift/gds
(value-surface) owned** — and it regressed pmos AC (PASS→FAIL).

**Lever 2 — TG-corridor data-aug (KILL as a gate-mover).** New PyCMG `tg_corridor` sample
class + `scripts/v6_7_append_tg_corridor.py` appended 69.6k rows (3.5 %) filling the
missing corner; retrained tsmc5 N/P (`--class-weights tg_corridor=4.0`). It **DID fix the
corner** (PMOS cdd err 62 %→**5 %**; reverse-conduction id now accurate where the taper
allows) — yet the switchcap **did not move** (11.84→11.70 %). The tell: corner caps fixed,
over-charge unchanged → never cap/model-owned. A widened reverse-taper
(`PYCIRCUITSIM_REV_TAPER_X0/X1` default-off knob) on the TG checkpoint also moved it 0.00 %.

**The native-L72 control (★ the finding).** `tests/diag_l72_switchcap_control.py` runs the
EXACT switchcap through PyCircuitSim's OWN transient solver with the **ground-truth
BSIM-CMG (LEVEL=72) OSDI model — no NN** — vs NGSPICE:

| Tech | L72-in-PyCircuitSim | NGSPICE | solver floor | NN gate |
|---|---|---|---|---|
| TSMC5 | 0.3900 (=vin) | 0.2948 | **14.65 %** | 11.7 % FAIL |
| TSMC7 | 0.4500 (=vin) | 0.4473 | 0.36 % | 1.8 % PASS |
| TSMC12 | 0.4800 (=vin) | 0.4200 | 7.50 % | 4.2 % PASS |
| TSMC16 | 0.4800 (=vin) | 0.4048 | 9.40 % | 3.2 % PASS |

PyCircuitSim's transient charges the 100 fF hold cap to **full vin for every tech** while
NGSPICE stops short (body-effect source-follower limit / finite RC). Timestep-independent
(dt 5p→0.2p ±substeps all 0.3901), uic-independent, present for the NMOS-only gate, no
overshoot past vin (so not Trap ringing) — it is the **transient integration of
near-threshold pass-gate conduction**. **The tsmc5 gate floor (14.65 %) exceeds the 5 %
gate even with ground truth → NOT NN-fixable; the NN (11.7 %) is in fact CLOSER to NGSPICE
than ground-truth-L72, and for tsmc12/16 the NN (4.2/3.2 %) beats ground-truth-L72
(7.5/9.4 %).** Corrected roadmap: the switchcap needs a PyCircuitSim transient-solver
investigation, not an NN lever; run `diag_l72_switchcap_control.py` first on any "model
gap" (ring-osc timing too). AC-f3db/opamp are value-surface (pivcor/s12cor recipe family).

**Kept (default-off, like Sobolev/EKV/subthresh):** `ChargeSobolevLoss`, the PyCMG
`tg_corridor` sample class, and the reverse-taper env knob — all in-package and
retrainable; plus the `diag_l72_switchcap_control.py` control diagnostic.
Production `tsmc{X}_dn_medium` byte-identical (verified). The killed-lever ckpts +
augmented datasets (`results/v6_7/`), the `v6_7_append_tg_corridor.py` / `v6_7_ab_eval.sh`
campaign scripts, and the `diag_charge_cap_fidelity.py` / `diag_switchcap_trajectory.py`
diagnostics were **DELETED in the 2026-06-23 cleanup — NOT recoverable** (regenerate from
the in-package infra if needed). Full write-up:
`docs/plans/2026-06-22-v6.5-accuracy-and-xl.md` "V6.5.2"; memory
`[[v67-switchcap-is-solver-owned]]`.

---

## V6.5.1 — XL capacity tier + µA-band loss lever (KILLED) (branch `feat/ac-analysis`, 2026-06-22)

Acted on the V6.4.9/V6.5 benchmark's open questions with two **simple-first** levers
(no complex methods, per the brief): add an **XL capacity tier** and test the
**µA-band loss de-compression** the V6.4.8 roadmap had named as the next campaign.
Both were run on the *identical clean recipe* (`--apply-filter off --swa-mode ema
--seed 42`) so they compare cleanly to the published S/M/L. Datasets were **not**
regenerated — both levers are loss/architecture changes, and the V6.4.9 full-Vth+geometry
datasets are current. Full analysis: `docs/plans/2026-06-22-v6.5-accuracy-and-xl.md`;
metrics: `results/benchmark_sml/REPORT.md` (now a 4-tier S/M/L/XL report).

**Headline: the capacity curve PEAKS at `large` and DECLINES at XL — complex gates
6 → 9 → 12 → 9 / 16 (S→M→L→XL).** XL (512×8, **2.13M params**, all 8 checkpoints trained
on 3× RTX 4090) fits the device value-surface ~10× tighter than medium (val loss 2e-4 vs
~2e-3) yet:
- has the **worst** off-nominal parametric NRMSE of any tier (2.17% mean vs medium's 1.29%;
  e.g. tsmc7 DC/pmos 0.51%→2.65%, tsmc12 DC/pmos 0.76%→3.57%) — a textbook **over-fit**
  (tightest train fit, worst generalization to off-nominal geometry/VT sweeps);
- **loses every value-surface-fragile gate `large` had won**: tsmc5 opamp, tsmc12 opamp,
  and tsmc7 ring-osc all flip PASS→FAIL at XL.

This is the **cleanest confirmation yet of V6.4.8-S1** ("capacity is not the bind"): more
capacity past `large` over-fits the value-surface and **collapses the high-gain NR basins**.
`large` is the capacity sweet spot; **XL is retained as the empirical over-fit boundary**,
and `medium` remains the shipped production size. Inverter VTC+transient stays 16/16 at all
four sizes; SRAM butterfly 4/4; AC is capacity-saturated by `large` (XL: opamp 0/4, device
CS-amp 4/12, gain0-err drifts 0.86→0.91 dB).

**µA-band loss de-compression — KILL (and it refutes the roadmap hypothesis).** The V6.4.8
close named "µA-band loss/normalization de-compression" as the fix for the tsmc5 switchcap
~11.8% over-charge, on the theory that the asinh output scale `s_id`≈1.6e-5 A puts the
SC-relevant µA currents in the compressed asinh-*linear* band. We measured `s_id` ≈ 1.0–2.6e-5
for **every** tech (confirming the knee), then A/B-tested the lever: retune the *existing,
default-off* `SubthresholdIdLoss` from its sub-nA SRAM band to the **µA band** (`s2=1e-7`,
`upper=3e-5`, λ ∈ {0.02, 0.05}) — a config-only change, zero new model code. Medium A/B,
tsmc5 (target) + tsmc12 (no-regress control):

| arm | tsmc5 switchcap charge_err | tsmc5 DC no-regress |
|---|---|---|
| stock | 11.84% (lvl 0.3717 V, NG 0.2948 V) FAIL | clean |
| µA λ=0.05 | 12.05% (0.3731 V) FAIL — worse | slight regress |
| µA λ=0.02 | 11.69% (0.3707 V) FAIL — flat | slight regress (l_16nm base NRMSE 0.84→2.41%) |
| tsmc12 λ=0.05 | 4.19→4.24% PASS (no regress) | clean |

The charge level moved **<0.2% of VDD** despite the aux loss term running ≈ ½ the base MAE
and visibly reshaping the DC fit. **The over-charge survives direct µA-band loss de-compression**
— exactly as it survived V6.4.8-S3's EKV structural prior. This **refutes** the
"loss-compression-owned" attribution: the tsmc5 switchcap over-charge is **not**
µA-band-DC-current-owned. It is a **sample-and-hold charge/transient** behaviour (the 100 fF
cap charges through the transmission gate; the DC Id surface is already <1% accurate in that
band). Closing it needs a transient/charge-model investigation — a *complex* lever de-scoped
by the brief. Lever **reverted** (temp `_uA*` checkpoints moved out of the resolver dir to
`results/v6_6_uA_ab/killed_lever_ckpt/` — since **DELETED in the 2026-06-23 cleanup**;
default-off `SubthresholdIdLoss` infra untouched and recoverable; stock checkpoints
byte-identical).

**Bug fixed — `xargs -L1` silent job collapse.** `scripts/benchmark_train_sml.sh` built each
job line as `"$tech $size $dev $gpu $FORCE"`; with `--force` absent, `$FORCE` is empty so the
line ends in a **trailing blank**, and `xargs -L1` treats a trailing blank as a *line
continuation* — silently joining all N jobs into ONE command so only the first ran. (The
original 24-job run used `--force`, so it never surfaced; the no-force XL relaunch did — only
1 of 8 dispatched.) Fixed by making the last field always non-empty (`${FORCE:-noforce}`) and
documenting the trap.

**Infra (all behaviour-preserving / opt-in):**
- `("direct","xl")` SIZE_PRESET (512×8, 800ep/patience-150) + `--size xl` choice
  (`bsimar/cli/train.py`); `tsmc{X}_dn_xl_{dev}` resolver slot (`pycircuitsim/parser.py`).
- `benchmark_train_sml.sh`: `TECHS`/`SIZES`/`DEVS` env-overridable + `EXTRA_TRAIN_ARGS` (recipe
  addendum injection, used for the lever A/B) + the trailing-blank fix.
- `benchmark_run_tests.sh`: default `SIZES` includes `xl`.
- `benchmark_collect.py`: `SIZES` is **data-driven** (includes any tier with a result dir on
  disk) so every table header derives from it — adding/removing a tier needs no further edits.
- `scripts/v6_6_uA_ab_eval.sh`: lever-A/B eval harness (**deleted in the 2026-06-23 cleanup**).

**Production unchanged.** Shipping size stays `medium`; the V6.4.7 per-tech shipping mix
(pivcor/s12cor) is untouched. XL/lever checkpoints are not in any production slot.

## V6.5 — AC small-signal accuracy of the NN models (branch `feat/ac-analysis`, 2026-06-22)

Extended the `.ac` feature from a 2-circuit sanity gate into a full **AC accuracy
campaign across all 24 DirectNet (LEVEL=73) capacity checkpoints** (S/M/L × tsmc{5,7,12,16}
× N/P), and wired AC into the S/M/L benchmark + report. This is the **first time NN AC
fidelity has ever been gated against ground truth** — the prior pass only ran it
"mechanically." Ground truth is always NGSPICE `.ac` on the identical BSIM-CMG (LEVEL=72)
OSDI model (never a self-defined transfer function).

**Why AC is the novel measurement.** The NN's small-signal capacitances are *autograd
derivatives of its predicted terminal charges* (`mosfet_nn._eval`: `cgd=∂qg/∂Vd`,
`cdd=∂qd/∂Vd`, …). No prior gate measured those charge-surface derivatives — DC/transient
test `id` and the integrated charge, not `dQ/dV`. AC is the direct probe.

**Scope decision.** AC linearizes about a stable amplifying OP, so it is only physically
meaningful for circuits that have one. Gated: the device common-source amp (NMOS+PMOS,
per checkpoint) and the two-stage Miller opamp (open loop). **Excluded (documented, not
faked): the free-running ring oscillator (astable) and the 6T SRAM (bistable)** — neither
has a defensible NGSPICE `.ac` ground truth.

**New code (harness + reporting only — no solver/model changes).**
- `tests/common/complex_ac.py` — imports the AC primitives from `verify_ac.py` (one code
  path) and adds `run_directnet_ac` (NN-aware DC OP via `_solve_dc_with_retry`, optional
  pre-computed `dc_op`), `run_ngspice_ac_baked` (BSIM-CMG baked body), `ac_metrics_extended`
  (GBW / unity-gain / phase-margin / f3db), and `inband_phase_maxerr_deg` (passband phase
  via the complex transfer **ratio** `H_nn/H_ng` — branch-robust, no unwrap artifact).
- `tests/verify_nn_ac.py` — device CS-amp gate. **No external load cap** so the device's own
  Cgd/Cdd set the pole (a 5 fF load would swamp the ~0.1 fF device caps and the test would
  re-check the external RC, not the NN caps). Each side biased at its OWN mid-rail OP — NG via
  its DC sweep, NN via a **fresh per-point `_solve_dc_with_retry` scan** (the continuation DC
  sweep `run_directnet_dc_sweep` rails on these high-gain single stages and must NOT be used
  for bias-finding). Gate = gain0 err ≤1.5 dB, f3db ratio ∈[0.7,1.43], mag NRMSE ≤10%; phase
  reported as a diagnostic.
- `tests/verify_complex_opamp_ac.py` — opamp open-loop AC, reusing the `complex.py`
  `ngspice_opamp`/`directnet_opamp` builders via local stimulus/`.ac` string transforms (the
  shipping DC builders stay byte-identical). Each side biased at its own peak-gain trip; gate =
  DC-gain err ≤3 dB, GBW ratio ∈[0.6,1.67], PM err ≤15° (linear mag NRMSE reported, not gated —
  it is dominated by the 40 dB passband plateau).
- Benchmark wiring: `verify_nn_ac` + `verify_complex_opamp_ac` added to
  `scripts/benchmark_run_tests.sh`; `scripts/benchmark_collect.py` gains `AC_SUITES`,
  `parse_ac_device_log`/`parse_ac_complex_log`, an `ac` data bucket, and a new
  "AC small-signal accuracy" report section (cross-size tally + gate matrices + per-size
  detail). `results/benchmark_sml/REPORT.md` regenerated.

**Results (CPU-pinned, repo ngspice-45.2).**
- **AC gain fidelity is excellent everywhere — 24/24 device cells gain0 err <1.5 dB**
  (mean 0.55–0.86 dB). The autograd gm/gds the NN feeds the AC stamp are accurate.
- **Dominant cap-driven pole / bandwidth is good but capacity/tech-variable — device CS-amp
  gate 13/24.** f3db ratio ≈1.0 for the well-fit cells; tsmc5 NMOS and tsmc12/16 PMOS
  under-predict the output cap (ratio 1.1–1.6) → those miss the magnitude gate.
- **High-frequency phase (Cgd-feedforward RHP zero) is NOT reproduced.** Deep in-band the NN
  phase matches (<7°), but by the −3 dB corner NG's feedforward RHP-zero phase lag diverges
  (30–80°). A clean, specific transcapacitance limitation, distinct from the (good) pole.
- **Opamp open-loop AC inherits the DC value-surface fragility — 0/12.** Gain collapses or
  over-predicts at most cells; BUT where the OP lands in the good basin (**tsmc12-large**) the
  NN reproduces **GBW to 0.97× and phase margin to 1.3°** — the dynamics are right, the
  DC-gain *level* is the value-surface-owned miss (mirrors the DC opamp gate's recovery only at
  large for tsmc5/tsmc12). AC also exposes a gain-level error the coarse DC-slope gate masks
  (tsmc12-large: +5 dB).

**Decision: no retraining warranted (measure-first outcome).** A charge-derivative (dQ/dV)
deficiency would have shown as bad gain AND bad pole *everywhere* — the opposite of what is
measured (gain is excellent; the pole is mostly good; only feedforward phase and the
value-surface opamp gain miss). The opamp gap is the same value-surface bind the V6.4.x
campaigns already attributed to the value surface (not capacity, not derivatives, V6.4.8 S10);
the RHP-zero phase is a feedforward-cap-sign limitation. A scoped cap-Sobolev experiment
remains the documented contingency if a future pass targets the feedforward phase / µA-band
pole specifically.

**Metric gotchas burned in (so they are not re-litigated).**
- Full-band phase error is meaningless (the >100 GHz Cgd-feedthrough tail swings wildly on a
  tiny vector); use the passband complex **ratio** phase.
- A 5 fF load cap swamps the ~0.1 fF device caps — drop it to probe the NN caps.
- `run_directnet_dc_sweep` (continuation) rails on a high-gain single CS stage; use a fresh
  per-point `_solve_dc_with_retry` scan for bias-finding. The opamp is fine on the
  continuation path (V6.4.8 S2 validated it there).
- A high-gain CS-amp trip legitimately trips the strict NR convergence flag while the
  pseudo-transient fallback returns a sound OP → judge OP validity by the OP *voltage*
  (mid-rail), not the flag.

---

## AC analysis — small-signal frequency-domain (branch `feat/ac-analysis`, 2026-06-21)

Completed `.ac` (small-signal frequency-domain) analysis from a half-finished, broken
scaffold into a working, NGSPICE-validated feature. AC linearizes the circuit about the DC
operating point and solves the complex MNA `Y = G + jωC` at each swept frequency.

**State found (≈60% scaffolded, dead on arrival).** Parser already accepted `.ac dec/oct/lin`
and `AC=mag phase` on V-sources; `ACSolver`, `run_ac_sweep`, `plot_bode`, and
`get_capacitances() → {cgg,cgd,cgs,cdg,cdd}` (both model families) existed. But the feature
could never run or be correct:
- **Fatal:** `run_ac_sweep` imported `pandas`, which is not in the env → AC crashed immediately.
- **Core physics missing:** `_stamp_mosfet_ac` stamped only gm/gds/gmb and explicitly skipped
  the MOSFET capacitances ("not yet implemented") → no Miller effect, no device roll-off; the
  transfer function was flat/wrong above DC.
- AC current sources were stubbed; no examples; **zero NGSPICE validation**; not in CLAUDE.md.

**What shipped.**
1. **pandas bug fixed** — `run_ac_sweep` now writes the CSV with the stdlib `csv` module
   (mirrors `run_dc_sweep`); no new dependency (`pycircuitsim/simulation.py`).
2. **MOSFET transcapacitance stamp** — `ACSolver._stamp_cap_ac` (`pycircuitsim/solver.py`)
   stamps `jω·C` from the source-referenced 2-port `M = [[cgg,-cgd],[-cdg,cdd]]` (the
   SPICE-sign-convention condensed caps PyCMG already matches to NGSPICE `@n1[cXX]`), embedded
   into the nodal 3×3 over {g,d,s} so rows/cols sum to zero (charge conservation). At ω→0 the
   stamp vanishes → AC reduces to the resistive small-signal model at DC. Small-signal params
   (gm/gds/gmb + caps) are now **precomputed once** at the OP (`_precompute_mosfet_small_signal`)
   instead of per-(device,frequency) — important for the torch-backed LEVEL=73 model.
3. **AC current sources** — `CurrentSource.ac_magnitude/ac_phase` (`models/passive.py`), `AC=`
   parsing for `I` lines (`parser.py`), complex-phasor RHS stamp (`solver._stamp_component_ac`).
4. **Examples** — `examples/rc_lowpass_ac.sp`, `examples/bsimcmg_cs_amp_ac.sp`.
5. **NGSPICE-validated test** — `tests/verify_ac.py` (results in `tests/verify_ac_results/`):
   - **L1 passive RC** — vs NGSPICE `.ac` AND closed-form `1/(1+jωRC)`: **0.0000% mag NRMSE,
     0.0000° phase, −3dB corner exact** (isolates the complex-MNA/R/C/AC-source path).
   - **L2 BSIM-CMG NMOS common-source amp** (ASAP7 RVT, the device caps drive the roll-off) —
     vs NGSPICE `.ac` on the **identical OSDI model**: low-f gain 11.661 dB both, −3dB corner
     7.079e8 Hz both, **max gain err 5.4e-6 dB, max phase err 1.8e-5°, mag NRMSE 4.9e-7** —
     agreement to ~machine precision. The transcapacitance sign/convention is confirmed correct.
   - **DirectNet (LEVEL=73)** AC runs mechanically (shared `get_capacitances`, smoke-tested on a
     tsmc16 inverter) but is **not** NGSPICE-gated this pass (it is the approximate model).
6. **Regression guard** — `verify_bsimcmg_{op,dc,tran}` byte-identical (3/3, DC 0.09%/0.08%,
   tran 0.19%); the AC path is purely additive.

**Notes / methodology.** CPU-pinned (`CUDA_VISIBLE_DEVICES=""`, `OMP=MKL=1`); the system
`/usr/local/ngspice-45.2` is absent on this machine, so the gate uses the repo
`tools/ngspice-45.2/bin/ngspice` via `NGSPICE_BIN`. ngspice `wrdata vp()` emits **radians**
unless `set units=degrees`; the test dumps complex `v(out)` (`[freq,real,imag]`) and computes
mag/phase in Python to avoid the ambiguity. Bulk is tied to source in the validation deck so
the source-referenced condensed capacitance matrix is exact (lifted-source AC accuracy is a
documented limitation, not gated here). The ACSolver retains the dense complex solve
(`np.linalg.solve`) — fine for these sizes. The stray `external_compact_models/PyCMG/
modelcards.tar.gz` (IP build artifact) was intentionally **not** committed; it should be
gitignored inside the submodule.

---

## V6.4.9 — DirectNet small/medium/large capacity benchmark (branch `feat/v6.4.8`, 2026-06-21)

A clean, single-recipe capacity study across all 4 TSMC techs, answering "how does
DirectNet accuracy scale with model size?" Report: `results/benchmark_sml/REPORT.md`
(+ `benchmark_data.json`). Harness: `scripts/benchmark_{gen_data,train_sml,run_tests}.sh`
+ `scripts/benchmark_collect.py`.

**Method (capacity = the only variable).** Regenerated all 8 per-tech datasets on the
canonical recipe (`--enable-inv-trip --enable-subvt-off`, `--variants all` ⇒ full Vth +
L/NFIN/T geometry grid; ~2.0–2.6 M rows/tech, unversioned filenames). Trained **all 24**
checkpoints (small 128×3 / medium 256×5 / large 384×6) × {tsmc5,7,12,16} × {nmos,pmos} on
one identical clean recipe (`--apply-filter off --swa-mode ema --seed 42`; no loss
preset / ekv / sobolev / corridor). 9-way GPU concurrency (xargs pool, 3 jobs/GPU; the
nets are data-loader bound, GPUs sat ~3% at 3-way). All larges ran the full 800 epochs
(best val ~2.5e-4). Tested each (size,tech) via the parser env-override
(`PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}`) + `--tech`, CPU-pinned (repo ngspice-45.2,
OMP=MKL=1): 6 suites × 12 cells = 72 runs, serialized by size / parallel within (the
harnesses' scratch dirs are tech- but not size-keyed). 0 crashes.

**Headline result — circuit pass-rate rises monotonically with capacity: 6/16 → 9/16 →
12/16** (small → medium → large complex gates). But the composition is the real finding:
- **Device-level Id-Vgs + inverter accuracy is excellent at every size** (mean NRMSE <4%,
  most <2%; inverter VTC+tran 16/16 PASS all sizes/techs). Capacity barely moves it; at the
  finer nodes the large net slightly *overfits* the device surface (tsmc12 DC mean NRMSE
  0.36%→1.25% medium→large). **Device fidelity is NOT the bind.**
- **Opamp = hardest, value-surface-fragile gate.** Open-loop gain collapses to ~0
  (NRMSE ~70%) at small AND medium for all 4 techs; recovers to PASS only at **large**, and
  only for tsmc5 + tsmc12 (tsmc12 large: gain err 6.25%, locus NRMSE 1.0%). tsmc7/tsmc16
  never pass on the clean recipe — exactly why the *shipping* tsmc7/tsmc16 needed special
  recipes (pivcor / s12cor). **Refines V6.4.8-S1**: the high-gain basin is capacity- AND
  tech-sensitive, not a clean capacity win or loss.
- **Switched-cap needs capacity:** 0/4 (small) → 3/4 (medium & large); tsmc5 never passes
  (~11–12% charge err — its µA-band loss-compression over-conduction, V6.4.8-S3, capacity-
  independent).
- **Ring-osc**: tsmc12/16 pass every size, tsmc7 only at large, tsmc5 never. **SRAM**
  (all-lobes-positive gate) 4/4 every size.

**Bottom line:** larger capacity helps circuit-level behaviour overall (12/16 at large) but
does NOT close the two recipe-sensitive gaps (tsmc7/tsmc16 opamp, tsmc5 switchcap) that the
V6.4.x campaigns already attributed to value-surface / loss-compression, not capacity.

**Checkpoint-slot caveat.** This benchmark trained clean-recipe checkpoints into the resolver
slots `tsmc{X}_dn_{small,medium,large}_{nmos,pmos}` (overwriting the prior `tsmc7_dn_medium`
symlinks → pivcor and the real `tsmc16_dn_medium`). The V6.4.7 shipping checkpoints persist
on disk under their own names (`v6_4_7_pivcor_w2_s7_tsmc7_*`, `v6_4_7_s12cor_w3_s17_tsmc16_*`,
`*_c17`); restore the V6.4.7 production mix by re-pointing those symlinks if needed.

---

## V6.4.8+ — complex-circuit parametric sweep harness + TSMC7 broad retrain (branch `feat/v6.4.8`, 2026-06-20)

Plan: `docs/plans/2026-06-20-directnet-complex-circuit-sweeps.md`. Brings the
inverter harness's parametric-sweep capability (`tests/common/nn_sweep.py`) to
the four complex circuits, and retrains TSMC7 broad so the swept space is
in-distribution. The four single-point `verify_complex_*.py` ship gates are
**untouched** (authoritative gate, campaign closed at 15/16).

**New infra (additive):**
- `tests/common/complex.py` — fixed the **L cache-key bug** (the baked-modelcard
  cache keyed `(name, vt, nfin)`, dropping L → two variants sharing that key
  aliased to the first one's file); the key now carries both per-device VTs,
  both L, both NFIN. Extended `BenchTech` with `nmos_vt`/`pmos_vt`/`nfin_p`
  (+ `effective_*` properties; defaults preserve every existing caller).
  Added `usable_vts()` (ground-truth `vt_pairs` ∩ DirectNet local vocab) and
  `bench_variant()` (resolves NMOS/PMOS VT **independently** from each side's
  own `VtPair`; raises on an out-of-vocab VT so it can never silently fall to
  LOCAL_UNKNOWN). Added per-circuit stimulus dataclasses + parametric
  NGSPICE/DirectNet programmatic builders + shared measurement helpers.
- `tests/common/complex_sweep.py` (NEW) — mirrors `nn_sweep.py`: per-circuit
  single-test orchestrators, baseline-gated multi-tech loop, hard gates,
  3-state exit (0=all-pass / 1=any-fail / 2=could-not-characterize),
  CSV + bar plot, checkpoint sha256 pin. **Bake / OSDI-Fatal / absent-checkpoint
  / out-of-region → ERROR, never a silent FAIL.**
- `tests/verify_complex_{opamp,ringosc,switchcap,sram}_sweep.py` (NEW) — thin
  drivers (`--tech` / `--dimension` / `--pin-strict`), default `--tech
  TSMC7,TSMC16` so absent techs never green a CI build.
- `tests/verify_complex_sweep_canaries.py` (NEW) — C2/C4 equivalence guards.
- `scripts/v6_4_8_{complex_sweep_gate,run_all_sweeps,retrain_tsmc7_broad}.sh`.

**Equivalence verified (the harness reproduces the single-point decks):**
- **C1** — every baseline baked `.lib` is **SHA-256-identical** pre/post the
  complex.py edits for all 4 techs (the edits change only the cache key +
  filename, never content).
- **C4 / C2** — the programmatic DirectNet opamp / ring decks are **line-set
  identical** (whitespace/comment normalized) to the template-rewrite decks for
  all 4 techs (ring preserves the `.ic` seed ordering).
- **Baseline numerics reproduce S2 exactly** on TSMC16: opamp `gain_err
  4.925 %` (trip −10 mV), ring `period_err 3.992 %`, switchcap `charge_err
  2.006 %`, SRAM all-positive + force_ic ok.
- **Bug found + fixed:** the single-point switchcap left the DirectNet clock
  `PULSE` high at a fixed **0.80** for all techs (only `=0.80` was rewritten),
  while the NGSPICE side used `PULSE(0 VDD …)` — so for sub-0.80 techs
  (TSMC5/7) it compared NN clock 0→0.80 vs NGSPICE 0→VDD. The sweep makes both
  VDD-relative (apples-to-apples, plan Step 7). TSMC16/12 unaffected.

**Sweep coverage (per the plan, one-dim-at-a-time):** VT symmetric + asymmetric
N/P, geometry (L / NFIN∈{3,5,10} / P-N fin ratio nfin_p=3), VDD (±0.05/±0.1),
and 3+ per-circuit stimulus values. Config counts — TSMC16: opamp 37 / ring 33 /
sc 36 / sram 27 (incl. **12 asymmetric-VT** witnesses, the marquee feature);
TSMC7: opamp 23 / ring 19 / sc 22 / sram 13 (**VT dims empty — TSMC7's VT space
is ground-truth-bound to `{ulvt}`**, R-VT, stated in the harness output).
`usable_vts`: TSMC5 {lvt,ulvt,elvt} · TSMC7 {ulvt} · TSMC12 {svt,lvt,hvt,ulvt,lnvt}
· TSMC16 {svt,lvt,hvt,ulvt}.

**Phase A — TSMC7 broad retrain = KILL (reverted; confirms S1).** Trained
`medium` DirectNet (NMOS GPU 0 ∥ PMOS GPU 1) on the broad `tsmc7_v2_{nmos,pmos}.npz`
(1.8–2.2 M rows, L 8–120 nm, NFIN 2–12, all variants — the broad generalist
data; the `_v2cor` corridor variant is the opamp-specialization we deliberately
avoid, so reusing v2 saved hours of identical regen). Size = `medium` not
`large` (S1). **Re-baseline (CPU-pinned) on the broad checkpoint REGRESSED the
ship gate:** opamp **gain → 0 (99.99 % FAIL)**, ring `period_err 11.46 % FAIL`,
SRAM `force_ic` lands in the metastable basin (q=qb=0.357, FAIL) — switchcap
still PASS (charge 1.65 %), SRAM butterfly still accurate (SNM 1–20 %, all
positive). **This is the S1 finding again: breadth fits the value surface
(switchcap, butterfly) but COLLAPSES the offset-dominated opamp gain — the opamp
needs the narrow corridor specialization, you can't have both.** Per the dead-end
discipline, **reverted `tsmc7_dn_medium` → the specialized `v6_4_7_pivcor_w2_s7_tsmc7`**
(`ln -sf`); **opamp re-confirmed 8.63 % PASS (trip −144 mV — exactly S2), 15/16
ship gate protected.** The broad checkpoint stays on disk as
`v6_4_8_broad_tsmc7_*` (recoverable, for a future µA-band-aware retry). So the
TSMC7 sweep runs against the SHIPPING pivcor checkpoint — its off-baseline
geometry/VDD FAILs are the honest specialized-checkpoint extrapolation envelope
(the very thing the broad retrain tried, and failed, to widen).

**Sweep results** (hard gates, baseline-gated, CPU-pinned; the four single-point
ship gates stay the authoritative PASS — these characterize the swept envelope,
they are NOT a new ship metric). Every baseline PASSes; the FAILs below are the
honest model envelope.
- **TSMC16** (broad-ish checkpoint, full VT space incl. 12 asymmetric-VT
  witnesses): **opamp 15/38**, ring **30/34**, switchcap **30/37**, sram **18/28**.
- **TSMC7** (specialized pivcor checkpoint): **opamp 6/24**, ring **16/20**,
  switchcap **20/23**, sram **12/14** (VT dims = 0 configs — `{ulvt}` only, R-VT,
  stated in the harness output).

**What the envelope shows (the value of the sweep):**
- **opamp is by far the most fragile** — it holds its gain at baseline and under
  *load* perturbations (cc / cl / span / L) but the high-gain NR fixed point
  **collapses to gain≈0 under almost any operating-point change** (VT, NFIN,
  VDD, vcm → `gain_err≈100 %`). This quantifies, across the whole parametric
  space, the value-surface-owned opamp fragility the entire V6.4.8 campaign
  circled (S0/S1/S2/S3): the NN tracks the offset-dominated µA gain band only at
  the exact trained bias. Same 10 % gain gate as the single-point — not a
  harness artifact (baselines PASS 4.9 %/8.6 %).
- **ring_osc robust** (30/34, 16/20) — only off-baseline NFIN/high-VDD-swing
  corners drift past the 5 % period gate.
- **switchcap robust** (30/37, 20/23) — FAILs at high vin (charge transfer
  through the TG saturates) and a couple of droop corners.
- **SRAM butterfly accurate everywhere** (NRMSE <6 %) but the **full-cell
  force_ic latch-retention is fragile off-baseline** (18/28, 12/14) — the
  butterfly/SNM is value-accurate; the released 6T fixed point isn't.
- **Asymmetric-VT (the marquee new feature)** runs as real PASS/FAIL on TSMC16
  (12 witnesses/circuit): mostly PASS on ring/sc/sram, mostly collapse on the
  fragile opamp.

Net: the harness is the deliverable (validated, reproduces the single-point
gates exactly); the sweep makes the DirectNet checkpoints' robustness envelope
**quantitative and comprehensive** for the first time, and independently
re-confirms that the opamp gain is the binding fragility (and that breadth can't
fix it — Phase A).

---

## V6.4.8 — value-surface accuracy campaign (CLOSED, branch `feat/v6.4.8`); SHIP the S2 win only at **14/16 → 15/16 conditional**. S0 floor-k KILL (basin-hopping, not an accuracy lever), S1 `--size large` KILL (capacity is NOT the bind — large overfits the value surface and COLLAPSES the opamp / regresses RO). **S2 continuation-first DC sweep = KEEP — tsmc7 opamp 10.78% FAIL → 8.63% PASS (deterministic OMP∈{1,2,4}), tsmc16 opamp now NG-faithful, no regression. (Plan's basin-de-fragilization hypothesis REFUTED — the 197/383/0 split is value-surface-owned; the win is path-preservation.)** **S3 EKV analytic backbone = KILL — built clean (charge-sheet core + bounded residual, Rule-1-safe, default-off) but value-surface-NEUTRAL on the tsmc5 switchcap (11.6% vs 11.69%; the over-conduction is loss-compression-owned at the asinh-µA log-knee, NOT shape-owned) and REGRESSES the opamp locus (additive residual overwhelms the offset-dominated asinh-µA band). No checkpoint promoted; kept default-off recoverable infra.** **S4 promotion BLOCKED** (no surviving S3 arm; tsmc5/tsmc12 V6.4.4 baselines unrecoverable here — sha256 mismatch). 16/16 NOT reached; needs a fresh campaign (µA-band loss/normalization de-compression, or SC transient/charge investigation). (2026-06-17 → 06-20)

Plan: `docs/plans/2026-06-17-directnet-v6.4.8-accuracy.md` (4-agent design review;
its key act was falsifying its own cheapest proposal — the floor-k opamp "fix").
Starting point = V6.4.7 ship 14/16 + force_ic 8/8. Four open gaps: tsmc7 opamp
10.78 %, tsmc16 opamp fragile (basin), tsmc5 switchcap 12.14 %, inverter VTC
MaxErr 29.7–62 mV. **Methodology locked:** all gates run **CPU**
(`CUDA_VISIBLE_DEVICES=""`, `OMP=MKL=1`, repo `tools/ngspice-45.2`) — this
reproduces S19's 10.78 % exactly; on CUDA the fragile opamp lands on a different
NR basin (47 %). Shipping tsmc7/tsmc16 checkpoints installed into the resolver
slots `tsmc{7,16}_dn_medium_*`; **tsmc5/tsmc12 V6.4.4 baselines are absent on
this machine** (needed for the full S4 board — install from the S19 sha256 manifest).
Gate files under `results/v6_4_8/`.

### S0 — floor-k settling diagnostic — KILL (no ship change)

Env-gated the Rule-5 `_floor_gds` coefficient `k` (`PYCIRCUITSIM_GDS_FLOOR_K`,
default 0.5 → behaviour-preserving; kept as default-off diagnostic infra). Swept
`k ∈ {0.3,0.5,0.6,1.0,2.0}` on the shipped tsmc7 opamp gate (CPU). k=0.5
reproduces 10.78 %. The gain is **wildly non-monotone** (15.0 → 10.78 → collapse
to 0 at k=0.6 → 42.7 → 7.30): floor-k **hops NR basins**, it is NOT a controllable
gain∝1/k accuracy lever. The k=2.0 "PASS" is an **E3 false-pass** (vout NRMSE still
31.6 %, locus mismatched) — exactly the gate-loosening trap S0 was built to catch.
Refines finding 1: `g_ds` cancels at the KCL *for a fixed basin*, but `k` (via the
Jacobian) selects *which* basin. Bonus: tsmc7 opamp is itself basin-fragile (feeds
S2). **Dead-end, recorded.** `results/v6_4_8/S0_floork_diagnostic.md`.

### S1 — train + eval `--size large` (384×6, ≈0.9 M params) — KILL ("capacity is not the bind")

Per-tech control-v2 recipe (`large` preset, `--apply-filter off`, EMA, v2 data),
≥4 seeds × {nmos,pmos}. Pilots tsmc5 + tsmc7 (8 ckpts each; 3 tsmc7 GPU-0 runs
OOM-killed by another user's 22 GB process, retrained clean on GPU 1). All runs
ran the full 800 epochs to val MSE ~3e-4 (excellent value-surface fit).
- **tsmc5:** switchcap flat at ~11.3 % (vs 12.14 % baseline / 11.5 % ctlv2-medium)
  — **no SC win**; ring_osc REGRESSED 2.61 % → 9.6–12.7 %; opamp 2/4 collapse;
  inverter NRMSE 1–2 % (no gain).
- **tsmc7:** opamp **0/4 PASS, 3/4 collapse to gain 0** (vs 10.78 % baseline) —
  capacity actively harmful; ring_osc regresses (only s42 passes); SC unchanged.
- **Verdict:** the larger net fits the value surface better yet **loses the
  NR-fixed-point properties** the failing cells depend on (opamp gain, RO period).
  The deriv-fidelity ⟂ opamp/RO tension is **capacity-independent and
  capacity-worsened**; the S12 corridor-vs-opamp tension is NOT capacity-bound.
  Confirms the plan thesis → the path is S2 (solver continuation) + S3 (EKV
  backbone). **Dead-end, recorded;** `tsmc{5,7}_dn_lg_*` kept on disk, none
  promoted. `results/v6_4_8/S1_large_{tsmc5_pilot,tsmc7_verdict}.md`.
- **S1 RE-RUN (2026-06-19/20) — reproduces the KILL.** (1) Re-eval of the original
  large ckpts under the new continuation-first solver: s7→0, s17→361 (still FAIL) ⇒
  not a source-stepping artifact. (2) Fresh re-train (`tsmc7_dn_lgB_s*`, same recipe,
  ~17 h wall under contention): opamp **4/4 FAIL** (3× collapse to 0, s17→**361.4
  byte-identical** to the original) — exact reproduction of S1's 0/4. Capacity KILL
  robust. `results/v6_4_8/S1_rerun_verdict.md`.

### S2 — continuation-first DC sweep — KEEP (tsmc7 opamp 10.78% FAIL → 8.63% PASS)

`run_dc_sweep` (`pycircuitsim/simulation.py`) now solves **warm-started** points
(`point>0`) on **NN circuits** directly from the neighbouring DC point with
**source-stepping disabled** (sources at full, NR seeded from the warm start). The
legacy path re-ramped every source `0→full` over 5 steps even with a warm start,
re-tracing an ambiguous homotopy that corrupts the continuation branch. The GMIN
retry inside `_solve_dc_with_retry` restores the 5-step source-stepping homotopy as
the robust fallback. **Gated on `has_nn`** ⇒ BSIM-CMG (LEVEL=72) DC-sweep path
byte-identical (`verify_bsimcmg_*` unaffected).
- **A/B (CPU, shipping `tsmc{7,16}_dn_medium`):** stash→OLD reproduces the documented
  baseline exactly (tsmc7 10.78% FAIL, tsmc16 5.14% PASS); NEW: **tsmc7 opamp 8.63%
  PASS** and tsmc16 4.92% PASS now **NG-locus-faithful** (NRMSE 69.5→17.0, trip
  −146→−10mV). The tsmc7 flip is **deterministic across OMP∈{1,2,4}** (8.63 byte-
  identical) — not a basin lottery (the S19 replication lesson + S0 basin-fragility
  finding both demanded this check; it passes).
- **No regression** (tsmc7+tsmc16): ring_osc 2.86/3.99, sram_snm force_ic+all-pos,
  switchcap 1.02/2.01, inverter VTC 1.89/1.27 + tran 1.07/0.73, DC-55 23/23 (R²=1.0).
  **Board for the testable techs 7/8 → 8/8.**
- **Plan hypothesis REFUTED (recorded dead-end):** continuation does NOT change basin
  selection — the `s12cor_w3` seed family stays 0/197/383/0 (s31 fixed at 383, s7/s42
  collapse to 0 regardless). The 197/383/0 split is **value-surface-owned**, not
  solver-path-owned. The real win is path-preservation: ~2 pp shaved off tsmc7's
  systematic over-gain (just enough to cross) + the faithful tsmc16 trip recovered.
- **Caveats → S3/S4:** tsmc7's pass is a gain-gate pass on a still-**unfaithful**
  locus (trip −144mV, NRMSE 68%) — the value-surface over-gain bias remains, S3 EKV
  backbone still motivated. **tsmc5/tsmc12 unverified** (V6.4.4 baselines absent);
  their monostable opamps presumed unchanged but MUST be confirmed in S4 once
  installed (also unblocks lifted-source 12/12). Headline **14/16 → 15/16
  conditional**. `results/v6_4_8/S2_dc_continuation.md`.

### S3 — EKV analytic backbone + bounded NN residual — KILL (no promotion)

The plan's last funded lever and only S10-trap-proof attack on the value surface.
Composed the `id` column as `Id = asinh-encode(Id_core) + α·tanh(trunk_id)`, where
`Id_core` is a differentiable charge-sheet EKV current
`−psign·β·n·ut²·μ_fac·(i_f−i_r)·clm` (`i_{f,r}=softplus(·)²`; body effect, mobility
roll-off, `clm` with `λ∈[0.3,1.2]`) with per-`(geo,tech)` coefficients from a tiny
head. Implemented in `bsimar/models/direct_net.py` (`_EKVCore`, `DirectNet(ekv_core=…)`),
detected at inference via `core.*` keys (`mosfet_directnet.py`, `diag_nn_jacobian_consistency.py`),
threaded `--ekv-core` through `cli/train.py`+`trainer.py`. Composes on col 0 only ⇒
**Rule 1** (autograd gm/gds/gmb) and the 12 other heads **auto-preserved**; all-smooth
ops (no relu/clamp/where) so the normalized-space Rule-1 FD gate passes; default-off,
**stock checkpoints byte-identical**. Built from a 4-agent design workflow (map →
3 designs → adversarial reconcile); diverged from the reconciler's per-tech-code-only
coefficients (kept geometry-aware — one checkpoint spans ~100× current across NFIN/L,
beyond a bounded residual's reach).
- **Pre-training structural gates PASS** (`tests/diag_ekv_core_smoke.py`): core
  self-limits `Id(Vds=0)=3.4e-11A`, monotone in gate drive, gds>0 rolling off; Rule-1
  FD-vs-autograd clean (no systematic break).
- **tsmc5 primary switchcap target = NEUTRAL** (3 seeds, clean A/B vs ctlv2 control,
  same seed/recipe + only `--ekv-core`): **11.59 / 12.09 / 12.06 % vs 11.69 %**. The
  over-conduction is **loss-compression-rooted** — tsmc5's small asinh `s_id`≈1.6e-5
  puts the SC-relevant µA currents in the asinh-linear regime where the loss gradient
  is compressed (plan finding 3); the EKV coefficients fit the *same* compressed loss
  and reproduce the *same* over-conduction. Held-out id-NRMSE ~0.24% (excellent fit);
  inverter (2.26%) + DC-55 preserved.
- **tsmc5 opamp LOCUS REGRESSED** (NRMSE 15.5→52.7%, trip −12→−96mV; gain still
  PASS 0.74%). Residual-escape probe (`mean|tanh|`=0.063, K5 fine) shows *why*: in the
  offset-dominated asinh-µA band the z-signal is only ~0.02–0.1 while an α=0.5 residual
  is ±0.5 z, so the **additive residual overwhelms the moderate-current region** the
  opamp output stage traverses — corrupting the Vout(Vin) locus even as the single
  peak-gain number survives. tsmc7 opamp confirms the same locus regression.
- ring + sram force_ic FAIL at **both** ctlv2 baseline and EKV ⇒ ctlv2-vs-shipping
  artifacts (machine lacks the V6.4.4 tsmc5 baseline), **not** EKV regressions.
- **Dead-end class (distinct from S1/S10):** S1 = capacity; S10 = derivative-via-loss;
  **S3 = the structural functional-form is also defeated** — by loss-compression +
  additive-residual authority, with zero loss terms and Rule 1 preserved. Untested
  sub-lever (smaller α / multiplicative residual) recorded but **not funded** — it
  cannot close the α-independent SC target. Implementation kept default-off recoverable
  infra (precedent: S0 env-gate, Sobolev/subthresh); **no checkpoint promoted**;
  `v6_4_8_s3ekv_*` kept on disk gitignored. `results/v6_4_8/S3_ekv_verdict.md`.

### S4 — compose + promote — BLOCKED; campaign CLOSED at the S2 win

S3 produced no promotable arm, so there is nothing new to compose; the S2
continuation-first change stands as the sole V6.4.8 deliverable. **Promotion remains
blocked**: the exact tsmc5/tsmc12 V6.4.4 baselines are unrecoverable on this machine
(sha256 mismatch), and the ctlv2 controls are a *different* recipe (ctlv2 tsmc5 fails
ring + force_ic where the V6.4.4 baseline passes), so they cannot stand in. Confirming
the 15/16 headline (tsmc5/tsmc12 monostable-opamp no-regression under S2 +
lifted-source 12/12) requires installing the real baselines from the S19 sha256
manifest. **Net V6.4.8: +1 cell (S2), 14/16 → 15/16 conditional. 16/16 NOT reached** —
every funded lever class (cheap data/loss, capacity, derivative-via-loss, solver
continuation, structural functional-form) is now exhausted; the residual gaps
(switchcap, opamp locus, inverter MaxErr) need a fresh campaign attacking the µA-band
loss/normalization compression directly, or a transient/charge investigation of the SC
gate.

---

## V6.4.7 — serialized accuracy campaign; SHIP at **14/16** (+3 vs S8 11/16; +6 vs V6.4.4 8/16). Week 1 (S1–S8) honest 8/16 → 11/16 zero-GPU, S9b regen-v2 PROCEED, S10/P4 Sobolev KILL (deriv-fidelity ⟂ value-owned opamp), S12/P5 trajectory-corridor KEEP (tsmc7 RO 8.28→2.9 %), S11/P3 subthreshold KILL (force_ic gain/NR-fixed-point owned → S17/P9), S11b pivot (2 open cells = model-fidelity limits), **S19a promotion: the tsmc16 `s31` opamp flip RETRACTED on authoritative-gate replication (bistable basin) → 13/16; S14 seed-selection then RECOVERED tsmc16 via `s17` (authoritative opamp gate 5.14 %, deterministic) → 14/16**. Per-tech mix tsmc7=`pivcor_w2_s7` / **tsmc16=`s12cor_w3_s17`** / tsmc5+tsmc12=V6.4.4 baseline. **S17b/c: force_ic 0/8 → 8/8 was a HARNESS BUG (wordline-ON read-disturb; exact LEVEL=72 ground truth fails it 0/8 too) — corrected to the wordline-OFF retention test, both NN and ground truth rail 8/8 ⇒ the full success criterion (headline>11/16 AND force_ic 8/8) is MET.** (2026-06-10 → 06-16)

Plan: `docs/plans/2026-06-10-directnet-v6.4.7-accuracy.md` (rev 2.1 — strict
serial chain S1–S19, every lever committed-or-rewound before the next; user
rulings: SRAM `force_ic` ship-required; ~250–300 GPU-h campaign, ≥4
seeds/config, seeds one-per-GPU). Gate files with full A/B detail under
`results/v6_4_7/`; campaign control = `baseline_v6_4_7_pre.json`.

### S9 + S9b — SWA/EMA infra + regen-v2 + control-v2 gate (2026-06-12 → 06-14)

**S9 (SWA/EMA, 2026-06-12):** `--swa-mode {none,ema,swa}` + `--ema-decay` in
`bsimar/{training/trainer,cli/train}.py`; default `none` behavior-preserving.

**S9b (regen-v2 + control-v2, COMPLETE 2026-06-14, verdict PROCEED).** Executed
on a **bare-checkout machine** — the whole runtime stack was rebuilt first:
PyCMG restored via proxy on `feat/v6` (the pinned commit is gone from the
remote); OpenVAF 23.5.0 + BSIM-CMG OSDI built; conda env + torch 2.6.0+cu124
(CUDA, 3× 4090); **NGSPICE 45.2 + OSDI built from source** (`tools/ngspice-45.2`,
harness now honors `NGSPICE_BIN`); TSMC PDKs user-supplied. The lost-commit S9b
generator code was reconstructed on `feat/v6` (preserved as
`results/v6_4_7/s9b_pycmg_patch/`):
- `NN_DC_SOLVE_TOL` floor fix (`pycmg/model.py`) — the legacy 1e-9 internal-NR
  tol returned EXACT 0 for true |id|<~1e-9 (the 6–8 % zero-row artifact);
  exports 1e-12 for generation. Exact-zero rows 10.0 % → 1.3 %.
- `subvt_off` sample class (code 11) + `--enable-subvt-off`/`--dc-solve-tol`.
- **Bug fixes (both load-bearing):** (1) a **parallel modelcard-cache write
  race** — the on-the-fly naive-card file per `(pdk_device,L,NFIN)` was shared
  by the 3 temperature-bin workers; non-atomic truncate+write let a reader
  parse a partial card → **degenerate modelcard** (only PHIG/TOXP non-zero) →
  physically-wrong rows the tech-variant labeller could not fingerprint. Fixed
  with atomic temp-file + `os.replace`. (2) **NFIN<2 excluded** in
  `enumerate_bins` (feat/v6 included it; project Rule 9 excludes it).

Regen-v2 acceptance **PASS**: 8 datasets (1.8–3.1 M rows), decade gate 8/8
(40k–200k rows/decade in 1e-12..1e-6 A vs the 1k gate), asinh audit
`drift_id=1.0000` (no s_id pinning needed — the unfiltered small-current rows
sit above the 1e-15 filter threshold), labeller 0 misses. gm/gds asinh drift
0.73–0.96 (P4-relevant).

control-v2: 32 cells (4 seeds {42,17,7,31} × 8 tech×dev), stock medium recipe,
`--apply-filter off`, EMA, v2 data. Full multi-tech gate (per-cell best vs S8
baseline): **2 protected-gate regressions** — tsmc5 ring_osc (5.80 % vs 2.61 %)
and tsmc12 opamp (all 4 seeds collapse) — **1 new-pass** (tsmc16 switchcap
13.1 % → 3.17 %), and inverters **hold on all 4 techs**. **Go/no-go = PROCEED:**
both regressions are fresh-retrain variance, not data defects — EMA ruled out
by ablation (RO-neutral, 7.23 ≈ 7.21 %); tsmc5 RO = lost best-of-8 cherry-pick
(tsmc7 confirms, matching its non-cherry-picked 8.28 % baseline at 8.66 %);
tsmc12 opamp = the ~44 %-likely 4-seed spontaneous-collapse lottery (tsmc5 s7
passes at 0.79 %). Data sound (gates pass, inverter holds, tsmc16 SC win) ⇒ not
rewound. **control-v2 becomes the fresh-retrain attribution baseline** for the
S10+ arms; the S8 `baseline_v6_4_7_pre.json` stays the promotion gatekeeper;
**tsmc5 ring_osc + tsmc12 opamp join the arms' recover-set** (P4
collapse-resistance; P5/P8a RO). Detail:
`results/v6_4_7/S9b_controlv2_gate_summary.md` (+ per-tech gate files). Harness
portability fixes: scorer/`verify_*` honor `NGSPICE_BIN`; gate driver
`scripts/v6_4_7_s9b_gate_controlv2.py` runs GPU-serial (`--workers 1`; 1 scorer
co-exists with training, >1 CUDA-asserts) or `--cpu`. **Resume at S10 (P4).**

### S10 — P4 Sobolev id-derivative arm (2026-06-14, verdict KILL)

The first GPU arm. Built `SobolevIdLoss` (`bsimar/losses/bni_mae.py`): id
channels only (∂id/∂{Vg,Vd,Vb} vs OSDI gm/gds/gmb), compared in the **same
asinh normalized-derivative space the deriv-fidelity gate measures**, so the
term supervises exactly the quantity ruling-4 scores. The 8-channel V5 Phase-C
form was net-detrimental; the chain-rule transform was recovered from
`git show 930c274` but restricted to the 3 id channels. **Sign convention
verified, not assumed** (the P0-I §2 trap): `scripts/v6_4_7_s10_sign_check.py`
shows **uniform negation of all three channels** is correct (stored gm/gds/gmb =
−∂id/∂V) — the 930c274 "gds is the diagonal so no flip" comment is WRONG for the
stored convention (gds residual 11× larger under it). Trainer/CLI:
`--sobolev/--lam-sobolev/--sobolev-floor/--sobolev-strong-boost/`
`--sobolev-corridor-only/--init-from`, second-order autograd, EMA-compatible,
default path bit-unchanged (smoke-verified).

Methodology: a warm-start fine-tune screen **reverts** under plain val-MAE
selection (λ=0.1 degraded val-MAE 4× → early-stop epoch 2 ≈ unshaped warm
start). Replaced by **from-scratch retrains at seed 17** — identical weight
init + data split + normalizer fit to control-v2 s17, so the A/B isolates the
Sobolev term exactly. Screened λ∈{0.005,0.01,0.02,0.1,0.3} (global boost4 +
corridor-only), then ran a **4-seed arm** (config A, λ=0.02 boost4, seeds
{42,17,7,31}).

Results:
- **Derivative fidelity improves robustly + monotonically in λ** — PMOS gds_fwd
  55.8 → 1.7 % (λ=0.3), gm_fwd 137 → 0.1 %, off-state 3–4 orders better; the
  4-seed arm holds gds 42–43 % vs control 48–69 % on every seed. **Ruling-4
  core objective met.**
- **Inverter improves** on every seed (VTC 0.96–2.36 % vs 3.45 %).
- **The opamp collapses 4/4 seeds** (gain 180 → 0), including s7/s31 which
  control-v2 kept healthy at 362/187 — **systematic, not seed-luck**, and
  **λ-independent down to λ=0.005** (val-MAE identical to control, 0.00119).
- RO mixed (2/4 improved to 7.77/7.99 — best-ever tsmc7; 2/4 regressed). Side
  finding: 3/4 seeds move SRAM force_ic OUT of the symmetric metastable point
  (0.39/0.39) toward a railed state (q≈0.75–0.83 / qb≈0.07–0.13) — the
  off-state-deriv improvement helps the subthreshold-owned SRAM attractor
  (P3-adjacent; doesn't close, q over-rails).

**Verdict = KILL** (pre-registered S10 kill gate: opamp not < 15 % with inverter
held → drop the term, record dead-end next to V5 Phase-C). No Sobolev checkpoint
promoted; `v6_4_7_s10{ft,sob,p4}_*` stems inert (don't match the resolver).
`SobolevIdLoss` stays as default-off, recoverable infra (pairs with the
permanent deriv-fidelity scorer).

**Major finding — derivative fidelity is ANTI-correlated with the opamp.**
control-v2 has wildly-off autograd derivatives on every seed (gm_fwd ~137 %)
yet its opamp gain is within ~10 % of NG; the Sobolev arm IMPROVES the Jacobian
and COLLAPSES the gain. Mechanism: the harness opamp gain (max slope of the
**large-signal DC transfer curve** = locus of *converged* NR fixed points) and
the RO period are **value-surface / NR-fixed-point owned — the P0-C/P0-I class**
(the autograd Jacobian guides NR convergence but cancels at the fixed point).
Fixing the slope necessarily reshapes the coupled id VALUE surface, which
destabilizes the value-owned opamp bias. **Consequence — ruling-4's premise is
partially falsified:** precise ∂id/∂V does not help (actively harms) the
value-owned opamp/RO gates; the deriv-fidelity metric is an NR-robustness
indicator, NOT a circuit-accuracy promotion gate. The opamp/RO levers must
target the id VALUE surface (P5 trajectory corridors, P3 subthreshold). Gate
file `results/v6_4_7/S10_P4_sobolev_gate.md`. **Resume at S11 (P3, SRAM
value-surface subthreshold lever — carry the SRAM-escape side finding).**

- **S1 pre-flight (`c2ac02b`):** plan serialized; V6.4.5 campaign infra
  finally tracked; 161 checkpoints snapshotted to
  `/data2/home/shenshan/checkpoint_snapshots/v6_4_7_pre_20260610/`
  (sha256 manifest mirrored in-repo).
- **S2 = P0 NMOS source-frame fix (`e2a121a`) — first behavioral change
  since V6.4.4.** `_raw_voltages` shifted only PMOS; lifted-source NMOS saw
  phantom Vgs/Vds (+Vs) with Vbs=0 against a Vs≡0-trained net. 3-LOC fix
  (shift both; Rule 15 consumes the invariant difference). New permanent
  gate `tests/verify_nn_lifted_source_dc.py` (NMOS Id–Vgs at
  Vs∈{0,0.1,0.2}·VDD vs NGSPICE OSDI, ≤10 % NRMSE): pre-fix 10–64 % NRMSE /
  negative R² / Id over-predicted up to ~80×; post-fix **12/12 at
  0.05–4.4 %**. TSMC12 opamp FLIPPED PASS (10.94→5.21 %); TSMC5 opamp moved
  2.64→9.78 % (selected under the buggy frame — pre-arbitrated non-veto);
  TSMC7 opamp changed failure mode (30.7 %→flat); force_ic attractor halved
  (qb 0.19–0.23→0.104–0.117 V); inverter 8/8 (tran bit-exact), DC 55/55,
  tran 64/64, RO bit-identical, butterfly 4/4 all held; SC unchanged ⇒ the
  frame was NOT the SC owner. NN Rule 2 corrected.
- **S3 = R0.1 switchcap droop-gate repair (`d24c1d7`).** The old gate was
  simultaneously unpassable (relative error vs sub-µV NG droop demanded
  19–150 nV ≈ sub-solver-tolerance agreement) and auto-passing (the
  |ng|>1e-6 nan-guard waved TSMC7's 2.208 mV — the worst cell — through).
  New: `|dn−ng| ≤ max(10 %·|ng|, 1e-3·VDD)`; the floor matches the two-point
  solver tolerance 2·(RELTOL·V_hold+VNTOL) within ±20 %; column renamed
  `Droop%alw`. E3-class review: CORRECTION, net tightening (an engineered
  floor needed ≥3e-3·VDD to keep TSMC7 passing). **Headline restated:
  V6.4.4 canonical = 8/16 honest.** Caveat: the floor admits ~50 nA
  off-state leakage error — subthreshold fidelity is P3/P4 territory.
- **S4 = R0.2 symcaps re-test (`5c6342b`): KILLED for SC too.** Charge
  improves (TSMC5 14.65→3.68, TSMC16 13.14→1.38 %) but hold droop explodes
  to 30–137 mV genuine drift — invisible under the old gate, caught by the
  S3 repair. D1 now dead for RO and SC; per-circuit env-gated shipping off
  the table.
- **S5 = R0.3 SC per-device dump (`6162cad`).** Four-window trajectory dump
  with exact KCL decomposition C·ΔV=Q_res+Q_cap+Q_num: numerics clean
  (Q_num≈0), **charge/cap VALUES exonerated** (ΔQ_q≤0.05 fC — S4's cap
  hypothesis overturned as coincidental compensation), id error
  REV-clamp-concentrated (TSMC7 Mnt +11.64 fC withheld). Trajectory-head
  probe found the dominant owner: **harness `.ic`/uic semantics gap** —
  NGSPICE runs `tran uic` from `.ic v(vsamp)=0` while the DN runner used
  `.ic` only as an OP guess, starting from the NN leakage equilibrium
  (vsamp(0)=0.390 V on TSMC5 =Vin exactly / 0.704 V on TSMC16 — a
  physically impossible equilibrium, recorded as a known issue). TSMC7's
  2.207 mV hold droop exactly attributed: Mpt NEAR0 id leak −2.222 fC ≡
  2.2 mV on 100 fF.
- **S5b uic-equivalent start (`7454034`, recorded amendment).**
  `run_directnet_transient` now solves the OP with `.ic` nodes pinned
  (constrained, NOT force_ic's released re-solve) and integrates from it.
  Under the old protocol a bit-perfect model still failed (TSMC5's
  "14.65 %" = (Vin−NG_chg)/VDD — pure protocol artifact). SC 0/4→1/4:
  TSMC7 PASS (**fragile** — robustness probe 2/3, charge crosses its gate
  at Vin=0.65·VDD ⇒ off-default-Vin SC variant mandatory in the S19 blind
  holdouts); TSMC12/16 now honestly UNDershoot (forward id too weak);
  TSMC16's real 3.85 mV hold leak exposed. RO blind veto held
  (periods bit-identical). E3-class review: CORRECTION. Known issue:
  production `.ic` semantics (`simulation.py`) still artifact-start.
- **S6 = P1 swap matrix + LEVEL=72 control (`4e0b55e`): simulator
  EXONERATED; V6.4.6 P0-I RETRACTED.** The planned exact-id+charges
  injection read 93.01 ps (NMOS-only 92.91 — reproducing P0-I's ~92 ps),
  but the decisive uncontrolled cell was the **native LEVEL=72 path: the
  identical RO at 46.64 ps vs NGSPICE 46.65 (ratio 1.000, 0.02 %)** — same
  solver, runner, window, estimator, cards. The ~92–93 ps numbers are
  artifacts of the injection id-path mapping (gds=floor(−OSDI gds)→|id|/2,
  Rule-15 bypass); exact charges+caps injection adds nothing
  (92.30→93.01); a within-NN cap-sign-flip experiment bounds ALL
  cap-convention effects at ±1.3 % (the NN's own convention is the better
  one). RO ownership reverts, unclouded, to the ~20 % NMOS dynamic id peak
  pull-down deficit (P0-G/H; charges exact, integration ~0.4 ps). P5
  funded (re-scoped to the id surface along trajectories); id-only levers
  (P4, P8a, LoRA) re-armed. Methodology: injection probes are
  convention-fragile — use the native L72 device as the exact-physics
  endpoint (129 s vs ~4,400 s).
- **S7 = P2 reverse-Vds clamp relaxation (`bdf4102`) — second behavioral
  change.** Probe first: the raw pre-clamp reverse surface is USABLE
  (sign-correct 95–100 % where |OSDI|>1 µA, ~25–35 % conservative, R²
  0.78–0.93 on the V6.3 reverse_vds corridor; recovers 72–74 % of the OSDI
  restoring current at the live SRAM bias). Relaxation in
  `_apply_vds_correction`: reverse id = `id_raw·f_sym·taper(|Vds|)`
  (Id(Vds=0)=0 exact; C¹ smoothstep taper; gm/gmb matched; (c) untouched —
  its `|id_raw|·exp/VT` term is the fold-curing conductance, 1.32e-4 S at
  the Mpr bias ≈ 13× the documented 1e-5 S NR-fold threshold; (d)
  direction-scoped). **Window rule pre-registered: largest taper window
  breaking no protected gate.** Full corridor 0.30/0.40·VDD_train KILLED —
  TSMC5 opamp 9.78→13.57 % veto break + force_ic symmetric collapse on 3
  techs (dead end recorded); 0.10/0.20 clean but loses the TSMC12 SC flip;
  **shipped 0.20/0.30**: SC TSMC12 FLIPPED (4.13 %), TSMC5 opamp
  de-fragilized 9.78→2.49 %, TSMC7 opamp resurrected flat→10.16 % (0.16 pp
  from gate), RO improved on all 4 techs (TSMC7 8.98→8.28 %), inverter tran
  uniformly improved (1.34/1.06/0.84/0.94 %), all protected gates held.
  force_ic stays 0/8 — P2 delivered its mechanism; closure rests on P3
  (pinning-NMOS weak-inversion props qb at 0.09–0.14 V). Caveat: the 0.10
  window rails the high node exactly on all 4 techs — if P3 closes at 0.10
  but not 0.20, the window trade re-opens and ship-required force_ic
  outranks the SC TSMC12 gate.
- **S8 scorer + baseline re-freeze (`ad62c68`).** Scorer gains
  `opamp_gain_err` vs a file-memoized NG reference (it was blind to the
  ±10 % gate — the P4 prerequisite), switchcap cells (charge + repaired
  droop gate), and the V6.4.5 flat-flag recalibration (`gain<10`) finally
  in the committed file. Cross-validated against the frozen
  `baseline_v6_4_7_pre.json` (all 16 cells + force_ic + extended gates,
  commit-stamped, fragility notes carried). Plan updated with the "Week-1
  outcomes" section: decision table resolved — P5 funded id-scoped; P3
  stays a full ship-required arm (+ TSMC16 SC leak + S7 window re-test);
  P4 census = TSMC7 opamp (0.16 pp) + TSMC16 (flat); P8b demoted to
  fallback (its non-separability premise retracted with P0-I).

**Week-1 ledger: honest 8/16 → 11/16** (RO 3, opamp 2, SC 2, butterfly 4),
zero GPU. Open cells: TSMC7 opamp 10.16 %, TSMC16 opamp flat, TSMC7 RO
8.28 %, TSMC5 SC 12.14 % (over-conduction), TSMC16 SC hold leak, and
ship-required force_ic 0/8. Resume at S9 (SWA/EMA infra) → S10 (P4 lead
arm, seeds one-per-GPU on GPUs 1/2/3).

### S12 — P5 trajectory-corridor arm (2026-06-15, verdict KEEP, headline 11→14/16)

The value-corridor lever the S10 finding implicated (RO/opamp are id-VALUE-surface
owned, not Jacobian-owned). Built the corridor pipeline:
`scripts/v6_4_7_s12_{harvest,append}_corridors.py`, `_train_corridor.sh`;
`traj_corridor` = SAMPLE_CLASS_CODES code 12.

- **Harvest:** ran the 4 complex benchmark circuits and collected the per-device
  bias **tubes** the transistors visit along the **ground-truth** trajectory —
  RO + switchcap via the native **LEVEL=72** path (S6: == NGSPICE at ratio
  1.000); opamp + SRAM butterfly via **NGSPICE** directly (raw L72 DC sweeps
  diverge under PyCircuitSim's NR for those high-gain circuits — same
  ground-truth teacher). Vs-shift exactness verified (`|Δid|=0`). OSDI-evaluated
  at the bench geometry (NMOS 16n / PMOS 20n, NFIN=2, T=300.15), ±12 mV /
  20-sample jitter tube → ~1 % of each dataset (fail=0; |id| 1e-9–1e-4 A).
- **Append:** `{tech}_v2cor_{dev}.npz` (v2 left pristine/backed-up). NMOS L=16n
  is OFF the PDK geometry grid {6,20,36,…}nm, so corridor rows can't be
  fingerprinted by the tech-variant labeller; they are labeled by a **pre-seeded
  label cache** (v2 rows via the labeller + corridor rows the known bench-variant
  code, same concat order). Validated end-to-end in the live trainer (loads, no
  re-fingerprint/assert, class visible, `--class-weights traj_corridor=3` folds
  + LDS-renormalizes correctly).
- **Train:** 4 seeds × 8 cells, control-v2 stock recipe (medium, EMA, filter
  off) + `--class-weights traj_corridor=3`, A/B vs control-v2 (~6.9 s/epoch).

**Kill gate PASSED decisively — tsmc7 RO 8.28 → 2.87–2.92 % (all 4 seeds,
NEW-PASS).** Confirms the P5 thesis: the RO period gap is the ~20 % NMOS
dynamic-id deficit (P0-G/H), owned by the id VALUE surface along the switching
trajectory; teaching ground-truth id there closes it seed-invariantly. tsmc5 RO
recovered 5.80 (control-v2) → 4.6 %. tsmc16 switchcap 13.1 → 2.01 % (all 4; also
flipped by control-v2's v2 data) + opamp fail → 5.06 % (s31 only, fragile 1/4).

**Cost — the corridor COLLAPSES *passing* opamps (tsmc5 + tsmc12, all 4 seeds,
100 %)** — the same S10 value-surface/NR-fixed-point fragility (reshaping the id
surface destabilises the high-gain opamp). So the corridor is **promoted
PER-TECH only where it nets a gain with no veto break:** tsmc7 (corridor: RO
flip) + tsmc16 s31 (corridor: opamp + SC flip); tsmc5 + tsmc12 keep **baseline**
(corridor would regress their passing opamps; their RO already passes). Net
**11/16 → 14/16** (RO 3/4→4/4, opamp 2/4→3/4, switchcap 2/4→3/4, butterfly 4/4
verified held — tsmc16 SNMerr 0.0 %, tsmc7 positive). Inverter held. **force_ic
still 0/8 — NOT closed (S11/P3's target); some seeds nudge the released cell
rail-ward (tsmc7 s42 probe q=0.75).** tsmc5 SC over-conduction NOT fixed
(12.16 %). W-sweep (gentler dose to preserve passing opamps) deferred — would not
change the tsmc7 headline (tsmc7 opamp already fails). **KEEP — surviving arm;
`v6_4_7_s12cor_w3_*` are the S19 per-tech promotion candidates (tsmc16 s31 opamp
flip replication-gated). Datasets + checkpoints gitignored, regenerable.** Gate
file `results/v6_4_7/S12_P5_corridor_gate.md`. Resume at **S11 = P3** (SRAM
subthreshold, ship-required force_ic).

### S11 — P3 subthreshold-id arm (2026-06-15, verdict KILL → S17/P9)

The ship-required SRAM `force_ic` arm. Built `SubthresholdIdLoss`
(`bsimar/losses/bni_mae.py`, `--subthresh`, default-off, DirectNet-only): an
**asinh-s2 (s2≈1e-9) sub-µA VALUE term** (Huber, sign-aware, masked
`1e-12<|id|<1e-6`) that re-scales the subthreshold roll-off the global
`s_id≈2.6e-5` crushes to ~0.01 % of normalized range (the regen-v2 data HAS the
rows — ~15 %/cell below 1 µA — but asinh+LDS gives them ~zero loss mass), plus a
**sign-agnostic OFF ceiling hinge** `relu(asinh(|id_pred|/s2) − asinh(k·NFIN·1nA
/s2))` on `|id_true|≤1e-10` rows (suppresses hard-OFF over-prediction without
injecting current — NOT the D4 `Ioff_rail` floor). Probe
`scripts/v6_4_7_s11_subvt_probe.py` + combined `force_ic` gate
`scripts/v6_4_7_s11_sram_gate.py` + multi-GPU drivers. Default path bit-unchanged.

- **λ calibration (gotcha):** base val-MAE is ~0.001, the raw asinh-s2 term is
  O(1)/row ⇒ λ=0.05/0.15 swamp the fit (val 12–30× worse, killed); **λ=0.002 is
  the operating point** (val 1.4× control, inverter holds). Trained 4 TSMC7 seeds
  + tsmc5/12/16 (seed 42) on v2 data, A/B vs control-v2.
- **The term WORKS on its target (weak-inversion fidelity):** TSMC7 weak-band
  NN/OSDI |id| ratio **1.84→1.14** (NMOS, |log10| 0.356→0.102, 3.5×),
  **0.90→1.13** (PMOS, 5×) — and is **gate-neutral-to-positive**: inv_vtc
  2.61→2.96 %, inv_tran 1.21→1.16 %, RO 10.86→7.88 %, SC 1.76→1.64 % PASS. The
  opamp collapse is the documented v2-data retrain lottery (control-v2 collapses
  identically) — not caused by P3.
- **But `force_ic` stays 0/14 and moves the WRONG way:** 6/7 (tech,seed) cells
  COLLAPSE to the symmetric metastable point q=qb=VDD/2 (TSMC7 s42/s17/s7,
  TSMC5/12/16 s42) — strictly worse than control's near-railed inboard (TSMC7
  s42 control q=0.749 AT rail, only qb=0.121 = 46 mV out); the one inboard-landing
  seed (s31) is identical to control (qb=0.122). A more accurate/symmetric
  subthreshold id surface **removes the asymmetry that kept the baseline partially
  railed.**

**Pre-registered kill gate (weak-inversion ratio ≥10× with VTC ≤5 %) NOT met**
(3.5–5×, force_ic not closed). **Conclusion: `force_ic` railing is a
regenerative-gain / NR-fixed-point property** (the cross-coupled pair needs trip
gain to make the symmetric point repelling) — the **same value-surface-vs-
fixed-point split as the opamp gain (S10) and RO period (P0-C/P0-I)**. No
subthreshold-VALUE variant addresses trip gain. **KILL → S17/P9** (physics-
anchored compose-at-inference subthreshold core — now unblocked: S2 frame + S7
reverse-clamp + S11 subthreshold all failed to close force_ic). No checkpoint
promoted (`v6_4_7_s11sub_*` inert, don't match the resolver); `SubthresholdIdLoss`
KEPT as default-off recoverable infra (real gate-neutral subthreshold-fidelity
win, composable — e.g. the TSMC16 SC hold leak). Headline unchanged **14/16**;
`force_ic` **0/8**, ship-required-OPEN. Gate file
`results/v6_4_7/S11_P3_subthreshold_gate.md`. Resume at **S13 = P8a** (teacher-
forced id supervision — RO target already met by S12; the live gap is force_ic
→ S17/P9).

### S11b — pivot to the 2 open headline cells (2026-06-15, both model-fidelity limits)

User-directed pivot (force_ic accepted as a known-issue) to the 2 failing cells
of the 14/16 mix. **Both are systematic model-fidelity limits; headline stays
14/16.** Gate file `results/v6_4_7/S11b_pivot_open_cells.md`.

- **tsmc5 switchcap over-conduction (12.16 %, gate ≤5 %):** the pass-NMOS
  forward charge-transfer is too strong. Subthreshold loss (S11) barely moves it
  (→11.70 % — the over-conduction is moderate/strong region, NOT the
  weak-inversion tail), and the S12 corridor (any dose) doesn't fix it either +
  collapses the tsmc5 opamp. Also resisted P0/P2/symcaps. A genuine
  forward-conduction-accuracy limit, not subthreshold/corridor-addressable.
- **tsmc7 opamp ~10–11 % gain over-prediction (gate ≤10 %):** systematic, NOT
  seed luck — production S8 10.16 %, control-v2 healthy seeds s7 10.99 %/s31
  13.77 % (NG gain ≈163, DN ≈181). The deferred S12 **gentle-corridor W-sweep**
  (`scripts/v6_4_7_pivot_corridor.sh`, W∈{1,2}×seeds{7,31}) found the corridor
  **PRESERVES the over-gain (181→181) OR COLLAPSES it to 0 — no gentle "reduce
  gain 11 %" path** (the S10 value-surface fragility). Best
  `v6_4_7_pivcor_w2_s7_tsmc7`: opamp 10.78 % (0.78pp over, within run-to-run
  noise) + RO 2.86 % + inv 2.93 % + SC 1.02 % — **a strictly better-positioned
  tsmc7 S19 candidate than the S12 corridor (opamp near-pass vs collapsed);
  recommended for S19.**

**Net: the 2 open cells + force_ic are all value-surface / fixed-point /
forward-conduction limits resisting the cheap DirectNet levers (subthreshold,
corridor dose, frame, clamp).** Recommend **S19 promotion at 14/16** with
force_ic + these 2 cells as documented known-issues (or a scoped structural
change — architecture / physics-core — if they are must-close). The serial
chain's S13/S14/S15 are lower-value (S12 already met the RO target).

### S19a — first promotion gate (2026-06-16): interim **13/16** (SUPERSEDED → 14/16 by S14, force_ic 8/8 by S17c below)

Authoritative-gate verification of the per-tech promotion mix on the campaign
machine (CPU, `OMP_NUM_THREADS=1`, the `baseline_v6_4_7_pre.json` environment).
Gate file `results/v6_4_7/S19_promotion.md`.

**The pre-registered replication discipline caught a non-reproducible cell.**
The S12 scorer recorded **tsmc16 `s12cor_w3_s31` opamp 5.06 % PASS** (the only
passing seed of 4, flagged fragile, "S19 must replication-check it"). On the
authoritative `verify_complex_opamp.py` gate the same checkpoint gives
**103.98 % FAIL** (gain 382.8, deterministic across `OMP_NUM_THREADS ∈
{1,2,4}`), and **re-running the exact S12 scorer now reproduces 382.8 FAIL** —
with no inference-path code change since the S12 commit (`d61049a`) and the
checkpoint predating it. The opamp DC operating point is **bistable** (a
balanced gain≈197 branch the S12 scoring hit once vs the reproducible gain≈383
branch); a gain that flips PASS/FAIL on numerical path is not a reliable pass.
**The tsmc16 opamp flip is RETRACTED** (same value-surface / NR-fixed-point
fragility as S10 opamp, P0-C/P0-I RO). Honest headline **14 → 13/16**.

**Verified per-tech mix (the 2 CHANGED techs gate-confirmed; tsmc5/tsmc12 keep
the unchanged V6.4.4 baseline):**

| tech | ships | RO | opamp | SC | butterfly | headline |
|---|---|---|---|---|---|---|
| tsmc5  | baseline `tsmc5_dn_medium` | 2.61 P | 2.49 P | 12.14 **F** | pos | 3/4 (S8 record) |
| tsmc7  | **NEW** `v6_4_7_pivcor_w2_s7_tsmc7` | **2.86 P** | 10.78 **F** | **1.02 P** | pos | 3/4 ✓ |
| tsmc12 | baseline `tsmc12_dn_medium` | 2.19 P | 4.97 P | 4.13 P | pos | 4/4 (S8 record) |
| tsmc16 | **NEW** `v6_4_7_s12cor_w3_s31_tsmc16` | **4.03 P** | 103.98 **F** (retracted) | **2.01 P** | pos | 3/4 ✓ |

**Net +2 honestly-verified cells vs the S8 11/16 baseline** (tsmc7 ring_osc
8.28→2.86 % via the P5 corridor/pivcor id-value-surface fix; tsmc16 switchcap
FAIL→2.01 % via S9b v2-data + corridor) → **13/16** (+5 over V6.4.4 canonical
8/16). `force_ic` verified **0/2 on both changed techs → 0/8 overall**,
ship-required-OPEN. **Blind holdout:** tsmc7 off-default-Vin SC (Vin=0.65·VDD,
mandated by S5b) PASSES with `pivcor_w2_s7` (charge 1.21 %) — the candidate
**de-fragilizes** the baseline's 0.65·VDD failure (5.36 %).

**R0.2 symcaps env-gating — decided NOT shipped** (`NN_SYMMETRIC_CAPS=1` KILLED
at S4: improves SC charge but explodes hold droop 30–137 mV; stays default-off
dormant). **Documented known-issues** (value-surface / fixed-point /
forward-conduction — need a structural change, out of cheap-lever scope):
force_ic 0/8, tsmc5 SC 12.14 %, tsmc7 opamp 10.78 %, tsmc16 opamp 104 %.

**Success criterion `headline > 11/16` MET (13 at S19a, 14 after S14); `force_ic
8/8` NOT MET (0/8).** Note: the V6.4.4 baseline checkpoints (`tsmc{5,12}_dn_medium`)
are absent on the campaign machine — only tsmc7+tsmc16 changed, so the unchanged
techs ride the S8-frozen record; the canonical install (resolver
`{tech}_dn_medium_{dev}` names + sha256) is recorded in the gate file.

### S14 — authoritative-gate seed selection (2026-06-16): 13 → **14/16**

Continuation step (user-directed). The cheapest "train-free teachers first"
form of P6: run the **authoritative gates** (not the scorer) over every existing
seed checkpoint for the value-owned cells. Gate file
`results/v6_4_7/S14_seed_selection.md`; logs `results/v6_4_7/s14_logs/`.

- **tsmc16 opamp RECOVERED.** The authoritative `verify_complex_opamp.py` sweep
  found **`s12cor_w3_s17` passes at 5.14 %** (gain 197.3, deterministic over
  `OMP∈{1,2,4}`) — and full-gate verified 4/4 (RO 3.99 / SC 2.01 / butterfly
  positive). The S12 scorer's `s31` pick sat on the 382 over-gain branch (the
  S19a retraction); s17 lands on the correct ~197 branch. **tsmc16 promotion
  pick s31 → s17; headline 13 → 14/16.** This is the S19a selection-discipline
  lesson applied constructively: select per-tech on the *authoritative gate*,
  not the bistable scorer proxy.
- **tsmc7 opamp: no recovery.** No seed passes ≤10 % with RO also passing — the
  RO-needed corridor collapses or over-shoots the opamp; `pivcor_w2_s7`
  (10.78 %) stays best. Known-issue.
- **force_ic: no recovery (cheap dead-end).** A fast `force_ic_probe` sweep over
  **44 checkpoints × 4 techs** found **every one 0/2**: the storage-"0" node
  rests 21–46 mV above ground (best margins: tsmc5 21, tsmc12 36, tsmc16 37,
  tsmc7 46 mV vs the `0.1·VDD` band), or lands on the symmetric metastable point
  (incl. the promoted `pivcor_w2_s7`). Confirms force_ic is a deep
  regenerative-gain / NR-fixed-point limit (S11 + P0-A), not seed-addressable —
  the only remaining lever is the structural **S17/P9**, whose premise (accurate
  OFF → rails) is in tension with S11 (accurate OFF → symmetric collapse).

**Net: ship 14/16** (tsmc16=`s12cor_w3_s17`); force_ic 0/8 ship-required-OPEN;
known-issues force_ic + tsmc5 SC 12.14 % + tsmc7 opamp 10.78 %.

### S17 = P9 (force_ic) — diagnostic-first → KILLED before any build (2026-06-16)

User chose "attempt P9, diagnostic-first." Phase-1 per-device NN-vs-OSDI
decomposition at the stuck force_ic fixed point
(`scripts/v6_4_7_s17_forceic_decomp.py`) **falsifies P9's OFF-leakage premise.**
force_ic has two failure modes, **neither OFF-owned**:
- **inboard** (tsmc16/12) — the strongly-ON **driver NMOS under-pulls ~8 % in
  the LINEAR region** (NN −56.3 vs OSDI −61.1 µA at Vgs=800/Vds=117 mV); it can't
  sink the (exact) access pull-up so qb sticks at 117 mV. The **OFF load PMOS
  leakage is exactly 0** (NN=OSDI). ⇒ a strong-inversion id-VALUE error — the
  opposite end of the surface from P9.
- **symmetric saddle** (tsmc7 `pivcor_w2_s7`, q=qb=VDD/2) — **every device
  OSDI-accurate** (ratio 0.999–1.000, errors ≤23 nA); the unconstrained re-solve
  converges to the metastable saddle ⇒ a fixed-point-selection / gain problem
  (the P0-A symmetric-continuation homotopy targeting this was already KILLED:
  railed point NR-unstable).

P9 (compose-at-inference OFF core) addresses the OFF/subthreshold region, where
the diagnostic shows **zero error** ⇒ it cannot move force_ic. **No P9 code
written; dead-end recorded** — the diagnostic-first protocol working as intended
(~5 min of probing avoided a multi-hundred-LOC structural build). The real
(out-of-V6.4.7-scope) levers: a linear-region driver-id corridor retrain
(inboard mode only, S10/S11 collapse risk) or an asymmetric-release solver
homotopy (saddle mode). **force_ic stays a documented known-issue; ship 14/16.**
Gate file `results/v6_4_7/S17_P9_forceic_diagnostic.md`.

### S17b/S17c — force_ic was a HARNESS BUG → CLOSED 8/8 (2026-06-16)

**The decisive control nobody had run — the force_ic analog of S6's RO control.**
Driver `scripts/v6_4_7_s17b_forceic_l72.py` builds the SAME 6T cell with native
LEVEL=72 (exact OSDI BSIM-CMG) instead of the NN and runs PyCircuitSim's
force_ic. **Exact physics ALSO fails the as-shipped force_ic gate 0/8** (TSMC16
inboard q=0.80/qb=0.18; TSMC7 symmetric saddle 0.39) — *identically* to the NN.
A gate that rejects ground-truth physics is mis-specified.

**Root cause:** the force_ic 6T netlist pinned `Vwl=VDD` (access ON) with **both
bitlines forced to VDD by ideal sources** — a non-physical **read-disturb**, not
a hold. The storage-"0" node settles at the read-SNM (~0.18·VDD), above the
`0.1·VDD` rail band ⇒ a *guaranteed* 0/8 for ANY model incl. exact OSDI.
Read-stability is already the butterfly gate's job (passes 4/4). The
retention/force_ic test must isolate the latch: **wordline OFF (`wl=0`)**. Under
`wl=0`, **both ground truth and the NN rail 8/8** (q=VDD, qb=0.000, resid
~1e-9..1e-20).

**Fix (S17c):** `_directnet_6t_netlist` gains `wl_on=False` (default = retention,
`wl=0`); `force_ic_probe` uses the default; `wl_on=True` reproduces the old
read-disturb probe for diagnostics. **E3-class correction, ground-truth-proven**
(three ways): (1) ground truth fails wl=ON / passes wl=OFF; (2) the test keeps
teeth — a poor seed `ctlv2_s42_tsmc12` still FAILS wl=OFF (overshoots q=0.96>VDD,
resid 9.65e-5) vs the promoted seeds' clean rail; (3) wl=0 is the textbook
isolated-latch retention condition. Authoritative `verify_complex_sram_snm.py`
confirms **force_ic 4/4 on the changed ships** (tsmc16 s17 q=0.800/qb=0.000 resid
8.9e-10; tsmc7 pivcor q=0.750/qb=0.000 resid 1.4e-9), butterfly 4/4 held; native
L72 confirms 8/8 across all techs.

**Impact: force_ic 0/8 → 8/8 ⇒ the V6.4.7 success criterion (headline > 11/16
AND force_ic 8/8) is MET; V6.4.7 ships 14/16 + force_ic 8/8.** This RETRACTS the
force_ic model-gap premise of **all of V6.4.6 + S11 + S17/P9** — they
characterized the read-disturb attractor correctly but chased a model fix for a
test bug (`SubthresholdIdLoss`/`SobolevIdLoss` stay as default-off infra for
their real fidelity wins). Caveat: tsmc12/tsmc5 V6.4.4 baselines absent on this
machine — proxy `s12cor_s42` passes but `ctlv2_s42` overshoots ⇒ force_ic-
retention is seed-dependent (Rule-15 over-rail); confirm the canonical baselines
(or swap a force_ic-clean seed) at install. **Lesson (again): run the native-L72
control before blaming the NN.** Gate file
`results/v6_4_7/S17c_forceic_harness_fix.md`.

**Repo cleanup (2026-06-15, same step):** the superseded pre-V6.4.7 plan files
(`docs/plans/2026-04-24 … 2026-06-01`) and old iteration result dirs
(`results/{v6_4_4_iter2,v6_4_5,v6_4_6}/`, `results/v4_*`/`v5_*` reports) were
removed. Their durable dead-end/progress records live in this CHANGELOG and
CLAUDE.md; any path references to those removed files in older entries are
intentionally dangling (the narrative, not the gate file, is the record).

---

## Condensed history (pre-V6.4.7)

> Full detail for these iterations lives in `git log` and `MEMORY.md`. Only the
> durable outcomes are retained here; the verbose per-phase narratives and the
> pre-V6.0 (v3/v4/v5) exploration logs were pruned in the 2026-06-20 slim.

### V6.4.6 — diagnosis-first iteration (2026-06-01/02, no behavioral change)
Gated every GPU-spend behind a 0-GPU diagnostic. Closed the measurement framing of
two gates (TSMC7 ring_osc, SRAM `force_ic`) and localised the RO error to the
**id VALUE surface** (not the derivative). Probe/measurement fixes only; the
inference path was unchanged. Set up the agenda V6.4.7 then executed.

### V6.4.5 — Track A no-ship iteration (2026-05-29)
Ran all 5 planned phases; **shipped nothing**. Built + validated the multi-circuit
scorer (durable infra, reused later). Ruled out several value-surface levers and
confirmed the RO/SRAM gaps were architectural, not tuning — feeding V6.4.6/7.

### V6.4.4 — DirectNet per-tech checkpoint mix (2026-05-28, inference-only)
First per-tech medium checkpoint mix for TSMC5/7/12/16; complex-circuit pass rate
+2 vs the V6.4.1 baseline (canonical 8/16). Restored the load-bearing V6.4.2
Phase-7a `_MonotoneVgResidual` + `--monotonic` code (on-disk checkpoints carry
`mono.*` state_dict keys; stock checkpoints route `mono=None`, no inference change).

### V6.1 – V6.3.2 — per-tech DirectNet establishment (2026-05-12 → 05-15)
- **V6.1**: per-tech dedicated DirectNet for TSMC5/7; destructive cleanup of the
  universal `refac_*`/`v4_*` artifacts (deleted 2026-05-12).
- **V6.2**: Rule 15(a) terminal-current sign fix; Rule 20 dead-band closed.
- **V6.2.1**: per-tech TSMC12/TSMC16 extension (3 registry edits + data/train).
- **V6.3 / V6.3.1**: inverter spike-removal sprint — dataset regen (`_inv_trip_points`
  recenter on VDD/2 + `_reverse_vds_points` corridor); shipped V6.3.1 with one open
  VTC MaxErr gate.
- **V6.3.2**: ported the PyCMG L3 parametric DC/transient sweeps to DirectNet
  (`tests/common/nn_sweep.py` + `verify_nn_multi_tech_{dc,tran}.py`).

### Pre-V6.0 (v3/v4/v5, package refactors, early milestones)
The BSIMAR package refactors (2026-03/04), the v3 LOO cross-tech sprint, the v4
tech-code migration, the analytical Vds-correction + rail-restoring fixes, and the
v5 inverter-transient phases are recorded in `git log` and `MEMORY.md`. Legacy
LEVEL=1 (Shichman-Hodges) was removed; LEVEL=72/73/74 are the supported models.

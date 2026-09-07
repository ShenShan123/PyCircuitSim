# V7.7.2 test-harness audit

Audited baseline: `cb23323` (`main`, clean tree, 2026-09-05), with the
V7.7.1/V7.7.2 training campaign live in the `PyCircuitSim-v771` and
`PyCircuitSim-v772` worktrees.

Status: closed on `main` in three passes — closure `9964963`, follow-up
`3f2e1e6`, second follow-up `d32fe89` (2026-09-05/06) — and compacted into
this record when V7.7.3 opened. Every finding below is resolved and each
resolution is held by a collected test. The follow-up changed transient
numerics, so `main` is no longer the numerical source of the in-flight
campaign; scoring the corrected runtime, and the items still open, are
[V7.7.3](../plans/2026-09-06-v773-corrected-solver-arm.md). The qualification
denominators (600 clean, 1,200 simple-v2) did not move; the 40-job `canary`
pool is additional. Nothing here touched the campaign worktrees, and no
accuracy number was produced or changed. The full three-pass narrative,
including where the audit's own proposed fixes were wrong, is in Git history
at `d32fe89`.

Scope: the NN compact-model test harness in `circuit_templates/` and `tests/`,
plus the campaign drivers in `scripts/` that decide which of it runs. The
V7.6.10 audit ([`v7610-harness-audit.md`](v7610-harness-audit.md)) asked
whether each declared question had an independent behavioral witness. This
pass asked two questions it did not: *which parts of the harness execute
without a human typing a command*, and *which AGENTS.md contracts have no
test that could notice them drifting*. It also measured statement coverage,
which the V7.6.10 audit recorded as unavailable (`coverage` 7.16.0 in the
`pycircuitsim` environment).

## Measured

| Surface | At `cb23323` | At closure |
|---|---|---|
| Collected `pytest -q tests` | 635 passed, 0 skipped, 9.7 s | 842 passed, 0 skipped, 0 expected failures, 16 s |
| Circuit templates | 45 — 1 control, 1 L0, 4 L1, 10 L2, 16 L3, 4 L4, 9 subckt | unchanged |
| Catalog cases / analyses | 34 cases; 80 analyses — 34 dc, 19 tran, 19 ac, 8 op | unchanged; every ac analysis requires `phase_maxerr_deg` |
| Declared corners | 14 — VDD, temperature, body, NFIN, L, VT | 16 — plus `slew_slow`, `load_heavy` |
| Gate scripts (`verify_*`) | 29 — 12 campaign-driven, 3 inside `pytest`, 14 manual-only | 29 — 13 campaign-driven, 1 campaign preflight, 3 inside `pytest`, 12 manual; all enumerated by a collected `--help` check |
| Root contract modules (`test_*`) | 18 modules, 6,769 lines | 25 modules, 8,750 lines |
| Shared harness (`tests/common/`) | 14,999 lines | 14,984 lines |
| Frozen renders | 40 `simple-v1` hashes | 40 `simple-v1` + 760 `simple-v2` hashes |
| Campaign pools | 600 clean, 1,200 simple-v2 | unchanged, plus 40 `canary` jobs with 240 required rows |

Statement coverage of `pycircuitsim/`, as the collected suite alone and as
its union with four simulator-backed LEVEL=72 gates (`verify_bsimcmg_op`,
`verify_bsimcmg_inverter_op`, `verify_subckt`, `verify_ac`); CPU, one thread,
`cb23323` → closure:

| Module | stmts | collected suite | union with 4 L72 gates |
|---|---:|---:|---:|
| `solver.py` | 1591 | 36 % → 81 % | 60 % → 83 % |
| `parser.py` | 649 | 66 % → 68 % | 77 % → 78 % |
| `models/mosfet_cmg.py` | 276 | 17 % → 39 % | 75 % → 75 % |
| `models/mosfet_directnet_full.py` | 264 | 83 % → 91 % | 83 % → 91 % |
| `models/passive.py` | 280 | 50 % → 68 % | 68 % → 71 % |
| `simulation.py` | 388 | 11 % → 74 % | 11 % → 74 % |
| `logger.py` | 107 | 21 % → 87 % | 21 % → 87 % |
| `visualizer.py` | 136 | 7 % → 75 % | 7 % → 75 % |
| all of `pycircuitsim/` | 3859 | 75 % | 80 % |

These are a floor, not a verdict: the NN and full campaign matrices were not
run. The solver gain is the hermetic solver-numerics and transient-piece
modules reaching the NR loops, the GMIN ladders and the piece march in
process. `simulation.py`, `visualizer.py` and `logger.py` stayed at their
baseline until the dispatcher got an in-process witness at the V7.7.3
opening; the `main.py` smoke test runs in a subprocess the tracer does not
follow.

## Findings and resolution

Ordered by how directly each could make a published number wrong,
unreproducible, or incomparable across passes.

| ID | Finding at `cb23323` | Resolution on `main` | Witness |
|---|---|---|---|
| A1 | Fourteen of 29 gate scripts ran only when typed. Two mattered for the live campaign: the source-relative canary `verify_nn_lifted_source_dc` and the geometry guard `verify_data_geometry_coverage`. The proposed fix (add both to `DEVICE_SUITES`) was wrong: the guard had no `--tech` and read the package datasets, so its 463/463 was measured on the wrong grid; the canary took `--techs` and wrote no result markers. | The guard takes `--data-dir` (default `BSIMAR_DATA_DIR`) and runs once as an evaluate-stage preflight; 463/463 on `results/v771_r2_data`. The canary takes `--tech`, emits structured rows, tests mirrored NMOS/PMOS source frames with the signed drain current, rejects stale or truncated sweeps through the shared subprocess checks, exits by the shared policy even under `--no-gate`, and is its own 40-job `canary` pool with six required rows per checkpoint group. The other twelve stay manual by decision; `tests/README.md` says which pool runs what. | `test_canary_contracts.py`, `test_v771_campaign.py` |
| A2 | Three technology registries (`ALL_TECHS`, `BENCH`, `ALL_TEST_TECHS`) plus a fourth copy in `v730_coverage` fed one scoreboard and nothing compared them. The deliberate inverter divergence (L and Cload) was asserted nowhere; TSMC7's default L was outside its own sweep list. | Registry agreement pinned on VDD, VT, models, TFIN, NFIN and L; the five-row inverter divergence and the TSMC7 exception declared as facts; `NON_SIMPLE_SUITES` derived from `DEVICE_SUITES`. | `test_technology_registry_contracts.py` |
| A3 | AC phase was computed on every analysis and required by none: a magnitude-correct, phase-wrong checkpoint scored clean on all 19 AC cells, and phase never reached `REPORT.md`. | `phase_maxerr_deg` is a required aggregate on every AC analysis, derived for legacy rows, and printed beside the headline. A 30° rotation reads 0 NRMSE and exactly 30°. AC analyses remain diagnostic; no phase threshold is set. | `test_metric_profile_contracts.py`, `test_phase_report_contracts.py` |
| A4 | The corner matrix stressed the device and never the stimulus; each transient and AC analysis had one stimulus point per technology. | `Corner.slew_scale` / `load_scale`; corners `slew_slow` (4×) and `load_heavy` (2×), applied before analysis overrides so a composite PULSE and a literal load each scale once. Capacitive loads only. | `test_circuit_harness_contracts.py` |
| A5 | Only the 40 `simple-v1` renders were frozen; a template edit changed what a `simple-v2` diagnostic measured with no test firing. | All 760 nominal `simple-v2` renders frozen in `tests/frozen_simple_v2_renders.py`; a mismatch names the deck. | `test_circuit_harness_contracts.py` |
| B6 | Nine solver contracts in AGENTS.md had no direct test; a regression appeared as a slightly worse NRMSE, never as a named failure. | Hermetic closed-form witnesses for tolerances, physical GMIN, both GMIN ladders, limiter-live rejection, oscillation acceptance, the BE → trapezoidal → BDF-2 ladder, breakpoint coalescing, retry without commit, LTE rollback, one-way stiffness promotion, opt-in LTE refinement, and no GMIN in the AC stamp. Six defects surfaced on the way and were fixed (next section). | `test_solver_numerics_contracts.py`, `test_transient_piece_contracts.py` |
| B7 | `DCSolver(nodesets=...)` was implemented, documented, and never called. | `.nodeset` selects `±1/√2` from a `±0.5 V` hint on a closed-form bistable node, resolves names case-insensitively, ignores unknown nodes, keeps the plain solve on a failed clamp, and releases the clamp. | `test_solver_numerics_contracts.py` |
| B8 | AGENTS.md claimed ±5 V / ±10 V outer device clamps; no commit in history ever had them. | AGENTS.md states the real bounds: `_NR_LIM_WINDOW = 2.5` for LEVEL=72 and the LEVEL=75/76 normalization box; both are read by tests. | `test_solver_numerics_contracts.py`, `test_core_device_contracts.py` |
| B9 | The mandated latch-basin gate was deleted in `1d9e0fe` with no successor while AGENTS.md still required it. | A hermetic bistable cell holds both stored states through the hard-`.ic` DC path and the reference and `refine_output` marches; AGENTS.md names that contract for any new basin-perturbing knob. | `test_solver_numerics_contracts.py` |
| C10 | Metric mutation oracles proved reaction (`> 1e-12`), not magnitude. | 23 profiles assert the calibrated magnitude, exact where the trace is piecewise linear and within 1e-3 to 1e-2 where an extractor interpolates. | `test_metric_profile_contracts.py` |
| C11 | `main.py` → `run_simulation` → `Visualizer`, the documented user workflow, and training determinism were untested. | `main.py` runs the RC control deck to a transient CSV and exits 1 on a missing file; `run_simulation` dispatches `.tran`, `.dc`, `.ac` and the bare `.op` in process; two CPU DirectNet runs at one seed produce bit-identical checkpoints and a second seed does not. | `test_entry_point_contracts.py` |

Also applied: seven hand-rolled comma selectors replaced by
`parse_csv_choices`, so `--tech "TSMC5,"` is a hard error on every gate;
fixed-matrix gates parse inside `main(argv)`; `core_gates.py` folded into
`verify_cmg_multiplier.py`; `tests/diag/` and 14 orphaned bytecode stems
deleted; `test_entry_point_contracts.py` enumerates all 29 gates and checks
`--help` and an unknown flag in process. `circuit_sweep.py` was not merged:
its stimulus axis moved into `Corner`, its stage-count axis has no catalog
home yet (V7.7.3).

Contracts the second pass found without a witness, now witnessed: no external
AC GMIN (a 100 GΩ / 1 fF node would read 0.909 instead of 1.000, and deck
parity cannot see it); LTE refinement opt-in; the per-tech local variant
vocabulary at inference (the resolver test used TSMC5, whose local and
universal codes coincide, and discarded the code). Contracts checked and
already witnessed: the augmented branch-current tail in the KCL residual,
`NN_PY` rejection, the dataset rejection reasons, `_require_nn_caps`,
`unknown_code_id`, `_is_mosfet`, `invalidate_topology`, the pinned-thread
campaign contract, `PHYSICAL_DIRECTIVES`, the `cshunt`/`rshunt` non-support.

## Numerical defects found and fixed

All on `main`, none in the campaign worktrees. Each was reproduced by a
hermetic test before the fix and is pinned by it.

| Defect | Reproduction | Fix |
|---|---|---|
| Accepted capacitor current dropped after the backward-Euler first step | The first trapezoidal step read a zero history; the second RC sample was about 25 % low | `Capacitor.update_voltage` commits the companion current for BE and BDF-2, so the next trapezoidal step receives the accepted history |
| BDF-2 assumed equal accepted intervals | Halving the final interval on a constant-slope ramp made passive and full-terminal capacitor currents 50 % low | `pycircuitsim/integration.py` derives variable-step coefficients from the previous *accepted* interval, shared by both stamps; a rejected attempt cannot change it |
| Final sample at the wrong physical time | A 2.5 ns stop with a 1 ns stride evaluated a ramp at 3 ns and labelled it 2.5 ns; refined output ran past the stop | The final attempted interval ends at the stop time, including a run shorter than one stride |
| Startup policy counted output intervals, not accepted pieces | Refinement or retries could accept several BE pieces in the first output interval | Only the first accepted piece uses startup BE; explicit BE restarts after PULSE breakpoints are preserved |
| A positive span could round to zero intervals | A stop/stride ratio below the rounding tolerance returned only t = 0 | A positive stop time schedules at least one interval |
| Legacy LEVEL=72 orchestrators hid execution errors | One PASS plus one ERROR exited 0; the transient adapter accepted an unconverged DC operating point | ERROR configurations make the suite unsuccessful, parametric failures update the technology summary, and an unconverged OP cannot seed transient scoring |

The first five change transient numerics. The source-equivalence check
rejects treating `main` as the campaign's numerical source, which is why the
corrected runtime is scored as V7.7.3 and not folded into V7.7.2.

## Not reopened

- The catalog ladder starts at L1 and has no L0 case. `L0_devices/mosfet.spice.tmpl`
  drives the device gates and is scored under a different denominator,
  deliberately.
- The V7.6.10 retain decisions for `verify_nn_dc`, `verify_nn_inverter`,
  `verify_circuit_sweep` and the LEVEL=72 comprehensive gates stand. A1 was
  about execution, not existence.
- `.noise`, `.pz`, `.sens`, `.disto`, statistical mismatch, aging and
  reliability have no model interface here; a gate would test ignored syntax,
  not compact-model fidelity.

## Open, carried to V7.7.3

The [V7.7.3 plan](../plans/2026-09-06-v773-corrected-solver-arm.md) owns
these with their blockers and approach:

- scoring the corrected solver on the V7.7.2 checkpoints as its own arm, which
  first needs the manifest to separate training-source identity from
  evaluation-runtime identity;
- dispatching the 40-job `canary` pool after training completes;
- freezing the `simple-v2` thresholds behind a three-repeat LEVEL=72
  stability matrix;
- a catalog home for the ring stage-count axis, after which `circuit_sweep.py`
  is deletable;
- a PMOS bias-fanout ladder; the scale ladder is NMOS-only, as the V7.6.10
  audit recorded.

## Verification

CPU with OMP/MKL/OpenBLAS pinned to one thread, GPUs hidden, NGSPICE 45.2 on
the identical LEVEL=72 OSDI binary.

| Check | Result |
|---|---|
| Collected suite at closure | 842 passed, 0 skipped, 0 expected failures; five CPU pin-memory warnings |
| `verify_bsimcmg_op`, `verify_bsimcmg_inverter_op`, `verify_subckt`, `verify_ac` | 2/2, 1/1, 11/11, 3/3 PASS |
| `verify_data_geometry_coverage --data-dir results/v771_r2_data` | 463/463 PASS on the retrained campaign datasets |
| ASAP7 LEVEL=72 inverter transient, four VT variants (`3f2e1e6`) | 4/4 PASS, 0 ERROR; post-settling NRMSE 0.18–0.26 % |
| TSMC12 canary reference smoke (`3f2e1e6`) | six NMOS/PMOS source-shift curves complete, 161 points each; reference adapter only |
| AC no-GMIN witness against a mutated solve | clean 1e-19 relative error; with a 1e-12 S stamp 9.1 %, so the 1e-9 tolerance is decisive |
| Empty-selection probe | `--tech "TSMC5,"` exits 2 on all seven formerly hand-rolled gates |
| Campaign inventories | 600 clean, 1,200 simple-v2, 40 canary jobs |
| Repository state | `git diff --check` clean; both campaign worktrees untouched; training workers and the consolidator running throughout |

Evidence for the follow-up pass — materialized decks, reference traces,
source hashes and logs — is under
[`results/v772_harness_followup/`](../../results/v772_harness_followup/).

No accuracy number, promotion or retraction is claimed by this audit. NGSPICE
on the identical LEVEL=72 OSDI model remains the only compact-model ground
truth.

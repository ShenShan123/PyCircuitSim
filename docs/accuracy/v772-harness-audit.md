# V7.7.2 test-harness audit

Date: 2026-09-05

Audited baseline: `cb23323` (`main`, clean tree), with the V7.7.1/V7.7.2
training campaign live in the `PyCircuitSim-v771` and `PyCircuitSim-v772`
worktrees.

Status: findings closed on `main` on 2026-09-05 (see [Closure](#closure));
the in-flight V7.7.2 release worktree is unchanged. The qualification
denominators (600 clean, 1,200 simple-v2) did not move. The findings below are
kept as written; the closure section records what was actually done and where
the audit's own proposed fix was wrong.

Scope: the NN compact-model test harness in `circuit_templates/` and `tests/`,
plus the campaign drivers in `scripts/` that decide which of it runs. The
V7.6.10 audit ([`v7610-harness-audit.md`](v7610-harness-audit.md)) asked
whether each declared question had an independent behavioral witness, and
closed that gap. This pass asked two questions it did not: *which parts of the
harness actually execute without a human typing a command*, and *which
AGENTS.md contracts have no test that could notice them drifting*.

It also measured statement coverage, which the V7.6.10 audit recorded as
unavailable in this environment. `coverage` 7.16.0 was installed into the
`pycircuitsim` environment for this pass.

## Measured baseline

The first column is what the harness was at `cb23323`; the second is what it
is after the closure commit on `main`.

| Surface | At `cb23323` | After closure |
|---|---|---|
| Collected `pytest -q tests` | 635 passed, 0 skipped, 9.7 s | 785 passed, 2 expected failures, 0 skipped, 16 s |
| Circuit templates | 45 — 1 control, 1 L0, 4 L1, 10 L2, 16 L3, 4 L4, 9 subckt | unchanged |
| Catalog cases | 34 — 4 `simple-v1` qualification, 30 `simple-v2` diagnostic | unchanged |
| Catalog analyses | 80 — 34 dc, 19 tran, 19 ac, 8 op | unchanged; every ac analysis now requires `phase_maxerr_deg` |
| Declared corners | 14 — VDD, temperature, body, NFIN, L, VT | 16 — plus `slew_slow`, `load_heavy` |
| Gate scripts (`verify_*`) | 29 — 12 campaign-driven, 3 run inside `pytest`, 14 manual-only | 29 — 13 campaign-driven (+ `canary` pool), 1 campaign preflight, 3 inside `pytest`, 12 manual-only; all 29 enumerated by a collected `--help` check |
| Root contract modules (`test_*`) | 18 modules, 6,769 lines | 21 modules, 7,953 lines |
| Shared harness (`tests/common/`) | 14,999 lines across two parallel stacks | 14,985 lines (probe helper folded into its caller) |
| Harness total vs `pycircuitsim/` | 30,609 vs 9,935 lines | 33,411 (incl. the 1,530-line render freeze) vs 9,951 |
| Frozen renders | 40 `simple-v1` hashes | 40 `simple-v1` + 760 `simple-v2` hashes |

Statement coverage of `pycircuitsim/`, as the collected suite alone and then
as the union of that suite with four simulator-backed LEVEL=72 gates
(`verify_bsimcmg_op`, `verify_bsimcmg_inverter_op`, `verify_subckt`,
`verify_ac`):

| Module | stmts | collected suite | union with 4 L72 gates |
|---|---:|---:|---:|
| `solver.py` | 1586 | 36 % | 60 % |
| `parser.py` | 649 | 66 % | 77 % |
| `models/mosfet_cmg.py` | 276 | 17 % | 75 % |
| `models/mosfet_directnet_full.py` | 264 | 83 % | 83 % |
| `models/passive.py` | 277 | 50 % | 68 % |
| `simulation.py` | 388 | 11 % | 11 % |
| `logger.py` | 107 | 21 % | 21 % |
| `visualizer.py` | 136 | 7 % | 7 % |

These are a floor, not a verdict. The NN and full campaign matrices were not
run, and they reach solver and `simulation.py` paths these numbers do not.
`simulation.py` stays at 11 % because the gates call `run_dc_sweep`,
`run_transient` and `run_ac_sweep` directly; the `run_simulation` dispatcher
above them is called by nothing but `main.py`.

## Findings

Ordered by how directly each one can make a published number wrong,
unreproducible, or incomparable across passes — not by fix cost.

### A1. Fourteen of twenty-nine gate scripts never execute on their own

`scripts/v710_regate_jobs.py` drives twelve gates, and
`test_hermetic_gate_suites.py` runs three simulator-free suites inside
`pytest`. The remaining fourteen run only when someone types the command.
Five of those have their argument parser
imported by `test_gate_cli_contracts.py`, which exercises selection, not
science. Nine have nothing at all.

Reachable from neither the campaign nor the collected suite:

| gate | note |
|---|---|
| `perf/verify_ar_cache.py` | `level1()` runs via a meta-test in `test_full_terminal_model_contracts.py`; `level0` and `level2` do not |
| `simple_circuits/verify_ac.py` | |
| `simple_circuits/verify_bsimcmg_inverter_op.py` | |
| `simple_circuits/verify_nn_inverter.py` | |
| `simple_circuits/verify_subckt.py` | |
| `single_devices/verify_bsimcmg_op.py` | |
| `single_devices/verify_cmg_multiplier.py` | |
| `single_devices/verify_data_geometry_coverage.py` | 463/463 PASS in 64 s; needs no NGSPICE and no checkpoint |
| `single_devices/verify_nn_dc.py` | |

CLI-selection coverage only: `verify_bsimcmg_dc_comprehensive`,
`verify_bsimcmg_tran_comprehensive`, `verify_multi_tech`,
`verify_circuit_sweep`, `verify_nn_lifted_source_dc`.

Two of the fourteen matter for the campaign now in flight. AGENTS.md names
`verify_nn_lifted_source_dc.py` as *the* canary for the source-relative
inference contract. `verify_data_geometry_coverage.py` is the guard that every
benchmark bias point still resolves inside the training grid — it exists
because a grid change once silently invalidated a benchmark, which is exactly
the hazard a full retraining reintroduces.

The V7.6.10 audit retained `verify_nn_dc`, `verify_nn_inverter`,
`verify_circuit_sweep` and the LEVEL=72 comprehensive gates for stated
reasons. This finding does not reopen those decisions. Retained is not
executed.

Smallest fix (as proposed): add `verify_data_geometry_coverage` and
`verify_nn_lifted_source_dc` to `DEVICE_SUITES` in `v710_regate_jobs.py`. Both
take `--tech`, so `v710_regate.sh` already resolves their paths. The remaining
twelve are a documented decision to make, not a defect.

Correction at closure: neither gate took `--tech`. The geometry guard had no
technology flag at all and read `neural_network/data/datasets`, never the
campaign root, so the 463/463 above was measured on the Sep 2 datasets and
not on the retrained grid; the canary took `--techs` and wrote no result
markers. Adding either to `DEVICE_SUITES` would have produced 40 or 80
infrastructure exits. See [Closure](#closure) for what was done instead.

### A2. Two harness stacks feed one scoreboard from two technology registries

Five of the seven campaign device suites resolve geometry through
`BenchTech`/`BENCH` in `tests/common/circuit_benchmarks.py`.
`verify_nn_multi_tech_dc` and `verify_nn_multi_tech_tran` resolve it through
`TestTechConfig`/`ALL_TEST_TECHS` in `tests/common/nn_gate.py`, a second
hand-maintained table of the same facts. `ALL_TECHS` in `tests/common/base.py`
is the nominal master. No test compares any two of them.

They agree today on VDD, default VT and modelcard names. They already differ
on the inverter, and both results are reported as the CMOS inverter in the
same campaign:

| tech | catalog `inverter_energy` | `verify_nn_multi_tech_tran` |
|---|---|---|
| TSMC5 | L=16n/20n, Cload 5 fF | L=20n/20n, Cload 1 fF |
| TSMC6 | L=16n/20n, Cload 5 fF | L=20n/20n, Cload 1 fF |
| TSMC7 | L=16n/20n, Cload 5 fF | L=20n/20n, Cload 1 fF |
| TSMC12 | L=16n/20n, Cload 5 fF | L=16n/20n, Cload 1 fF |
| TSMC16 | L=16n/20n, Cload 5 fF | L=16n/20n, Cload 1 fF |

The divergence is deliberate — `nn_gate.py` documents the inverter geometry as
aligned to the per-tech training bins — but it is recorded in one stack's
comments and asserted nowhere, so a later cleanup of either registry would
move a bias point silently.

The master registry also carries an unasserted internal inconsistency:
`ALL_TECHS["TSMC7"].default_l_nmos` is 16 nm, which is not a member of its own
`l_values = [20, 24] nm`.

Smallest fix: one contract test asserting the three registries agree on VDD,
default VT, default L/NFIN and model names per technology, that each declared
default is a member of its own available set, and that the inverter divergence
is exactly the five rows above.

### A3. AC phase is computed on every analysis and required by none

Nineteen catalog AC analyses. None lists a phase quantity in
`required_metrics`. `compare_traces` computes `<signal>_phase_maxerr_deg` for
every complex signal (`simple_circuit_harness.py:2677`) and the aggregate
NRMSE is taken on `|H|`, so a checkpoint with correct magnitude and wrong
phase scores clean on every AC cell of the topology screen. The only
phase-bearing entry in any case's `required_metrics` is
`phase_aligned_nrmse_pct` on `ring_osc`, which is amplitude alignment, not
phase error.

For a full-terminal model, phase is the observable the learned
transcapacitance matrix controls most directly. `verify_circuit_opamp_ac` does
gate phase margin at 15°, and it is campaign-driven, so this is a hole in the
19-analysis screen rather than in every AC gate.

Smallest fix: promote `phase_maxerr_deg` into `required_metrics` for the
AC-bearing cases and give it a mutation oracle in
`test_metric_profile_contracts.py`. The quantity is already computed.

### A4. The corner matrix stresses the device and never the stimulus

All 14 corners vary supply, temperature, body bias, fin count, channel length
or VT. `Corner` has no field for input slew, output load, pulse width, duty
cycle or common mode, although `circuit_templates/README.md` lists Stimulus
and Loading as first-class token groups and the templates carry
`INPUT_RISE` and `OUTPUT_LOAD`. Each of the 19 transient and 19 AC catalog
analyses is therefore characterized at exactly one stimulus point per
technology.

Those axes exist in the older stack only — `circuit_sweep.py` declares
`"ringosc": ["n_stages", "cload"]` and the inverter Cload/slew/pulse-width
sweeps — reachable through `verify_circuit_sweep.py`, which no campaign runs.
Dynamic fidelity is where charge and transcapacitance errors surface, and the
catalog is the part of the harness that no longer probes them.

Smallest fix: add `slew_scale` and `load_scale` to `Corner` and declare two
corners. This is harness plumbing, not new topology, and it also moves the
last non-duplicated capability out of the legacy driver.

### A5. Only the four `simple-v1` renders are frozen

`SIMPLE_V1_RENDER_SHA256` pins 40 hashes: 4 cases x 5 technologies x 1
analysis. The 30 `simple-v2` cases and their 76 analyses have no byte-level
freeze. Two narrow guards exist in `test_template_tier_contracts.py`
(`test_frozen_opamp_deck_still_renders_plain_dc_sources`,
`test_frozen_current_mirror_output_biases_are_unchanged`) but they check
specific rendered lines, not the deck.

A template edit therefore changes what a `simple-v2` diagnostic measures with
no test firing, and comparability across campaign passes is the reason those
diagnostics carry a denominator at all.

Smallest fix: extend the frozen set to the nominal corner of every
`simple-v2` case, or store one aggregate digest over the sorted
`(case, tech, analysis) -> sha256` map so the diff stays a single line.

### B6. The solver-numerics contracts in AGENTS.md have no direct test

Each of these is stated in AGENTS.md as a solver contract. A search of
`tests/` returns no assertion for any of them; they are exercised only
indirectly through circuit gates that need NGSPICE and a checkpoint, where a
regression appears as a slightly worse NRMSE rather than as a named failure.

- the wide GMIN ladder, and accepting convergence only at the final
  physical-GMIN step
- physical GMIN pinned at 1e-12 S
- the five-snapshot oscillation detector and its variance check
- the convergence test `|dV| < VNTOL + RELTOL*max(|V_old|, |V_new|)` with
  `RELTOL=1e-4` and `VNTOL=1e-7`
- backward Euler for the first accepted step, trapezoidal afterwards, and
  one-way BDF-2 promotion on stiffness — `test_core_device_contracts.py`
  checks which method *names* are accepted, not the ordering behaviour
- a retry reducing the actually attempted time step
- committing device charge and solver history only after a step is accepted
- PULSE breakpoint alignment with `CKTminBreak`-style tolerance
- an iteration that still applied a limiter not counting as converged

Smallest fix: the pattern is already proven in this repository.
`test_core_device_contracts.py` answers exactly this class of question
hermetically, against closed forms of linear networks, and it is the module
the V7.6.9 audit created after five deleted gates left `Inductor`,
`integration_method` and `set_temperature` with no coverage. Each item above
is one or two tests in that style.

### B7. `DCSolver(nodesets=...)` is implemented, documented, and never called

AGENTS.md specifies `.nodeset` as an initial clamp followed by an
unconstrained solve, correctly noting it is a `DCSolver` argument and not a
parser card. `_presolve_nodesets` (`solver.py:710`) implements it. Every
reference to `nodesets` outside `solver.py` is either the parser naming it as
an ignored directive or `test_deck_engine_compatibility.py` asserting that the
parser drops it and warns. No caller anywhere passes `nodesets=`.

Smallest fix: one hermetic test — a bistable network where the clamp selects a
branch and the released solve stays in it, plus the stated guarantee that a
nodeset hint never makes a solve worse. Or delete the feature and its contract
together.

### B8. The outer device safety clamps in AGENTS.md are not in the code

AGENTS.md: "Keep the outer device safety clamps at Vgs ±5 V and Vds ±10 V."
They could not be found. There is no `np.clip` anywhere in `pycircuitsim/`,
and no `VGS_MAX`/`VDS_MAX`-shaped constant. LEVEL=72 limiting uses
`_NR_LIM_WINDOW = 2.5` (`mosfet_cmg.py:325`) applied symmetrically to the
normalized gs, ds and bs pairs alike. LEVEL=75/76 bound their inputs with
`_check_support` against the persisted normalization box
(`mosfet_directnet_full.py:240`).

Those may be the better mechanisms. The finding is that a stated hard bound
left the code and nothing said so, because no test reads it.

Smallest fix: decide which statement is true, then make the surviving bound a
named constant with a test that reads it.

### B9. The mandated latch-basin gate was deleted and not replaced

AGENTS.md: "Any change that can alter a nonlinear solution basin requires a
latch-basin gate before use." `tests/perf/verify_latch_basin_gpu.py` and
`tests/perf/verify_batched_tail.py` were removed in `1d9e0fe`
("retire reduced compact-model families") with no successor; `tests/perf/`
now holds one gate. The V7.6.8 audit
([`v768-template-harness-audit.md`](v768-template-harness-audit.md)) still
reports "latch basin 2/2".

This is the V7.6.9 finding recurring: a deletion took a required gate with it,
and the requirement stayed in the documentation.

Smallest fix: restore a 6T-latch basin check for both stored states as a
hermetic contract on a synthetic checkpoint. `sram6t_modes` already declares
`hold` and `hold_state0`, so the topology and both initial states exist.
Failing that, strike the requirement from AGENTS.md rather than leave it
uncheckable.

### C10. Metric mutation oracles prove reaction, not magnitude

`test_every_catalog_analysis_detects_its_headline_behavior` drives all 80
layouts through a targeted mutation and asserts `headline > 1e-12`. That
catches a dead extractor. It does not catch an extractor off by three orders
of magnitude, or one reporting the right number in the wrong unit. Exactly one
profile has a calibrated oracle: the PMOS `self_bias_cascode` asserts `50.0`.

Several mutations have closed-form answers already — a 0.85x
transmission-gate slope, a 1.2x supply current, a 0.06 V droop ramp.

Smallest fix: convert the mutations whose magnitude is analytically known into
exact assertions and leave the rest as sign-of-life. Roughly a third of the 80
qualify.

### C11. The documented user entry point and training determinism are untested

`main.py` -> `run_simulation` -> `Visualizer` is the workflow README.md
documents for running a netlist directly. Nothing in `tests/` calls it.
`run_simulation` (`simulation.py:240`) has no caller outside `main.py`;
`visualizer.py` sits at 7 % and `logger.py` at 21 % in the union measurement.

Separately, campaign provenance is thoroughly covered —
`test_campaign_source_equivalence`, `test_campaign_training_handoff`,
checkpoint-mutation detection, dataset checksums — but nothing asserts that
the same seed, configuration and data reproduce the same checkpoint.
`neural_network/cli/train.py:267` calls `set_seed`; `trainer.py:181` builds
its `DataLoader` with `shuffle=True` and no explicit `generator=`. During a
retraining campaign this is the difference between "the new model is worse"
and "this run was noisier".

Smallest fix: a smoke test running `main.py` on a two-resistor deck, checking
the exit code and one written artifact. Separately, a short-epoch training run
repeated twice at a fixed seed, asserting identical weights.

## Merge, deletion and addition candidates

Applied at closure: the selector merge (seven gates, not four — the two
parametric NN suites had the same empty-field hole), the `core_gates.py` fold,
both deletions, and the addition. Not applied: the `circuit_sweep.py` merge.
The stimulus axis moved into the catalog `Corner` (A4), but the driver and the
3/7/9-stage ring templates stay: the stage-count axis has no catalog home,
and the templates are also rendered by the catalog harness's
`ring_n_stages` path, so they were never the driver's alone.

### Merge candidates

- `circuit_sweep.py`'s stimulus axes into the catalog `Corner` (A4). 940 lines
  plus a 137-line render canary plus five contract tests currently maintain a
  parametric driver no campaign runs. Its one non-duplicated capability is the
  slew / load / pulse-width / stage-count axis. Move that axis and the driver,
  together with `ring_oscillator_{3,7,9}stage.spice.tmpl`, becomes deletable
  rather than orphaned. The V7.6.10 audit retained the driver precisely
  *because* those axes are not duplicates of the corner matrix; moving them
  settles the question rather than reopening it.
- The four hand-rolled comma selectors into `parse_csv_choices`. Nine gates
  use the shared selector and are contract-tested; `verify_device_integrity`,
  `verify_terminal_integrity`, `verify_nn_ac`, `verify_circuit_opamp_ac` and
  `verify_circuit_topologies` each reimplement it. The drift is already
  measurable: `--tech "TSMC5,"` is a hard error in the shared selector and
  exits 0 in `verify_device_integrity`, `verify_terminal_integrity` and
  `verify_circuit_topologies` (verified). AGENTS.md requires empty selections
  to fail before a campaign starts.
- `tests/common/core_gates.py` into its single consumer. 39 statements, one
  caller (`verify_cmg_multiplier.py`), and a docstring that is now largely a
  record of five gates deleted two releases ago. The ASAP7 baking helper
  belongs in `base.py`; the probe belongs in the gate.

### Deletion candidates

- `tests/diag/` — a package containing only `__init__.py`. Both diagnostic
  probes were deleted (`111518e`, `1d9e0fe`) while `tests/README.md` still
  describes the directory as holding explanatory probes. Either delete the
  package and the README line, or restore one probe.
- Fourteen orphaned `__pycache__` stems for deleted modules, including
  `verify_latch_basin_gpu`, `verify_batched_tail`, both `diag_*` probes and
  eight retired `test_v7*` modules. Cosmetic, but they are why a grep for a
  deleted gate still returns hits.

### Addition candidate

- A self-enumerating entry-point contract. The V7.6.10 audit verified by hand
  that "all 32 `verify_*` / `diag_*` modules answer `--help` with exit 0".
  There are 29 now and nothing re-checks it. A collected test that globs
  `tests/*/verify_*.py`, asserts each answers `--help` and rejects an unknown
  flag, keeps that inventory honest as modules come and go, and would have
  surfaced A1 and B9 when they landed.

## Suggested sequencing and outcome

As proposed: before the V7.7.2 pass publishes numbers, wire
`verify_data_geometry_coverage` into the campaign (A1) and add the
three-registry test (A2); before the pass is compared against V7.7.1, freeze
the `simple-v2` renders (A5).

Outcome: all three landed on `main` before the V7.7.2 evaluate stage started,
and the geometry guard was run by hand against the campaign datasets
(`results/v771_r2_data`, 463/463 PASS). The in-flight release worktree
predates the closure, so its evaluate stage runs the 600 + 1,200 jobs only;
the 40-job `canary` pool is to be dispatched from the audited source, into its
own output root, before the V7.7.2 reports are finalized. The render freeze
protects comparison from this pass onward.

## Not reopened

- The catalog ladder starts at L1 and has no L0 case. That is not a hole.
  `L0_devices/mosfet.spice.tmpl` is the most-referenced template in the
  repository, driving `verify_device_integrity`, `verify_terminal_integrity`,
  `verify_cmg_multiplier` and the NN device gates. The L0 rung is scored under
  a different denominator, deliberately.
- The V7.6.10 retain decisions for `verify_nn_dc`, `verify_nn_inverter`,
  `verify_circuit_sweep` and the LEVEL=72 comprehensive gates are sound as
  stated. A1 is about execution, not existence.
- The V7.6.10 ruling that `.noise`, `.pz`, `.sens`, `.disto`, statistical
  mismatch, aging and reliability have no model interface here — so a gate
  would test ignored syntax rather than compact-model fidelity — still holds.
  Nothing has made those reachable.

## Still not covered

- The bias-fanout scale ladder remains NMOS-only, as the V7.6.10 audit
  recorded.
- `simple-v2` remains diagnostic; its thresholds and three-repeat LEVEL=72
  stability matrix are still not frozen.
- Of the B6 contracts, the stiffness trip to BDF-2, a retry shrinking the
  attempted step, and commit-after-accept have no hermetic witness: each
  needs a nonlinear step that fails, which no closed-form device produced.
- The backward-Euler to trapezoidal seam in `Capacitor.update_voltage` is a
  known defect, pinned as a strict expected failure and not fixed in this arm.
- The `circuit_sweep.py` driver is still reachable only by hand; its stimulus
  axis now has a catalog equivalent, its stage-count axis does not.
- Statement coverage was not re-measured after closure.
- No numerical campaign was run in this pass. Every accuracy number in
  `docs/accuracy/` is untouched by it.

## Verification evidence

| Surface | Result |
|---|---|
| Collected unit/contract suite | 635 passed, 0 skipped, 9.7 s; 2 CPU-only Torch warnings |
| `verify_bsimcmg_op` | 2/2 PASS |
| `verify_bsimcmg_inverter_op` | 1/1 PASS |
| `verify_subckt` | 11/11 PASS |
| `verify_ac` | 3/3 PASS |
| `verify_data_geometry_coverage` | 463/463 PASS in 64 s |
| Empty-selection probe | `--tech "TSMC5,"` exits 0 in `verify_device_integrity`, `verify_terminal_integrity`, `verify_circuit_topologies` |
| Coverage tooling | `coverage` 7.16.0, statement coverage only; NN and full campaign matrices not run |
| Repository state | unmodified; 66 artifacts written under `results/tests/`; no campaign worktree touched |

After closure (commit `9964963` on `main`):

| Surface | Result |
|---|---|
| Collected unit/contract suite | 785 passed, 2 expected failures, 0 skipped; 5 CPU-only Torch warnings |
| `verify_data_geometry_coverage --data-dir results/v771_r2_data` | 463/463 PASS on the retrained campaign datasets |
| `scripts/v710_regate_jobs.py` | 600 clean, 1,200 simple-v2, 40 canary jobs |
| Empty-selection probe | `--tech "TSMC5,"` exits 2 on all seven formerly hand-rolled gates |
| Gate inventory | 29 modules answer `--help` with exit 0 and reject an unknown flag with exit 2, in process |
| Repository state | both campaign worktrees clean and untouched; no campaign process signalled |

No accuracy number, promotion or retraction is claimed by this audit. NGSPICE
on the identical LEVEL=72 OSDI model remains the only compact-model ground
truth.

## Closure

Applied on `main` after the audit, with the V7.7.2 training campaign left
running in its own worktrees. Nothing here reached the in-flight evaluation;
the [plan](../plans/2026-09-05-v772-full-retraining.md) records what to run
for V7.7.2 from the audited source afterwards.

| Finding | Outcome |
|---|---|
| A1 | The audit's fix was wrong in two ways: `verify_data_geometry_coverage` has no `--tech` and read `neural_network/data/datasets`, not the campaign root, so its 463/463 was measured on the Sep 2 grid, not the retrained one; `verify_nn_lifted_source_dc` took `--techs` and emitted no result markers. The guard now takes `--data-dir` (default `BSIMAR_DATA_DIR`) and runs once as an evaluate-stage preflight; run by hand against `results/v771_r2_data`: 463/463 PASS. The canary accepts `--tech`, emits structured rows the collector checks for completeness, and is its own 40-job `canary` pool so the clean denominator is unchanged. The twelve manual gates stay manual; `tests/README.md` now says which pool runs what. |
| A2 | `test_technology_registry_contracts.py` pins VDD, VT, models, TFIN, NFIN and L across `ALL_TECHS`, `BENCH`, `ALL_TEST_TECHS`, the five-row inverter divergence, and the TSMC7 sweep-list exception as a declared fact. Also found: `v730_coverage.NON_SIMPLE_SUITES` was a fourth copy of `DEVICE_SUITES`; it is now derived from it. |
| A3 | `compare_traces` reports `phase_maxerr_deg` (worst per-signal phase error) and `validate_analysis_metrics` requires it on every AC analysis. A 30° magnitude-identical rotation reads 0 NRMSE and exactly 30° on all 19 analyses. The collector derives the aggregate from per-signal keys for rows written before this change. |
| A4 | `Corner.slew_scale` / `load_scale`; corners `slew_slow` (4×) and `load_heavy` (2×). Scaling happens before analysis overrides expand, so the inverter's composite PULSE and a literal `Cload out 0 10f` are each scaled once; a corner applies only when the rendered deck changes. Capacitive loads only; sampling/storage capacitors, resistive loads and the L4 composite PULSE specs are out of scope by design. `circuit_sweep.py` was not merged or deleted. |
| A5 | All 760 nominal simple-v2 candidate/reference renders are frozen in `tests/frozen_simple_v2_renders.py`; a mismatch names the deck. |
| B6 | `test_solver_numerics_contracts.py`: pinned RELTOL/VNTOL/GMIN, the convergence formula through a shared helper now used by both NR loops, both GMIN ladders by recorded levels, limiter-live rejection, oscillation acceptance within tolerance and rejection of a 1 V two-cycle, BE first step in every mode, exact BE→BDF-2 for `gear2`, PULSE breakpoint coalescing. Not covered: the stiffness trip itself, step-shrinking retries, and commit-after-accept, which need a failing nonlinear step no hermetic device produced. **Defect found**: after the BE first step `Capacitor.update_voltage` leaves `_i_prev` at zero, so the first trapezoidal step drops the BE current (10% low on an RC step after five steps; the LEVEL=75/76 charge path is correct). Pinned as a strict expected failure; a numerical fix requires a fresh re-gated arm. |
| B7 | `.nodeset` on a closed-form bistable node selects `±1/√2` from a `±0.5 V` hint, resolves names case-insensitively, ignores unknown nodes, keeps the plain solve on a failed clamp, and releases the clamp components. |
| B8 | No code ever implemented ±5 V/±10 V clamps (no commit in history has them). AGENTS.md now states the real bounds: `_NR_LIM_WINDOW = 2.5` for LEVEL=72, read by a test on the tracked ASAP7 card, and the LEVEL=75/76 normalization box, read by a test on the synthetic checkpoint. |
| B9 | The deleted GPU/batch flags no longer exist. A hermetic bistable cell holds both stored states through the DC hard-`.ic` path and through the reference and `refine_output` transient marches; AGENTS.md names that contract and asks any new basin-perturbing knob to join its parametrization. |
| C10 | 23 profiles now assert the calibrated magnitude of their mutation, exact where the trace is piecewise linear and within 1e-3 to 1e-2 where an extractor interpolates a smooth curve. |
| C11 | `main.py` runs the RC control deck to a transient CSV and exits 1 on a missing file; two CPU DirectNet runs at one seed produce bit-identical checkpoints and a second seed does not. |
| Merges | Seven selectors replaced by `parse_csv_choices`, with `--tech "TSMC5,"` now a hard error on every gate; `core_gates.py` folded into `verify_cmg_multiplier.py`; fixed-matrix gates parse inside `main(argv)`. |
| Deletions | `tests/diag/` and 14 orphaned bytecode stems removed. |
| Addition | `test_entry_point_contracts.py` enumerates all 29 gates and checks `--help` and an unknown flag in process. |

Collected suite after closure: 785 passed, 2 expected failures, 0 skipped.

# PyCircuitSim agent instructions

These instructions apply to the entire repository. They preserve the
architecture contracts and debugging lessons whose violation has produced
plausible-looking but numerically wrong simulations.

## Document ownership

Keep each fact in one authoritative place:

- Put installation, commands, netlist usage, and the five-stage user workflow
  in `README.md`.
- Put durable implementation rules and debugging lessons in this file.
- Put release outcomes, measurements, retractions, and dead ends in
  `docs/CHANGELOG.md`.
- Put gate definitions and scoreboard summaries in `docs/accuracy/`.
- Put all simulation artifacts in `results/`.
- Put every test-circuit topology in one parameterized template under
  `circuit_templates/`.

Link to the owner instead of copying its content into another document.

## Non-negotiable goal

Use NGSPICE on the identical BSIM-CMG OSDI model as circuit ground truth.
Simplified equations, hand-written CMG approximations, and PyCircuitSim output
cannot serve as independent references. Keep LEVEL=72 as the reference model
for LEVEL=75/76 training and gates.

## Read before editing

Read the edited symbol, its immediate callers, and the shared test utility
before changing behavior. Start with these ownership boundaries:

- `pycircuitsim/solver.py`: MNA assembly and nonlinear/AC/transient solves.
- `pycircuitsim/models/`: device currents, conductances, and charges.
- `pycircuitsim/simulation.py`: analysis orchestration.
- `pycircuitsim/parser.py`: netlist expansion and checkpoint resolution.
- `PDKs/`: source technology modelcards; only ASAP7 is tracked.
- `external_compact_models/bsim_cmg/`: BSIM-CMG evaluation and dataset creation.
- `external_compact_models/neural_network/`: shared NN data, models, loss, and training.
- `tests/common/`: authoritative deck rendering and comparison infrastructure.
- `scripts/`: campaign-level evaluation scripts.

Keep the solver free of device equations and device models free of matrix
assembly. Update `_is_mosfet()` in `solver.py` whenever a new MOSFET class is
introduced.

## MOSFET current and Jacobian contract

Use OSDI terminal current `id`, not channel current `ids`; in this backend
`ids = id - is` is approximately twice the intended drain terminal current.

LEVEL=72, 75, and 76 expose four terminal currents and charge derivatives.
Stamp the full
terminal-current Jacobian and full transcapacitance matrix, including bulk and
gate leakage. Enforce KCL/charge closure at the interface rather than dropping
rows. The device/runtime boundary uses positive current leaving each terminal;
scalar `calculate_current()` exists only for comparison consumers and is not a
solver stamp.

Evaluate OSDI terminals in a source-relative frame:
`(Vd - Vs, Vg - Vs, 0, Vb - Vs)`. Apply voltage limiting to evaluation
voltages without corrupting the physical MNA voltages. A nonlinear iteration
that still applied a limiter is not converged; require an unlimited follow-up
iteration.

Keep the outer device safety clamps at Vgs ±5 V and Vds ±10 V. The tighter
LEVEL=72 per-iteration limiting window is an NR aid, not a replacement for the
outer bounds.

## Solver contracts

### DC and operating point

- A residual evaluated from node voltages must first solve the augmented MNA
  branch-current tail for ideal voltage sources. Never fill those unknowns
  with zero, and scale a KCL tolerance only from current-valued node RHS rows;
  voltage-source constraint rows are measured in volts. Cache the
  topology-stable source-incidence fit rather than recomputing its dense
  least-squares factorization in every nonlinear iteration.
- Apply the standard voltage convergence test:
  `|ΔV| < VNTOL + RELTOL * max(|V_old|, |V_new|)` with `RELTOL=1e-4` and
  `VNTOL=1e-7` unless a gate explicitly studies tolerances.
- Keep physical GMIN at `1e-12 S`.
- Use the wide LEVEL=72 GMIN ladder for difficult reference circuits. Accept
  convergence only at the final physical-GMIN step.
- Preserve the five-snapshot oscillation detector and its variance check.
- Let `_solve_dc_with_retry` try the fast NN path first and enter GMIN retry
  only when `_last_solve_converged` is false. LEVEL=72 follows its own solver
  path.
- Implement `.nodeset` as an initial clamp followed by an unconstrained solve.
  A permanently clamped node changes the circuit. This is a `DCSolver`
  argument; the parser does not read a `.nodeset` card.
- The parser implements `.ac`, `.dc`, `.tran`, `.temp`, `.ic`, `.model`,
  `.include`, `.subckt`/`.ends`, and treats a bare `.op` as the no-analysis
  operating point. It ignores every other directive. Ignoring one that NGSPICE
  acts on makes the two engines solve different circuits from one deck text,
  and no deck-to-deck parity check can see it, so `Parser.PHYSICAL_DIRECTIVES`
  names those and the parser warns when it drops one. Add a directive to that
  set when it changes the circuit, not when it changes the output listing.
- `.options cshunt` and `.options rshunt` are circuit elements, not knobs:
  NGSPICE stamps a capacitor or resistor from every node to ground at parse
  time, worth 14% on a measured amplifier slew rate (V7.5.10). The only
  implementation lived in the AnalogGym bench translator removed in V7.6.6, so
  the core parser does not apply them today. If they are reimplemented, add
  the elements after subcircuit flattening so every resolved node receives
  one.
- Values use SPICE scale factors, matched longest-first and case-insensitively
  with trailing unit text ignored: `m` is milli and `meg` is mega. Reading a
  value differently from NGSPICE is not a loud failure — both decks render
  identically and both engines converge — so
  `tests/test_deck_engine_compatibility.py` holds the parser against a table
  measured on NGSPICE rather than against itself.

### Transient

- Use backward Euler for the first accepted step, trapezoidal integration
  afterward, and one-way BDF-2 promotion on stiffness.
- Make a retry reduce the actual attempted time step. Commit device charge and
  solver history only after a step is accepted.
- Align PULSE breakpoints using `CKTminBreak`-style tolerance instead of exact
  floating-point equality.
- Keep LTE refinement opt-in. Treat output stride and breakpoint refinement as
  fidelity controls that require re-gating, not presentation-only knobs.
- In hard `.ic` mode, stamp initial-condition nodes as temporary voltage-source
  constraints, then release them and solve the physical circuit. This is
  required for bistable SRAM initialization.

### AC

- Linearize around a converged DC operating point and solve
  `Y = G + jωC`.
- Stamp the full LEVEL=72 terminal-current and transcapacitance matrices.
- Do not add an external AC GMIN that is absent from the NGSPICE problem.

Call `Circuit.invalidate_topology()` after any direct mutation of
`Circuit.components`; node maps and solver caches are keyed by the topology
version.

## NN compact-model contracts

Apply the source-relative voltage frame to NMOS and PMOS alike. Training uses
`Vs=0`; inference must shift every terminal by `-Vs`. Keep
`verify_nn_lifted_source_dc.py` as the canary for this rule.

Import full-terminal contract names and column order from
`neural_network.data.contracts`; `data.normalize` owns transforms and persisted
statistics, not schema re-exports.

Per-technology models use a local embedding vocabulary. Derive
`unknown_code_id` as `num_tech_codes - 1`; never reuse the universal UNKNOWN
identifier. At inference, map `(scope, tech, variant)` through
`neural_network.config.local_variant_code()`.

For LEVEL=75/76:

- Require `TECH` and `VT` in the netlist model declaration.
- Keep ASAP7 outside the NN checkpoint scope.
- Read an explicit checkpoint environment pin before automatic resolution.
  Raise when the pinned path is missing.
- Resolve per-tech stems large-first before considering a universal fallback.
- Require architecture sidecars where the family loader needs them.
- Compute source current and `qs = -(qg + qd + qb)` analytically to preserve
  KCL and charge conservation.
- Reject NaN/Inf, any terminal current over 1 A, and failed PyCMG internal-node solves during
  dataset creation. Keep unstable NFIN=1 bins out of training.

LEVEL=73/74 are retired. Do not restore reduced current/derivative heads,
classic drain-source stamps, or fallback checkpoint aliases for them.

DC/OP should skip NN charge Jacobians. `TransientSolver` and `ACSolver` must
declare `_require_nn_caps`; any new capacitance consumer must do the same.
`get_charge_stamp()` remains a self-healing fallback so a missed declaration
causes slowness rather than incorrect results.

## Verification discipline

Order `circuit_templates/` by what a circuit demands of a compact model, not
by application: `L0_devices` biases one device from ideal sources,
`L1_primitives` adds a passive load, `L2_stages` couples devices while every
gate rail stays ideal, `L3_blocks` makes the model determine an operating
point, and `L4_systems` closes a negative-feedback loop. Declare each case's
`tier` in the catalog and let `template_deck()` verify it against the file's
location; a topology present in two tiers is an error, not a convenience.

Treat `circuit_templates/subcircuits/` as the representation-equivalence
exception to the one-topology rule. A flat/hierarchical fixture pair may
describe the same physical circuit because the netlist representation is the
tested input. Keep these fixtures out of catalog denominators, and reuse the
canonical `controls/` or L0–L4 flat template when one already exists. Read the
subcircuit fixture seam in `circuit_templates/README.md` before adding or moving
a hierarchy fixture.

At least one `L4_systems` transient must run without `uic`. Every other
transient in the catalog is handed its initial state through `.ic`, so that is
the only place cold-start basin entry is exercised.

Report convergence separately from error, always. A solve that did not reach a
physical fixed point is an `ERROR` row that keeps its denominator slot and
never enters a numeric aggregate; recovering its numbers for reading is
allowed only under a key no scoring path consumes. Folding the two together is
how a 0/10 AC score came to mean an unconverged DC operating point.

Use one `.spice.tmpl` file in `circuit_templates/` as the single source for
each tested circuit. Render both candidate and reference decks through `tests.common.base`;
verification modules must not embed or privately copy netlists. Store every
materialized netlist and simulation artifact under `results/`, never `tests/`
or `circuit_templates/`.

Before interpreting a numerical mismatch:

1. Render both engine decks.
2. Diff topology, models, sources, options, analysis limits, and measurements.
3. Confirm that both engines solved the same problem.
4. Only then inspect solver or device math.

Pin scored NN campaigns to CPU with one OpenMP, MKL, and Torch thread. A report
must come from one complete campaign pass and must preserve its checkpoint and
commit provenance. Do not mix partial logs or compare totals across a gate-
denominator change without rescaling.

Train a new clean matrix into an isolated `BSIMAR_CHECKPOINT_DIR`; never
overwrite the preserved control while generating a comparison. A checkpoint
is campaign-ready only when its model, required normalization/config sidecars,
and completion marker all exist.

Keep diagnostics separate from gates. A diagnostic may explain a failure but
does not count as evidence. Make each gate answer one question, and test NMOS
and PMOS through single-device OP/DC, inverter VTC/transient, and derivative-
sensitive circuit behavior before promoting a model.

Accuracy CLIs must reject unknown technologies, devices, analyses, and NN
levels before running so a typo cannot shrink a denominator. The campaign
family selector accepts only LEVEL=75/76, and every banner must name the
selected family.

The clean re-gate driver must reject a missing or dependency-incomplete
`NN_PY`; never silently fall back to another interpreter. An unhandled Python
traceback is infrastructure failure, not a scientific exit-1 verdict. Keep
explicit `ERROR` configurations in parametric denominators while aggregating
numeric metrics only over rows that produced them.

For a high-gain opamp AC gate, locate the linearization bias with a physical
fine sweep around the coarse maximum-gain point. Do not interpolate a coarse
grid into a reference. Gate only after the reference bias is off-rail and the
NN DC operating point has converged; an unconverged response remains a
diagnostic.

Report model error with MRE, R², NRMSE, and maximum voltage error per
technology. Store the detailed results in `docs/accuracy/`, not in this file.

## Performance discipline

Classify every optimization before enabling it:

- A bit-identical optimization may ship enabled after its focused gates pass.
- A floating-point-perturbing optimization ships disabled until a full
  accuracy re-gate clears it.

Keep CPU, flags-off execution as the scored contract. Autoregressive caching
is opt-in and requires its focused numerical-equivalence gate. Any change that
can alter a nonlinear solution basin requires a latch-basin gate before use.

## Change workflow

- State assumptions and success criteria before non-trivial work.
- Prefer the smallest change that satisfies the stated intent.
- Match existing code style; add type hints to every Python signature and
  docstrings to non-obvious algorithms.
- Verify each significant step and report skipped or unavailable checks.
- Add a compact changelog entry for a versioned behavior change. Include
  reverted approaches when they prevent repeated investigation.
- Train independent jobs across available GPUs; keep scored inference on the
  pinned CPU contract.
- Preserve unrelated working-tree changes and generated evidence.
- Keep raw TSMC `PDKs/TSMC*/*.l` cards and `PDKs/modelcards.tar.gz*` out of Git.

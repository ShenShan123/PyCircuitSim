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
- Put gate definitions and score evidence in `docs/accuracy/`.
- Put detailed AnalogGym results in
  `examples/complex_circuits/RESULTS_TSMC.md`.

Link to the owner instead of copying its content into another document.

## Non-negotiable goal

Use NGSPICE on the identical BSIM-CMG OSDI model as circuit ground truth.
Simplified equations, hand-written CMG approximations, and PyCircuitSim output
cannot serve as independent references. Keep LEVEL=72 as the reference model
for LEVEL=73–75 training and gates.

## Read before editing

Read the edited symbol, its immediate callers, and the shared test utility
before changing behavior. Start with these ownership boundaries:

- `pycircuitsim/solver.py`: MNA assembly and nonlinear/AC/transient solves.
- `pycircuitsim/models/`: device currents, conductances, and charges.
- `pycircuitsim/simulation.py`: analysis orchestration.
- `pycircuitsim/parser.py`: netlist expansion and checkpoint resolution.
- `external_compact_models/PyCMG/`: BSIM-CMG evaluation and dataset creation.
- `external_compact_models/bsimar/`: shared NN data, models, loss, and training.
- `tests/common/`: authoritative deck rendering and comparison infrastructure.

Keep the solver free of device equations and device models free of matrix
assembly. Update `_is_mosfet()` in `solver.py` whenever a new MOSFET class is
introduced.

## MOSFET current and Jacobian contract

Use OSDI terminal current `id`, not channel current `ids`; in this backend
`ids = id - is` is approximately twice the intended drain terminal current.

At the device boundary:

- NMOS `calculate_current()` returns `-result["id"]`.
- PMOS `calculate_current()` returns `result["id"]`.
- Solver stamping uses one convention: positive current leaves the drain.
- Preserve the signs of `gm` and `gmb`.
- Floor the stamped drain conductance with `max(gds, 1e-12)`; `abs(gds)` turns
  a large negative derivative into a large positive one and destabilizes NR.

The reduced NN stamp is:

```python
i_leaving = -i_ds if is_pmos else i_ds
i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
rhs[d_idx] -= i_eq
rhs[s_idx] += i_eq
```

Stamp every VCCS with all four matrix entries. A partial stamp breaks the
Jacobian even when single-device current values look correct.

LEVEL=72 exposes four terminal currents and charge derivatives. Stamp the full
terminal-current Jacobian and full transcapacitance matrix, including bulk and
gate leakage. Enforce KCL/charge closure at the interface rather than dropping
rows to imitate the reduced NN model.

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
  A permanently clamped node changes the circuit.
- Apply `.options cshunt` and `.options rshunt` as circuit elements after
  subcircuit flattening so every resolved node receives the intended element.

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

Per-technology models use a local embedding vocabulary. Derive
`unknown_code_id` as `num_tech_codes - 1`; never reuse the universal UNKNOWN
identifier. At inference, map `(scope, tech, variant)` through
`bsimar.config.local_variant_code()`.

For LEVEL=73–75:

- Require `TECH` and `VT` in the netlist model declaration.
- Keep ASAP7 outside the NN checkpoint scope.
- Read an explicit checkpoint environment pin before automatic resolution.
  Raise when the pinned path is missing.
- Resolve per-tech stems large-first before considering a universal fallback.
- Require architecture sidecars where the family loader needs them.
- Compute `qs = -(qg + qd + qb)` analytically to preserve charge conservation.
- Reject NaN/Inf, `|id| > 1 A`, and failed PyCMG internal-node solves during
  dataset creation. Keep unstable NFIN=1 bins out of training.

Preserve checkpoint-compatible optional structures even when their flags are
off. In particular, keep `_MonotoneVgResidual` and its `mono.*` keys, the EKV
backbone and `core.*` keys, and Sobolev/subthreshold/charge-Sobolev loss paths.
Existing checkpoints depend on those state-dict shapes.

DC/OP should skip NN charge Jacobians. `TransientSolver` and `ACSolver` must
declare `_require_nn_caps`; any new capacitance consumer must do the same.
`get_capacitances()` remains a self-healing fallback so a missed declaration
causes slowness rather than incorrect results.

## Verification discipline

Use the `.sp`/`.cir` pair in `examples/` as the single source for each tested
circuit. Render reference decks through `tests.common.base`; verification
modules must not embed or privately copy netlists.

Before interpreting a numerical mismatch:

1. Render both engine decks.
2. Diff topology, models, sources, options, analysis limits, and measurements.
3. Confirm that both engines solved the same problem.
4. Only then inspect solver or device math.

Pin scored NN campaigns to CPU with one OpenMP, MKL, and Torch thread. A report
must come from one complete campaign pass and must preserve its checkpoint and
commit provenance. Do not mix partial logs or compare totals across a gate-
denominator change without rescaling.

Keep diagnostics separate from gates. A diagnostic may explain a failure but
does not count as evidence. Make each gate answer one question, and test NMOS
and PMOS through single-device OP/DC, inverter VTC/transient, and derivative-
sensitive circuit behavior before promoting a model.

Report model error with MRE, R², NRMSE, and maximum voltage error per
technology. Store the detailed results in `docs/accuracy/`, not in this file.

## AnalogGym migration rules

Before citing AnalogGym as evidence, check the current model scope in
`README.md` stage 5. Treat any extension beyond that scope as new work requiring
its own gates.

- Treat the source design tree as the corpus. Run the source-tree preflight
  before regeneration and fail loudly when it is absent.
- Set `AG_TREE` before importing modules that lazily locate the corpus.
- Follow the technology interpretation in `README.md` when selecting campaign
  dimensions; do not inflate the independent ground-truth count.
- Curate by measured discrimination and reproducibility, not by deck-level
  aesthetics. Record quarantines explicitly.
- Keep `NOT_COMPARABLE` as a measurement outcome when the two engines did not
  produce commensurate values.
- Treat transient stride as a semantic sampling choice. Preserve the explicit
  charge-pump exception instead of globally forcing stride 1.
- Regenerate reports through the report builder only after preserving the
  underlying campaign artifacts; the builder overwrites its destination.

Detailed campaign verdicts belong in `RESULTS_TSMC.md`, and release-level
interpretation belongs in the changelog.

## Performance discipline

Classify every optimization before enabling it:

- A bit-identical optimization may ship enabled after its focused gates pass.
- A floating-point-perturbing optimization ships disabled until a full
  accuracy re-gate clears it.

Keep CPU, flags-off execution as the scored contract. Keep CUDA inference,
fused Jacobians, and autoregressive caching opt-in. Preserve exact per-element
gates for batched/fused paths and latch-basin gates for changes that can alter
the nonlinear solution basin.

Previous experiments found no useful production path in TF32, `torch.compile`,
or bfloat16 DirectNet inference. Revisit them only with a new hypothesis and a
predeclared metric, not as routine cleanup.

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
- Keep raw TSMC `cln*.l` modelcards and `modelcards.tar.gz` out of Git.

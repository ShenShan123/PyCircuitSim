# Verification tests

`tests/` contains verification intent and reusable harness code. Circuit
topology is owned by the parameterized templates in [`circuit_templates/`](../circuit_templates/README.md),
and every persistent simulator artifact is written below `results/`.

## Layout

- `common/` provides strict template rendering, technology profiles, candidate
  and LEVEL=72 adapters, trace comparison, and campaign result contracts.
- `single_devices/` verifies four-terminal currents and the 4x4 physical
  transcapacitance matrix, geometry coverage, lifted-source behavior,
  compact-model device sweeps, and — through
  `verify_device_integrity.py` — the output characteristic, subthreshold
  decades, triode region, and `gm`/`gds`/`gmb` against ground truth.
- `simple_circuits/` verifies operating point, DC, transient, AC, topology
  parity, parameter corners, and accuracy-campaign tooling.
- `perf/` contains opt-in performance and solution-basin checks.
- `diag/` contains explanatory probes that are not release gates.

## Test contract

- NGSPICE on the identical BSIM-CMG OSDI model at LEVEL=72 is ground truth.
- A test selects a canonical template and supplies technology, VT, geometry,
  P/N ratio, PVT, slew, load, bias, and analysis values through the shared
  renderer.
- Candidate and reference decks must have physical parity—including values,
  waveforms, temperature, ICs, analysis limits, and device bindings—before
  numerical differences are interpreted. PyCircuitSim LEVEL=72 is the
  attribution control, never a replacement reference.
- Qualification gates and diagnostics remain distinct; a diagnostic result is
  not promotion evidence.
- Convergence is reported separately from error. A solve that did not reach a
  physical fixed point is an `ERROR` row that keeps its slot in the
  denominator and is never averaged into an accuracy number; where the numbers
  behind it are recoverable they are filed under a key no scoring path reads.
- Unknown technologies, cases, corners, or analyses must fail before a campaign
  starts.
- A partial/nonconverged trace is always an `ERROR`; recovered values live only
  below a non-scoring diagnostic key. Required event metrics must be finite.
- Every structured row records model family/level, explicit checkpoint pins,
  campaign provenance when present, and CPU thread settings.
- Generated `.sp`, `.cir`, CSV, JSON, logs, plots, and reports belong under
  `results/tests/` or another campaign directory below `results/`, never here.

Minimal malformed strings that test parser rejection are grammar fixtures, not
runnable circuit topologies. They may remain next to their assertions, but any
valid deck sent to a simulator must be rendered from `circuit_templates/`.

The fast catalog contract in
`simple_circuits/verify_simple_circuit_catalog.py` checks the inventory,
declared difficulty tiers, strict rendering, topology parity, corner matrix,
derived-metric names, the cold-start rule for L4 systems, and repository
placement. The versioned simple-circuit entry point is
`simple_circuits/verify_circuit_topologies.py`.

Environment setup, supported commands, and the five-stage workflow are
maintained in the repository [README](../README.md).

Run the authoritative unit suite with:

```bash
conda run -n pycircuitsim python -m pytest -q tests
```

The root `pytest.ini` excludes `results/` so archived campaign worktrees and
materialized decks cannot be collected as live tests.

# Verification tests

`tests/` contains verification intent and reusable harness code. Circuit
topology is owned by the parameterized templates in [`examples/`](../examples/README.md),
and every persistent simulator artifact is written below `results/`.

## Layout

- `common/` provides strict template rendering, technology profiles, candidate
  and LEVEL=72 adapters, trace comparison, and campaign result contracts.
- `single_devices/` verifies terminal currents, geometry coverage, lifted-source
  behavior, and compact-model device sweeps.
- `simple_circuits/` verifies operating point, DC, transient, AC, topology
  parity, parameter corners, and accuracy-campaign tooling.
- `perf/` contains opt-in performance and solution-basin checks.
- `diag/` contains explanatory probes that are not release gates.

## Test contract

- NGSPICE on the identical BSIM-CMG OSDI model at LEVEL=72 is ground truth.
- A test selects a canonical template and supplies technology, VT, geometry,
  P/N ratio, PVT, slew, load, bias, and analysis values through the shared
  renderer.
- Candidate and reference decks must have topology parity before numerical
  differences are interpreted.
- Qualification gates and diagnostics remain distinct; a diagnostic result is
  not promotion evidence.
- Unknown technologies, cases, corners, or analyses must fail before a campaign
  starts.
- Generated `.sp`, `.cir`, CSV, JSON, logs, plots, and reports belong under
  `results/tests/` or another campaign directory below `results/`, never here.

Minimal malformed strings that test parser rejection are grammar fixtures, not
runnable circuit topologies. They may remain next to their assertions, but any
valid deck sent to a simulator must be rendered from `examples/`.

The fast catalog contract in
`simple_circuits/verify_simple_circuit_catalog.py` checks the inventory,
strict rendering, topology parity, corner matrix, and repository placement
rules. The versioned simple-circuit entry point is
`simple_circuits/verify_circuit_topologies.py`.

Environment setup, supported commands, and the five-stage workflow are
maintained in the repository [README](../README.md).

Run the authoritative unit suite with:

```bash
conda run -n pycircuitsim python -m pytest -q tests
```

The root `pytest.ini` excludes `results/` so archived campaign worktrees and
materialized decks cannot be collected as live tests.

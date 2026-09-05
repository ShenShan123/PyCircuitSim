# Verification tests

`tests/` contains verification intent and reusable harness code. Circuit
topology is owned by the parameterized templates in [`circuit_templates/`](../circuit_templates/README.md),
and every persistent simulator artifact is written below `results/`.

## Layout

Two kinds of test live here and they are collected differently.

**Gate scripts** are `verify_*.py` / `diag_*.py` entry points, grouped by the
tier of circuit they gate. They are run by hand or by a campaign, and most need
NGSPICE, a PDK card, and an NN checkpoint:

- `common/` provides strict template rendering, technology profiles, candidate
  and LEVEL=72 adapters, trace comparison, and campaign result contracts.
- `single_devices/` verifies four-terminal currents and the 4x4 physical
  transcapacitance matrix across the declared corner matrix, geometry
  coverage, lifted-source behavior, compact-model device sweeps, and — through
  `verify_device_integrity.py` — the output characteristic, subthreshold
  decades, triode region, and `gm`/`gds`/`gmb` against ground truth.
- `simple_circuits/` verifies operating point, DC, transient, AC, topology
  parity, parameter corners, and accuracy-campaign tooling.
- `perf/` contains opt-in performance and solution-basin checks.
- `diag/` contains explanatory probes that are not release gates.

**Contract modules** are the `test_*.py` files at this directory's root. They
need no simulator and run in the collected `pytest` suite. Each owns one seam:

| Module | Seam it holds |
|---|---|
| `test_circuit_harness_contracts.py` | the catalog-driven experiment seam: frozen renders, trace/metric validity, physical parity, provenance, collectors, and the gate CLIs |
| `test_metric_profile_contracts.py` | known-trace identity and mutation oracles for every live catalog metric profile |
| `test_gate_cli_contracts.py` | fail-closed empty, duplicate, and unknown gate selections |
| `test_template_tier_contracts.py` | tier resolution and the frozen token defaults |
| `test_deck_engine_compatibility.py` | cards and value syntax both engines must read identically |
| `test_core_device_contracts.py` | the non-compact-model core: `Inductor`, integration method, current-source sign, transient branch currents, temperature rebinding |
| `test_subcircuit_harness_contracts.py` | the standalone hierarchy harness |
| `test_full_terminal_solver_boundary.py` | mandatory four-terminal DC/transient/AC solver seam |
| `test_full_terminal_*` | full-terminal dataset, family, and corridor contracts |
| `test_full_terminal_model_contracts.py` | both NN families, closure, multipliers, parser selection, and artifact integrity |
| `test_dataset_and_campaign_contracts.py` | dataset splits, provenance, and campaign coverage |
| `test_v771_campaign.py` | V7.7.1/V7.7.2 resume, GPU identity, release isolation, and predecessor dependency |
| `test_release_metadata.py` | package/README release identity |
| `test_hermetic_gate_suites.py` | wiring, not assertions: runs the three simulator-free gate suites |

Three gate scripts need no simulator at all —
`verify_simple_circuit_catalog.py`, `verify_circuit_sweep_canaries.py`, and
`verify_accuracy_campaign_tools.py`. They stay runnable as scripts for a
campaign operator and are also executed by `test_hermetic_gate_suites.py`, so
the collected run covers them. A new simulator-free gate suite belongs in that
list.

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
- Empty, duplicate, or unknown technologies, devices, cases, corners, sweeps,
  or analyses must fail before a campaign starts.
- A partial/nonconverged trace is always an `ERROR`; recovered values live only
  below a non-scoring diagnostic key. Required event metrics must be finite.
- Every structured row records model family/level, explicit checkpoint pins,
  campaign provenance when present, and CPU thread settings.
- Every card and value in a rendered deck must mean the same thing to both
  engines. Deck-to-deck parity cannot see a card one engine honours and the
  other silently drops, so `test_deck_engine_compatibility.py` holds every
  template and rendered deck against the parser's real support surface. Its
  scale-factor table in `common/parser_support.py` was measured on NGSPICE
  45.2, not copied from the parser: a reference the candidate defines is not a
  reference. `Parser.PHYSICAL_DIRECTIVES` warns when a dropped card would have
  changed the circuit.
- A gate that takes no options still answers `--help` and rejects an unknown
  flag (`common/base.py:parse_no_options`). A silently ignored `--tech` would
  let an operator read a full-matrix result as the subset they asked for.
- Every catalog analysis layout is exercised with a known identical trace and
  a targeted mutation through `compare_traces`; exact polarity and derived-dB
  checks supplement those metamorphic tests. Deck parity alone cannot certify
  a trip-point, period, delay, droop, bandwidth, or rejection-ratio extractor.
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

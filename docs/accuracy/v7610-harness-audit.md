# V7.6.10 test-harness audit

Date: 2026-09-04

Audited baseline: `54a2803` (V7.6.9).

Status: harness coverage and maintenance release. No diagnostic was promoted,
no accuracy threshold moved, and the frozen `simple-v1` `/20` denominator is
unchanged.

## Outcome

| Surface | V7.6.9 | V7.6.10 |
|---|---:|---:|
| Collected `pytest -q tests` | 357 | 567 |
| Catalog analysis layouts exercised with known traces | ad hoc | 80/80 identity and mutation checks across 47 profiles |
| NN flat/nested analyses | transient only | DC, transient, and AC |
| Generated-cascode bias polarities | NMOS only | NMOS and PMOS |
| Version-stamped root contract modules | 7 | 0 |
| Full-terminal family contract modules | 2 partially duplicated | 1 parametrized family suite |
| Stale generic metric profiles | 2 | 0 |
| Selection/entry-point regressions | uncollected | 34 collected checks |

The catalog now contains 4 frozen `simple-v1` cases and 30 held-out
`simple-v2` cases. The 45 authoritative templates comprise 1 control, 1 L0,
4 L1, 10 L2, 16 L3, 4 L4, and 9 subcircuit fixtures. `simple-v2` owns 76
analyses; the complete catalog owns 80.

## Coverage gaps found and closed

### Metric extraction was not independently tested

Candidate/reference deck parity proves that both engines receive the same
experiment. It cannot prove that the harness extracts the right trip point,
delay, period, droop, bandwidth, rejection ratio, output resistance, or bias
metric from the resulting traces. Most of the 47 live metric profiles had no
known-input behavioral test; catalog validation checked only that their names
and promised output keys existed.

`test_metric_profile_contracts.py` now drives all 80 catalog analysis layouts
across the 47 live profiles through the public `compare_traces` seam twice:
identical physically shaped traces must produce finite zero-error evidence,
and a targeted mutation of the named behavior must move the profile's headline
metric. Exact tests additionally pin the PMOS compliance orientation and the
20·log10 CMRR/PSRR derivation. The same registry check removed the unused `ac`
and `transient` profile names; neither was referenced by any catalog analysis.

This work found one real polarity bug. `self_bias_cascode` always measured the
high-output-voltage half of its sweep, which is the NMOS compliance region.
The PMOS complement must use the low-output-voltage half. A synthetic trace
whose PMOS output resistance differs only below mid-rail now pins that choice.

### NN hierarchy covered only the transient solver path

The flat/nested NN buffer checked expansion, geometry, and transient execution,
but a hierarchy can flatten correctly for transient while its DC sweep or AC
linearization path is broken. The same fixture pair now runs:

- DC transfer, including the internal `mid` / `Xbuf.m` node;
- transient switching, including the same internal node; and
- AC response around a stable logic-zero operating point.

All three rows are provenance-bound and required by the campaign collector.
`--analysis dc,tran,ac` permits focused reruns and rejects empty, duplicate, or
unknown selections. No second hierarchy script or duplicate topology was
added.

### Generated self-bias had no PMOS-only witness

`self_biased_cascode` exercised only an NMOS-generated cascode rail. The new
`self_biased_cascode_pmos` L3 case is its polarity complement and uses a
resistor-fed two-diode PMOS bias stack. The passive load line is intentional:
an ideal 5 µA sink forced the lower generated rail to −78.8 mV in NGSPICE and
made the native LEVEL=72 current-fed diode solve diverge. Loads from 20 kΩ to
1 MΩ made the LEVEL=72 control converge; 100 kΩ keeps both generated rails
inside the supply and leaves the compact model, not an ideal gate source, to
select them.

The TSMC12 LEVEL=73 smoke remains an honest model `ERROR`: NGSPICE and the
PyCircuitSim LEVEL=72 control converge, while the served DirectNet checkpoint
does not reach a physical fixed point. The recovered, non-scoring diagnostic
reports 11.79% output-resistance error and 10.76 mV bias-node error. Keeping
that denominator row is the purpose of adding the missing polarity test.

### Gate selection and release identity could drift silently

The older LEVEL=72 matrix drivers and four `simple-v1` entry points accepted
empty or duplicate comma lists; mixed valid/invalid device and sweep lists
could run only the valid subset. `parse_csv_choices` now gives these gates one
fail-closed selection seam, and the collected suite checks invalid, empty, and
duplicate technology, device, analysis, sweep, and NFIN inputs.

V7.6.9 also advertised one version in the README while
`pycircuitsim.__version__` remained at V7.6.6. A collected metadata contract
now keeps them equal.

## Merge and deletion decisions

### Merged

`test_v760_full_terminal_directnet.py` and
`test_v761_full_terminal_bsimar.py` were the deferred merge from the V7.6.9
audit. `test_full_terminal_model_contracts.py` now asks closure, scalar-current
sign, certified support, artifact sharing, checksum, explicit-family,
completion-marker, force-level, and temperature questions across both family
adapters. Family-specific DirectNet lazy-capacitance and Transformer target-
shape checks remain local. Both polarities are now exercised for both families.

### Renamed

The remaining root `test_v*` modules were live contracts with stale release
names, not obsolete behavior. They now have responsibility-based names for
dataset/campaign, full-terminal dataset, full-terminal corridor, NN multiplier,
and template-tier seams. No assertion was dropped by those moves.

### Deleted

- `diag_l72_circuit_control.py` was a closed one-off ring/opamp attribution
  probe. The catalog harness now owns the reusable `--level72-control` adapter,
  and the release outcome remains in the changelog.
- The two old full-terminal family files were removed after their replacement
  suite passed.
- The two unused generic metric-profile declarations were removed.

### Retained

- `verify_nn_dc` / `verify_nn_inverter` remain distinct from their parametric
  counterparts because their verdict thresholds differ.
- `verify_circuit_sweep` remains because its stage count, load, compensation,
  clock, and pulse-width axes are not duplicates of the catalog corner matrix.
- The LEVEL=72 comprehensive gates remain independent validation of the
  PyCircuitSim reference adapter across VT/L/NFIN; they do not grade an NN
  against itself.
- The `v710`/`v730` campaign tools and the reduced/full-terminal corridor
  scripts retain active README, training, collector, or test callers. Their
  release-stamped names are historical, but deleting them would remove live
  workflows rather than stale code.

## Still not covered

- `simple-v2` remains diagnostic. Its thresholds and three-repeat LEVEL=72
  stability matrix are not frozen, and a complete five-technology/all-corner/
  four-family numerical campaign was not run in this audit.
- The new PMOS self-biased cascode closes the smallest polarity gap; the
  3/5/9/17-device generated-bias fanout ladder is still NMOS-only.
- `.noise`, `.pz`, `.sens`, `.disto`, controlled-source, statistical mismatch,
  aging, and reliability categories have no simulator/model interface here.
  Adding a gate before implementing those semantics would test ignored syntax,
  not compact-model fidelity.
- The extended hierarchy gate was executed on TSMC12 LEVEL=73 in this audit.
  LEVEL=74–76 and the other technology nodes remain campaign work, although
  the renderer and collector enumerate the same three analyses for them.

## Verification evidence

| Surface | Result |
|---|---|
| Collected unit/contract suite | 567 passed; 2 CPU-only Torch warnings |
| Catalog contract | 4 `simple-v1` + 30 `simple-v2` cases |
| Static render/parity canary | 4,911 applicable cells |
| Campaign tooling | 600/600 unique clean jobs |
| Metric profiles | 80/80 catalog layouts accept identity and detect a targeted mutation; exact PMOS and CMRR/PSRR oracles pass |
| NN hierarchy, TSMC12 LEVEL=73 | DC/transient/AC characterized; flat/nested max differences 0 V, 0 V, and 1.06e-22 V |
| PMOS generated-bias control, TSMC12 | NGSPICE and PyCircuitSim LEVEL=72 converged; LEVEL=73 explicit nonconvergence |
| Gate selection | 33 invalid/empty/duplicate cases reject with exit 2; top-level sweep `--help` exits 0 |
| Standalone entry points | all 32 `verify_*` / `diag_*` modules answer `--help` with exit 0 |
| Lint/build | new contract modules pass the configured Ruff rules; repository fatal subset `E9,F63,F7,F82,F401,F841` and `compileall` pass |
| Type/coverage tooling | not available in the `pycircuitsim` environment (`pyright`, `pytest-cov`, and `coverage` absent); no type-clean or percentage-coverage claim |
| Generated report check | reduced DirectNet/BSIM-AR reports checksum-verified; all-family check unavailable because local LEVEL=75/76 matrices are incomplete or drifted |

Numerical diagnostic errors above are not promotion results. NGSPICE on the
identical LEVEL=72 OSDI model remains the only compact-model ground truth.

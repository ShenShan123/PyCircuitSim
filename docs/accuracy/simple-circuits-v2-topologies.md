# Simple-circuit topology evaluation

This document defines the held-out `simple-v2` evaluation set for NN compact
models. They exercise device composition more deeply than a single transistor
or inverter while remaining inside the compact-model qualification scope.

The existing four-cell score is versioned as `simple-v1` and remains `/20` per
tier. `simple-v2` is diagnostic until its reference stability and thresholds
are frozen; no exploratory result in this document is a published score.

## Topology ladder

| case | analyses | behavior exposed beyond pointwise device error |
|---|---|---|
| `source_follower` | NMOS/PMOS DC | body effect, source-relative lifting, gain and rail approach |
| `common_gate` | NMOS/PMOS DC | source-driven bias, transconductance and signed input current |
| `current_mirror` | NMOS/PMOS DC | mirror ratio, compliance and output resistance |
| `inverter_chain` | transient | open-chain FO4 loading, delay, rise/fall, amplitude and phase-aligned shape |
| `transmission_gate_dc` | forward/reverse DC | complementary conduction, bidirectionality and on-resistance |
| `transmission_gate_hold` | transient | charge hold, off-state droop and clock feedthrough |
| `diffpair_ideal` | steering DC, differential/common-mode AC | current steering and differential versus common-mode gain with an ideal tail |
| `diffpair_active` | steering DC, differential/common-mode AC | the same observables with device-device tail interaction |
| `cascode_stack` | NMOS/PMOS compliance DC, AC | stacked-device bias, internal nodes and output resistance |
| `nand2` | both input DC paths, transient | series NMOS stack, internal-node history, trip and delay |
| `nor2` | both input DC paths, transient | series PMOS stack, internal-node history, trip and delay |
| `sram6t_modes` | hold/read/write transient modes | full cross-coupled feedback, read disturb, write time and retention |

Every case declares its expected signals and domain metrics in
`tests/common/simple_circuit_catalog.py`. DC, transient and AC are all present;
current signals keep their sign rather than being converted to magnitudes.

## Unified-template contract

Each experiment has one `.spice.tmpl` source in `examples/simple_circuits/`.
The candidate and LEVEL=72 reference adapters render it with different compact-
model binding and engine-required syntax while preserving components,
connectivity, sources, initial conditions, and analysis limits. Rendering is
strict: missing and unused tokens are errors. Before either simulator runs,
the harness compares a canonical topology signature, including MOS polarity
and terminal order, source kind, and initial conditions.

The legacy opamp, ring oscillator, SRAM-SNM and switched-capacitor builders,
plus the scored inverter VTC/transient builders, now render these authoritative
templates as well. Programmatic ring generation remains only for the
stage-count sweep and is parity-checked at 3, 5, 7 and 9 stages.

## Corner matrix

The default diagnostic is nominal. The declared stress matrix adds:

- `temp_cold` at −25 °C and `temp_hot` at 125 °C;
- `vdd_low` at 0.85× nominal and `vdd_high` at 1.10× nominal;
- `body_reverse` at 0.10×VDD reverse body bias through explicit body rails in
  every applicable simple-v2 deck;
- `pn_n3p2` and `pn_n2p3` fin-ratio asymmetry;
- `joint_hot_lowvdd`: 125 °C, 0.90×VDD, LNMOS=LPMOS=20 nm,
  NFIN-N=3 and NFIN-P=2.

The older parametric four-circuit sweep also exposes `temp` and `joint`
dimensions so qualification and diagnostic campaigns share these stress axes.
The dataset-geometry guard derives the unique L/NFIN/temperature requests from
the catalog and checks that the corresponding training grid and PDK bin exist.
Voltage and body-bias support is assessed separately from accepted LEVEL=72
trajectories.

## Harness and campaign flow

`tests/common/simple_circuit_harness.py` provides an engine-neutral `Trace`,
multi-signal DC/transient/AC adapters, interpolation, aggregate MRE/R²/NRMSE/
maximum error, domain metrics, reference-repeat stability and support
diagnostics. Support is computed only from accepted LEVEL=72 physical node
trajectories in the source-relative terminal frame; candidate Newton trial
overshoot is never classified as a dataset gap.

Run a subset directly:

```bash
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_circuit_topologies.py \
  --case current_mirror,inverter_chain --tech TSMC5 --corner nominal
```

Use `--case all --corner all` for the declared matrix and
`--reference-repeats 3` when gathering promotion evidence. Results go under
`results/tests/simple_circuits/simple-v2/` by default and every requested
analysis emits a structured marker. An `ERROR` stays in the denominator while
numerical aggregates include only characterized rows.

For checkpoint campaigns, `scripts/v710_regate_jobs.py` writes a separate
nominal-corner `jobs_simple_v2.txt` screening pool. Full corner characterization
uses the unified CLI's `--corner all` mode. The shell driver maps catalog suite
IDs to the unified CLI, the collector stores structured results, and
`scripts/v730_coverage.py --simple-version simple-v2` audits coverage. The
default coverage mode remains `simple-v1`, preserving historical reports.

## Promotion rule

Do not fold these diagnostics into `simple-v1`. Create a new score version only
after all of the following are true:

1. LEVEL=72 produces three repeatable, complete traces at every proposed cell.
2. Each gate answers one declared question with a frozen threshold and unit.
3. The exact case/technology/corner/analysis denominator is immutable.
4. Candidate and reference topology parity, artifact completeness, checkpoint
   hashes, commit and CPU thread settings are recorded in one campaign.
5. No partial logs or infrastructure failures are treated as scientific
   passes or silently removed from the denominator.

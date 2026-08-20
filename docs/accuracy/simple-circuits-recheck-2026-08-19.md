# DirectNet / BSIM-AR simple-circuit recheck — 2026-08-19

## Scope and provenance

This is the prerequisite recheck before any new AnalogGym/complex-corpus
claim. It runs every clean **S/M/L/XL** checkpoint for DirectNet (LEVEL=73)
and BSIM-AR (LEVEL=74), across TSMC5/6/7/12/16 and all eight scored suites.
Ring and Miller-opamp DC cells run at OMP 1, 2 and 4; the other six suites run
at OMP 1. The complete matrix is **480 isolated CPU-pinned runs**.

| item | value |
|---|---|
| simulator, model and deck code | `24c181a` (`feat/directnet-analoggym`) |
| campaign validation and report tooling | this report's containing commit |
| checkpoints | retained V7.4.0 clean S/M/L/XL matrix |
| checkpoint artifacts | 280 (`.pt`, normalization/config sidecars, completion markers) |
| checkpoint manifest SHA-256 | `8e4245f1ab563cd116a789cb02388e0f7b736186141694d3242ede2a7ed07868` |
| scored execution | CPU only; OpenMP, MKL and Torch explicitly pinned per cell |
| raw evidence | `results/simple_recheck_24c181a/` (local, gitignored) |

The retained matrix is the only complete on-disk S/M/L/XL checkpoint set and
is the exact set behind the checked-in V7.4.0 tables. A separate
`results/v742_regate/` tree measures later, differently trained checkpoints,
but those weights are no longer present. Its cells are historical evidence
and are not mixed into this pass.

## Result against the checked-in old reports

Strict complex score requires every ring/opamp OMP run to pass.

| family | tier | old report | current recheck | change |
|---|---|---:|---:|---:|
| DirectNet | S | 11/20 | **8/20** | -3 |
| DirectNet | M | 12/20 | **11/20** | -1 |
| DirectNet | L | 14/20 | **12/20** | -2 |
| DirectNet | XL | 15/20 | **12/20** | -3 |
| BSIM-AR | S | 18/20 | **13/20** | -5 |
| BSIM-AR | M | 17/20 | **12/20** | -5 |
| BSIM-AR | L | 15/20 | **12/20** | -3 |
| BSIM-AR | XL | 13/20† | **12/20** | not a valid direct delta |

† The old BSIM-AR XL numerator came from 13 passes among the 16 valid
non-TSMC12 cells. All four TSMC12 complex cells were backed by race-corrupted
logs, yet the report included them in the denominator as failures. The honest
old result is **13/16 plus four invalid cells**, not 13/20.

The current per-tier circuit structure is simple:

| family | tier | ring | opamp DC | SRAM | switchcap |
|---|---|---:|---:|---:|---:|
| DirectNet | S | 2/5 | 0/5 | 5/5 | 1/5 |
| DirectNet | M | 2/5 | 0/5 | 5/5 | 4/5 |
| DirectNet | L | 2/5 | 0/5 | 5/5 | 5/5 |
| DirectNet | XL | 2/5 | 0/5 | 5/5 | 5/5 |
| BSIM-AR | S | 4/5 | 0/5 | 5/5 | 4/5 |
| BSIM-AR | M | 2/5 | 0/5 | 5/5 | 5/5 |
| BSIM-AR | L | 2/5 | 0/5 | 5/5 | 5/5 |
| BSIM-AR | XL | 2/5 | 0/5 | 5/5 | 5/5 |

Device-suite totals explain the remaining report changes:

| family | suite | S | M | L | XL | comparison |
|---|---|---:|---:|---:|---:|---|
| DirectNet | parametric DC | 69/69 | 69/69 | 69/69 | 66/69 | unchanged |
| DirectNet | parametric transient | 80/80 | 80/80 | 80/80 | 80/80 | unchanged |
| DirectNet | device AC | 0/10 | 0/10 | 0/10 | 0/10 | old: 7, 10, 9, 9 /10 |
| DirectNet | opamp AC | 0/5 | 0/5 | 0/5 | 0/5 | old: 0, 0, 2, 2 /5 |
| BSIM-AR | parametric DC | 67/69 | 68/69 | 65/69 | 67/69 | unchanged |
| BSIM-AR | parametric transient | 80/80 | 80/80 | 80/80 | 80/80 | unchanged |
| BSIM-AR | device AC | 0/10 | 0/10 | 0/10 | 0/10 | old report: 9, 10, 10, 10 /10 |
| BSIM-AR | opamp AC | 0/5 | 0/5 | 0/5 | 0/5 | old report: 1, 0, 1, 1 /5 |

## Bugs found

1. **Intermediate homotopy convergence became a final verdict.** The old
   solver kept a sticky `final_converged` flag: a solved source/GMIN step could
   make a later-diverged physical step look converged. Commit `c96dd09` fixed
   that. With the same checkpoints, every Miller opamp now fails final-step
   convergence; the old DirectNet 11→12→14→15 and BSIM-AR 18→17→15→13
   capacity narratives therefore included false-positive fixed points.
2. **AC response errors were scored at non-fixed states.** Commits `104d7af`
   and `efe9455` require a converged DC operating point before linearization.
   Device AC is consequently 0/10 and opamp AC 0/5 at every tier in both
   families. The old response-shape “passes” are diagnostics, not gate
   evidence.
3. **Race-corrupted logs counted as failures.** Every one of the 12 old
   `tf/xl/tsmc12` logs has two completion markers. The collector recorded
   `rc=RACED`, but the report path treated any nonzero string as a scientific
   failure. Coverage also accepted invalid/timeout entries from raw logs. The
   driver now locks each log and retries invalid prior markers; the collector,
   coverage map and report completeness check exclude `RACED`, `no-ckpt`,
   timeout 124 and process failures ≥126 from gate denominators.
4. **The clean campaign was not expressible as one complete job pool.** The
   generator omitted the TSMC6 repeat from its normal technology list and had
   no DirectNet+BSIM-AR S/M/L/XL clean pool. It now emits exactly 480 unique
   jobs; the collector also renders TSMC6 instead of silently dropping it.
5. **Checkpoint archive pins were ignored.** The gate driver and coverage map
   hard-coded the active checkpoint directory. Both now honor
   `BSIMAR_CHECKPOINT_DIR`, so a preserved matrix can be selected without
   copying or overwriting current checkpoints.
6. **Generator selection and checked-in reports had drifted.** The report
   builder had been repinned to V7.4.2, while the checked-in clean tables still
   contained V7.4.0 output. The clean reports are now pinned to this one
   complete current-code pass, and generation fails closed when any required
   run is partial, invalid, or lacks the suite-specific metrics consumed by
   its table.
7. **Coverage could look complete after an invalid rerun.** A stale
   `data.json` verdict could survive a newer raced, timed-out, killed or
   unfinished raw log for the same cell. Raw logs now override the same-pass
   snapshot even when they carry no scientific verdict. Lock contention also
   returns an infrastructure failure instead of success, and `--fail-on-gaps`
   makes missing measurements or checkpoint groups fail the coverage command.

## Deck and model audit

The changed verdicts were investigated only after checking problem identity.
All preserved paired PyCircuitSim source-deck artifacts are identical:
**60/60 device-AC decks** and **45/45 Miller-DC decks** match byte for byte
between the old and current runs, including topology, sources, models, options
and measurements. Paired NGSPICE decks have the same electrical statements
after stripping later comments/library-location changes. The TSMC12-XL race
group's fresh SRAM metric is also identical (2.44% lobe NRMSE) while the
isolated verdict is PASS.

No unexplained device/model regression remains. Across parametric DC,
transient, ring, SRAM and switchcap, **273/273 paired runs with valid old and
new verdicts agree**. All other changes are accounted for by the corrected
convergence contract, the AC prerequisite, or an invalid old race cell.

## Reproduce

Run the [complete clean checkpoint matrix](../../README.md#run-the-complete-clean-checkpoint-matrix)
from the repository root.

# Simple-circuit evaluation and dataset coverage audit

Date: 2026-08-25

## Resolution in V7.5.17

Resolved. Scientific DC/VTC gates now reject unconverged solves, execute and
account for every declared matrix cell, and compare signed terminal current.
The scored device matrix adds temperature endpoints, both body-bias signs,
reverse VDS, and a joint L/NFIN/temperature corner; inverter gates use exact
dataset-supported N/P pairs for ratios 0.5, 1.5, and 2.0.

Dataset generation now pins `max_l_ratio=1.35`, generates the transmission-
gate corridor, fails on any rejected point or bin by default, and records a
per-bin requested/kept/rejected manifest plus command, source commit, OSDI and
modelcard hashes, and a completion marker. The trajectory corridor remains an
external harvested class and is named explicitly in the dataset variant
metadata. Training caps and fine-tune subsets are stratified, while the default
train/validation/test split holds out complete `(technology, VT, L, NFIN, T)`
groups. The geometry guard now verifies the exact variant, temperature, PDK
bin, and joint `(L, NFIN)` proximity for every scored configuration; the
authoritative gate definition and result are recorded in
[`docs/accuracy/simple-circuits-dataset-coverage-v7517.md`](../accuracy/simple-circuits-dataset-coverage-v7517.md).

## Scope

This is a read-only audit of:

- the DirectNet/BSIM-AR simple-circuit parametric evaluation;
- the BSIM-CMG dataset generator and current TSMC dataset artifacts;
- coverage of input bias, polarity, VT, gate length, NFIN, temperature, and
  PMOS/NMOS sizing ratio;
- tests that are intended to prove that coverage.

No simulation campaign or retraining was run. The checks performed were source
inspection, dataset metadata/class/label inspection, generator-bin enumeration,
the existing geometry guard, and two focused parser tests.

## Executive verdict

The current artifacts have good *marginal* coverage: five TSMC technologies,
both device polarities, every declared VT variant, three temperatures, a dense
forward-bias grid, and selected L/NFIN knots. The current `max_l_ratio=1.35`
artifacts pass the baseline geometry guard 10/10.

They do **not** establish full combination coverage. The evaluator varies most
axes one at a time, NFIN 5 and 10 are not exact generator knots, only one
PMOS/NMOS ratio is actually exercised, temperature/body-bias/reverse-bias
corners lack scored simple-circuit gates, and the train/validation/test split is
row-random rather than combo-held-out. Two harness defects can also turn
incomplete evidence into a scientific verdict: unconverged DC points are
stored and scored, and a failed baseline removes all of that technology's
parametric cases from the denominator.

## Coverage matrix

| Axis | Dataset generation/current artifacts | Simple-circuit evaluation | Verdict |
|---|---|---|---|
| Technology | TSMC5/6/7/12/16 | Same five in `NN_TECHS` | Covered |
| Device polarity | Separate NMOS and PMOS artifacts for every technology | Single-device DC runs both; inverters combine both | Covered |
| VT | All declared variants are present in current label sidecars | Single-device DC varies VT one polarity at a time | Marginally covered; no joint VT x geometry grid |
| Temperature | 248.15, 300.15, 398.15 K | Simple-circuit decks use 27 C | Training-only; extremes are not gated |
| Gate length L | Current artifacts use intra-bin knots with `max_l_ratio=1.35` | Baseline plus a small per-tech L list, one axis at a time | Baselines pass proximity guard; sweep points are not all exact knots or guarded |
| NFIN | PDK NFIN-bin boundaries only | Baseline 2; single-device 5/10; complex suites also use 3/5/10 | 2 and 3 are exact; 5 and 10 are interpolation points |
| PMOS/NMOS ratio | No paired concept: NMOS and PMOS are generated independently | Default N=2/P=2 plus only N=2/P=3 survives pruning | Only ratio 1.5 is exercised; advertised 0.5 and 2.0 are skipped |
| Bias input | Source-relative `Vs=0`; forward Vg/Vd grid to 2 VDD, Vbs sampling, targeted reverse Vds | DC directly sweeps only Vgs at Vds=0.5 VDD and Vbs=0; circuit trajectories add indirect coverage | No scored direct Vds/Vbs/reverse-bias matrix |
| Joint combinations | Full cross-product only among each variant's *selected* L knots, NFIN boundaries and three temperatures | Predominantly one-factor-at-a-time | Not covered as an interaction matrix |
| Holdout/generalization | Rows from every sampled bin are mixed | Reported validation/test metrics use that row split | No held-out L, NFIN, VT, temperature, or bias trajectory |

### Exact generated NFIN knots

The current `max_l_ratio=1.35` recipe enumerates the following exact NFIN
values after excluding NFIN < 2:

| Technology/device | Exact NFIN knots |
|---|---|
| TSMC5 NMOS/PMOS | 2, 3, 4, 6, 12.001 |
| TSMC6/7 NMOS | 2, 3, 6, 12, 20.888 |
| TSMC6/7 PMOS | 2, 3, 6, 12, 20, 24.888 |
| TSMC12/16 NMOS/PMOS | 2, 3, 4, 6, 20.888 |

Therefore the evaluator's NFIN 5 and 10 cases are useful interpolation tests,
but they are not direct sampled-combination tests. N=2/P=3 is directly
supported by the separate polarity datasets; a ratio itself is not a dataset
feature.

## Findings

### P1 - DC/VTC evaluation scores unconverged points

`run_dc_sweep()` receives the solver object and its honest
`_last_solve_converged` flag, but does not inspect it for either the initial OP
or each sweep point. It appends the returned voltages/currents unconditionally.
The retry path explicitly permits returning best-available voltages with the
flag false. The parametric harness then treats those arrays as normal numeric
curves.

Evidence:

- [`_pseudo_transient_dc`](../../pycircuitsim/simulation.py#L54-L57) documents
  the false-flag return contract.
- [`run_dc_sweep`](../../pycircuitsim/simulation.py#L396-L397) accepts the OP
  without a convergence check.
- [`run_dc_sweep`](../../pycircuitsim/simulation.py#L483-L504) stores every
  point without checking `point_solver._last_solve_converged`.
- [`run_single_nn_dc`](../../tests/common/nn_sweep.py#L314-L343) and
  [`run_single_nn_inv`](../../tests/common/nn_sweep.py#L346-L383) score the
  returned curves without convergence metadata.

Impact: a numerically plausible non-solution can pass the DC or inverter-VTC
gate. This violates the project's rule that unconverged responses are
diagnostics, not gate evidence.

### P1 - Baseline failure silently shrinks the parametric denominator

`run_nn_multi_tech()` adds one failed baseline row, then skips every L, NFIN,
VT, ratio, VDD, load, slew, and pulse-width case for that technology/analysis.
The summary denominator is simply `len(results)` and has no expected-matrix
completeness check.

Evidence: [`tests/common/nn_sweep.py`](../../tests/common/nn_sweep.py#L389-L416)
and its summary counting at [lines 443-463](../../tests/common/nn_sweep.py#L443-L463).

Impact: model quality changes the number of configurations evaluated. A report
cannot distinguish a deliberately complete matrix from one pruned after early
failure, and difficult combinations disappear precisely when they matter most.

### P1 - Dataset generation succeeds after losing requested rows or bins

An invalid point increments `n_failed` and is omitted. A wholly failed bin is
represented by `None` and is omitted. Assembly raises only when *every* bin
fails; otherwise generation writes the artifact and exits successfully. The
metadata does not preserve the failed coordinates, requested-bin manifest, or
per-bin kept/failed counts.

Evidence:

- point omission: [`generate_one_bin`](../../external_compact_models/bsim_cmg/pycmg/nn_generate.py#L939-L946);
- permissive assembly: [`_assemble`](../../external_compact_models/bsim_cmg/pycmg/nn_generate.py#L1035-L1059);
- incomplete metadata: [`generate_dataset`](../../external_compact_models/bsim_cmg/pycmg/nn_generate.py#L1215-L1236);
- the batch driver declares success solely from process exit status:
  [`benchmark_gen_data.sh`](../../scripts/benchmark_gen_data.sh#L53-L63).

The available generation logs record successful artifacts despite 1,370
failed TSMC5 PMOS points and 335 failed points in each TSMC6/7 PMOS artifact:
[`gen_tsmc5_pmos.log`](../../results/benchmark_sml/gen_logs/gen_tsmc5_pmos.log#L77-L90),
[`gen_tsmc6_pmos.log`](../../results/benchmark_sml/gen_logs/gen_tsmc6_pmos.log#L57-L70),
and [`gen_tsmc7_pmos.log`](../../results/benchmark_sml/gen_logs/gen_tsmc7_pmos.log#L57-L70).

Impact: “all datasets complete” does not mean the requested matrix is complete.
The lost bias coordinates cannot be reconstructed from the saved artifact.

### P2 - The dataset geometry guard can pass a structurally incomplete artifact

The guard reduces the artifact to two global independent sets, `unique(L)` and
`unique(NFIN)`. It does not verify that an `(L, NFIN)` pair occurs together,
does not partition by VT variant or temperature, and checks only each
technology's baseline geometry. A file missing an entire VT, a temperature, or
the target pair can still pass if its separate L and NFIN values appear
somewhere else.

Evidence: [`_dataset_geometry`](../../tests/single_devices/verify_data_geometry_coverage.py#L75-L82)
and [`check`](../../tests/single_devices/verify_data_geometry_coverage.py#L85-L123).

The current artifacts pass this limited guard 10/10; that result should be
read as baseline-axis proximity, not full combination coverage.

### P2 - The known-required L densification is optional and off by default

`--max-l-ratio` defaults to `None`, reproducing the known-defective
lower-corner-only L grid. The nominal batch script also relies on optional
`GEN_EXTRA` instead of pinning `--max-l-ratio 1.35`. Running the documented
command without that environment variable can overwrite the current good
artifacts with the old sparse grid.

Evidence: [`generate_nn_data.py`](../../external_compact_models/bsim_cmg/scripts/generate_nn_data.py#L187-L200)
and [`benchmark_gen_data.sh`](../../scripts/benchmark_gen_data.sh#L14-L19).

Current artifacts are not affected: their metadata reports
`meta_max_l_ratio=1.35`, and the baseline geometry check passes 10/10. This is
a regeneration/reproducibility defect.

### P2 - Test and training splits do not measure combo generalization

The loader randomly permutes individual rows, so bias points from every
`(technology, VT, L, NFIN, T)` bin normally appear in train, validation, and
test. `max_rows`, when used, is also an unstratified global sample.

Evidence: [`load_and_split_bsimar`](../../external_compact_models/neural_network/data/dataset.py#L151-L157)
and [the split](../../external_compact_models/neural_network/data/dataset.py#L198-L204).

Impact: test metrics measure interpolation among neighboring points from the
same bins, not generalization to an unseen geometry, variant, temperature, or
circuit trajectory.

### P2 - Current artifacts omit two known circuit-trajectory sample classes

The generator declares `traj_corridor` and `tg_corridor`, but explicitly does
not generate them in `generate_one_bin`; separate post-processing scripts are
required. All ten current TSMC artifacts contain classes 0-5, 7, 10 and 11,
but none contain class 12 or 13.

Evidence: the class contract in
[`nn_generate.py`](../../external_compact_models/bsim_cmg/pycmg/nn_generate.py#L76-L98)
and the batch recipe, which invokes no corridor append step,
[`benchmark_gen_data.sh`](../../scripts/benchmark_gen_data.sh#L28-L48).

Impact: the checked-in comments identify deep reverse-Vds/body-bias
transmission-gate states as a known gap, yet the default/current data recipe
does not cover them. This is especially relevant to the switch-capacitor
simple-circuit suite.

### P2 - The advertised PMOS/NMOS-ratio sweep contains one effective point

The builder iterates ratios 0.5, 1.5 and 2.0, but with baseline NFIN=2 it clamps
0.5 back to 2 and rejects 2.0 because PMOS NFIN=4 leaves the allowed PDK group.
Only N=2/P=3 (ratio 1.5) remains.

Evidence: [`build_inv_parametric`](../../tests/common/nn_sweep.py#L236-L257).

This limitation is documented in the function, so it is not a hidden control-
flow bug. It is nevertheless insufficient evidence for plural “N/P ratios.”
The dataset cannot solve this at pair level because polarities are generated
independently; circuit gates must deliberately compose more legal pairs.

### P2 - Single-device DC magnitude scoring masks current-sign regressions

Both references and NN results are converted to absolute current before the
metric. A device returning the correct magnitude with the wrong terminal-current
sign passes this gate. The separate sign routine is a diagnostic and therefore
cannot substitute for a scored gate under the repository's verification rules.

Evidence: NMOS magnitude conversion in
[`nn_gate.py`](../../tests/common/nn_gate.py#L730-L732) and PMOS conversion in
[`nn_gate.py`](../../tests/common/nn_gate.py#L1043-L1048).

### P3 - “Stratified” fine-tune subsets are unstratified random subsets

`_save_finetune_split()` calls a global random permutation “stratified.” A
small subset can omit complete VT/L/NFIN/T bins and rare sample classes; the
same risk applies to `max_rows` in the loader.

Evidence: [`generate_nn_data.py`](../../external_compact_models/bsim_cmg/scripts/generate_nn_data.py#L57-L74).

### P3 - Dataset provenance is insufficient to bind artifacts to a recipe

Artifact metadata records major sampling flags but not the source commit,
OSDI/modelcard hashes, expected/actual bin manifest, generator command, or a
completion marker. The available logs' output paths refer to the former
`external_compact_models/bsimar/data/datasets` location, whereas current files
live under `external_compact_models/neural_network/data/datasets`. Row counts
and recipe metadata agree, but the repository cannot cryptographically prove
that those logs produced these exact files.

## Existing automated coverage

- `test_scan_pdk_geometry_combos_tsmc7` checks one TSMC7 PMOS variant's legacy
  boundary enumeration. It does not test `max_l_ratio`, other technologies,
  NMOS, temperatures, or generation assembly.
- `verify_data_geometry_coverage.py` checks ten baseline technology/device
  geometries against current files, with the limitations described above.
- No test imports `pycmg.nn_generate` to assert the variant x L x NFIN x T
  matrix, polarity mirroring, sample-class counts, failure policy, or metadata.
- No test verifies combo-held-out splitting or fine-tune stratification.

Focused checks run during this audit:

```text
verify_data_geometry_coverage.py: 10/10 PASS
test_scan_pdk_geometry_combos_tsmc7: PASS
test_tech_registry_all_techs: PASS
```

## Recommended test plan

1. Make generation produce a requested-bin manifest and fail on unexpected
   missing bins/rows; preserve expected, kept, rejected, and reason counts in
   metadata.
2. Replace the geometry guard's independent axes with exact keys
   `(tech, device, VT, L, NFIN, T)` and validate every evaluation-requested
   geometry, including interpolation-distance policy where exact sampling is
   intentionally absent.
3. Add an expected-matrix assertion before reporting evaluation totals. Keep
   every skipped/error configuration in the denominator.
4. Require converged OP and every converged DC sweep point before scoring.
5. Add signed-current assertions to the scientific gate.
6. Add explicit held-out suites: leave-one-L-bin-out, leave-one-NFIN-interval-
   out, temperature holdout, body-bias/reverse-Vds, and legal asymmetric N/P
   pairs. Keep the existing random-row split only as an interpolation metric.
7. Pin the safe L recipe in the canonical generation command and either include
   trajectory/TG corridors or make their absence an explicit dataset variant.
8. Make fine-tune/max-row sampling stratified by at least
   `(VT, L, NFIN, T, sample_class)` and assert no required stratum is lost.

## Bottom line

The current dataset is materially broader than the original sparse grid and is
adequate for the baseline geometries. It does **not** cover the requested space
as a rigorously tested multidimensional matrix. Until convergence, denominator,
exact-key coverage, and held-out-combination checks are added, reported passes
should be interpreted as evidence for the sampled one-dimensional slices—not
for arbitrary L x NFIN x device x input x N/P-ratio combinations.

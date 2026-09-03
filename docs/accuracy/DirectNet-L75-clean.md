# DirectNet-Full (LEVEL=75) — clean large simple-circuit qualification

Date: 2026-09-01

This is the latest clean, source-pinned qualification of the four-terminal
`large` DirectNet-Full model. It learns the independent OSDI surfaces
`i_d`, `i_g`, `i_b`, `qd`, `qg`, and `qb`; source current and charge
are closed analytically. NGSPICE LEVEL=72 on the identical BSIM-CMG OSDI model
is ground truth.

The campaign covers the declared gates backed by canonical parameterized
templates in `circuit_templates/`. Results from earlier checkpoints are
not mixed into this campaign.

Gate definitions and denominator rules are owned by
[`methodology.md`](methodology.md).

## Outcome

**The requested simple-circuit qualification passes: 20/20 strict cells with
zero OMP verdict flips.** The same checkpoints also pass 100/100 parametric
inverter configurations and 10/10 common-source AC cells.

This is not a general LEVEL=75 production promotion: parametric device DC is
115/129 and Miller open-loop AC is 2/5.

## Final strict gate table

Ring and Miller DC were repeated at OMP 1, 2, and 4. SRAM and switch-cap used
the declared deterministic OMP=1 contract. Every declared job completed and
all thread-count verdicts agreed.

| technology | ring oscillator period error | Miller DC gain error | SRAM worst-lobe NRMSE | switch-cap charge error / VDD | strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSMC5 | **PASS** 0.02% | **PASS** 0.51% | **PASS** 7.35% | **PASS** 0.04% | **4/4** |
| TSMC6 | **PASS** 0.09% | **PASS** 3.43% | **PASS** 1.96% | **PASS** 0.03% | **4/4** |
| TSMC7 | **PASS** 0.09% | **PASS** 3.43% | **PASS** 1.96% | **PASS** 0.03% | **4/4** |
| TSMC12 | **PASS** 0.04% | **PASS** 0.01% | **PASS** 3.20% | **PASS** 0.00% | **4/4** |
| TSMC16 | **PASS** 0.08% | **PASS** 0.87% | **PASS** 3.50% | **PASS** 0.17% | **4/4** |
| **all** | **5/5** | **5/5** | **5/5** | **5/5** | **20/20** |

The strict thresholds are period error ≤5%, Miller DC gain error ≤10%, SRAM
worst-lobe NRMSE ≤10% with positive lobes, and switch-cap charge error ≤5% of
VDD together with the gate's convergence and state checks.

## Device and inverter error

Numeric aggregates include only configurations that returned commensurate
metrics. Explicit failures remain in each denominator.

### Parametric device DC — 115/129

| technology | pass | mean NRMSE | mean MRE | minimum R² | maximum error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSMC5 | 22/26 | 4.542% | 28.649% | -1.97134 | 494.3 µA |
| TSMC6 | 22/26 | 5.915% | 28.594% | -4.21186 | 483.6 µA |
| TSMC7 | 17/21 | 7.267% | 35.207% | -4.21186 | 483.6 µA |
| TSMC12 | 29/30 | 1.966% | 7.809% | -2.24170 | 198.3 µA |
| TSMC16 | 25/26 | 2.751% | 9.643% | -2.14903 | 198.3 µA |
| **all** | **115/129** | — | — | — | — |

The 14 failures are concentrated in the +125 °C and joint
length/NFIN/temperature configurations for TSMC5/6/7, plus the TSMC12/16 PMOS
joint configurations. Increasing circuit closure therefore does not erase
the remaining hot/joint device error.

### Parametric inverter VTC/transient — 100/100

| technology | pass | mean NRMSE | mean MRE | minimum R² | maximum voltage error |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSMC5 | 20/20 | 0.838% | 3.215% | 0.99114 | 184.6 mV |
| TSMC6 | 20/20 | 0.827% | 3.438% | 0.99094 | 245.6 mV |
| TSMC7 | 20/20 | 0.827% | 3.438% | 0.99094 | 245.6 mV |
| TSMC12 | 20/20 | 0.724% | 3.685% | 0.99196 | 218.0 mV |
| TSMC16 | 20/20 | 0.779% | 3.582% | 0.99242 | 210.7 mV |
| **all** | **100/100** | — | — | — | — |

## AC gates

### Common-source device AC — 10/10

Cells report DC-gain error / f3dB ratio / magnitude NRMSE.

| technology | NMOS | PMOS |
| --- | --- | --- |
| TSMC5 | **PASS** 0.614 dB / 1.000 / 7.01% | **PASS** 0.255 dB / 0.891 / 2.82% |
| TSMC6 | **PASS** 0.512 dB / 1.122 / 5.40% | **PASS** 0.240 dB / 1.000 / 2.58% |
| TSMC7 | **PASS** 0.512 dB / 1.122 / 5.40% | **PASS** 0.240 dB / 1.000 / 2.58% |
| TSMC12 | **PASS** 0.251 dB / 1.000 / 2.65% | **PASS** 0.394 dB / 1.000 / 4.11% |
| TSMC16 | **PASS** 0.315 dB / 1.000 / 3.39% | **PASS** 0.488 dB / 1.000 / 5.08% |

### Miller open-loop AC — 2/5

The gate requires DC-gain error ≤3 dB, GBW ratio in [0.6, 1.67], phase-margin
error ≤15°, a valid refined LEVEL=72 reference bias, and a converged NN
operating point. Magnitude NRMSE is diagnostic.

| technology | verdict | DC-gain error | GBW ratio | phase-margin error | magnitude NRMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| TSMC5 | **FAIL** | 7.34 dB | 0.999 | 0.207° | 71.2% |
| TSMC6 | **PASS** | 2.60 dB | 0.986 | 1.09° | 17.6% |
| TSMC7 | **PASS** | 2.60 dB | 0.986 | 1.09° | 17.6% |
| TSMC12 | **FAIL** | 9.41 dB | 0.974 | 2.48° | 41.2% |
| TSMC16 | **ERROR** | — | — | — | — |

TSMC16 remains an explicit error because NN DC sweep point 51 at 0.4391 V did
not converge.

## Training and provenance

- Source, data generation, training, and scoring commit:
  `9cac5e9af8d603fe35231a26e11adfc0494e1b56` (clean).
- Training matrix: five technologies × NMOS/PMOS, 10 fresh full-terminal
  datasets and 10 `large` checkpoints, 58,940,980 accepted rows total.
- Recipe: `--output-contract full-terminal --apply-filter off --swa-mode ema
  --seed 42`, trained from scratch in an isolated
  `BSIMAR_CHECKPOINT_DIR`.
- Checkpoint bundles: model, normalization sidecar, dataset provenance, and
  completion marker hashes all validated.
- Focused pre-gate tests: 46/46 passed. Two CPU pin-memory warnings were
  emitted; no tests were skipped.
- Scored campaign: all 60/60 fixed jobs collected; 52 returned exit 0 and
  eight returned scientific exit 1. Every log has exactly one numeric
  completion marker, and no log contains an unhandled traceback.
- Campaign manifest SHA-256:
  `e97ae8864ba2eb0a42cdf76e221a6f6c8c2018beefb6bdbc245790a2497386a9`.
- Fixed job-list SHA-256:
  `cb94baf83a0674f25e94957dfad7c1250329e7b8e1546164537458c8a4a7e281`.
- Collected data SHA-256:
  `9c8baa2b9ca2fd2b336cbd76d96c76503378c0bf19c75348adf3e1b59e827e70`.
- Generated report SHA-256:
  `781e0993a9bc782806ded4cd95d77dcd3466a16fff66c490f605a09a694ad8af`.
- LEVEL=72 OSDI SHA-256:
  `f089f17d5d5b1178c48932ff699960dab3ab509c33b34c798421eccbbf14a78b`.
- NGSPICE 45.2 SHA-256:
  `3b931f4ecb53a9e2e650087be470b3decc49654ce2237a204c6cb6b4ab45d764`.

Raw evidence is under
`results/directnet_full_simple_20260831/eval_9cac/`; checkpoint and dataset
artifacts are under `results/directnet_full_simple_20260831/`.

## Rejected infrastructure attempt

The first `eval/` attempt is not evidence. Its detached worktree could not
resolve the repository's untracked OSDI binary, so NGSPICE reported an unknown
`bsimcmg` device. The scored `eval_9cac/` campaign started fresh after an
NGSPICE smoke test and pins the exact OSDI hash above. No rows from the
rejected attempt are mixed into this report.

## Qualification decision

The four-terminal `large` DirectNet-Full checkpoints are **qualified for the
declared simple-circuit matrix**: 20/20 strict cells, 100/100 inverter
configurations, and 10/10 device AC cells pass. They are **not yet qualified
as a general LEVEL=75 replacement** because 14/129 device-DC configurations
and three of five Miller open-loop AC cells remain open.

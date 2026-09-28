# Faster NN compact-model evaluation

Status: proposed 2026-09-28; revised the same day with measured costs. This plan
changes development-time case selection only. It claims no accuracy result and
does not change the qualification denominator.

## Goal and boundary

Shorten the feedback loop for LEVEL=75/76 compact-model changes. Keep the full,
CPU-pinned campaign for qualification. A quick run must produce an explicit
case inventory, keep every requested result or `ERROR` row, and never be
described as a clean score.

The [catalog](../../tests/common/simple_circuit_catalog.py) declares four
`simple-v1` qualification cases and 30 `simple-v2` diagnostic cases (76
analyses). `main.py evaluate` selects `clean` and `canary` by default;
`simple_v2` is optional. The [job generator](../../scripts/v710_regate_jobs.py)
selects pools and cases; the [README](../../README.md#nn-workflow-from-the-project-root)
owns the commands.

## Where the time goes

Measured on V7.7.5 rounds 1–3, which together ran the generator's 1,840 nominal
cells ([round 2 runtime](../accuracy/v775-round2.md#runtime)); wall time from
log creation to verdict on a shared host, weighted by thread count:

| slice | thread-hours | share |
|---|---:|---:|
| full campaign | 1,892 | 100% |
| BSIM-AR-Full | 1,863 | 98.5% |
| BSIM-AR-Full xl | 969 | 51% |
| OMP 2/4 repeats of ring and Miller (160 cells) | 692 | 37% |
| TSMC6 | 395 | 21% |
| DirectNet-Full, all cases and sizes | 29 | 1.5% |

The default `clean` + `canary` run costs 901 thread-hours, of which 77% are the
OMP 2/4 repeats. The longest cell took 52.7 h (BSIM-AR xl `ring_osc_supply`),
and wall time is bounded by the longest cell. A cut therefore matters only if
it removes BSIM-AR cells, OMP repeats, or the longest cells.

## Quick profile

Keep every `.spice.tmpl` file; this is a selection, not a deletion.

| axis or case | quick decision | evidence | coverage kept |
|---|---|---|---|
| OMP 2/4 repeats | skip; OMP=1 only | 77% of the default run; V7.7.5 round 2 found one flip in 80 ring/Miller cells | qualification runs 1/2/4 ([methodology §3](../accuracy/methodology.md#3-determinism-and-execution)) |
| TSMC6 | skip | reference is TSMC7's ([§7](../accuracy/methodology.md#7-tsmc6-controlled-repeat)); in 5 of 8 family/size groups, 156 of 172 rows are bit-identical to TSMC7 | qualification `/20` |
| `bias_tree_fanout_3t`, `5t`, `9t` | defer; keep `17t` | every metric within 0.33% of 3T | full diagnostic pool |
| `ring_osc_supply` | defer | same template as scored `ring_osc`; 15% of the campaign; 12–53 h xl cells | scored ring gate |
| `switchcap_multicycle` | defer | same template as scored `switchcap`; only adds cycle drift | scored switch-cap gate |
| `beta_multiplier` | defer | unique start-up basin, but 11% of the campaign and cells up to 38 h; restore for self-bias or solver changes | full diagnostic pool |
| `inverter_chain` | defer | NRMSE is set by the NGSPICE `uic` start-up spike: 0.78% for all 8 models on TSMC5 | `inverter_energy`, `inverter_vtc_tran` |
| `transmission_gate_hold` | defer | 0.000% NRMSE for all 8 models on all 5 technologies; no discrimination | `transmission_gate_dc` |
| BSIM-AR sizes | explicit option: all four, small/medium/large, or the size under study | xl is 51% of the campaign | DirectNet always runs all four sizes (9 thread-hours) |

Kept: the four `simple-v1` gates at OMP=1, every L0 device suite, the canary,
and 26 catalog cases (L1 4, L2 9, L3 9, L4 4). The L4 transients of
`ota_5t_buffer`, `multistage_buffer_12t` and `ldo_regulator` start without
`uic`, as AGENTS.md requires.

Effect on the V7.7.5 measurements:

| selection | cells | cases | thread-hours | longest cell |
|---|---:|---:|---:|---:|
| full campaign | 1,840 | 42 | 1,892 | 52.7 h |
| A: defer `ring_osc_supply`, `switchcap_multicycle`, `beta_multiplier`, bias 5T/9T | 1,640 | 37 | 1,310 | 20.6 h |
| B: A + OMP=1 only | 1,480 | 37 | 618 | 16.9 h |
| C: B + no TSMC6 | 1,184 | 37 | 494 | 16.9 h |
| D: C + defer `inverter_chain`, `transmission_gate_hold`, bias 3T | 1,088 | 34 | 448 | 16.9 h |
| E: D + BSIM-AR small/medium/large | 952 | 34 | 208 | 12.3 h |
| F: D + one BSIM-AR size (large shown) | 680 | 34 | 109 | 9.0 h |

`quick` is profile D; E and F are its BSIM-AR size options.

A quick run cannot detect OMP flips, TSMC6 training variability, multi-cycle
drift, supply-dependent oscillation, the self-bias start-up basin, or the
behaviour of unselected BSIM-AR sizes. Those stay with the full campaign.

## Execution and verification

1. Re-derive the tables above from the complete V7.7.6 two-arm campaign
   (`ff4234b`, 1,840 cells per arm) and replace the V7.7.5 figures.
2. Implement `quick` as a named profile in `scripts/v710_regate_jobs.py`, with
   the BSIM-AR size option explicit. Record the profile in the campaign
   manifest. Add a contract test for the inventory: its cases and
   technologies, OMP=1, a case at every level L0–L4, and a `uic`-free L4
   transient.
3. Validate without a new campaign: quick cells are a subset of the full
   campaign, so filter the complete V7.7.6 evidence to the profile and list
   every failure class (support, convergence, qualification failure) the
   subset misses. Add the smallest case that exposes each missed class before
   adopting the profile.
4. Store quick runs under `results/` with a distinct selection identity; never
   combine them with a full-pool report.

## Out of scope

Deleting templates, or shrinking a published pool, needs a new versioned
denominator, complete evidence and updated gate documentation. The other lever
is BSIM-AR's per-cell cost, a median 39× DirectNet in V7.7.5 round 2; that is
runtime work, not case selection.

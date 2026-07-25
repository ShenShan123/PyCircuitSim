# Systematic-audit fix waves — plan and triage (2026-07-24)

Companion to `docs/2026-07-21-systematic-audit.md` (the finding register) and
`docs/plans/2026-07-24-a3-regate-status.md` (the V6.13.0 campaign). The audit's
P0 items are shipped: **A1** (ngspice return code) and **A2** (spsolve NaN) in
`e756481`, **A3** (gds sign + guard F) in `8ed35bd`. This file covers everything
that was left.

## How the remaining findings were triaged

All 43 open P1/P2 findings were re-verified against `d2ea720` by a 7-way
file-disjoint analysis pass — each one re-located in the current source (the
audit's line numbers had drifted), re-reproduced where a cheap pure-Python
repro was possible, and given a prescribed fix and a risk class. Three were
dropped on the evidence; the rest split by **whether the change can alter a
number a gate reports**.

That split is the whole point of the ordering. Every accuracy number published
in `docs/accuracy/` was measured on `d2ea720`. A gate-affecting fix invalidates
them, so it cannot land until the V6.13.0 re-gate is collected and committed,
and it carries a re-gate obligation of its own.

## Dropped — verified NOT to need a fix

| id | why dropped |
|---|---|
| `C2` | The NaN mechanism is real but unreachable — the softplus clamp is in no autograd graph (both callers cut the graph after it). See the note below. |
| `B5k` | Closed by `38c47d8` (TSMC6 retire), which post-dates the audit. `NCELL`/`NGATES` were always derived from `TECHS`; dropping TSMC6 made them 16. Verified by import: `gate_grid.NCELL == 16`, `recipe_retest_collect.NGATES == 16`, `grep -rn tsmc6 scripts/` empty. |
| `C6p` | Moot: TSMC6 is out of the registry (`38c47d8`) and its datasets are deleted, so the colliding fingerprint can no longer be constructed. A hardening idea (raise on an unregistered dataset stem instead of silently falling through to the universal scan) is recorded in the finding. |

`C2` deserves the detail: the NaN mechanism the audit describes is real, but the
softplus clamp is not in any autograd graph (both callers cut the graph after
it), so it cannot fire. More importantly the audit's prescribed rewrite
`F.softplus(bx, beta=1)/beta` was measured **not** forward-bit-identical — 23 of
401 samples differ in the last fp32 bit, because the linear branch becomes
`bx/beta` instead of `v_raw - v_min`. On a codebase whose opamp and ring basins
are documented as sensitive at that magnitude, shipping it would be a
gate-affecting change in exchange for nothing. If defence-in-depth is wanted
later, the argument-clamp variant measured bit-identical at 401/401.

## Wave 1 — gate-neutral, lands immediately (21 findings)

These make silent failures loud or fix paths no gate exercises. None changes a
value, a tolerance, or a numeric path.

| id | eff | what the fix does |
|---|---|---|
| `B3` | M | Two halves, both needed (fixing only one leaves the hole). (1) Workers: distinguish `verdict recorded` from `no verdict possible`. Keep `exit 0` when the cell … |
| `B4` | M | scripts/benchmark_collect.py, four changes: 1. :25 — drop the fabrication. `SIZES = [s for s in _SIZE_ORDER if (BASE / s).is_dir()]`; in main(), if not SIZES: … |
| `B5a` | S | Preferred: delete scripts/train_per_tech_8cells.sh. It is fully subsumed by `TECHS='tsmc5 tsmc7' SIZES='small medium' bash scripts/benchmark_train_sml.sh`, which … |
| `B5c` | S | tests/verify_complex_opamp.py. (1) next to GAIN_TOL at :44 add `OPAMP_MIN_GAIN = 5.0 # ng_gain below this V/V = out-of-region bias, not an opamp (mirrors … |
| `B5d` | S | tests/verify_nn_dc_tran.py, in get_available_checkpoints. Make a pin that resolves to nothing fatal, matching the parser's no-silent-fallback rule. DirectNet arm … |
| `B5e` | S | tests/verify_nn_dc_tran.py: add one helper next to _cascade_handles_stem (:1077) and use it at all four print sites (:1937, :1975, :2094, :2132): ` def … |
| `B5g` | M | tests/verify_subckt.py: add a `level0()` batch (no NGSPICE, no solver) and register it first in main's list at :485-487 as `("Level 0 (loud parse errors)", … |
| `B5h` | S | tests/diag_l72_complex_control.py. Ring, replace :151-152 with a three-way verdict: ` if not np.isfinite(err): verdict = ("INCONCLUSIVE (L72 period undefined: <3 … |
| `B5i` | M | tests/verify_nn_dc_tran.py, run_idvds_diagnostic. (1) Kill the vacuous pass at :3142-3143: `else: passed = False; reason = 'no Vds>0.01 samples in the NN sweep'` … |
| `B5j` | S | tests/diag_l72_switchcap_control.py. Add `CHARGE_TOL_PCT = 5.0 # % of VDD — mirrors verify_complex_switchcap.CHARGE_TOL` next to TRAN_TSTEP (:23) and replace the … |
| `B5l` | M | Validate up-front in all five main() functions, mirroring the pattern verify_nn_dc_tran.py:3207-3211 already uses. Immediately after `techs = [...]` (opamp :166, … |
| `B5n` | S | tests/verify_nn_lifted_source_dc.py main(). (1) Validate the filter at :154-156: ` want = {t.strip().upper() for t in args.techs.split(",") if t.strip()} unknown = … |
| `C1` | M | Rewrite the first pass in `Parser.parse_file` (pycircuitsim/parser.py:408-455) to buffer EVERY logical line, not just `.model`. Both existing branches (:435-446 … |
| `C6h` | S | pycircuitsim/parser.py. Add `self._seen_inst_paths: set = set()` to `Parser.__init__` (near :375-386), then in `_expand_instance` immediately after :1300: `python … |
| `C6i` | M | pycircuitsim/parser.py — canonicalize ground at the parser boundary so nothing downstream needs to change. Add near the top of the class: `python GROUND_ALIASES = … |
| `C6j` | S | pycircuitsim/parser.py, in `_parse_model`, replace :1122-1126 with a conflict check that stays idempotent for a doubly-`.include`d library: `python new_def = … |
| `C6l` | S | pycircuitsim/parser.py. First capture which env var supplied the value so the message can name it — change :106-108 to: `python _src_env, _override = next( ((n, … |
| `C6m` | S | pycircuitsim/parser.py. Add `LOCAL_UNKNOWN_CODE_ID` to the `from bsimar.config import (...)` at :86-89, then replace :303-309: `python _unknown = (UNKNOWN_CODE_ID … |
| `C6o` | M | In `eval/loo_labels.py::get_or_build_tech_variant_labels` (167-200): move `tech_filter = _infer_tech_filter(data_path_p.stem)` above the cache branch, then … |
| `C6q` | S | Guard the range; do NOT enlarge the vocabulary. Returning `NUM_TOTAL_CODES` (22) from `tech_scope_vocab_size('universal')` would change `nn.Embedding` row count … |
| `C6r` | S | Stop depending on call ordering. In `models/tabpfn.py`, next to the existing overrides (after line 521) add both: (1) `def _load_from_state_dict(self, *a, **kw): … |

## Wave 2 — gate-affecting, lands after V6.13.0 is committed (19 findings)

| id | eff | fix | re-validation it obliges |
|---|---|---|---|
| `B1` | M | pycircuitsim/solver.py::DCSolver._solve_newton. (1) Replace the latch with last-attempt semantics: at :983-984 write … | Re-run /tmp/claude-1001/-data2-shenshan- … |
| `B2` | L | pycircuitsim/solver.py, module level: add `_kcl_residual_inf(mna_matrix, rhs, x, num_nodes) -> (resid, i_scale)` next to … | /tmp/claude-1001/-data2-shenshan-PyCircuitSim/84b9ddae-37cf-4ec2-b9c6-fe290567dd45/scratchpad/b2_repro.py + … |
| `B5f` | M | scripts/gate_matrix_iso.sh: (a) Worker — invalidate before measuring. Insert `rm -f "$cell"` immediately before line 73 (after … | Re-run the /tmp harness at /tmp/claude-1001/-data2-shenshan- … |
| `B5m` | S | Two one-liners per script, no new machinery. (1) Reorder: move the checkpoint-existence test (benchmark_run_tests.sh:56-58, … | Predicate unit check, no gate: `printf '===BENCH_DONE no-ckpt===\n' | grep -q '===BENCH_DONE rc=' ; echo $?` … |
| `B6` | S | Two edits. (1) tests/common/complex.py:43-48 and tests/common/complex_ac.py:39-44 — reject empty, and stop swallowing a … | `PYCIRCUITSIM_TORCH_THREADS= python -c "import sys;sys.path.insert(0,'.');import tests.common.complex;import … |
| `C3` | M | pycircuitsim/parser.py — make the suffix table SPICE-correct and longest-match, and update all three consumers in one commit … | Pure-Python, no gate run: ``` /data1/shenshan/.conda/envs/pycircuitsim/bin/python -c " import sys; … |
| `C4` | S | pycircuitsim/models/passive.py:727-729, replace the two-line body with a three-way branch so the BE step hands its true current … | Cheap offline check first: /data1/shenshan/.conda/envs/pycircuitsim/bin/python /tmp/c4_repro.py — asserts … |
| `C5` | S | pycircuitsim/simulation.py, replace lines 401-417 in `run_dc_sweep` with a tolerance-terminated accumulator that snaps the final … | Cheap unit check (no solver): `python -c "from pycircuitsim.simulation import run_dc_sweep"` is not enough — … |
| `C6a` | M | `pycircuitsim/solver.py`, inside the `for step in range(1, num_steps)` loop. Derive the interval from the (already clamped) … | Re-run `/tmp/.../scratchpad/c6a_repro.py` (no ngspice, no NN, <1 s): after the fix the `dt=3e-10` line must … |
| `C6b` | L | `pycircuitsim/solver.py:2136-2246`. Replace the single-attempt retry with a real sub-interval walker so the clock and the … | Re-run `/tmp/.../scratchpad/c6b_repro.py` (<1 s, no ngspice/NN): the trace must become `3 @ 2.50e-09 … |
| `C6c` | S | pycircuitsim/solver.py::DCSolver._solve_newton, :854-867. Keep the damped update, but record the raw step for the test: inside … | Cheap self-check: temporarily assert at the `if all_converged:` break (:935) that `max(raw_deltas.values()) … |
| `C6d` | S | pycircuitsim/solver.py: delete the `_has_nn_device` guard in both places — :973-974 becomes `residual_ok = avg_residual[0] <= … | Telemetry pass first: `PYCIRCUITSIM_DEBUG_OSC=1 python tests/verify_bsimcmg_dc_comprehensive.py` (81 cases) … |
| `C6e` | S | Three small, independent pieces. 1. `pycircuitsim/solver.py`, `ACSolver.__init__` (:2398-2407): add `op_converged: … | No gate reads `ACSolver.op_converged`, `warnings`, or the `.lis` text, and no default-path numeric changes — … |
| `C6f` | M | Carry the previous step size and use the variable-step BDF-2 stencil. With h = t_n - t_{n-1}, h1 = t_{n-1} - t_{n-2}, w = h/h1: … | First, prove neutrality at HEAD's configuration: with constant dt, w == 1.0 exactly and the new expressions … |
| `C6g` | S | pycircuitsim/simulation.py, replace lines 798-810 with a geometric grid that reproduces NGSPICE's `f_k = fstart * base**(k/N)` … | `python -c "import numpy as np; f=10*10.0**(np.arange(101)/20); print(len(f), f[0], f[1], f[-1])"` -> `101 … |
| `C6k` | L | Three separable changes in pycircuitsim/parser.py; land (a)+(b) together, (c) separately. (a) Exact-token dispatch. In … | Parts (a)+(b), pure-Python and cheap: ``` /data1/shenshan/.conda/envs/pycircuitsim/bin/python -c " import … |
| `C6n` | M | Make selection track deployment, staged so existing records stay reproducible. (1) `trainer.py:149-166`: add `ar_val: bool = … | Cheap correctness check without a campaign: `python -m bsimar.cli.train --model transformer --size small … |
| `C6s` | S | `pycircuitsim/solver.py`, `_stamp_mosfet_dc`. Compute the effective gmb ONCE, before the stamp, and use it in both places: … | Re-run `/tmp/.../scratchpad/c6s.py` (<1 s, no ngspice/NN): after the fix the `g_mb=5.0e-13` row must print … |
| `C6t` | S | pycircuitsim/models/mosfet_cmg.py:227-230, replace the `abs()` with the same negative-only guard the NN path uses, plus a loud … | Static: `grep -n 'abs(g_ds)' pycircuitsim/models/mosfet_cmg.py` must return nothing after the fix. … |

### Wave 2 ordering constraints

- **B2 before B1.** B1's honest-convergence check wants a residual it can trust;
  landing it first would OR in the broken metric B2 exists to fix.
- **B2 is telemetry-first.** Ship a pass that logs the old and the new residual
  side by side on every acceptance, calibrate `_RESID_ABS_FLOOR` against the
  observed distribution on a passing gate, and only then flip the thresholds.
  1e-6 A is plausibly too tight for the mA-scale complex cells and too loose for
  nA SRAM retention, and the current gate is a no-op in both directions
  (measured: at the exact solution of a divider the residual scores *worse* than
  a 100 mV-perturbed point, and any circuit drawing >~100·reltol·VDD amps can
  never pass).
- **B1 needs its NR budget change measured separately.** Per-step budget is
  `max_iterations // num_steps` = 50//20 = **2** iterations, so the full-supply
  step is starved. Fixing the latch without giving the last step the remaining
  budget will push most NN operating points into the GMIN + pseudo-transient
  ladder — a wall-clock and basin risk, not a correctness one, but it must be
  observed rather than assumed.
- **C3 is one commit across three consumers.** `UNIT_SUFFIXES`, `_parse_value`
  and `_eval_expr` must not disagree about what `m`/`meg`/`mil` mean.
- **B5m before any resume-based re-run.** The `===BENCH_DONE no-ckpt===` pill
  contains its own skip sentinel, so it is permanent. Requiring `rc=` in the
  resume predicate retro-un-poisons every pill log already on disk.

### What wave 2 obliges

A full complex-matrix re-gate of at least the production DirectNet stems, plus
the device/parametric/L72 suites, on the post-wave-2 code — the same shape as
the V6.13.0 campaign. Until that lands, `docs/accuracy/` describes `d2ea720`
and must say so.

### Re-training obligations

**None from the V6.13.0 campaign itself.** All 36 checkpoint sets on disk — 28
per-tech (`tsmc{5,7,12,16}` × {dn, tf, pfn} × sizes and recipes) and 8 universal
stems — already existed and were re-*evaluated*; the gds fix is inference-side
and does not touch training data, loss or optimization. Nothing needed retraining
and nothing was retrained.

**One optional experiment comes out of wave 2.** `C6n` — LEVEL=74 selects its
checkpoint on **teacher-forced** validation loss while deployment is
free-running autoregressive, and the audit measured `gds` 33 % worse under AR.
The fix adds an opt-in `--val-mode ar`, so it changes nothing for existing
weights; realizing the benefit means **retraining the BSIM-AR family** under
AR-mode selection and re-gating it. That is a campaign in its own right, not
part of the fix wave, and it should be run against the post-wave-2 baseline so
the two effects are not confounded.

## Execution record

- Wave 1 implemented in the git worktree `/data2/shenshan/pcs-fixes` (branch
  `audit-fixes`), seven file-disjoint packages, so nothing lands in the
  V6.13.0 measurement commit on `main`.
- Additional finding not in the audit's numbered register, verified by hand and
  folded into wave 1: `external_compact_models/bsimar/config.py` does
  `sys.path.insert(0, PYCMG_DIR)`, so after any `import bsimar` the name
  `tests` resolves to **PyCMG's** `tests` package, shadowing the repo's own.
  Reproduced at HEAD. `sys.path.append` fixes it.

## Deferred to a follow-up pass (after the V6.13.0 campaign)

- **`scripts/gate_matrix_iso.sh`** — the 12th B3 dispatcher, plus all of **B5f**
  (numerator greps the all-time SUMMARY while the denominator is this
  invocation, giving ratios like "8/1"; and a `.cell_*` is never invalidated, so
  a SIGKILLed worker's prior PASS folds back into the rebuild). It was left
  byte-identical because it was *driving* the campaign while the fixes were
  written. Note the stale-cell arm did not affect V6.13.0: every `.cell_*` in
  `results/a3_regate/` was checked to post-date the gds fix commit, so no
  pre-fix verdict could have been folded in.
- **Cosmetic mislabel in `tests/verify_nn_ac.py`** (not in the audit): under
  `PYCIRCUITSIM_NN_FORCE_LEVEL=74/75` the report still prints the DirectNet
  banner and labels its own column `DN=`, so `nn_ac_tf.log` and `nn_ac_pfn.log`
  read as DirectNet results. Harmless to the verdict, actively misleading to a
  reader. Label from the resolved level.

# AnalogGym simulator-fix session notes (2026-08-10)

**Task:** worktree `feat/analoggym-migration`. Make all AnalogGym test circuits
pass gate, metrics match NGSPICE. Fix simulator (solver / parser / BSIM-CMG
L72 path) — NN models parked. Then update CLAUDE.md, CHANGELOG.md, README.md.
User: workflows allowed, max 5 agents.

## Source of truth
`examples/analoggym/RESULTS_TSMC.md` §"PyCircuitSim versus NGSPICE" — 7-deck
TSMC5 pilot. Three root causes recorded there:

1. **Subthreshold current floor** (blocks sensing_front_end + voltage_reference,
   75 decks): L72-in-PyCircuitSim returns id=0.0 *exactly* at Vgs=55.3mV,
   2.16nA at 61.9mV; NGSPICE (same OSDI) nonzero. m-independent. Suspect
   PyCMG internal-node solve: `pycmg/core.py solve_internal_nodes` (200 it,
   tol from env NN_DC_SOLVE_TOL default **1e-9 A absolute residual**, NR step
   clamp ±0.2V) — at sub-nA current levels the 1e-9 A tolerance accepts the
   initial guess / wrong internal-node state. `model.py:570` reads the tol.
2. **Amplifier tb_dc 125C first-point divergence** (85 decks): sweep 125→-40C,
   0/67 points; same solve at 25C fine. Newton start problem.
   Harness has unmeasured recovery (`compare_with_recovery`), but goal is
   simulator-side fix.
3. **No per-terminal limiting** (charge pump dead 5 decks, transients partial
   ~110): `dv_limit` bounds Newton step only; gate walks to -2.96V on 0.65V
   rail, OSDI rejects point. Need damped fetlim/limvds-style limiting for L72
   in solver NR loop (NOT hard clamp — kills derivative).

## Key files
- `pycircuitsim/solver.py` (3457 lines) — MNA+NR, dv_limit, retry ladder.
- `pycircuitsim/models/mosfet_cmg.py` (442) — L72 wrapper; `_eval_dc` cache;
  NOTE line 288: `g_ds = abs(g_ds)` (contradicts CLAUDE.md rule "never abs")..
- `external_compact_models/PyCMG/pycmg/core.py` — `solve_internal_nodes`
  (line 575), `_nr_step` clamp ±0.2 (line 571), INIT_LIM set EVERY iteration
  (suspicious; should be first-iter only?).
- `external_compact_models/PyCMG/pycmg/model.py:533` `eval_dc` — tol env
  `NN_DC_SOLVE_TOL` default 1e-9; raises RuntimeError on internal divergence.
- Harness: `examples/analoggym/pycircuitsim_bench/{translate,measure,run_compare}.py`.
- Uncommitted (pre-existing, keep): run_compare.py fix — `_op_worst` reads
  `worst_abs` (was `worst` = always None, silently disabled recovery), narrow
  segment forced stride=1.

## Pilot deck status (from RESULTS_TSMC.md)
| deck | agree | note |
|---|---|---|
| amplifier tb_gain | 8/8 | OK |
| ldo tb_load | 11/11 | lr/lr_pp measurement-definition gap only |
| ldo tb_loop_max | 8/8 | OK |
| amplifier tb_tran | 2/6 | NR exhausts on falling slew edge (cause 3) |
| ptat_1 tb_dc | 5/13 | subthreshold floor (cause 1), splits at ~50C |
| amplifier tb_dc | 0/15 | 125C first-point divergence (cause 2) |
| chargepump tb_tran | dead | first step, all dt (cause 3) |

## Env facts
- conda env `pycircuitsim` at /home/shenshan/.conda/envs/pycircuitsim
- NGSPICE /usr/local/ngspice-45.2/bin/ngspice; worktree at
  /data2/home/shenshan/PyCircuitSim/.claude/worktrees/analoggym-migration
- Gates CPU-pinned: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

## Progress log
- [x] Read RESULTS_TSMC.md, mosfet_cmg.py, core.py internal solve, located tol.
- [ ] Task 1: repro subthreshold floor (single device PyCMG vs NGSPICE, Vgs 0-100mV)
- [ ] Task 2: fix floor
- [ ] Task 3: per-terminal limiting in solver NR
- [ ] Task 4: 125C DC start
- [ ] Task 5: re-run 7 pilot decks + repo gates
- [ ] Task 6: docs (RESULTS_TSMC.md, CLAUDE.md, CHANGELOG.md, README.md)

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
- [x] Task 1: root cause = PyCMG internal-solve tol 1e-9 A abs (model.py); NGSPICE ABSTOL=1e-12.
- [x] Task 2: default tol -> 1e-12 (committed 544e9f4). ptat_1 5/13 -> 9/13 @stride10
      (rest = stride artifacts + hot-end flags 115/120C, recheck full stride).
- [~] Task 3 (in progress), findings chain:
      1. Added fetlim/limvds pair-space limiting + eval about limited point in
         _stamp_mosfet_dc (limit kwarg; probes pass limit=False), _nr_limited
         convergence gates in DC+tran loops + both oscillation-acceptance paths,
         reset at DCSolver.solve + TransientSolver.solve entry, retreat-to-anchor
         loop (8x bisect) on eval RuntimeError. PMOS via _nr_sign=-1.
      2. Charge pump STILL died: eval fails at pairs (2.5,2.5,2.5) when ABSOLUTE
         source = -16.1V. Probe (nch_svt_mac l68n nfin4): same pairs eval OK at
         s=0 / s=-8.5, FAIL s=-16.1 warm AND cold => OSDI internal solve not
         shift-robust. FIX: _eval_dc now evaluates SOURCE-REFERENCED (subtract
         v_s; mirrors NN Rule 2). Device name/L/NFIN/m now appended to eval errors.
      3. Next failure: pure NR ping-pong +-0.65V (dv cap) at t=4e-11, resid
         pins at 2.048e3 at "min dt". ROOT CAUSE: dt-cut ladder halves
         current_sub_dt but keeps SAME target sub_time => companion demands
         full-interval dV in dt/2^n => STIFFER each halving. Broken by design.
      4. IN PROGRESS: rewrite retry ladder as true interval subdivision
         (2^n pieces walked sequentially, commit per piece, snapshot/rollback
         cap v_prev/v_prev2/_i_prev + mosfet _q_prev/_q_prev2/_v_prev_tran/
         _i_prev_gate/_i_prev_drain on failure). n=0 path byte-identical.
      5. Diagnostic added: PYCIRCUITSIM_NR_TRACE=1 traces transient NR.
- [~] Task 4 (in progress) — full failure chain unwound:
      a. Eval RuntimeErrors at 125C: internal-solve tol 1e-12 ABS unreachable in
         float64 when junction currents ~1e2 A (542A forward d-b diode at 2.5V/
         125C, pch_svt_mac l26 nfin2). FIXED: voltage-delta acceptance in
         solve_internal_nodes (accept step<1e-9 AFTER >=1 NR step; floor safe).
      b. Internal-state RATCHET: failed internal solves leave sim.solve garbage
         poisoning later evals. FIXED: reset internal nodes to cold + retry once;
         also reset before raising (pycmg/model.py eval_dc).
      c. pnjlim added on (b,s) & (b,d) pairs (normalized frame; b-d honored by
         adjusting ds), vcrit=0.6, vt from device T. _NR_LIM_VCRIT const.
      d. abs(gds) removed from get_conductance (Critical Rule 4; floor at stamp).
      e. gmin ladder widened for non-NN: [1e-2..1e-11, gmin] (NN keeps V5 2-level).
         FIXED pre-existing sticky final_converged bug (intermediate gmin level
         could mark whole solve converged → flag True with nodes at -666V);
         verdict now = last level, last source step; failed level restarts from
         last good homotopy point. NOTE: DCSolver default use_source_stepping=
         True/20 steps DIVIDES max_iterations (500//20=25 its/step) — harness
         passes False; probes must too.
      f. STILL failing at gmin<=1e-3: NR limit cycle, vout6 residual pinned
         +0.1716 A. ROOT CAUSE FOUND: stamp uses channel-only gds/gm/gmb opvars
         but id includes JUNCTION currents at 125C (measured M.xop5.Mm8:
         i_ds=+1.8e-3 A with gds=4.3e-13 S — d(i_junction)/dv ~ 0.07 S missing
         from Jacobian; also junction current d->b not routed to bulk terminal).
         NGSPICE stamps the FULL OSDI Jacobian. FIX IN PROGRESS: 4-terminal
         stamp for L72 via PyCMG get_jacobian_matrix (condensed 4x4 dI/dV,
         already NGSPICE-verified) + all four terminal currents (id,ig,is,ie),
         evaluated at the limited bias; NN path keeps 3-cond stamp.
         Sign conventions MUST be verified numerically (finite diff + inverter).
      g. FULL STAMP LANDED (commit c96dd09): PyCMG Instance.condense_last_jacobian
         (no re-eval condensation, == finite diff of (id,ig,is,ie) wrt (d,g,s,e));
         MOSFET_CMG.get_terminal_stamp -> (i_out, g4) with i_out=-I_pycmg*m,
         g4=-jac4*m; _stamp_mosfet_dc full 4x4 branch with gmin across d-s/d-b/
         s-b, i_eq about limited eval point. RESULT: 125C plain-NR cold start
         converges in 1.8 s; Alfio tb_dc stride50 no-recovery 11/15 (4 misses =
         hot-end branch difference vs NGSPICE monolithic), WITH recovery 15/15,
         worst node 2.6 uV, 41 s. verify_bsimcmg_{op,dc} green.
- [x] NR trace diagnostics: PYCIRCUITSIM_NR_TRACE=1 (transient + DC tails + gmin
      per-level summary).

## Task 5 status (2026-08-11)
Batches launched via $SCR/run_batches.sh {cp,bench1,bench2,gatesA,gatesB,gatesC,nn}
logs at $SCR/batch_<name>.log where SCR=/tmp/claude-1001/-data2-home-shenshan-
PyCircuitSim/2c8d61c4-d247-4adb-b589-7f06d9d42622/scratchpad.
- cp: chargepump tb_tran stride 20 (LONG; watch for SimStepLimit)
- bench1: ptat_1 full stride, ldo_1 tb_load + tb_loop_max stride 1
- bench2: amp tb_gain stride1, tb_dc stride25+recovery, tb_tran stride4
- gatesA: op/dc/dc_comprehensive/multi_tech_dc/tran/subckt/cmg_multiplier/ac
- gatesB: tran_comprehensive, multi_tech_tran
- gatesC: complex ring_osc/opamp/sram_snm/switchcap/opamp_ac
- nn: verify_nn_dc_tran (NN path expected untouched)
Amp tb_dc 15/15 evidence already at $SCR/amp_dc3.json.

### Results so far (2026-08-11)
- bench2: tb_gain 8/8 | tb_dc 15/15 (recovery, engine 0/0 segments) | tb_tran
  stride4 11/11 (was 2/6). ALL GREEN.
- bench1 first pass: ptat_1 full-stride 13/13 (was 5/13; mono_violations 0,
  9.6s vs 74s) | tb_load 11/11 (5.2s vs 678s!) | tb_loop_max REGRESSED 0/1
  (OP fell into fake equilibrium vo=-3.9V; plain NR cycle, honest flag).
  FIX: auto GMIN-homotopy fallback in DCSolver.solve (commit after c96dd09);
  loop_max back to 8/8, OP worst 2.7e-7V. bench1 rerunning.
- gatesA: op PASS, dc 2/2, dc_comprehensive 81/81 PASS... (running)
- gatesB: tran_comprehensive 45/45 PASS; multi_tech_tran running.
- gatesC/nn: first run void (worktree lacked bsimar/checkpoints — SYMLINKED
  to main repo copy), rerunning.
- cp (chargepump stride 20): running, watch $SCR/batch_cp.log.
NOTE monitor task bbi2mpkui already marked bench1/gatesC/nn done (old runs) —
track reruns manually.

### FINAL STATE (2026-08-11, after fallback+trigger fixes; commits through 5c18101)
Pilot: tb_gain 8/8 | tb_load 11/11 5.2s | tb_loop_max 8/8 (0.27uV) | ptat 13/13
| amp tb_dc 15/15 (branch-fork recovery trigger added to harness) | amp tb_tran
11/11 (1101/1101 steps) | chargepump STILL RUNNING (b81m96jr3 watches).
Gates: op 3/3, dc 2/2, dc_comp 81/81, multi_tech_dc 53/53 (known ERROR FIXED),
tran 1/1, tran_comp 45/45, multi_tech_tran 86/86, ac 2/2, subckt 11/11,
multiplier 6/6, ring_osc 5/5, sram 5/5, switchcap 5/5, canaries PASS,
nn_dc_tran 30/30. verify_nn_ac RUNNING. opamp+opamp_ac 0/5 PRE-EXISTING at
base (bit-identical repro with base solver files) — NN-side, out of scope.
Docs committed (5c18101): CHANGELOG V7.5.1, RESULTS_TSMC.md (cp row pending),
CLAUDE.md, README. Worktree checkpoints SYMLINKED from main repo (gitignored).

### Charge-pump saga part 2 (2026-08-11 morning; commits 97e8a42..4e42971)
- verify_nn_ac 10/10, verify_nn_lifted_source_dc 15/15, canaries PASS.
- Stride-20 cp ran 4.3 CPU-h without finishing; py-spy blocked (yama).
- Fail-fast added to transient NR (97e8a42): bail to dt-cut when no 2x
  progress in 30 its and max_delta > max(100*tol, 1e-2) — the 1e-2 floor
  protects the oscillation-average path. Tran gates re-ran green.
- Trace showed march stuck at t=1.5e-14s: acceptance-gate fix (e684bd9) —
  KCL residual gate on oscillation-average acceptance extended to L72
  (was NN-only; a quiet-garbage average could commit and poison the charge
  history through 1/dt).
- STILL stuck at fs scale: FD probe caught the real killer — the transient
  transcap 3x3 block (SPICE 5-cap + conservation shortcuts, no bulk row/col)
  stamps SIGN-FLIPPED off-diagonals for FLOATING-BULK devices (stamped
  +0.758 S vs true -0.758 S at dt=1e-15). Wrong-signed Jacobian at small dt
  => every NR iteration amplifies ~15x. Rail-tied bulks masked it in all
  prior gates. FIX (4e42971): full 4-terminal charge companion for L72 from
  Instance.condense_last_react() (== FD of (qd,qg,qs,qb), C[g,g]==cgg);
  per-terminal BE/Trap/BDF2 histories (_i_prev_source/_i_prev_bulk added);
  NN keeps 3x3 block bit-identically. Stamped system == FD to ~1e-6.
- RESULT: cp stride-20 COMPLETES 5001/5001 in 443 s: 4/6 agree
  (lo/up_iavg 7e-5/3.6e-4, lo_imin, up_imax); misses are lo_imax & up_imin
  = 10ps switching-spike extrema unsampled on a 40ps grid (ng up_imin is a
  -4uA reversal spike). Re-runs green: tran_comp 45/45, multi_tech_tran
  86/86, amp tb_tran 11/11, switchcap 5/5, sram 5/5.
- IN FLIGHT: cp stride-4 ($SCR/cp_s4.log) and stride-1 ($SCR/cp_s1.log) for
  the spike metrics. Then: RESULTS cp row + CHANGELOG item 9 + final summary.
- FOLLOW-UP (not this sprint): AC solver still uses the 5-cap 3x3 expansion
  for L72 — same floating-bulk hazard in principle; AC gates green today.
- [ ] Task 5: re-run 7 pilot decks + repo gates (incl. verify_bsimcmg_tran, ac,
      subckt, complex x4; L72 numbers all perturbed by tol+src-ref changes —
      expected within gate tolerances, sha256 sweep baselines may need rebase)
- [ ] Task 6: docs

## Uncommitted so far (beyond 544e9f4)
- mosfet_cmg.py: limiter (+fetlim/limvds/reset/retreat), src-ref eval, err context
- solver.py: limit kwarg plumbing, convergence gates, resets, NR_TRACE, (ladder WIP)

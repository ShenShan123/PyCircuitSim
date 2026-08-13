# AnalogGym simulator-fix notes (opened 2026-08-10, branch `feat/analoggym-migration`)

**Task:** make the AnalogGym test circuits agree with NGSPICE by fixing the
*simulator* (solver / parser / BSIM-CMG L72 path) — never by weakening a gate or
editing a deck. NN models (L73/74/75) are parked. Docs to update on each
version: `docs/CHANGELOG.md`, `CLAUDE.md`, `README.md`,
`examples/complex_circuits/RESULTS_TSMC.md`.

> **This file is the working scratchpad: current state + what is still open.**
> The narrative history, every dead end and all per-version evidence live in
> **`docs/CHANGELOG.md` §V7.5.0–V7.5.5** — that is the record, and it is not
> duplicated here.

## Current state (end of V7.5.5, 2026-08-12)

Five sprints done on this branch: **V7.5.1** (11 root-caused defects),
**V7.5.2** (AC full stamp + opt-in LTE output refinement), **V7.5.3**
(charge-state LTE, campaign driver, comparison-fairness harness fixes),
**V7.5.4** (the internal-solve current floor; two record corrections),
**V7.5.5** (the refine dt-controller rebuilt on ngspice dctran semantics with
a legacy corner guard; the 2026-08-10 open list closed).

**The 2026-08-10 open issues are all closed or characterized:**

1. *Refine cost on LDO load-step decks* — **closed** (CHANGELOG §V7.5.5.1).
   Open-water marching is now dctran-exact (CKTterr factors, per-test
   order-matched suggestions, 0.9h accept band, suggestion-verbatim growth);
   a 2-local-gap corner guard keeps the V7.5.3 flavor at PULSE corners —
   the only flavor measured to hold the charge pump at both strides.
   Basic_LDO 540→329 s, ldo_2 3005→208 s (1/5→3/5), cp 6/6 both strides
   with stride-independent up_imin (1.42/1.43 %). ITL4 iteration-count
   control measured dead (NR ≤6 iterations corpus-wide). Dead ends
   (tmax cap, BE/BDF-2 guard holds, loose-in-guard, honest-everywhere) all
   measured and recorded.
2. *ldo_2 both benches* — **characterized, reference side** (§V7.5.5.2).
   Loop AC agrees 8/8 both benches; tb_load `lr` is NGSPICE's forward-sweep
   first-point transient (reverse sweep at default tol → our value, <1 %);
   tb_tran metrics move toward ours as NGSPICE's own reltol tightens, and
   its overshoot ladder brackets our value. Same standing as Fan_SMC cmrrdc:
   quote with caveat, do not chase.
3. *Amplifier slew-edge metrics* — both V7.5.3 marginal regressions fixed
   (DFCFC2, Peng_IAC → 11/11); four new shallow 2.2–2.7 % crossings appeared
   elsewhere; 7/17 fully agree at 3–7× less cost. The 2 %-gate composition
   on these never-validated designs is scatter at the gate edge, not a
   solver axis (§V7.5.5.3).
4. *Transient campaign, other techs* — **done, both modes** (§V7.5.5.4).
   Flags-off: 1/1/6/2 of 23 (tsmc6/7/12/16). Refine-on: 10/8/8/8/9 of 23
   (tsmc5/6/7/12/16), net-positive everywhere. tsmc6 ≡ tsmc7
   verdict-identical in both modes. Evidence:
   `v755_campaign_tsmc*_tran{,_refine}/`.

Pilot regression basket (unchanged rows verified where touched): amplifier
`tb_gain` 8/8 · ldo_1 `tb_load` 11/11 · ldo_1 `tb_loop_max` 8/8 · amplifier
`tb_tran` 11/11 flags-off · ptat_1 `tb_dc` 13/13 · amplifier `tb_dc` 15/15 ·
chargepump `tb_tran` 6/6 refine+trap at strides 100 AND 20 ·
front_end_25_6T `tb_dc` 11/13.

Gate basket, all green at V7.5.5: op PASS · dc 2/2 · dc_comp 81/81 ·
subckt 11/11 · tran 1/1 · tran_comp 45/45 · ac 3/3 (re-run this sprint;
the untouched remainder — multi_tech_dc 53/53, cmg_multiplier 6/6,
multi_tech_tran 86/86, ring_osc/switchcap/sram_snm, sweep canaries,
nn_dc_tran 30/30, nn_lifted_source 15/15, PyCMG's own 314 — last verified
V7.5.4; the NN path and flags-off transient path are untouched by V7.5.5
by construction).

---

## OPEN ISSUES

### 1. Remaining per-deck misses — characterized metric/reference caveats, NOT solver work

Quote these with the caveat; do not chase them in the solver.

- **`min_slope_25_75c` / `max_step_frac_25_75c`** (18 decks). A *minimum
  over 100 adjacent steps* of a 0.5 °C staircase whose steps are ~225 µV;
  one bad sample sets the statistic, and the bad sample is the
  *reference's* (measured: NGSPICE's own curve moves 83.8 µV where its
  neighbours move ~230 µV, and at reltol=1e-5 its value goes negative).
  **Read the median slope** — it agrees to 0.5 %.
- **`Fan_SMC/tb_cmrr`** (1 deck). NGSPICE's default-tolerance early stop
  from the deck's own `.nodeset`; six-seed probe + its own tolerance ladder
  converge on our −36.0255 dB.
- **`ldo_2` tb_load `lr` + tb_tran excursions; `Basic_LDO` overshoot**
  (V7.5.5): reference-tolerance artifacts — NGSPICE's own reltol ladder
  moves each of them by ±40 % or more while our values sit inside its
  scatter (details + probes: CHANGELOG §V7.5.5.2).
- **Slew-edge 2 %-gate crossings on ~6 amplifier tb_tran decks**: at
  2.2–2.7 % these are the gate's noise floor on never-validated designs;
  the deep (4.6–7.3 %) misses of V7.5.3 are gone.
- **`Song_DACFC` operating point on tsmc6/7/12/16** (0.30–0.50 V from
  NGSPICE's; tsmc5 agrees to 9.6e-05 V) and **`Sau_CFCC` on tsmc6**
  (0.07 V): basin/multi-OP differences on never-validated designs —
  flagged for a future basin study, not scored as solver defects.
- **tsmc12 charge pump under refine: 3/6** — first-ever number on that
  tech (the cp gate has always been tsmc5-pinned). Coverage, not
  regression; a root-cause pass would start from the tsmc12 reversal-spike
  waveform.

### 2. Pre-existing, out of charter

`verify_complex_opamp` and `verify_complex_opamp_ac` are **0/5**, reproduced
bit-identically at base *before* any of this work. NN-model gap (L73), not
solver, and they do not call PyCMG. NN is parked for this task.

### 3. Refine default-on promotion — not attempted

Refine stays opt-in (`PYCIRCUITSIM_TRAN_REFINE=1`). Promotion needs the
§Performance-Discipline full re-gate (fidelity is a CPU flags-off property)
plus a decision about the campaign stride policy; nothing here requires it.

---

## Env facts, artifacts, and traps

- conda env `pycircuitsim` at `/home/shenshan/.conda/envs/pycircuitsim`;
  NGSPICE `/usr/local/ngspice-45.2/bin/ngspice`; worktree at
  `/data2/home/shenshan/PyCircuitSim/.claude/worktrees/analoggym-migration`.
- Gates CPU-pinned: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1`. `bsimar/checkpoints` is a symlink to the main repo copy
  (gitignored) so the NN gates can run here.
- **Campaign evidence is untracked and only on disk:** per-deck JSONs under
  `examples/complex_circuits/pycircuitsim_bench_results/v753_campaign_tsmc{5,6,7,12,16}/`
  (679 decks, layouts differ: tsmc5 nests under `acdc/`, `tran/`, `pilot/`)
  **plus the V7.5.5 sweeps** `v755_campaign_tsmc{6,7,12,16}_tran/` (flags-off)
  and `v755_campaign_tsmc{5,6,7,12,16}_tran_refine/`. Copy them out before
  any worktree cleanup. These are **pre-V7.5.6 full-corpus** records: 420 of
  the 794 scored deck records they hold belong to designs the V7.5.6
  curation removed, so they are the only surviving measurements of those
  decks — and their denominators do not match a post-V7.5.6 run.
- `run_compare.py --out` is a **directory**. Per-deck JSON carries `verdict`,
  `op_delta` and `notes`; `op_delta` (worst node vs NGSPICE) is the field to
  trust, not `_last_solve_converged` and not the metric columns alone — see
  the tsmc16 front_end case in V7.5.4.
- **Comparing two code states: never put the probe script inside the
  worktree.** `sys.path[0]` is the script's own directory (and cwd for `-m` /
  `-c`), so a PYTHONPATH-staged package copy is silently ignored and the two
  states measure bit-identically. It produced one wrong conclusion in V7.5.4.
  Run probes from the scratchpad through a wrapper that inserts the chosen
  package at `sys.path[0]` and **prints the resolved `solver.__file__`** as
  proof.
- Bit-identity of a flags-off change is checked by sha256 over the full
  float64 waveform (time axis + every node), not by eyeballing metrics.
- **Diagnostic knobs added in V7.5.5** (refine mode only):
  `PYCIRCUITSIM_REFINE_TRACE=<path>` — JSONL per-piece march trace
  (t, dt, r_v, r_q, NR iters, reject, binding device);
  `PYCIRCUITSIM_TRAN_REFINE_MAXDT` / `TransientSolver(refine_max_dt=...)` —
  ngspice's tmax rule (diagnostic-only, see the V7.5.5 dead ends);
  `PYCIRCUITSIM_BENCH_DUMP_WAVE=<path>.npz` — py transient axes from
  run_compare. The bench can pin `PYCIRCUITSIM_BENCH_TRAN_TMAX=1` to apply
  the tmax rule from the native tstep.
- When probing NGSPICE by hand, go through `designs_tsmc5/tools/meas.run_deck`
  (it preloads the OSDI binary and pins `num_threads=1`); a bare
  `ngspice -b` run rejects every `.model ... bsimcmg` card.

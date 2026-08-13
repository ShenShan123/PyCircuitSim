# AnalogGym simulator-fix notes (opened 2026-08-10, branch `feat/analoggym-migration`)

**Task:** make the AnalogGym test circuits agree with NGSPICE by fixing the
*simulator* (solver / parser / BSIM-CMG L72 path) — never by weakening a gate or
editing a deck. NN models (L73/74/75) are parked. Docs to update on each
version: `docs/CHANGELOG.md`, `CLAUDE.md`, `README.md`,
`examples/analoggym/RESULTS_TSMC.md`.

> **This file is the working scratchpad: current state + what is still open.**
> The narrative history, every dead end and all per-version evidence live in
> **`docs/CHANGELOG.md` §V7.5.0–V7.5.4** — that is the record, and it is not
> duplicated here.

## Current state (end of V7.5.4, 2026-08-12)

Four sprints done on this branch: **V7.5.1** (11 root-caused defects),
**V7.5.2** (AC full stamp + opt-in LTE output refinement), **V7.5.3**
(charge-state LTE, campaign driver, comparison-fairness harness fixes),
**V7.5.4** (the internal-solve current floor; two record corrections).

Pilot decks — **7/7 fully agreeing**, and they are a regression basket now:

| deck | 2026-08-10 | now |
|---|---|---|
| amplifier `tb_gain` | 8/8 | **8/8** (dcgain rel 1.5e-07) |
| ldo_1 `tb_load` | 11/11 | **11/11** (lr reconciled, py-ng 0.18 %) |
| ldo_1 `tb_loop_max` | 8/8 | **8/8** (GBW rel 7.4e-06) |
| amplifier `tb_tran` | 2/6 | **11/11** flags-off |
| ptat_1 `tb_dc` | 5/13 | **13/13**, 0 mono violations |
| amplifier `tb_dc` | 0/15 | **15/15** monolithic |
| chargepump `tb_tran` | dead | **6/6** refine+trap, stride 100 *and* 20 |
| *(new)* front_end_25_6T `tb_dc` | >1 h timeout | **11/13**, 281/281 pts in 3.2 s |

Gate basket, all green at V7.5.4: op PASS · dc 2/2 · dc_comp 81/81 ·
multi_tech_dc 53/53 · subckt 11/11 · cmg_multiplier 6/6 · tran 1/1 ·
tran_comp 45/45 · multi_tech_tran 86/86 · ac 3/3 · ring_osc 10/10 ·
switchcap 10/10 · sram_snm 12/12 · sweep canaries PASS · nn_dc_tran 30/30 ·
nn_lifted_source 15/15 · PyCMG's own suite 314 passed. (`verify_nn_ac` 10/10
last verified in V7.5.3, not re-run in V7.5.4 — the NN path is untouched.)

---

## Done, condensed (detail in CHANGELOG)

**V7.5.1** — the three original root causes and everything they unwound into:
the subthreshold current floor (PyCMG internal-solve tolerance), the 125 °C
first-point divergence, and the missing per-terminal limiting. What shipped:
the **full 4-terminal Newton stamp** for L72 (channel-only opvars are blind to
junction/gate-leakage conductances, which at 125 °C carry the drain current —
measured id=+1.8 mA against gds=4.3e-13 S); the **full 4-terminal charge
companion** (the old 3×3 SPICE-cap expansion stamps sign-flipped transcap
off-diagonals for floating-bulk devices → a ~15× Newton error amplifier at
small dt); SPICE-style damped limiting (fetlim/limvds/pnjlim, retreat-to-anchor
on eval failure); **source-referenced OSDI evaluation** (the internal solve is
not shift-robust); a wide DC gmin ladder with automatic fallback and honest
`final_converged`; and a transient retry ladder that re-walks the interval with
a locally halved dt instead of stiffening companions against a fixed target
time. `abs(gds)` removed (Critical Rule 4). `PYCIRCUITSIM_NR_TRACE=1` added.

**V7.5.2** — **AC stamps the full `Y = G4 + jωC4`** for L72 (no external gmin
in AC: it measurably pollutes high-impedance bulk nodes); **opt-in LTE output
refinement** (`PYCIRCUITSIM_TRAN_REFINE=1`, flags-off byte-identical) with
PULSE corners as breakpoints and depth-1 un-commit rollback;
`integration_method='trap'` pinnable. `verify_ac.py` gained the L3
floating-bulk case that earlier AC gates had masked by rail-tying every bulk.

**V7.5.3** — **per-device charge-state LTE** (CKTterr shape, CHGTOL=1e-18
because the stock 1e-14 sits 100× above a FinFET terminal charge and never
fires) closed the charge pump to 6/6 stride-independently; the **campaign
driver** (`campaign.py`, 159 decks/tech) and the AC/DC families swept on all
five techs (650/679 fully agreeing; **tsmc6 ≡ tsmc7 verdict-identical across
all 136 decks** — the relabelled-tech control at campaign scale); and the
harness became NGSPICE-exact (sample-exact `.meas` windows, `dctrcurv.c` grid
rule with Kelvin accumulation, grid-matched strided extrema, corroborated fork
recovery, `altns` fallback for NGSPICE-side basin failures, honest `ng_ran`).
Also `59aadb7`: PyCMG's `_read_opvar` descriptor rescan was **67 % of every
L72 transient's wall** — memoized, bit-identical, ~3×.

**V7.5.4** — the internal-node solve accepted on an **absolute** current
residual (1e-12 A) evaluated on the entry state *before any Newton step*, so
any device quieter than that constant kept its internal nodes unsolved. Same
defect V7.5.1 had "fixed" by re-tuning 1e-9 → 1e-12; it simply failed one
decade lower on deep-subthreshold FinFETs. `eval_dc` now binds on
`min(tol, max(1e-18, RELTOL·max|i_terminal|))` (`NN_DC_SOLVE_RELTOL`, default
1e-9), making `NN_DC_SOLVE_TOL` a ceiling the test can only tighten past.
front_end_25_6T: >1 h timeout → 281/281 converged in 3.2 s, worst node
0.28 mV, and seed-independent. Answer-preserving elsewhere (tsmc5 15/15,
tsmc16 14/14, tsmc12 13/14 verdict-identical), but it also revealed that on
tsmc16 that deck had been scoring **13/13 while 49 mV off** NGSPICE's OP —
the `.meas` cards do not probe the node that was wrong. Two corrections to the
record are folded in: Basic_LDO under refine is **4/5, not 5/5**, and
`min_slope_25_75c` measures the *reference's* noise (see open item 5).

---

## OPEN ISSUES

Ordered by substance. Each says what is wrong, what the numbers are, what has
already been ruled out, and what to try next.

### 1. Refine-mode cost on LDO load-step decks — solver, blocks the campaign

**Symptom.** `ldo/Basic_LDO/tb_tran` under `PYCIRCUITSIM_BENCH_TRAN_REFINE=1`
costs **662 s / 19 595 pieces** against NGSPICE's 4 141 steps in 1.9 s, and
still scores **4/5**: `overshoot` reads 0.85–1.11 mV against NGSPICE's
3.04 mV (rel 0.63–0.72). Flags-off is 28 s but reads overshoot as ~0 (rel 1.0),
so refine is the only mode that sees the feature at all. `ldo_2/tb_tran` is
worse (~3000 s).

**Measured march** (0–6 µs window, gear2, dt=20 ns): 100 pieces at the full
20 ns grid before the load step, 991 pieces across 2.0–2.5 µs, then
**12 868 pieces at median 47 ps** for the remaining 3.5 µs, with **zero** NR
failures. It collapses at the load step and never recovers — and it parks 20×
finer than the "~1 ns" V7.5.3 reported.

**Mechanism** (this is the correction to V7.5.3's diagnosis): a **dead zone in
the growth law**, not a noise-limited estimator.
`_grow = min(2, max(1, 0.9·r^(-1/3)))` exceeds 1 only when r < 0.9³ = 0.729,
so **any accepted ratio in [0.729, 1) freezes dt exactly**, with no escape
path. That is the same line V7.5.3 reported as "growth clamps at 1.0 for 32 %
of accepted pieces".

**Already ruled out — do not retry:**
- The V7.5.3 recorded fix path (secant-matched exponent + disarm the LTE tests
  on states moving below the solved tolerance). Built, measured, **reverted**:
  it holds cp 6/6 at both strides and is flags-off bit-identical, but costs
  **3.3×** (630 s → 2067 s, 19 897 → 78 298 pieces) and **flips no verdict**.
  Re-measured on the fixed PyCMG: 2138 s, overshoot rel 0.096 — same picture.
- Raising CHGTOL (V7.5.3): 1.4×, and it forfeits the charge pump's 6/6.
- The PyCMG internal-solve fix: independent of this (HEAD controller + fixed
  PyCMG = 662 s, overshoot rel 0.720). Same numbers, different problem.
- Moving the safety factor inside the exponent (`(0.9/r)^(1/p)`) shrinks the
  dead zone to [0.9, 1) but is worth only **~6 %** in dt at order 3 — correct,
  but not the cost lever.

**Worth knowing:** finer steps move overshoot monotonically toward NGSPICE
(3.28 mV at 78 k pieces vs 0.85 mV at 19.6 k), so **refine at HEAD is
under-resolving this deck ~4×** on that metric. This is a fidelity finding
independent of the cost question — the cheap answer is not "march coarser".

**Next candidates:** NGSPICE-style **ITL4 iteration-count** timestep control
(stock NGSPICE resolves fast features that way, not via charge terr — measured
from `cktterr.c` in V7.5.3); or a genuine local-order estimate feeding both
shrink *and* growth; or accept the cost, budget ~1 h/LDO deck, and say so.
Any attempt must re-gate the charge pump at **both** strides.

**Blocks:** campaign-wide refine-on runs, and promoting refine to default-on
(which additionally needs the §Performance-Discipline re-gate).

### 2. `ldo_2` disagrees on both its benches — deck-level, untouched

`tb_load` `lr` sits on a ~110 µV-flat replica-regulated curve (py holds it
**16× flatter** than NGSPICE, so a genuine small residual is amplified by a
peak-to-peak-of-a-flat-curve metric); `tb_tran` excursions differ **11–57 %**.
The corpus's most delicate local-loop design. Start at the deck level — read
the loop's own stability, not the solver — and only then decide whether this is
a comparison artifact or a real gap.

### 3. Slew-edge metrics on ~10 amplifier `tb_tran` decks — integrator policy

**7/17** designs fully agree under refine (up from 4/17 flags-off, composition
net-positive: Fan_SMC / Leung_DFCFC1 / Yan_AZ each 5→8 metrics, Qu2017_AZC
0/0→11/11 via the `altns` fallback). Two marginal 2 %-gate crossings went the
other way under charge-LTE's step pattern: Leung_DFCFC2 `sr_rise`
1.88 %→2.45 %, Peng_IAC `sr_fall` 1.93 %→2.17 %. Every underlying flags-off
miss is *improved* by refine (DFCFC2 7.3 %→4.6 %). These are never-validated
designs on the same integrator-policy axis as item 1; diminishing returns until
item 1 is settled.

### 4. Campaign coverage gap: the transient family on tsmc6/7/12/16 — compute

AC/DC families are done on all five techs. Remaining: **4 techs × 23 transient
decks**. Runnable **flags-off today**; refine-on is gated behind item 1 (or
budget ~1 h per LDO deck). Driver:
`python -m examples.analoggym.pycircuitsim_bench.campaign` (resumable, N
workers, writes `summary.md`).

### 5. Metric-definition caveats — characterized, NOT solver work

Quote these with the caveat; do not chase them in the solver.

- **`min_slope_25_75c` / `max_step_frac_25_75c`** (18 decks — the campaign's
  largest miss family). A *minimum over 100 adjacent steps* of a 0.5 °C
  staircase whose steps are ~225 µV, so one bad sample sets the statistic. On
  front_end_25_6T, NGSPICE's own curve moves **83.8 µV** across 31.5→32.0 °C
  where its neighbours move 234/226/279 µV — a one-sample ~100 µV wobble in
  the *reference's* DC solution, dropping its `min_slope` to 1.68e-4 against
  our smooth 4.27e-4. The Fan_SMC treatment (re-run the reference tighter)
  does **not** rescue it: at `reltol=1e-5` NGSPICE's value goes **negative**
  (−4.08e-4), so it is not tolerance-stable on the reference side at all.
  The **median** slope — same physical property, robustly estimated — agrees
  to **0.5 %** (4.485e-4 / 4.461e-4 / ours ~4.4e-4), `mono_violations` is 0
  both sides, and the 281-point OP agrees to 0.28 mV worst over 1124 node
  comparisons. **Read the median slope; do not tighten the reference to chase
  it.** (Our curve is also smoothed by per-point continuation seeding —
  NGSPICE continues too, but converges each point only to its own default
  tolerance.)
- **`Fan_SMC/tb_cmrr`** (1 deck). `cmrrdc` 2.27 % against the 2 % gate is
  NGSPICE's default-tolerance early stop from the deck's own `.nodeset`: a
  six-seed probe lands NGSPICE on our −36.0255 dB for four seeds including no
  seed, and its own tolerance ladder converges to it (reltol ≤ 3e-4). Ours is
  seed-independent to 0.002 dB. A deck-hygiene `.options` fix would move
  reference numbers, so it stays a caveat.

### 6. Pre-existing, out of charter

`verify_complex_opamp` and `verify_complex_opamp_ac` are **0/5**, reproduced
bit-identically at base *before* any of this work. NN-model gap (L73), not
solver, and they do not call PyCMG. NN is parked for this task.

---

## Env facts, artifacts, and traps

- conda env `pycircuitsim` at `/home/shenshan/.conda/envs/pycircuitsim`;
  NGSPICE `/usr/local/ngspice-45.2/bin/ngspice`; worktree at
  `/data2/home/shenshan/PyCircuitSim/.claude/worktrees/analoggym-migration`.
- Gates CPU-pinned: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1`. `bsimar/checkpoints` is a symlink to the main repo copy
  (gitignored) so the NN gates can run here.
- **Campaign evidence is untracked and only on disk:** per-deck JSONs under
  `examples/analoggym/pycircuitsim_bench_results/v753_campaign_tsmc{5,6,7,12,16}/`
  (679 decks). Copy them out before any worktree cleanup. Note the layouts
  differ: tsmc5 nests them under `acdc/`, `tran/`, `pilot/`; the other techs
  keep them flat.
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

# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology. (Compressed 2026-07-03, re-condensed 2026-07-23,
2026-07-30 and 2026-08-10 — every entry and verdict retained, verbose prose
pruned; the full original text lives in git history.)

---

## V7.5.3 — pilot closed 7/7; the campaign begins and the harness learns to compare fairly (branch `feat/analoggym-migration`, 2026-08-12)

**Goal: the V7.5.2 open issues** — charge-state LTE for the last charge-pump
metric, campaign expansion beyond the 7-deck pilot, and the three deck-level
anomalies. All closed or precisely classified; full gate basket re-run green
(81+53+45+86 comprehensive suites, complex/NN/subckt/canaries, AC 3/3,
nn_dc_tran 30/30, nn_ac 10/10, lifted_source 15/15).

1. **Per-device charge-state LTE closes the charge pump to 6/6** (`6421c79`):
   NGSPICE-CKTterr-shaped truncation control on MOSFET terminal-charge states
   inside `refine_output` — solver-side 3-deep accepted (t, q[4], i_cap[4])
   histories, third-divided-difference trap LTE converted to a current error,
   `tol = max(ABSTOL + RELTOL·max|i|, RELTOL·max(|q|,CHGTOL)/h)`. The /h
   loosener is what the V7.5.2 branch-current dead end was missing: below the
   h where DD3 is NR-noise-dominated it grows as 1/h and disarms the test.
   Two measured deviations from stock NGSPICE, both deliberate: CHGTOL=1e-18
   (stock 1e-14 sits 100× ABOVE a FinFET terminal charge, the test never
   fired — up_imin unchanged at 5.2 %; 1e-17 → 2.06 %), and the true trap
   error (h³/12)|6·DD3| (~7× tighter than CKTterr's factor·|DD3| +
   0.9·√ reject rule — reading `cktterr.c` settled that stock NGSPICE
   resolves the spike via ITL4 iteration-count timestep control, not charge
   terr; the tightened charge test is this solver's equivalent). **Charge
   pump refine+trap: 6/6 at BOTH strides** — up_imin −3.971 µA vs −4.031 µA
   (1.49 %) at stride 100/417 s, 1.84 % at stride 20/670 s. Flags-off
   byte-identical (all new code behind `refine`). **The 7-deck pilot is
   fully closed.**
2. **Qu2017_AZC was an NGSPICE Newton-basin artifact — the fallback existed,
   unwired** (`71237cc`): NGSPICE fails every transient-op homotopy from the
   deck's PRIMARY `.nodeset` seed (0.6 mV from the true OP — proximity does
   not predict basin membership; four of six probed seeds INCLUDING no seed
   at all land on the answer), while PyCircuitSim solves BOTH seeds with
   metrics agreeing to 1.7e-7. Blast radius 4/85 primary tb_tran decks
   (tsmc5 Qu2017_AZC, tsmc6+tsmc7 Yan_AZ — the relabelling confirming
   itself — tsmc16 Leung_DFCFC2), all recovered by the shipped
   `tb_tran_altns.cir` twin the reference runner falls back to;
   `translate.altns_deck()` existed with zero callers. Wired: on an
   NGSPICE-side failure the twin is scored, the row keeps the primary name,
   `altns`/`notes` record which seed won — 11/11. Verdicts also gain
   `ng_ran`: a run where the REFERENCE produced nothing used to read
   `ran=True, engine_ok=True, 0/0` and exit 0.
3. **The lr/lr_pp "definition gap" was one dropped sample** (`e021afd`):
   NGSPICE reduces `.meas` windows over the SAMPLES enclosed by [from,to]
   with exact double comparisons; the engine interpolated the window edges
   into the trace. The accumulated dc grid's last abscissa sits six ulp
   above `to=0.055`, NGSPICE drops it, and V(vo) is monotone so that sample
   IS the minimum: 1.1 % engine-control error on lr_pp, and the exclusion
   footnote on lr. Sample-exact windows: engine control 1.2e-07/1.6e-06,
   and the py-vs-ng lr gap collapses 1.6 % → **0.18 %** (both simulators
   drop the point symmetrically) — lr/lr_pp are evidence again. Bonus kill:
   the old 1e-9 edge tolerance was ABSOLUTE 1 ns on a seconds axis and
   replaced 1000 real samples of the charge pump's ps-scale grid with two
   chords (the source of lo_iavg's 4.9e-05 control error). NGSPICE's window
   bound parsing is spelling-dependent at the ulp (0.0055 ≠ 5.5e-3 ≠ 5.5m),
   measured and accepted as the bit-exactness limit.
4. **Comparison fairness** (`a04a40c`, `7c9387e`) — what the HoiLee tb_dc
   "12/15" unwound into (no fork existed; 15/15 monolithic now, recovery
   not even needed):
   - `plan_points` reproduces NGSPICE's sweep loop (`dctrcurv.c`): emit
     while sgn(step)·(value−stop) ≤ 1000·DBL_EPSILON, temperature sweeps
     accumulated in KELVIN (`CKTtemp += step`). The old
     floor((stop−start)/step)+1 gave 1651 points against NGSPICE's 1650 on
     `dc temp 125 -40 -0.1`, the extra endpoint a few ulp outside the
     deck's own `.meas from=-40` edge. Verified point-for-point against
     dumps (worst deviation = one Kelvin ulp = rawfile print precision).
   - Extremum/PP metrics compare on the SHARED grid when py runs strided
     (ng re-measured on py's abscissae; `ngspice.metrics` and the engine
     control stay full-grid): HoiLee's max_hot "2.84 %" was the 0.62 V
     peak at 38.4 °C falling between 2.5 °C samples — 1.29e-5 grid-matched.
   - The V7.5.0 branch-fork recovery trigger demands per-point
     corroboration (op-delta worst >1 % of rail, or asymmetric point sets)
     — it had fired on a 206 µV/0.03 %-of-rail stride artifact and the
     segments then traded one grid artifact for another.
   - `METRIC_ATOL` for relative-comparison blowups on near-zero values:
     vos25 (= vout25 − 0.195, ~1e4 cancellation amplification of a µV-scale
     node agreement) gets 10 µV; LDO overshoot/undershoot get 1 µV
     (ldo_folded_cascode: 2.4e-10 V vs 8.0e-10 V, rel 0.70, both meaning
     "the output does not move").
5. **NGSPICE-side monolithic failure is an explicit recovery case**
   (`0f7f3d2`): NGSPICE's own 125 °C cold start dies mid-sweep on
   Song_DACFC and Qu_LEC tb_dc ("DC: Timestep too small; temp = 45.7")
   while PyCircuitSim solves 67/67. The old code recovered these
   accidentally — the fork trigger fired on the None-vs-number entries of
   the empty NGSPICE column — and the corroboration gate closed that path.
   The reference flow itself re-runs wide sweeps as two continuations from
   25 °C, so the failure now names itself as the recovery reason and both
   sides re-score on segments: 15/15 each.
6. **Campaign driver + first tsmc5 campaign** (`d2368d3`):
   `python -m examples.analoggym.pycircuitsim_bench.campaign` — 159 decks
   per tech, pilot stride policy, N worker subprocesses, resumable,
   summary.md. First full tsmc5 pass under the fixed harness (per-deck
   JSONs preserved untracked in `examples/analoggym/
   pycircuitsim_bench_results/v753_campaign_tsmc5/`; summary tables in
   RESULTS_TSMC.md):
   **AC 88/89, dc_source 14/15, dc_temp 28/31 scored (+1 timeout), tran
   family first-ever scores**. The residual misses are each classified:
   Fan_SMC cmrrdc (NGSPICE-side early stop, item 7), ldo_2 lr (genuine
   ~1e-4 V flat-curve residual, pre-existing), min_slope_25_75c on three
   sensors (derivative metric at the µV node-agreement floor),
   front_end_25_6T timeout (item 8).
7. **Fan_SMC tb_cmrr caveat — the reference is the unconverged side**: the
   one AC miss (cmrrdc 2.27 % vs the 2 % gate) is NGSPICE's
   default-tolerance early stop from the deck's own `.nodeset`: six-seed
   probe lands NGSPICE on PyCircuitSim's −36.0255 dB for four seeds
   including NO seed, and its own tolerance ladder converges to it
   (reltol ≤ 3e-4); py's answer is seed-independent to 0.002 dB and its AC
   engine reproduces NGSPICE's own value to 2 µdB when evaluated at
   NGSPICE's OP. cmrrdc is a −CMRR residual at 9.7 dB/mV OP sensitivity —
   quoted with that caveat, deck untouched (one cell of ~85).
8. **New findings from the campaign, open**: (a) front_end_25_6T tb_dc
   grinds at ~11 s/point full-stride (>1 h timeout; stride-50 probe: 3/7
   points flagged non-converged with values kept, worst node 8 mV at the
   COLD end — a genuine cold-end sensor OP robustness gap, the successor
   to the closed V7.5.1 subthreshold-floor class); (b) Basic_LDO tb_tran
   under refine costs 2684 s vs 28 s flags-off (96×) — but scores **5/5**,
   capturing the 3 mV load-step overshoot flags-off reads as exactly zero
   (rel error 1.0); the grind is a refine-mode step-control pathology
   under diagnosis; (c) the amplifier refine sweep holds at **7/17
   designs fully agreeing** with the composition shifted net-positive:
   Fan_SMC/Leung_DFCFC1/Yan_AZ improved 5→8 metrics each, Qu2017_AZC went
   0/0→**11/11** via the altns fallback, against two marginal
   gate-crossings the other way (Leung_DFCFC2 sr_rise 1.88 %→2.45 %,
   Peng_IAC sr_fall 1.93 %→2.17 % — slew metrics that sat just inside the
   2 % gate under V7.5.2's step pattern sit just outside under
   charge-LTE's; every underlying flags-off miss those decks had is
   IMPROVED by refine, e.g. DFCFC2 7.3 %→4.6 %); (d) ldo_2 disagrees on
   BOTH its benches (tb_load lr on a ~110 µV-flat replica-regulated
   curve; tb_tran excursions 11–57 % at 3006 s refine) — the corpus's
   delicate local-loop design is a genuine open comparison item.
9. **Dead ends (recorded):** CHGTOL=1e-14 (stock) — charge test never
   binds at FinFET scales, up_imin unchanged; CHGTOL=1e-17 — binds but
   stops at 2.06 % against the 2 % gate. Neither reverted so much as
   walked through; 1e-18 shipped.
10. **The LDO refine grind, diagnosed — and 67 % of every L72 transient
    recovered** (`59aadb7`). Instrumented on Basic_LDO (0–8 µs: 6,126
    solves, 2,100 rejects, median piece 1.1 ns): the slowdown is two
    independent multipliers. (a) *Per-solve cost, unrelated to refine*:
    PyCMG `_read_opvar` linearly scanned 942 OSDI descriptor entries
    with a UTF-8 decode per candidate on every read — 169 µs/read ×
    8 reads/eval = **67 % of total wall on every L72 transient,
    flags-off included**. Fixed: the (name, alias)→index map is now
    memoized per Model (bit-identical by sha256 over all solved
    waveforms; inverter probe 1.08 s → 0.38 s; charge pump stride-100
    refine 417 s → **94 s** with float-identical metrics, on a loaded
    box). (b) *A stable ~1 ns
    equilibrium in the refine step controller*: the 0.9·r^(−1/3)
    shrink/grow laws assume the LTE ratio falls as h³, but measured
    within reject sequences it falls as h^1.05 — the voltage-LTE
    estimator is noise-limited (once all history spacings equal h, a
    noise floor ε gives DD3 ≈ ε/6h³ and an h-INDEPENDENT error
    estimate ε/12; the binding node moves 6× less per piece than the
    tolerance the piece was solved to), and the charge test improves
    only as h¹ through its /h loosener. So rejects shrink too little
    (r=1.5 → r=1.13, 3–7 reject sequences routine), growth clamps at
    1.0 for 32 % of accepted pieces, and the march parks at ~1 ns for
    the remaining 42 µs. NOT charge-test-specific: charge test off =
    only 1.6× cheaper; the voltage plateau carries the rest. NGSPICE
    marches the same deck in 4,141 steps at median 7.2 ns — refine
    takes ~10× the steps at 6.5× smaller median. Recorded fix path
    (not yet implemented, needs a charge-pump re-gate): match the
    controller exponent to the measured local order (secant on the
    last two (h, r) samples), and disarm both LTE tests on states
    whose per-piece move is below the tolerance the solve resolved
    (preserves the charge pump exactly — its spike moves charge
    orders of magnitude above RELTOL·|q|). Raising CHGTOL is NOT the
    fix (1.4×, forfeits the 6/6). Also measured: refine's accuracy
    gain on this deck is march accuracy, not output density (+0.61 mV
    of the recovered undershoot vs +0.28 mV), and fixed-dt is not a
    safe substitute — its error is non-monotone in dt under BDF-2
    (dt=20 ns reads WORSE than 80 ns).

---

## V7.5.2 — the two V7.5.1 follow-ups closed: AC full stamp + LTE output refinement (branch `feat/analoggym-migration`, 2026-08-11)

**Goal: close the two follow-ups V7.5.1 recorded** — the AC solver still used
the 5-cap 3×3 expansion for L72, and fixed output grids could not see what
NGSPICE's LTE timestep control records. Both closed; every gate re-run green
(45+81+53+86 comprehensive suites, all complex/NN/subckt/canary gates, AC 3/3).

1. **Full 4-terminal AC stamp for L72** (`cd4a106`): ACSolver was the last
   consumer of channel-only gds/gm/gmb + the 3×3 transcap expansion — blind to
   junction/gate-leakage conductances and carrying the same floating-bulk
   sign-flip hazard the transient fix killed. Devices exposing
   `get_terminal_stamp` now stamp `Y = G4 + jωC4` once at the OP, exactly
   NGSPICE's AC load. Measured effect on the pilot AC decks: tb_gain dcgain
   3.9e-05 → **1.5e-07** rel, tb_loop_max GBW 1.4e-03 → **7.4e-06** (≈200×);
   both 8/8. NN AC path bit-identical (verify_nn_ac 10/10).
2. **No external gmin in the AC load** (same commit): the first cut mirrored
   the DC stamp's gmin across d-s/d-b/s-b and it *measurably polluted* the
   small-signal answer — 5.8 % NRMSE on an NMOS bulk behind 100k (1e-12 S
   against a true junction conductance of ~2e-11 S) and a fake 250 nV bulk
   response on a PMOS whose bulk NGSPICE holds at 1e-83 V. gmin is a Newton
   convergence aid; NGSPICE's AC load carries only the model Jacobian (OSDI
   handles gmin internally). `verify_ac.py` Level 3 (floating-bulk NMOS+PMOS,
   gating v(out) AND v(bulk) with a 1 pV-floor complex-residual check) now
   guards this permanently — every pre-V7.5.2 AC gate rail-tied the bulk.
3. **LTE-driven output refinement, opt-in** (`6b92f1a`, `58d01a4`):
   `refine_output=True` / `PYCIRCUITSIM_TRAN_REFINE=1` / bench
   `PYCIRCUITSIM_BENCH_TRAN_REFINE=1`; default off, flags-off byte-identical.
   Three mechanisms: (a) every committed march piece is emitted into the
   returned waveform — non-uniform time axis, all fixed grid points present
   exactly, branch currents on the same axis (NGSPICE saves every accepted
   internal point, so its `.meas` sees 10 ps events a fixed grid cannot);
   (b) PULSE corners become breakpoints — pieces land on them, and the next
   piece restarts small (min(sub_dt/64, local-corner-gap/8) — the gap scale
   matters: sub_dt/64 alone left the first piece spanning a third of the
   charge pump's spike) and integrates BE, since the trapezoid's current
   history is inconsistent across a slope discontinuity (this is what seeds
   corner ringing; NGSPICE restarts at order 1 the same way); (c) per-piece
   LTE (trap third divided difference on the non-uniform fine history,
   TRTOL=7) with depth-1 un-commit rollback (`_snapshot/_restore_tran_state`)
   and 0.9·r^(−1/3) dt scaling. Probe (L72 inverter, 10 ps edges, 40 ps output
   grid, vs 2.5 ps reference): fixed grid errs 147 mV max / 30 mV rms at
   shared points; refine 0.50 mV / 58 µV — **~500× rms** at 526 vs 151 points.
4. **`integration_method='trap'`** (`67930f9`): pinned trapezoid (BE step 1,
   no stiffness swap), mirroring `gear2` — NGSPICE's default integrator was
   not directly selectable, and 'auto' swaps to BDF-2 exactly on the decks
   where the comparison matters.
5. **Charge pump verdict** (refine+trap): **5/6 at stride 20 AND stride 100**
   — the fixed-grid axis is gone (V7.5.1 needed BDF-2 damping and still
   missed spike extrema; stride 20 previously lost `lo_imax` too). All four
   pump-defining averages/extrema at 3e-6…1.8e-3; `up_imin` (the ±4 µA,
   ~10 ps reversal spike) is now **captured at −3.84 µA vs NGSPICE's
   −4.031 µA (4.7 %)** — right sign, right width, amplitude just outside the
   2 % gate. What remains is integrator-policy sensitivity, not grid.
6. **Dead ends (recorded, reverted):** (a) *post-corner hold window* — pinning
   the restart dt for 2 local gaps after each corner over-resolved the spike
   into trapezoid micro-ringing: up_imin −5.70 µA (41 % OVER) and lo_imin
   regressed; fine uniform trap through a discontinuity response rings just
   like the fixed 2 ps grid did (−8.4 µA). (b) *branch-current LTE* — adding
   the V-source tail currents to the LTE state (reltol·|i|+1e-12 A) thrashed
   into the 4096-piece cap: NR-tolerance-level noise on MNA tail currents has
   a dt-INDEPENDENT third divided difference, so rejection never converges.
   NGSPICE runs its truncation control on smooth per-device CHARGE states;
   a charge-based LTE (needs 4-deep q histories per device) is the faithful
   follow-up if the last 4.7 % ever matters.
7. **Bonus visibility — full amplifier tb_tran category** (17 designs, first
   run beyond the pilot): flags-off stride 4, 5/17 fully agree (Alfio pilot
   regression intact at 11/11); the misses are slew-edge metrics on the
   coarse grid — the exact axis refine addresses (refine sweep in
   RESULTS_TSMC.md). One deck-level anomaly: Qu2017_AZC's NGSPICE run
   produces no data (ng 0 s, engine 0/0) — a campaign item, not a
   PyCircuitSim gap.

---

## V7.5.1 — AnalogGym parity: the solver learns real SPICE robustness (branch `feat/analoggym-migration`, 2026-08-11)

**Goal: every pilot deck of the V7.5.0 PyCircuitSim-vs-NGSPICE comparison
agreeing, by fixing the simulator — NN models parked.** The three recorded
causes in `RESULTS_TSMC.md` unwound into eleven distinct defects across the
solver, the L72 wrapper, and PyCMG. Scoreboard movement (TSMC5 pilot):
ptat_1 tb_dc **5/13 → 13/13** (9.6 s, was 74 s), amplifier tb_dc
**0/15 → 15/15** (worst node 2.6 µV vs the diverged 9.75 V; 67 s, was 1250 s
partial), amplifier tb_tran **2/6 → 11/11**, ldo tb_load 11/11 (5.2 s, was
678 s — 130×), tb_loop_max 8/8, tb_gain 8/8, charge pump tb_tran
**dead-at-step-0 → 5/6** (all 25001 steps solved; the sixth metric is a
±4 µA 10 ps current-reversal spike whose amplitude is integration-method
sensitive — trapezoidal over-rings it 2×, BDF-2 damps it out; the four
current averages/extrema that define pump function agree to 8e-6…2e-2).

1. **Subthreshold current floor** (blocked 75 weak-inversion decks): the PyCMG
   internal-node NR tolerance defaulted to 1e-9 A absolute, so any true
   |id| < ~1 nA "converged" without moving the internal nodes and returned
   exactly 0.0 A — the V6.4.7 zero-row artifact, patched then only inside the
   data generator. Default now 1e-12 (NGSPICE's ABSTOL), `NN_DC_SOLVE_TOL`
   still wins. ptat_1 vout25 went 59 % off → 8e-05.
2. **Full 4-terminal L72 Newton stamp** (the big one; blocked the 125 °C
   amplifier family): the 3-conductance companion linearizes only the channel
   — the gm/gds/gmb opvars are blind to body-junction and gate-leakage
   conductances, which at 125 °C carry the drain current (measured id=+1.8 mA
   vs gds=4.3e-13 S), so NR cycled around a permanent 1.4 mA KCL violation.
   L72 now stamps all four KCL rows from the condensed OSDI Jacobian (new
   `Instance.condense_last_jacobian`, no re-eval, verified == finite
   differences of (id,ig,is,ie)); GMIN across d-s/d-b/s-b. The 125 °C cold
   start converges in 1.8 s with plain NR. NN levels keep the 3-cond stamp.
3. **SPICE-style damped NR limiting for L72**: fetlim (vgs), sign-symmetric
   limvds (vds), pnjlim (both body-junction pairs) in the NMOS-normalized
   frame inside a ±2.5 V window; the device is EVALUATED at the limited bias
   and the companion linearizes about it (hard clamps zero the derivative);
   an iteration that limited is never accepted as converged (incl. both
   oscillation-average paths); anchors reset per NR sweep — a stale anchor
   distorts the retry's first evals and the charge companion amplifies that
   by 1/dt (measured ~1e2 A phantom residuals at a good OP); on eval failure
   the stamp bisects toward the last evaluable anchor (or zero bias).
4. **Source-referenced OSDI evaluation**: BSIM-CMG physics is pair-driven but
   the OSDI internal solve is NOT shift-robust — the identical
   (vgs,vds,vbs)=(2.5,2.5,2.5) evaluates at s=−8.5 V and diverges at
   s=−16.1 V (warm or cold). Eval frame now pinned to the source terminal
   (mirrors NN Rule 2). Eval errors carry device name/L/NFIN/m.
5. **Internal-solve float wall + state ratchet** (PyCMG): an absolute 1e-12 A
   residual is beyond float64 at amp-scale junction currents (a forward
   drain-body diode at 2.5 V/125 °C evaluates to 542 A), so the internal NR
   now also accepts a sub-nV voltage step — but only AFTER at least one step,
   which is what keeps the fix floor-safe. And a diverged internal solve used
   to leave the internal nodes hundreds of clamped steps into garbage,
   poisoning every later eval (the warm start became poison): failed solves
   now retry once from the cold state and always leave a cold state behind.
6. **Honest gmin homotopy**: wide ladder for non-NN circuits (1e-2 → GMIN
   decade by decade; NN keeps the measured V5 2-level schedule); fixed the
   PRE-EXISTING sticky `final_converged` (an intermediate gmin level could
   mark a diverged final level converged — observed flag=True with nodes at
   −666 V); the verdict is now the last level / last source step only, and a
   failed level restarts from the last good homotopy point. When a plain L72
   DC solve fails, `DCSolver` now retries the ladder **automatically** —
   with source stepping off (one homotopy at a time, not a 20-way split of
   the NR budget) and ≥200 iterations, returning the primary iterate if the
   ladder itself fails (never replace a near-solution with a wreck).
7. **Transient retry = a genuinely smaller step**: the old ladder halved the
   companion dt while keeping the SAME target time — the cap companions then
   demand the full interval's dV inside dt/2ⁿ, so every "halving" made the
   system STIFFER (charge pump: residual pinned ~2e3 A at "minimum dt"). The
   interval is now re-walked with a locally halved dt (down to `dt·2⁻²⁴`,
   doubling back after successes, 4096-piece budget via `SimStepLimit`),
   committing charge state piece by piece, exactly as NGSPICE's timestep
   control does. First-attempt successes are expression-identical.
8. **Critical Rule 4 violation removed**: `MOSFET_CMG.get_conductance` did
   `abs(gds)`; the floor lives at the stamp.
9. **True 4-terminal charge companion** (found by the charge pump, day 2):
   the transient transcap block rebuilt a 3×3 matrix from the five SPICE
   cap variables plus charge-conservation shortcuts — no bulk row or
   column. For FLOATING-BULK devices that stamps SIGN-FLIPPED
   off-diagonals (measured +0.758 S against a true −0.758 S at
   dt=1e-15), and a wrong-signed Jacobian at small dt makes every Newton
   iteration AMPLIFY the error ~15× — the adaptive march then commits
   femtosecond pieces forever (2 CPU-hours to reach t=15 fs). Rail-tied
   bulks kept the old block honest enough to pass every prior transient
   gate, which is why it survived. L72 now stamps the charge companion
   from the condensed reactive OSDI Jacobian (`condense_last_react`,
   verified == finite differences of (qd,qg,qs,qb) with C[g,g]==cgg),
   all four terminal rows with their own BE/Trap/BDF-2 histories; the
   stamped transient system matches finite differences to ~1e-6. The NN
   3×3 block is untouched. **Follow-up recorded:** the AC solver still
   expands the 5-cap view for L72 — same floating-bulk hazard in
   principle; AC gates are green today.
10. **Oscillation-average acceptance gated on KCL for L72 too**: the
   Phase 6b residual gate was NN-scoped, so a pure-L72 circuit could
   accept and COMMIT a quiet-but-garbage averaged point, poisoning every
   later warm start through the 1/dt companion (this is what first wedged
   the charge-pump march). Both DC and transient acceptance paths now
   probe the residual whenever the circuit has NN or full-stamp devices.
11. **Fail-fast out of stagnant transient NR** (perf, failure-path only):
   a step that is going to fail used to burn all 200 iterations before
   the march cut dt — hundreds of full NR runs per switching edge. NR now
   bails to the dt-cut when a far-from-converged iterate makes no 2×
   progress across 30 iterations (absolute 1e-2 V floor keeps this
   strictly above the oscillation-average acceptance ceiling).

**Harness (`pycircuitsim_bench`)**: `_op_worst` read a key `op_delta` never
carried, silently disabling the recovery path (the narrow segment also ran
strided, discarding the 25 °C point it exists to recover); and a **branch-fork
trigger** — with the solver fixed, the monolithic temperature sweep tracks
NGSPICE to 63 µV everywhere except the first point, where two valid Newton
roots exist; a metric disagreement on a dc_temp deck with the engine control
holding now triggers the outward-from-25 °C continuation on both simulators.

**Dead ends recorded**: fixed 2ⁿ-piece re-subdivision of the whole interval
(replaced by the adaptive march — exponential cost, still fails on bistable
chatter); interval-subdivision rollback snapshots (unnecessary once the march
only commits successful pieces); a wide gmin ladder alone (converged=True at
−666 V — that was the sticky-flag bug, not a cure); retreat-only limiting
without junction Jacobians (NR limit-cycles no matter how small the step).

**Verification (this hardware, CPU-pinned; transient gates re-run after
EVERY solver change, three times total)**: verify_bsimcmg_op 3/3, dc 2/2,
dc_comprehensive 81/81, **multi_tech_dc 53/53 — the pre-V7.5 known-ERROR
`TSMC5_lvt_inv_l_24nm` pure-L72 NR divergence is FIXED**, tran 1/1,
tran_comprehensive 45/45, multi_tech_tran 86/86, verify_ac 2/2 (after the
fallback learned to never undercut the primary result — the CS-amp OP is now
honestly converged instead of sticky-masked), subckt 11/11, cmg_multiplier
6/6, complex ring_osc 5/5 / sram_snm 5/5 / switchcap 5/5, sweep canaries
PASS, verify_nn_dc_tran 30/30, verify_nn_ac 10/10,
verify_nn_lifted_source_dc 15/15 (NN untouched; the lifted-source canary
also revalidates the source-referenced frame). complex_opamp{,_ac} 0/5 on
this hardware **pre-exists at the branch base** (NN-side, reproduced with
base-commit solver files bit-for-bit) and is out of this sprint's scope.
Diagnostics added: `PYCIRCUITSIM_NR_TRACE=1` (DC tail + gmin summary +
transient NR), and harness-side `PYCIRCUITSIM_BENCH_TRAN_METHOD` /
`PYCIRCUITSIM_BENCH_TRAN_SUBSTEPS` for integrator comparisons on one deck
(the charge-pump spike bracketing above came from these).

---

## V7.5.0 — AnalogGym in-repo: 190 analog designs scored against NGSPICE (branch `feat/analoggym-migration`, 2026-08-10)

**AnalogGym's five TSMC design trees are now `examples/analoggym/`.** 190
designs (amplifier, ldo, sensing_front_end, voltage_reference, charge_pump)
across TSMC5/6/7/12/16 with their 880 testbench decks, the shared `tools/`
harness, and the NGSPICE reference numbers the comparison is scored against
(`result.json` per design, `results/summary.csv`, `RESULTS_TSMC.md`). Tree names
stay `designs_tsmc{N}` because `tools/pycmg_lib.py` derives the technology from
the tree name; only `PYCMG_DIR`'s default changed, to the in-repo vendored
PyCMG. Excluded: `work/` (891 MB of sizing scratch), run logs, `.sweep` dumps.
The per-design `tsmc{N}_models.spice` and `models/cache/*.l` are PDK-derived and
**gitignored** (318 MB on disk, regenerable via `resolve_modelcard`); 1627 files
tracked. Migration verified by re-running tsmc5 `Alfio_RAFFC_Pin_3/tb_gain.cir`
under NGSPICE in its new location: dcgain 75.87195 dB and phase 55.3409 deg
reproduce the shipped reference exactly.

**These decks needed real capability work — none of the 880 parsed before.**
They are ngspice OSDI decks: geometry lives in per-geometry `.model` cards (the
OSDI binding rejects instance parameters), devices carry the `N` prefix and an
`m=` multiplier, values are `.param` expressions, metrics are `.meas`
statements, and the analysis command is not in the deck at all — `finalize.py`
injects it. Counts in one tree: 1029 `.meas`, 567 `.param`, 176 `.nodeset`,
648 `N` devices, 29 inductors.

**`examples/analoggym/pycircuitsim_bench/`** — the comparison harness, split so
one measurement semantics scores both simulators:
`__init__.py` (frozen contract + the CONTROLS tables), `translate.py`
(deck → netlist + AnalysisPlan; **all 880 decks translate, 0 failures**, audited
over 19405 emitted cards), `measure.py` (**all 5145 `.meas` cards in all five
trees parse**, 20 distinct forms; agrees with ngspice's own `.meas` to ~1e-8 and
reproduces the scored artifacts to ~1e-9), `run_compare.py` (drives both
simulators, writes per-deck JSON).

**Core support (`pycircuitsim/`):** an `Inductor` with its MNA branch row
stamped in DC (short), AC (jwL) and transient — the decks' 1T L/C feedback break
needs it; the BSIM-CMG `m=` multiplier, scaling the residual and both Jacobians
while the eval cache stays raw so the NR Jacobian remains the derivative of the
stamped residual; `N`-prefix dispatch (previously dropped **silently** at top
level — a correctness hazard, not just a gap); and `dv_limit`, a per-iteration
per-node trust-region cap on `DCSolver`/`TransientSolver`, **default `None` so
existing behaviour is bit-identical**. Also fixed a **pre-existing bug**:
`_solve_newton` restored the source-stepping ramp only on the returning path, so
any raise left every `VoltageSource` at 1/N of nominal and the next solve ramped
down again from there.

**PyCMG (bit-identical, all three to make 880 decks tractable):** `apply_param`
via a per-descriptor name→index map instead of a linear scan (was ~2.05 M string
compares per device, 808.8 ms/device); `get_shared_model` memoising Models per
(card, name, geometry), geometry in the key because L/NFIN/TFIN are MODEL-kind
params in this OSDI build; `Instance.set_temperature` rebinding in place at
0.255 ms/device versus 808.8 ms to rebuild.

**Verdict per family** (tsmc5 pilots, PyCircuitSim vs NGSPICE 45.2 on the same
OSDI model, `engine N/N` control passing on every deck so no row is a
measurement artifact):

| family | deck | result | worst OP error | py / ng seconds |
| --- | --- | --- | --- | --- |
| amplifier AC | `tb_gain` | **8/8 agree** (dcgain 3.9e-05) | 110 uV | 2.2 / 0.3 |
| ldo dc source | `tb_load` | **11/11 agree** (1.9e-05) | 148 uV | 681 / 0.3 |
| ldo AC | `tb_loop_max` | **8/8 agree** | 91 uV | 8.1 / 0.3 |
| amplifier tran | `tb_tran` | 2/6, 119/221 steps | 111 uV | 105 / 1.1 |
| sensor dc temp | `ptat_1/tb_dc` | 5/13 | 76 mV | 74 / 0.1 |
| amplifier dc temp | `tb_dc` | 0/15 monolithic | 9.75 V | 1217 / 0.7 |
| charge pump tran | `tb_tran` | dies at step 1 | 161 mV | — / 30.7 |

The AC path is essentially exact: solved about NGSPICE's own operating point it
returns dcgain within 0.004 dB. **Every failure above is the DC operating
point**, and the two causes are distinct:

- **Subthreshold current floor (blocks sensing_front_end + voltage_reference,
  75 decks).** At NGSPICE's own operating point PyCircuitSim's LEVEL=72 returns
  `id = 0.0 A` **exactly** at Vgs = 55.3 mV, and 2.16 nA at 61.9 mV. A
  weak-inversion stack whose devices all read zero has no operating point.
  Above ~50 C the same PTAT deck agrees to 2e-05. This is m-independent (m=1 and
  m=4 both return hard zero; above 0.08 V they agree to 0.00 % and scale exactly
  4x), so it is **not** the new multiplier — it is an OSDI/PyCMG evaluation gap
  below ~60 mV Vgs. `verify_cmg_multiplier.py` is 5/6 for exactly this reason
  and the failure is left visible rather than tolerance-hidden.
- **Newton start, not the model (amplifier `tb_dc`, 85 decks).** That bench's
  FIRST point is 125 C and it diverges there, after which the continuation
  carries garbage down the sweep. The identical solve at 25 C is sound: op delta
  114 uV, `vout6` 0.193383 versus 0.193383. `compare_with_recovery` ports
  `finalize.py`'s outward-from-25 C recovery to address this — **implemented and
  import-verified, but its end-to-end numbers are NOT yet measured**, so the
  0/15 row above still stands as the recorded result for this family. It
  re-scores **both**
  simulators on the segments, because `recombine_temp_segments` is deliberately
  bug-compatible with the reference (each segment's deck carries the other's
  window, collapsing to the single 25 C point), so a recombined `maxval` sits at
  0.1933828 V against the monolithic 0.3699788 V and scoring recombined against
  monolithic would report a 2x difference that is pure bookkeeping.
- **Transients need per-terminal limiting (charge_pump 5 decks dead, ~110
  partial).** `dv_limit` bounds the Newton STEP, not the terminal voltage, so
  capped iterations still walk a gate to -2.96 V and a drain to +3.94 V on a
  0.65 V rail and OSDI rejects it. Dies at step 1 at every dt tried (2 ps, 20 ps,
  200 ps, 1 ns). The fix is SPICE-style `fetlim`/`limvds` inside
  `models/mosfet_cmg.py`, which does not exist — CLAUDE.md lists "voltage
  clamping Vgs+/-5V, Vds+/-10V" as a standard the LEVEL=72 path never
  implemented. Must be damped limiting, not a hard clamp: a clamp zeroes the
  derivative and stalls NR.

**Dead end, measured and reverted (do not retry):** extending the
source-stepping homotopy to ramp CURRENT sources as well does **not** fix the
divergence — same OSDI raise at g = -25261 V, i.e. 1/20 of the original step.
The blow-up is the missing terminal limiting, not the homotopy's coverage.

**Also measured:** PyCircuitSim's own RELTOL 1e-4 / VNTOL 1e-7 is 10x tighter
than the NGSPICE it is scored against; the harness defaults to NGSPICE's
1e-3/1e-6 (a deck's `.options` still wins), which took the LDO sweep from
0/101 to 101/101 flag-converged with identical values. And `ok` is reported, not
obeyed: that same sweep matched NGSPICE to 2e-05 while
`_last_solve_converged` rejected all 101 points, so `op_delta` is the verdict.

Gates after every change, CPU-pinned: `verify_subckt.py` 11/11,
`verify_bsimcmg_dc.py` 2/2, `verify_ac.py` 2/2, `verify_bsimcmg_tran.py` 1/1
(NRMSE 0.19 %), `verify_nn_ac.py` 10/10 (checkpoints symlinked in for the run;
they are gitignored and absent from a fresh worktree).

**Campaign cost, from measured rates:** 795 scored decks — 445 AC, 160 dc_temp,
75 dc_source, 115 tran. AC ~0.7 h, sfe/vref dc_temp ~1.7 h, ldo dc_source ~12 h,
amplifier tran ~90 h at the deck's own dt (extrapolated, not measured at full
dt), amplifier dc_temp ~740 h as it stands, projected ~2 h if the recovery path
restores the 25 C pass's 3.1 s/point rate (unverified — see above).
Roughly 110 CPU-hours total, parallelisable per deck. What it scores today:
~520 of 795 decks producing trustworthy metrics, with the three causes above
naming the other 275.

---

## V7.4.1 — housekeeping: PyCMG vendored in-repo, docs compacted, stale plans pruned (branch `v720-gpu-scaling`, 2026-08-10)

**PyCMG is no longer a git submodule.** The gitlink, `.gitmodules` and the
local module clone are gone; the 65 files PyCMG tracked (its full working tree
minus `modelcards.tar.gz`) are tracked directly by this repo at
`external_compact_models/PyCMG/`. The conversion was performed at the pinned
commit `06a20b7` (verified pushed to `ShenShan123/PyCMG` first, no unpushed
work on any branch). Guardrails preserved: PyCMG's own `.gitignore` still
covers `build*/` and the IP-protected TSMC `cln*.l` cards (nested .gitignore
files apply regardless of repo boundaries); the parent `.gitignore` adds
`external_compact_models/PyCMG/modelcards.tar.gz*` so neither the archive nor
the local `*-backup` tarball (the only copy holding the TSMC6 raw card) can be
committed. All gitignored on-disk assets — TSMC modelcards, the OSDI binary,
the backup tarball — were untouched. README clone/setup instructions updated
(no `--recurse-submodules`); future PyCMG upstream changes are now ordinary
commits here, not pin bumps.

**House-clean (two waves, user-directed):** pruned 14 stale docs — 12
closed-campaign `docs/plans/` files + the V6.9.0 TSMC6 parse-audit doc + the
2026-07-21 systematic-audit register (verdicts live in the V6.12.1–V6.13.1
entries; the six references in CLAUDE.md/README/code comments were redirected
to CHANGELOG sections) — plus the V4-era `external_compact_models/bsimar/docs/`
and `bsimar/results/` (2026-04 improvement plans, paper PDF, v3 LOO report).
`docs/plans/2026-07-26-v720-gpu-scaling.md` is the one plan kept (still the
cited dead-end register). Tracked the previously untracked
`scripts/v740_tf_reaper.sh`; deleted 3 unreferenced example decks
(`bsimcmg_{inverter_tran_verify,nmos_only,pmos_only}.sp`) and **24
closed-campaign one-off scripts** (a3/tsmc6/uni/pfnxl campaign drivers and
collectors, the V6.6.x gate-matrix/recipe-retest stack, `recipe_eval.sh`,
`opamp_sweep_def.sh`, `v710_regate_control.py`), keeping the functional core:
v730 docs/coverage/control, the v740 drivers, the `v710_regate` trio,
`recipe_train.sh`, the benchmark suite, `v720_t3_flag_bundle.sh` +
`a3_omp_one.sh` (a live dependency), `bench_gpu/`, and the corridor data
pipeline `v6_4_7_s12_*` + its imported library `v6_4_7_s6_l72_ro_control.py`.
Untracked cruft deleted (`__pycache__`, generated `tests/verify_*_results/`,
finished-campaign master logs, stale V6.x bench dirs, and the 56 MB
`t3_gpu_bundle/scratch/` work dirs). Evidence kept: `results/v740_regate/`
whole (the docs-builder `--check` source) and the T3/T4 GPU-bundle
VERDICTs/logs/return-codes.

**Docs compacted:** CHANGELOG 1274 → ~800 lines (fourth condensation — every
entry, verdict, retraction and dead-end retained); CLAUDE.md 395 → 366
(model-family state and the TSMC6 story each told once).

---

## V7.4.0 — clean rebuild + GPU fidelity re-gate on new hardware (branch `v720-gpu-scaling`, 2026-07-30 → 2026-08-06)

**COMPLETE for the CPU accuracy axis.** The box changed (`memlab-gpu2`, fresh
clone, no datasets/checkpoints), so every NN number was re-earned: all 10
datasets regenerated (`--enable-inv-trip --enable-subvt-off`), DirectNet and
BSIM-AR rebuilt 40/40 checkpoints each on the one clean recipe
(`--apply-filter off --swa-mode ema --seed 42`), 240/240 suite runs per family,
reports regenerated and `--check` clean. Recipe rows and PFN were deliberately
NOT retrained — their reports stand as labelled V7.3.0 historical evidence
(an empty local source cannot reconstruct measured rows). Training ran on GPU;
headline gating stayed CPU-pinned (fidelity is a CPU/flags-off property).

**Infra:** `scripts/v740_campaign.sh` (drain-loop orchestrator),
`v740_tf_rescue.sh` (claims OOM-abandoned stems mid-wave), `v740_tf_fill.sh`
(post-wave refill), `v740_tf_reaper.sh` (two-cycle kill of dead-incomplete
stems — a killed run's leftover `_best.pt` otherwise makes every retry SKIP;
retrain is from scratch with `--force` so the control stays one uninterrupted
run). `v730_docs_build.py` gained `--only/--recipes`, a complete-matrix guard,
per-report pass pin and SHA-256 manifest, so a partial pass cannot overwrite a
coherent report; `v740_regate` registered as the V7.4 evidence pass.

### Verdicts — three V7.3.0 claims retracted

- **"DirectNet peaks at `large`" — RETRACTED.** Monotonic 11→12→14→**15/20**,
  `xl` best; but the climb is switchcap+opamp only (ring frozen 2/5, SRAM
  saturated 5/5). `large` stays production on cost (`xl` = 2.3× for one cell).
- **"`tsmc7-opamp` at `large` is open" — RETRACTED.** TSMC7 passes the opamp
  at all four tiers (3.9–6.1 %), strongest opamp column.
- **"`tsmc5-opamp` is the thinnest margin" — RETRACTED.** 0/4: three tiers
  rail, `large` misses 11.6 % vs the 10 % gate.

### Measured noise floor — the TSMC6 repeat paid off

TSMC6/TSMC7 train on bit-identical rows; they agree 14/16 (all rings, SRAMs,
switchcaps) and split twice, **both opamp** — a 2-cell spread in per-tech
totals. **A ≤2-cell opamp difference is noise, not a result.** Applied
immediately: `tsmc16-opamp` went 0/4 (passed twice in V7.3.0) — exactly 2
cells, so flagged *not established pending a second seed*, not a regression.

### New findings

- **`xl` breaks parametric DC for the first time** — 66/69 vs 69/69 at every
  other tier: capacity now damages the device surface *while the circuit score
  climbs* (device-vs-circuit inversion: `medium` best device fit, `xl` best
  circuit tier).
- **Low-VDD rings are an exception-free partition** — TSMC5/6/7 fail all 12
  ring cells, TSMC12/16 pass all 8; TSMC5 worsens with capacity.
- **`switchcap` at `small` fails 4/5 on hold droop**, not headline charge.
- **TSMC5 device AC reports `f3db = nan` at `large`/`xl`** — degenerate fit.

### BSIM-AR clean verdict — stronger, but not flat

Strict curve **18→17→15→13/20** (`small`→`xl`), zero flips — retracts the
V7.3.0 claim that BSIM-AR is flat across capacity. `small` misses only TSMC5
switchcap (hold droop) + the noisy TSMC7 ring. The decline is circuit-surface,
not device damage (transient 80/80 every tier; device AC 9–10/10; parametric DC
65–68/69). TSMC6/TSMC7 agree 15/16 (sole split inside ring noise) — the old
"all 16 repeat cells reproduce" claim is retracted. V7.3.0 corridor recipes
still record 20/20 but were not retrained; they are historical evidence, not
the current artifact set.

### GPU acceleration fidelity — remaining V7.2 gate closed

- **T3 full bundle** (`commit+gpu+stamp+NATURAL`, DirectNet clean `large`):
  **48/48 executed** (4 techs × 4 circuits × OMP {1,2,4}); binding
  SRAM+switchcap 24/24, Rule 2 15/15 on CUDA, zero flips/OOMs/errors.
  Report-only basket 12/16 strict — exactly the V7.4 CPU clean-`large` basket,
  metrics identical except one 0.02 pp opamp delta with the same verdict.
- **T4 full latch basin: 8/8 PASS, zero basin flips**; worst max|ΔV|
  0.1206 mV, worst q-NRMSE 0.0101 %VDD.
- CUDA stays opt-in because CPU/flags-off remains the scored compatibility
  contract, not because a gate is open. Evidence:
  `results/v720_gpu_regate/t{3,4}_gpu_bundle/`.

### TSMC6 ≡ TSMC7: §7 confirmed exhaustively

Of 1748 shared modelcard parameters 11 differ + 74 TSMC6-only + 12 TSMC7-only
= **97 keys, all with zero occurrences in the BSIM-CMG Verilog-A** (vs 333 of
the identical keys that do appear). All 97 are TSMC TMI layout-effect
parameters; no core device-physics parameter differs. Over-determined: OpenVAF
never compiles them AND `nn_generate.py` stamps only `{L, NFIN, TFIN,
DEVTYPE}`. The regenerated `tsmc6_*.npz` are bit-identical to `tsmc7_*` bar
the tech-name field.

### Dead end recorded

- **Round-robining BSIM-AR training over a shared GPU** (`GPUS='1 1 0'`;
  GPU 0 held ~9.5 GB free) — `large`/`xl` jobs OOM there at epoch 0. The wave
  was not restarted (would discard a 75-epoch in-flight job); recovery layered
  instead (rescue → fill → `--require-complete`). Pin big tiers to a GPU with
  real headroom.

---

## V7.3.0 — accuracy reports restructured, re-gated on one code state (2026-07-27 → 29)

**Nine cross-cutting accuracy documents became two per family**
(`{DirectNet-L73,BSIM-AR-L74,PFN-L75}-{clean,recipes}.md`): the clean report
answers per tech/scale/testcase, the recipes report carries the addenda,
cross-cutting material lives once in `methodology.md`, and the pre-fix archive
keeps only the register of retracted claims (frozen tables recoverable via
`git show 37cef77:docs/accuracy/archive-pre-gds-fix.md`). **All tables are
generated**: `v730_docs_build.py` (`--check` fails on stale files),
`v730_coverage.py` (coverage by pass + runnable gap job file),
`v730_control.py` (cross-pass comparison). 2173 → 1768 lines with more
measurement behind them.

**Denominators /16 → /20:** TSMC6 folds into the headline (complex /20, device
AC /10, opamp AC /5). No total is comparable across the boundary without
rescaling. TSMC6 remains TSMC7 relabelled; only presentation changed.

**Measured** (1536 cells, 12 clean + 14 recipe groups; the control reproduced
176/176 jointly-measured cells at OMP=1, licensing the V7.1.0 merge):

- **BSIM-AR clean strict at all four tiers, identical scores at each**, zero
  flips over a 22× capacity range; the open set is the low-VDD ring column only.
- **`corroft@medium`, `corro15@medium` and all four corridor recipes at `xl`
  sweep 20/20** — `corroft@medium` the only checkpoint passing every cell.
- **TSMC6 retires DirectNet's sweep:** `crit15m@xl` passes `tsmc7-opamp` and
  fails `tsmc6-opamp` — same data, same recipe, different training run/basin.
  A sweep a duplicate column can break was never a sweep. Every such split is
  in the bimodal opamp column.
- **PFN's first curriculum arm** (`corroft@small`): rings 3/5→5/5, opamps
  2/5→0/5 — total unchanged, failure set replaced. Cleanest instance of
  curricula *relocating* basins rather than composing them; third architecture
  where the corridor is the ring lever.
- TSMC6 recipe checkpoints trained (26); its freshly harvested ring corridor is
  `array_equal` to TSMC7's — a fourth reproduction of the duplicate finding.

**Fixed:** `--cuda` no longer silently falls back to CPU (a faulted 4090 turned
a run into a 50× slower CPU run that would have been gated normally; guard
exits 1 — note torch enumerates FASTEST_FIRST so nvidia-smi index ≠ CUDA
index). `v730_coverage.py --require-complete` keeps half-trained checkpoints
(bare `_best.pt`, no `.complete`) out of gates.

**Dead ends / non-fixes:** the opamp open-loop AC bias-resolution defect is
documented, not fixed (gate changes are decisions, not side effects). "Every
family is flip-free" retracted a second time — the rule is now "a nonzero flip
count is unbankable, re-measure".

---

## V7.2.0 — GPU-accelerated large-scale SRAM transient (branch `v720-gpu-scaling`, 2026-07-27/28)

**All phases of `docs/plans/2026-07-26-v720-gpu-scaling.md` landed.**
Bit-identical work ships default-on; perturbing levers ship default-off behind
env flags (same discipline as V7.0.x). Workload context: a 200-step SRAM write
op = 483 NR iterations + 200 commit points; the rev-4 discovery was the
post-step charge-history commit running one cache-cold batch-1 eval per device
per step = 75–85 % of transient wall.

### Bit-identical, default-on

- **Phase 1, parse** `4d76c22` — cache-first `torch.load`/norm/sidecars,
  memoized resolver. `PYCIRCUITSIM_NN_DEVICE` default **cpu** (fixes the
  silent-CUDA provenance bug). 32×32 parse ~44 → 4.67 s.
- **Phase 2a-lite + 2c** `146f05f` — one D2H block transfer, deduped constant
  tensors, value-keyed geometry/tech-code cache.
- **Phase 2a-full, batched denorm tail** `e6f8154` — per-device `_unpack_eval`
  → one float64 numpy pass, all three families. §8.1 bit-exactness enforced in
  code: per-element libm `math.exp` on the Vds-correction (np.exp differs 1 ULP
  on ~4.6 % of args, amplified ~60× by the `1−exp` cancellation), float64
  casts + dtype asserts. New gate `verify_batched_tail.py` 22/22 exact.
- **Phase 2d + 4a, NR-loop vectorisation** `1a576ed` — per-node loops
  vectorised; `Circuit` caches nodes/map behind `invalidate_topology()`; one
  LIL→CSR conversion per NR iteration. A/B byte-identical.

### Perturbing, default OFF (env-flag opt-ins)

- **Phase 2t** `d1b8e40`, `PYCIRCUITSIM_TRAN_BATCH_COMMIT=1` — batch-eval the
  commit loop. 4×4 write op 146 → 34.7 s (4.2×); 16×16 latch end-states
  512/512, max final ΔV 0.52 mV.
- **Phase 3a** `2496b9a`+`38b6920`, `PYCIRCUITSIM_NN_DEVICE=cuda[:N]` — GPU NN
  eval with runtime-enforced T0 determinism pins (TF32 off, deterministic
  algorithms, CUBLAS workspace).
- **Phase 3b** `4e9c396`, `PYCIRCUITSIM_BATCHED_STAMP=1` — NN stamps as one
  COO from cached index arrays; 7.0×/6.1× on stamp+convert; perturbation is
  last-bit (max rel 3e-15).
- **Phase 4a′** `4e9c396`, `PYCIRCUITSIM_MNA_ORDERING=<spec>` — explicit splu
  ordering. **Re-measure on real matrices REFUTED §5.2's synthetic claim:**
  COLAMD fill is benign, `MMD_AT_PLUS_A` is *slower* than shipped, and
  **NATURAL wins 2.4–30×** (128×64 `.op` factor 152.9 → 4.7 ms). Phase 4b
  (KLU-class) demoted.

### §8.4 gating — CPU flag bundle GATED AND PASSING

- **T4 latch-basin** (new gate `verify_latch_basin_gpu.py`): all five CPU flag
  configs 8/8, 0 basin flips, ≤60 µV; `commit+gpu` 8/8 (≤0.37 mV, RTX 4090).
- **T3 16-gate CPU-bundle campaign** (`v720_t3_flag_bundle.sh`, 48 cells):
  **15/16 strict, 0 flips — production cell-for-cell**; binding sram+switchcap
  deviate 0.00 pp. Flags stay default-off; the GPU-axis T3 pass remained open
  (closed in V7.4.0).
- **Phase 0 (partial)** `8a2a18b` — the guard-F discontinuity is the **bulk
  regime** of an SRAM array (37.4 % of evals hit the negative branch), so the
  T1 branch-disagreement counter can never bind; binding tiers are T2/T4.

**Version summary:** same 4×4 write op: 146 s baseline → 100.3 s flag-off
(default-on work) → 20.3 s with {commit, stamp} = 7.2×, latch basins 8/8 in
every T4 config. Regression state (CPU flags-off): reference CSVs
sha256-identical; subckt 11/11; L72 op/dc/tran PASS; AC 2/2; inverter 8/8.
Docs consolidation 2026-07-30: README gained §Performance & GPU Acceleration;
CLAUDE.md de-duplicated.

---

## V7.1.0 — accuracy pivots, pre-fix device/AC re-measure, TSMC6 restored, PFN xl (2026-07-25)

Four threads: `docs/accuracy/` reorganized into cross-cutting pivots
(by-tech/by-scale/by-recipe — retired again in V7.3.0); every number still
standing on pre-`gds`-fix code re-measured (resumable driver
`scripts/v710_regate*`; control confirmed HEAD reproduces the V6.13.0
verdicts); TSMC6 restored; PFN gained an xl preset (14.86 M params, mirroring
BSIM-AR xl; lr 3e-4 after the V6.10 divergence collapses).

- **"AC peaks at SMALL" — retracted.** DirectNet device AC 7/8·8/8·8/8·7/8
  across small→xl (pre-fix 5/12·4/12·4/12·4/12): level *and* shape were
  artifacts of the wrong-signed gds.
- **The production curriculum improved the charge surface too:**
  `v660clean@large` fails TSMC5-NMOS AC where `crit30f` weights are 8/8.
- **"Opamp open-loop AC is 0/4 everywhere" — false** (DirectNet `small` banks
  TSMC16; BSIM-AR banks more, un-railed OPs).
- **Finding (not fixed):** the opamp AC gate has a bias-resolution defect —
  2 mV sweep grid across a 3–14 mV transition, `op_valid` applied to the NN
  only.
- **TSMC6 restored as a controlled repeat, not a technology** — a bit-identical
  duplicate is the only instrument for run-to-run variance with data held
  fixed (the first repeat's 68.2 % vs 2.0 % SRAM disagreement collapsed to
  5.2 % vs 6.2 % once gds was fixed). Tail codes 22–24;
  `assert_tech_is_distinct()` kept with tsmc6↔tsmc7 the sole acknowledged
  duplicate. Scoring rule then: own /4 column, never folded into /16.

---

## V7.0.0–V7.0.4 — NN compact-model performance (2026-07-25)

**Inference DC solve 1.68× byte-identical; training 4.9×/epoch; BSIM-AR 1.6×
behind an opt-in flag. Full measurements + dead ends:
`docs/plans/2026-07-25-v700-nn-perf.md`.** Governing constraint: a last-bit NN
perturbation can land a different NR basin, so every change is bit-identical
(default-on) or perturbing (default-off flag, promoted only after a re-gate).

- **V7.0.0 scan:** inference is **bandwidth-bound** (DirectNet `large` streams
  3.6 MB of weights 4× per eval); training was loader-dominated.
- **V7.0.1:** DC/OP skips charge Jacobians (`_require_nn_caps` contract;
  `get_capacitances` self-heals). 1610 → 784 µs/eval; CSVs sha256-identical.
- **V7.0.2:** `_DeviceBatches` on-device slicing + fused AdamW; 3.4 → 0.7
  s/epoch. Changes shuffle order (retrain ≠ same weights); `BSIMAR_LOADER=torch`
  reproduces legacy.
- **V7.0.3:** fused analytic Jacobian `PYCIRCUITSIM_NN_FUSED_JAC=1`, DEFAULT
  OFF — transient 1.38×, DC slightly slower (transient/AC lever only). Not
  bit-identical; stays off until a 16-gate re-gate.
- **V7.0.4:** LEVEL=74 AR prefix cache `PYCIRCUITSIM_NN_AR_CACHE=1`, DEFAULT
  OFF — 1.60× DC / 1.56× tran. "Exact in exact arithmetic" refuted bit-wise:
  `F.linear` is not row-stable on CPU, so **no incremental AR form can be
  bit-identical in float32**; deviation ≤1.6 µV.

**Dead ends (do not retry):** TF32 / `torch.compile` / bf16 autocast all
*slower* for DirectNet (launch-overhead-bound); replica-batch gradient trick
dominated by the analytic Jacobian; larger training batch excluded (a recipe
change invalidating every comparison).

---

## V6.13.1 — systematic-audit fix wave 1: 22 gate-neutral findings (2026-07-24)

Closed the 22 findings from `docs/2026-07-21-systematic-audit.md` that cannot
change a gated number; the 19 gate-affecting ones staged behind a re-gate.
Classes: **silent-wrong parser** (C1 `+` continuation lines dropped outside
`.model` — now every logical line buffers, orphan `+` raises; case-sensitive
ground; duplicate `X` names merging internal nodes; duplicate `.model`
redefinition; polarity-mismatched env pins falling through); **silent-green
harness** (11 dispatchers exited 0 regardless of sub-jobs; report published
from an empty tree; opamp gate without a minimum-gain guard; typo'd `--tech`
silently SKIPped; production checkpoint slots writable with `--overwrite` —
deleted); **data-pipeline integrity** (geometry-sha256 `.meta.npz` sidecar
fingerprints; universal vocab-size guard; TabPFN `_ctx_cache` invalidation on
state-dict/EMA writes). **Dead end recorded:** the audit's prescribed
`F.softplus` NaN rewrite is measurably NOT forward-bit-identical (23/401
samples) — dropped; the argument-clamp variant measured 401/401 identical if
ever wanted.

---

## V6.13.0 — gds sign + guard fix shipped, TSMC6 retired, every checkpoint re-gated (2026-07-24)

**Shipped the audit's last P0 (`A3`, the NN `gds` sign bug), retired TSMC6,
re-gated all 36 checkpoint sets. Production DirectNet 14 → 15/16 strict.**

**The fix (`8ed35bd`):** inference negated `gm`/`gmb` but not `gds` — all
three derive from the same signed `id`; the loss had negated all three since
V6.4 and the correction never reached inference (autograd vs `-gds_head` =
0.12 rel err, vs `+gds_head` = 2.08 — the signature of a pure sign flip). The
old two-sided floor `max(gds, |id|*0.5)` asserted an Early voltage ≤2 V,
overriding the learned conductance at 90.9 % of amplifying points — load-
bearing only because it masked the sign error. Replaced by **guard F**:
positives pass bit-identical, negatives clamp to `|id|/50 V` (OSDI −d(id)/dVd
is positive at 100.0000 % of 111,630 conducting evals). **Sign and guard must
ship together** — sign alone is bit-identical, guard alone regresses device AC.

**TSMC6 retired** (`38c47d8` + PyCMG `23b0ace`): audit §D1 — bit-identical
datasets, identical L72 currents. 22 checkpoints + registry entries deleted;
codes were tail so nothing renumbered. (Restored as a controlled repeat in
V7.1.0.)

**Re-gate — one signature dominates: every gained cell is an opamp** (`gds`
sets small-signal output resistance; it cancels at the Newton fixed point
everywhere else). DirectNet production 14→15/16 strict zero flips
(`tsmc16-opamp` closes; `tsmc7-opamp` sole open cell at `large`).
**`crit15m@xl` = 16/16 STRICT, zero flips** — first full-matrix sweep under
one uniform recipe; not promoted (2.3× inference, no device gain). Clean
capacity curve 10/10/13/12. Universal: +3 strict and **all three OMP FLIPs
eliminated** (a wrong-signed Jacobian entry was the thread-count sensitivity).
PFN 11/11/9; device AC at `large` 5/8→8/8. **BSIM-AR clean 14/16 at every
tier** — "capacity peaks at medium" was largely this bug; ceiling moved opamp
→ ring. Corridor recipes: `corroft@medium`/`corro15@xl` 16/16 STRICT. **gds
moved opamps, the corridor moves rings — independent levers.** DC confirmed
exactly invariant (the single parametric-DC failure is bit-identical pre/post).

**Retracted:** "`tsmc7-opamp` reachable only by the V6.5.9 T3 fine-tune"; "the
three-basin simultaneous hold is the open 15/16 target"; BSIM-AR's
capacity-peak-at-medium.

**Also:** methodology note — the resumed re-gate half ran from a frozen rsync
snapshot of `d2ea720` (verified byte-identical), decoupling an 8-hour campaign
from the working tree; worth repeating.

---

## V6.12.1 — silent-green P0 branch merged + accuracy reports per family (2026-07-24)

Merged `fix/silent-green-p0`: `tests/common/base.py` never checked NGSPICE's
exit status — a dead binary left a stale CSV and everything passed against
**stale ground truth** (reproduced with `NGSPICE_BIN=/bin/false` → 8/8 PASS);
now unlink-before-invoke + raise. `solver.py`: `spsolve` returns NaN on
singular matrices so the `LinAlgError` guards were dead code; now detected.
Also landed the 5-area ~70-finding audit register
(`docs/2026-07-21-systematic-audit.md`, incl. gds §A3 and TSMC6 §D1) and
consolidated seven per-version accuracy reports into three per-family ones.

---

## V6.12.0 — .subckt/.ends hierarchical netlists (2026-07-18)

Added `.subckt`/`.ends` + `X` instances (flattening at parse time,
ngspice-style — solver untouched), `.ic` into subckt bodies, converted the
test circuits to hierarchy, re-ran the full matrix: **zero regressions**
(484/489 checks, all 5 non-passes pre-existing). New gate `verify_subckt.py`
(8/8 → 11/11 in V6.13.1), subckt ≡ flat at max|ΔV|=0. Three pre-existing CLI
defects surfaced by README smoke-tests fixed (trace-count log, missing
transient CSV/lis, duplicated final sample from an IEEE-754 quotient — now
snapped at rel-eps 1e-9); README brought up to date (was stranded pre-V6.5).

---

## V6.11.0 — TSMC6 NN family trained + gated at every scale (2026-07-14/17)

Completed the V6.9.0 deferral: all three NN families trained/gated on TSMC6
(22 clean ckpts). ⚠ TSMC6 later found ≡ TSMC7 — the splits below are basin
coin-flips, not distinct-tech fidelity. Complex 4-cell matrix: DN peaks
`large` 3/4 (opamp rails at every size), BSIM-AR peaks `medium` 3/4 (the only
family to pass the tsmc6 opamp), PFN flat 2/4. Device fidelity complete for
all 11 cells. **Bug fixed:** `tech_code_in_vocab` rejected TSMC6 — the
ASAP7-guard checked the *universal* code against an 18-ceiling, silently
SKIPping per-tech TSMC6; now any tech in `LOCAL_VARIANT_CODES` passes.
Campaign ran under sustained cluster overload (loadavg ~1400/192); gate
timeout raised 1800→7200 s; complex matrix 100 % resolved.

---

## V6.10.0 — TabPFN port: the "PFN" family, LEVEL=75 (2026-07-11/14)

Ported TabPFN v3 into the bsimar stack as a third family and ran the full
scale campaign (24 ckpts). **PFN clean small (0.69 M) = 11/16 STRICT with ZERO
flips** — the first family with no OMP multistability; strongest clean small
on record. Architecture: faithful scaled-down port of the three v3 stages +
two deviations (frozen learned context baked into the ckpt with context-KV
caching; direct 13-output value head for smooth autograd). Gate curve
**declines** 11/10/8 (s/m/l); device fidelity peaks medium; 15.6 ms/eval CPU
(4× faster than BSIM-AR). Root-caused: `nmos_nfin_10` off-grid geometry fails
at s/m (context-relative embedding interpolates the NFIN 6→21 gap poorly);
capacity repairs it. 8 divergence-collapse events at `large` (both lr tried);
diverged runs bank pre-divergence EMA bests. **Dead ends:** fp32 large @300ep;
"lr 3e-4 fixes divergence" refuted. Pretrained 58 M TabPFN ICL baseline: not
solver-viable (id 3.2–5.7 % — the from-scratch port beats it ~10× at 1.2 % of
params).

---

## V6.9.0 — TSMC6 (CLN6) onboarding + TSMC PDK parse audit (2026-07-12)

Onboarded the N6 iPDK card as sixth tech (LEVEL=72 + NN plumbing + datasets;
NN training deferred to V6.11.0) and audited PyCMG's card parsing across all 5
TSMC cards. ⚠ Later corrected: TSMC6 is TSMC7 relabelled. Universal codes
22–24 tail-appended (existing ckpts stay valid). **Real defect fixed:** the
12-param fingerprint labeller (`loo_labels.py`) used silent last-writer-wins
and tsmc6↔tsmc7 collide 108/108; per-tech datasets now label against their own
tech and the universal scan raises on cross-tech collision. Verification:
PyCMG pytest 314/314; TSMC6 DC 9/9, tran 14/14; full 6-tech regression clean
bar the pre-existing `TSMC5_lvt_inv_l_24nm` ERROR. **Parse audit PASS** (40
devices, 0 round-trip mismatches); durable verdicts: mid-line `*` in blocks is
multiplication NOT a comment (`_extract_model_params`'s no-strip is
load-bearing); bin selection inclusive-both-ends is self-consistent
PyCMG↔NGSPICE; TMI/stat params ignored identically by both.

---

## V6.8.1 — BSIM-AR xl-tier fill (2026-07-11/23)

Trained the xl preset (14.81 M) across the full Phase-B recipe mirror (48
ckpts) and gated. **xl TIES medium at 15/16 strict, does NOT exceed it, does
NOT basin-shuffle** (the DirectNet xl shuffle does not replicate); tsmc7-opamp
is the only miss for corroft/crit15m/corro15. **AC COLLAPSES at xl** (opamp-AC
0/4 every recipe; device AC weak; charge-Sobolev does not recover it — a tier
property, not recipe-fixable). **No-promote:** 3× params + ~30–100× AR
inference for the same 15/16; `corroft@medium` remains the validated best.
Ops lesson: never reuse a `gate_iso_*` output dir across model families.

---

## V6.8.0 — BSIM-AR Transformer (LEVEL=74) un-parked: recipe campaign (2026-07-06/07)

Shipped the DirectNet training/recipe/eval stack to the parked AR Transformer
and ran the full scale × recipe campaign. **`corroft@medium` (1.9 M) = 15/16
STRICT, beating DN production (14/16)** — banks tsmc16-opamp + both low-VDD
rings; misses only tsmc7-opamp. Port notes: `unknown_code_id` was hardcoded
universal 17 (local vocab would CUDA-assert) → `num_tech_codes-1`; aux losses
force the MATH SDPA backend (fused SDPA has no double-backward); LEVEL=74
parser cascade + `PYCIRCUITSIM_NN_FORCE_LEVEL=74`. **Latent L74 bug fixed:**
`_out_col` ranked the canonical column order before the BSIMAR layout → qg
denormed as id (~5× current) though the module scored 0.38 % on the trainer's
test set — classic silent-green, caught by the first real checkpoint. Scale
study: complex peaks MEDIUM; AC peaks small; **inv_trip anchor is INERT on
the Transformer** (opposite of DN); the corridor is the whole ring lever;
tsmc7-ring ⊥ tsmc7-opamp under the corridor at every tier.

---

## V6.7.1 — house-clean after the V6.7.0 campaign (2026-07-05)

~12.7 G reclaimed; nothing load-bearing touched; retired checkpoints archived
to `/data2/shenshan/v66x_v670_retired_ckpts_2026-07-05.tar.gz` before deletion
(recorded-loser recipe pool + failed fine-tunes pruned; production/alternates/
universal bases kept). Datasets 26 G → 19 G (regenerable concats deleted;
corridor sets kept — retrainability over reclaim). The 580-line collector
output merged into the DirectNet accuracy report as Part II.

---

## V6.7.0 — universal DirectNet + TSMC5 transfer study (2026-07-04/05)

Resurrected the universal-scope DirectNet (ONE 18-code model on TSMC16+12+7),
ranked the Core-4 recipes on the 12 shared complex gates, measured TSMC5
few-shot transfer. Headlines: (1) **universal is VIABLE** — device fidelity
per-tech-grade; **`corroft` = 10/12 strict, 0 FLIPs = per-tech parity with
full OMP determinism** (which per-tech large never had); corridor fixes
tsmc7-ring 14.89→3.61 %; anchor/csob basins do NOT survive the scope change
(recipe→basin maps are SCOPE-dependent). (2) **TSMC5 onboarding = ~1 M
stratified rows** (`plain@n1M` 4/4 STRICT at half the data; ≤10 k DIVERGE;
n1M beats nfull). (3) **No free retention** — source techs collapse at gate
level; fine-tune = de-facto per-tech ckpt. Phase 1b: the opamps-XOR-rings wall
reappears at universal xl → `corroft@large` is the final best universal
config. Env-pin-only; per-tech resolution untouched.

---

## V6.6.7 — 15/16 hunt round 1: csobcrit + crit30a1 both 13/16 (2026-07-03)

Both routed arms NEGATIVE; production stays `crit30f@large` 14/16. `csobcrit`:
the curriculum **relocates rather than composes** basins (csob's deterministic
tsmc16-opamp hold degrades to a FLIP while tsmc12-opamp is gained). `crit30a1`
(half anchor): reproduces corroft almost exactly — **the {16} → {5,12}
opamp-basin hop is DISCONTINUOUS in anchor weight ∈ (1.0, 2.0)**; the uniform
lever is exhausted for the 5+12+16 triple → 15/16 routes to structural levers.
Harness: crit-family `--init-from` at `large` redirected to the v660clean
archive (production slots carry crit30f — warm-starting from them would
silently stack curricula).

---

## V6.6.6 — xl curriculum ties production 14/16 strict + full test-infra audit (2026-07-03)

`corroft`/`crit10`/`crit15m`@xl = 14/16 STRICT (all bank tsmc16-opamp which
production fails); production unchanged. **The weight→basin map is
TIER-dependent**; curriculum warm-start rescues xl wholesale (clean@xl 10→14);
xl basins are OMP-deterministic unlike large's endemic opamp flips.
**Test-infra audit — 17 verified fixes**, silent-green class: >100 %
divergences scored as ERROR-skip → exit 0; all-ERROR suites exited 0; **an
absent `PYCIRCUITSIM_NN_CHECKPOINT_*` pin silently fell back to production**
(now raises); flat-reference `nrmse()` returned 0 = auto-PASS (now inf);
SUMMARY clobbering on subset re-runs; train-resume trusted `_best.pt` (now
`.complete` markers). The v664-P0 torch thread-pin landed in the complex
harness (verdict-neutral).

---

## V6.6.5 — recipe×size matrix completed: 13 recipes × 4 sizes (2026-07-03)

208 ckpts, 864 eval cells, zero blanks. `clean@large` 13/16 stays the unbeaten
in-matrix cell; **the corridor inverts the capacity curve** (dominates below
`large`, collapses above); **xl is basin-shuffled, not uniformly over-fit**;
AC peaks at SMALL across recipes; device-NRMSE bottoms at medium; tsmc7-opamp
0/52. Ops: 22 killed-run "best-so-far" ckpts quarantined + retrained (a
`_best.pt` on disk is NOT evidence of a completed run — gate on the log tail).

---

## V6.6.4 — crit30f PROMOTED to production (2026-07-02)

All 8 `tsmc{X}_dn_large_*` production slots replaced with the V6.6.3-validated
`crit30f` checkpoints (clean base + one curriculum fine-tune, ring-only
`corro` data); clean originals archived as `tsmc{X}_dn_v660clean_large_*`.
Production 13/16 (12 strict) → **14/16 strict**, verified on the default
resolver path. Checkpoints are gitignored — this entry is the record.

---

## V6.6.3 — full-recipe re-test: crit30 supersedes crit15 at 14/16 STRICT (2026-07-02)

All 22 on-disk recipes re-tested under one discipline (isolated matrix + OMP
determinism sweep). Best = `crit30` 14/16 STRICT; validated by `crit30f` (all
8 ckpts retrained to full spec — the honest rerun reproduces cell-for-cell).
The corridor-weight → tsmc5-opamp basin map is **non-monotone** (w1.0 FLIP,
w1.5/2.0 detFAIL, w3.0 detPASS — the inv_trip anchor makes w3.0 safe). csob
re-scoped to the AC/device alternative (12 strict). tsmc7-opamp: 100 % FAIL
across all 23 artifacts × all OMP.

---

## V6.6.2 — the cross-wall combo breaks 13/16: crit15 = clean+1 (2026-07-02)

REFUTED the V6.6.1 "13/16 uniform ceiling": `crit15` (corridor + inv_trip
curriculum — the two levers had only ever been tested separately) nets 13/16
strict = clean+1, the +1 the DETERMINISTIC tsmc5-ring opening (12.66→4.0 %).
Confirmed live that the opamp gate is a multistable OMP coin-flip — single-run
opamp passes are unbankable. Round-2 arms read NEGATIVE on single runs —
corrected by V6.6.3's strict re-test.

---

## V6.6.1 — uniform-recipe comparison sweep (2026-07-01)

Swept uniform recipes (csob/sob/ekv/seeds/combos) across all 4 sizes: **NO
uniform recipe beats clean's 13/16 at `large`**. The ceiling is mutual
exclusivity of value-surface basins — each recipe/seed lands a different
subset; combo stacking is zero-sum (EKV core breaks tsmc12/16 SRAM). `csob` =
best all-rounder → documented alternative; refutes the V6.5.x "charge-Sobolev
dead on arrival" verdict (measured only at `medium`). `sob` reconfirms
deriv-fidelity ⟂ opamp.

---

## V6.6.0 — house-clean + uniform-recipe reset (2026-06-29)

Deliberate reset from V6.5.9's hand-tuned 16/16 to the honest uniform
baseline: all 32 DirectNet ckpts retrained on ONE recipe; production = uniform
`large` at 13/16 (capacity curve 7/10/13/10 s→xl). The 3 open gates are the
true fidelity frontier the V6.5.x per-case specials had force-closed.
House-clean: datasets 16 G → 4.5 G; V6.5.x specials archived off-repo
(`/data2/shenshan/v6.5.9_production_specials.tar.gz`; rollback = `git checkout
V6.5.4` + untar); resolver prefers per-tech `large` first.

---

## V6.5.9 — ★ 16/16: T3 differentiable-DC-solver lands the tsmc7 opamp (2026-06-29)

First-ever tsmc7 opamp PASS (gain 178.0 vs 163.4, 8.92 %) → production 16/16.
Put the DC solve **inside the loss**: a differentiable unrolled Newton solver
supervises the emergent transfer curve against L72, so r_o is shaped by the
gain target. Broke the V6.5.8 "gain stuck ~370" wall — the gain-163 root DOES
exist; "370" was the gate's continuation landing on an over-flattened branch.
The gate gain is bimodal + sampling-noisy; **preservation, not existence, was
the binding work**. Installed via symlink (retired in V6.6.0). Memory
`[[v659-t3-solver-lands-opamp-16of16]]`.

---

## Test-infrastructure correctness sprint — 11 bugs fixed (2026-06-28)

Production pass-rates unchanged; every fix re-checked vs NGSPICE. **B1
(CRITICAL):** the per-tech device gates pinned **tsmc5's** net for ALL techs;
routed through `_cascade_handles_stem`. **B3/B5:** SRAM scored PASS when every
corner errored (`all([])==True`) and never compared to ground truth; now ANDs
point-by-point NGSPICE tracking, `force_ic` reconciled as a printed
diagnostic, not a gate. **B4:** a diverged inverter transient could
false-PASS. Plus sweep↔gate `uic` canary, ASAP7 skip, real-deck canary,
honest exit codes.

---

## V6.5.8 — EKV high-r_o core breaks the tsmc7-opamp rail (2026-06-28)

First non-railed tsmc7 opamp of the campaign (structural EKV core +
vout-weighted KCL fine-tune → real amplifying curve, gain ~350–381) —
**REFUTES the V6.5.6/7 "only T3" verdict**. BUT gain ⟺ existence are coupled
through the output-stage r_o: every calibration lever is a binary rail↔370
switch (the over-flattened r_o is *required* for reachability). Nothing
installed (15/16); routes to T3. Memory `[[v658-ekv-core-breaks-opamp-rail]]`.

---

## V6.5.7 — panel-review correction of the V6.5.6 opamp verdict (2026-06-27)

5-agent adversarial review found "no high-gain zero exists / only-T3"
over-strong (the probe measured reachability, not existence). The cheap
vout-prioritized existence retrain was then RUN & KILLED: the vout F_rel floor
compatible with preservation is ~20× above what a high-gain zero needs — the
soft-wall is near-hard for the KCL-loss family. fetlim also dead. Memory
`[[v657-vout-existence-retrain-kill]]`.

---

## V6.5.6 — 3-operator Phase-0 routing + T1 KCL-residual lever (2026-06-26)

**Durable organizing frame — the 3-operator taxonomy:** the solver reads the
one NN surface through three operators, each owning a different gap:
id-values→KCL→NR fixed point (opamp gain, ring period); autograd dQ/dV→pole
(f3db); off-diagonal cgd→RHP zero (HF phase). Charge-head retrains are
DC-safe; id-surface retrains are not; the recurring ledger failure is applying
the wrong fix-class. Phase-0 diagnostics: existence (not conditioning) is the
tsmc7 gap; f3db is OP-drift-owned (caps already match OSDI); the fixed-point
LOCATION is a pure function of `id` VALUES. T1 KCL-residual loss solved
existence but produced an unstable fixed point, and preservation is binding.
⚠ Its "only-T3" conclusion corrected by V6.5.7. Memory
`[[v656-t1-existence-to-contraction]]`, `[[nn-accuracy-3operator-taxonomy]]`.

---

## V6.5.5 — diagnostic-routed corridor retrain → 15/16 (2026-06-24/25)

tsmc5 ring = NMOS-conduction-owned → lifted 3/4→4/4 via `large` + ring
corridor + seed7 (capacity was the bind). tsmc7 opamp = value-surface-owned —
seeding the sweep from the L72 ground-truth OP at every point STILL rails
(the high-gain OP is unstable on the NN surface; PTC/homotopy/OP-seed cannot
fix it). Net 14→15/16. Memory `[[v655-corridor-retrain-15of16]]`.

---

## V6.5.4 — fresh full retrain + best-config-per-tech → 14/16 (2026-06-23/24)

Full capacity matrix retrained from scratch on regenerated data (one clean
recipe); best config per tech → 14/16, clean. **Native-L72 control (decisive
diagnostic, `diag_l72_complex_control.py`):** the exact gate circuits through
PyCircuitSim's own solver with the OSDI model match NGSPICE at ring 0.00 % /
opamp ≤0.10 % ⇒ the remaining gaps are genuinely NN-value-surface-owned, not
solver/harness. Memory `[[v653-l72-control-ring-opamp-model-owned]]`.

---

## V6.5.3 — ★ the switchcap gap was a HARNESS CLOCK BUG (2026-06-23)

**Overturns V6.5.2.** The tsmc5 switchcap "11.84 % over-charge" chased across
the entire V6.4.x–V6.5.2 campaign was two harness bugs: (1)
`render_directnet_netlist` rescaled supply rails but MISSED the
space-delimited PULSE clock rail — the NN clock over-drove tsmc5 pass gates to
0.80 V vs NGSPICE's 0.65 V → 11.84 % FAIL became 1.56 % PASS, switchcap 4/4;
(2) the "14.65 % L72 floor" was a control op with no `uic` pinning. **LESSON
(load-bearing): when an NN gate fails vs NGSPICE, FIRST diff the rendered NN
netlist against the NGSPICE deck token-by-token BEFORE blaming the model or
solver.** `uic` made first-class in the product path. Memory
`[[v652-switchcap-is-harness-clock-bug]]`.

---

## V6.5.2 — charge-derivative levers + the (refuted) switchcap-is-SOLVER-owned finding (2026-06-22)

> SUPERSEDED by V6.5.3 — the conclusion was two harness bugs.

Both switchcap levers KILLED (correctly — there was no model gap):
charge-Sobolev left switchcap unmoved and did NOT move f3db (⇒ OP-drift
owned); TG-corridor fixed PMOS cdd 62 %→5 % yet the charge didn't move. Valid
reference: NN autograd caps match OSDI ~0.3–2.5 %; per-channel sign map
`+cgg,−cgd,−cdg,+cdd`.

---

## V6.5.1 — XL capacity tier + µA-band loss lever (KILLED) (2026-06-22)

**XL (2.13 M) = the over-fit boundary:** pass-rate 6→9→12→9/16 (S→M→L→XL); XL
fits the device surface ~10× tighter yet loses every value-surface-fragile
gate `large` won. µA-band `SubthresholdIdLoss` retune KILL (moved switchcap
<0.2 %). Also fixed the `xargs -L1` trailing-blank silent job-collapse bug.

---

## V6.5 — AC small-signal accuracy of the NN models (2026-06-22)

First NN AC gate vs NGSPICE (24 ckpts + opamp): AC **gain** excellent
everywhere (autograd gm/gds accurate); cap-driven pole good but tech-variable;
the **Cgd-feedforward RHP-zero HF phase is NOT reproduced**; opamp AC inherits
the DC value-surface fragility, but where the OP lands well GBW is 0.97× —
dynamics right, DC-gain level the miss. No retrain warranted. Harness
`complex_ac.py`, `verify_nn_ac.py`, `verify_complex_opamp_ac.py`.

---

## AC analysis — small-signal frequency-domain (2026-06-21)

Brought `.ac` from dead-on-arrival to NGSPICE-validated: `ACSolver` solves
complex `Y = G + jωC` about the DC OP; added the missing MOSFET
**transcapacitance stamp** (source-referenced 2-port embedded in the nodal
3×3, charge-conserving). `verify_ac.py` 2/2 (RC 0.0000 %; CS-amp gain err
5.4e-6 dB). Gotcha: ngspice `wrdata vp()` emits radians.

---

## V6.4.9 — DirectNet S/M/L capacity benchmark (2026-06-21)

Pass-rate rises 6→9→12/16 (S→M→L) but device accuracy is excellent at EVERY
size (not the bind). The opamp is the value-surface-fragile gate (recovers
only at `large`, only tsmc5/12); switchcap needs capacity; SRAM 4/4
everywhere. More capacity does NOT close recipe-sensitive gaps.

---

## V6.4.8+ — parametric sweep harness + TSMC7 broad retrain (KILL) (2026-06-20)

Built the complex-circuit parametric sweep harness (baseline-gated,
sha256-pinned). **TSMC7 broad retrain = KILL:** breadth fits the value surface
but COLLAPSES the offset-dominated opamp (gain→0); reverted. The opamp holds
gain under load perturbations but collapses under almost ANY OP change.

---

## V6.4.8 — value-surface accuracy campaign; 14 → 15/16 conditional (2026-06-17→20)

**Methodology locked: all gates run CPU** (the fragile opamp lands a different
NR basin on CUDA). S0 floor-k KILL (gain non-monotone in the floor coeff — it
hops NR basins; gds cancels at the fixed point). S1 capacity KILL (larger net
fits better yet collapses the opamp). **S2 continuation-first DC sweep KEEP
(the sole win, load-bearing):** warm-started NN points with source-stepping
off, gated on `has_nn` so BSIM-CMG is byte-identical; tsmc7 opamp 10.78 %
FAIL → 8.63 % PASS — the win is path-preservation. S3 EKV backbone KILL.

---

## V6.4.7 — serialized accuracy campaign; SHIP at 14/16 + force_ic 8/8 (2026-06-10→16)

Strict serial S1–S19 chain from the V6.4.4 canonical 8/16. **Durable changes
still in code:** S2 — the NMOS source-frame fix (NN Rule 2; permanent canary
`verify_nn_lifted_source_dc.py`); S7 — reverse-Vds clamp relaxation (C¹ taper;
the wider corridor was KILLED: tsmc5 opamp veto). **Key findings:** S6
simulator EXONERATED (native-L72 ring control ratio 1.000); S9b regen-v2 data
+ two load-bearing data-gen fixes (the `NN_DC_SOLVE_TOL` floor causing the
zero-row artifact; an atomic-write fix for a modelcard-cache race); S10 —
MAJOR: **derivative fidelity is ANTI-correlated with the opamp** (the Jacobian
guides NR convergence but cancels at the fixed point)
`[[v647-s10-deriv-fidelity-vs-opamp]]`; S12 — trajectory-corridor KEEP
(11→14/16; ancestor of every later corridor recipe); S17c — the force_ic
0/8→8/8 "gap" was a HARNESS BUG (non-physical read-disturb both NN and ground
truth fail) — LESSON: run the native-L72 control before blaming the NN; S19 —
trust the `verify_complex` gate, not the scorer proxy.

---

## Condensed history (pre-V6.4.7)

> Full detail for these iterations lives in `git log` and `MEMORY.md`. Only the
> durable outcomes are retained here.

### V6.4.6 — diagnosis-first iteration (2026-06-01/02, no behavioral change)
Gated every GPU-spend behind a 0-GPU diagnostic; localised the RO error to the
**id VALUE surface** (not the derivative). Probe/measurement fixes only.

### V6.4.5 — Track A no-ship iteration (2026-05-29)
Ran all 5 planned phases; shipped nothing. Built the multi-circuit scorer
(durable infra); confirmed the RO/SRAM gaps were architectural, not tuning.

### V6.4.4 — DirectNet per-tech checkpoint mix (2026-05-28, inference-only)
First per-tech medium checkpoint mix (canonical 8/16). Restored the
load-bearing `_MonotoneVgResidual` + `--monotonic` code (on-disk checkpoints
carry `mono.*` keys).

### V6.1 – V6.3.2 — per-tech DirectNet establishment (2026-05-12 → 05-15)
- **V6.1**: per-tech DirectNet for TSMC5/7; universal `refac_*`/`v4_*`
  artifacts deleted.
- **V6.2**: terminal-current sign fix; dead-band closed. **V6.2.1**:
  TSMC12/16 extension.
- **V6.3 / V6.3.1**: inverter spike-removal sprint (dataset regen with
  inv-trip recenter + reverse-Vds corridor).
- **V6.3.2**: PyCMG L3 parametric DC/transient sweeps ported to DirectNet.

### Pre-V6.0 (v3/v4/v5, package refactors, early milestones)
The BSIMAR package refactors (2026-03/04), the v3 LOO cross-tech sprint, the
v4 tech-code migration, the analytical Vds-correction + rail-restoring fixes,
and the v5 inverter-transient phases are recorded in `git log` and
`MEMORY.md`. Legacy LEVEL=1 (Shichman-Hodges) was removed; LEVEL=72/73/74/75
are the supported models.

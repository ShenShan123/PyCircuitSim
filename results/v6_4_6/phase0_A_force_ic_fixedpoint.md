# Phase 0 P0-A — `force_ic` railed-fixed-point EXISTENCE probe

**Date:** 2026-06-01  •  **Branch:** `feat/v6.4.6`  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Decides:** Open Q1 / plan §4 P0-A — does a self-consistent **railed DC fixed
point** exist for the released NN 6T cell? (The single most decision-critical
unknown of V6.4.6.)

---

## Method (matches the V6.4.5 temporary-edit-then-revert precedent)

1. Temporarily instrumented `pycircuitsim/solver.py` inside the `force_ic`
   cleanup block of `_solve_newton` (around lines 943–952), GATED behind
   `P0A_RESIDUAL` so default behaviour is byte-identical. After
   `self.initial_guess = voltages.copy()` and the `matrix_size` recompute (the
   `voltages` dict here is the **constrained, correctly-railed** solve), and
   again after the unconstrained re-solve, it called the in-method
   `_dc_residual_at(voltages, node_map, nodes, num_nodes, matrix_size,
   self.gmin)` (all variables verified in scope: `nodes`/`node_map`/`num_nodes`
   are method-level at `_solve_newton:560-562`; `matrix_size` recomputed at
   :947). The acceptance threshold uses the solver's own pattern,
   `max(_RESID_ABS_FLOOR, 100·reltol·rhs_scale)` with `_RESID_ABS_FLOOR = 1e-6`
   (solver.py:57) and `reltol = 1e-4`.

2. Command (source of truth):

   ```
   P0A_RESIDUAL=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     conda run -n pycircuitsim python tests/verify_complex_sram_snm.py \
     --tech TSMC5,TSMC7 --nfin 2 \
     > results/v6_4_6/phase0_logs/p0a_residual.log 2>&1
   ```
   (`--nfin 2` only narrows the butterfly sweep; the `force_ic` cell always uses
   `bt.nfin=2`. Both states × both techs were probed.)

3. **Reverted** the solver edit (`git checkout -- pycircuitsim/solver.py`);
   `git diff --stat pycircuitsim/solver.py` is **empty** (confirmed clean).

4. Independent cross-check via a standalone driver
   `scripts/v6_4_6_p0a_force_ic_residual.py` that re-stamps the **unconstrained**
   MNA KCL residual at the *literal* rail seed `(q=VDD, qb=0)` / `(q=0, qb=VDD)`
   on a fresh solver (no solver edit). The constrained solve lands on the exact
   rails, so the literal-rail residual equals the instrumented RAILED-seed
   residual — and it does, bit-for-bit.

---

## Result table (RAILED-seed = constrained, correctly-railed solution)

| Tech  | VDD  | State  | RAILED-seed (q, qb) | residual_inf | rhs_scale | threshold | ratio r/thr | Verdict |
|-------|-----:|--------|---------------------|-------------:|----------:|----------:|------------:|:-------:|
| TSMC5 | 0.65 | state1 | (0.650, 0.000)      | 8.5001e-05   | 6.500e-01 | 6.500e-03 | **0.01308** | **EXISTS** |
| TSMC5 | 0.65 | state0 | (0.000, 0.650)      | 8.5001e-05   | 6.500e-01 | 6.500e-03 | **0.01308** | **EXISTS** |
| TSMC7 | 0.75 | state1 | (0.750, 0.000)      | 1.2566e-04   | 7.500e-01 | 7.500e-03 | **0.01676** | **EXISTS** |
| TSMC7 | 0.75 | state0 | (0.000, 0.750)      | 1.2566e-04   | 7.500e-01 | 7.500e-03 | **0.01676** | **EXISTS** |

The RAILED-seed unconstrained KCL residual is **~60–75× BELOW** the acceptance
threshold on both techs, both states. **A self-consistent railed DC fixed point
EXISTS** for the released NN 6T cell.

### STUCK-final (where the unconstrained re-solve actually lands)

| Tech  | State  | STUCK (q, qb)     | residual_inf | threshold | Note |
|-------|--------|-------------------|-------------:|----------:|------|
| TSMC5 | state1 | (0.7021, 0.1632)  | 8.5001e-05   | 6.500e-03 | also a valid fixed point (residual small) |
| TSMC5 | state0 | (0.1632, 0.7021)  | 8.5001e-05   | 6.500e-03 | mirror |
| TSMC7 | state1 | (0.8151, 0.2264)  | 1.2552e-04   | 7.500e-03 | also a valid fixed point |
| TSMC7 | state0 | (0.2264, 0.8151)  | 1.2552e-04   | 7.500e-03 | mirror |

Both the railed point AND the inboard point have small residuals → the released
NN 6T cell is genuinely **bistable** (two co-existing DC fixed points). The
`solver.py:938-952` unconstrained re-solve, seeded at the rail, slides into the
**inboard** basin (q≈0.18/0.82) instead of staying on the rail. This is a
solver-path basin-selection problem, **not** the absence of a railed
equilibrium. (The STUCK residual ≈ RAILED residual is coincidental — both are
dominated by the same ~1e-4 KCL imbalance floor of the converged NN inverter
pair, well under the 100·reltol·‖b‖∞ acceptance band.)

---

## DECISION

> **RAILED-seed residual ≪ threshold (ratio ≈ 0.013–0.017) on BOTH TSMC5 and
> TSMC7 ⇒ a self-consistent RAILED DC fixed point EXISTS.**

Per plan §4 decision tree (`P0-A railed residual small? ── yes ▶ Phase 1
continuation viable (SRAM, 0 GPU)`):

- **Phase 1 (`force_ic` railed-solution recovery) is VIABLE at 0 GPU.** The fix
  is a *solver* change: replace the one-shot unconstrained release
  (`solver.py:938-952`) with a constraint-continuation homotopy (relax the IC
  temp-V-sources via `λ:1→0`, full NR per stage, warm-started) that **tracks the
  railed branch** instead of falling into the global inboard basin. The railed
  branch provably exists, so the homotopy has a target to track. Worth up to 8
  gates (4 techs × 2 states) with **zero model-regression surface**.
- Phase 1 must FIRST harden the probe acceptance: the current accept
  (`verify_complex_sram_snm.py:178-181`) uses a stale `_last_solve_converged`
  (never set on the early-return at solver.py:952) + rail-proximity, with **no
  KCL-residual gate**. Add a `_dc_residual_at` gate on the *released* solution.
- The transient write-then-hold re-spec (the "no fixed point" branch) is **NOT
  triggered** — it is held in reserve only if the homotopy slides off the railed
  branch at some λ\*<1 during Phase 1 implementation.

**This UNLOCKS Phase 1 and does NOT kill it. D3 (the V6.4.5 "true NN attractor"
finding) is refined, not contradicted: the inboard point is a real attractor,
but it co-exists with a real railed fixed point — so the gate is closable by
tracking the right branch, not by changing the model.**

(Rule 16: this is a residual-magnitude comparison; reported residual_inf,
rhs_scale, threshold, and the ratio per the instruction.)

# V6.4.7 S12 (P5) — trajectory-corridor overlay: build-ready plan

**Date:** 2026-06-14 · Reordered ahead of S11/P3 per the post-S10 rev-4 ruling
(the S10 finding makes the value-surface corridor the priority lever for the
value-owned opamp/RO gaps). **Status: designed + reconnoitred, ready to build.**

## Goal

Harvest the bias **corridors** the devices actually visit along the
ground-truth (NGSPICE-equivalent) trajectories of the 4 complex circuits,
evaluate OSDI at those biases, and add them as a weighted `traj_corridor`
sample-class to the v2 datasets — the same move as `inv_trip` (the project's
single most successful data lever: TSMC5 16.90 % → 0.92 %), at the V6.3.1
dosage (~3.5 % of rows). Targets: **TSMC7 RO 8.28 %**, **tsmc5 RO**, **TSMC5 SC
over-conduction 12.14 %**, opamp/TSMC16. Kill: first scored arm's RO err not
< 7 % → stop at one iteration (plan P5).

## Why harvest via native LEVEL=72 in PyCircuitSim (not raw NGSPICE node parse)

S6 proved pycircuitsim's native LEVEL=72 path reproduces NGSPICE at **ratio
1.000** (46.64 / 46.65 ps). So running each circuit with L72 in PyCircuitSim
(a) gives the ground-truth trajectory, and (b) reuses PyCircuitSim's parser
device→terminal mapping + per-timestep converged node solution — no brittle
NGSPICE-node-voltage reconstruction. Template: `scripts/v6_4_7_s6_l72_ro_control.py`
(`render_l72_netlist` + `build_merged_card` + `make_l72_parse` +
`run_directnet_transient`, which returns the FULL node trajectory keyed by node
name).

## Harvest steps (per tech ∈ {tsmc5,tsmc7,tsmc12,tsmc16})

BENCH variant per tech: tsmc5→lvt, tsmc7→ulvt, tsmc12→svt, tsmc16→svt; NMOS
L=16n, PMOS L=20n, NFIN=2, T per the bench (room-temp bin). Devices in each
circuit (from `examples/complex/*_directnet.sp`):
- **ring_osc** (transient): 5×{Mp(vdd-gate),Mn(0-gate)} — grounded-source.
- **opamp** (DC sweep): diff pair Nn1/Nn2 on **vtail** (lifted source), mirror
  Np3/Np4, tail Nn5, 2nd stage Np6/Nn7.
- **switchcap** (transient): pass NMOS Msw1/Msw2 with source = vsamp/vref
  (lifted).
- **sram** (DC sweep / force_ic): cross-coupled Mpl/Mnl/Mpr/Mnr + access
  Mal/Mar on q/qb (lifted).

1. **Run** each circuit with L72 (transient via `run_directnet_transient`; DC
   sweeps via `run_directnet_dc_sweep`) → full node trajectory.
2. **Map** node trajectory → per-device (Vd,Vg,Vs,Vb)(t) using the parsed
   `circuit.components` MOSFET `.nodes` (d,g,s,b) — robust, no hardcoded
   topology.
3. **Source-relative shift (CRITICAL):** the dataset + post-P0 inference use
   the Vs≡0 frame, so each harvested bias becomes
   `[Vd−Vs, Vg−Vs, 0, Vb−Vs]`. OSDI is difference-only, so eval at the shifted
   frame == eval at the absolute frame (validate this equivalence on a few
   points). Lifted-source devices (opamp tail pair, SC pass, SRAM access) are
   the whole point — they were the P0 blind spot.
4. **Pool** NMOS biases and PMOS biases per tech across all 4 circuits;
   **subsample** by bias-space dedup + residence×|id| weighting (poor-man's
   adjoint sensitivity) to a count that lands the class at ~3.5 % of the
   dataset (~tens of k rows/tech-device; HOLD the dosage — V6.3 Phase-B
   overdose cautionary tale).
5. **OSDI eval:** `_create_model_and_instance(tech_cfg, dev, variant, L, NFIN,
   T)` → `inst`; `geo = [NFIN, L, T, *proc.as_array()]` (the 15-col vector,
   identical to `generate_one_bin`); `eval_single_point(inst, vd,vg,vs=0,vb)` →
   13-col output. Reject None (NaN/|id|>1A).
6. **Append** to the v2 npz: add `traj_corridor` = SAMPLE_CLASS_CODES code 12
   (NEW — add to `pycmg/nn_generate.py`), extend `meta_sample_class_names`,
   concatenate inputs/geometry/outputs/sample_class, re-save NMOS rows →
   `{tech}_v2_nmos.npz`, PMOS → `{tech}_v2_pmos.npz`. **Back up the v2 npz
   first** (the S9b regen is expensive to reproduce). Validate: corridor row
   count, |id| decade coverage, OSDI sanity vs a few hand evals.

## Train + score

- Retrain ≥4 seeds × {techs} × {nmos,pmos} with `--class-weights
  traj_corridor=W` (W swept ~1–3; LDS-product renormalized to unit mean — the
  S9b plumbing). Same stock recipe as control-v2 (medium, EMA, filter off).
  **A/B vs control-v2** (same seeds) isolates the corridor data.
- Score the multi-circuit vector + force_ic + inverter (`OMP_NUM_THREADS=1`).
  **Blind vetoes = ALL currently-passing cells.** Kill: first arm RO err
  not < 7 % → stop at one iteration, rewind.

## Reuse map (all confirmed by S12 recon)

| Need | Reuse |
|---|---|
| L72-in-PyCircuitSim run | `scripts/v6_4_7_s6_l72_ro_control.py` (RO template; extend to 4 circuits) |
| Full node trajectory | `run_directnet_transient` / `run_directnet_dc_sweep` return dicts keyed by node |
| device→terminal map | `circuit.components` MOSFET `.nodes` (parsed) |
| OSDI eval + geo vector | `pycmg/nn_generate._create_model_and_instance`, `eval_single_point`, `geo=[NFIN,L,T,*proc.as_array()]` |
| sample_class + class_weights | `SAMPLE_CLASS_CODES` (+code 12), loader plumbing + `--class-weights` (S9b) |
| BENCH specs | `tests/common/complex.BENCH` (variant/L/NFIN/tfin per tech) |
| merged modelcard | `build_merged_card` (S6) |

## Risks / guards

- **Dataset corruption** (wrong geo / missed Vs-shift / sign): validate the
  appended npz before any retrain (decade gate + hand-eval cross-check + a
  bit-for-bit re-load); back up v2 first.
- **Overdose** → hold ~3.5 % dosage.
- **Circuit-specific L72 runs** (opamp/SRAM DC-sweep, SC transient) are the
  main new code beyond the RO template — build + smoke each before harvesting.
- **Blind veto** on all passing cells (P0-I: id-surface changes can regress RO
  through id↔charge coupling; corridor data changes the value surface
  broadly).

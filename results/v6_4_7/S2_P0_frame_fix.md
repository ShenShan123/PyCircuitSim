# S2 = P0 — NMOS source-frame fix (V6.4.7, 2026-06-10)

**Change:** `pycircuitsim/models/mosfet_nn.py` `_raw_voltages` now source-shifts
BOTH device types (`return v_d - v_s, v_g - v_s, 0.0, v_b - v_s`); previously
only PMOS was shifted, so lifted-source NMOS was evaluated at phantom Vgs/Vds
(inflated by +Vs) with Vbs=0 against a Vs≡0-trained network. Shift invariance
makes the fix exact physics. All downstream consumers take the shift-invariant
difference `v_d_nn − v_s_nn` (Rule-15 unaffected), verified by grep audit.

**New permanent gate:** `tests/verify_nn_lifted_source_dc.py` — NMOS Id–Vgs at
Vs ∈ {0, 0.1, 0.2}·VDD, absolute terminals (Vd=VDD, Vb=0), 4 techs, vs NGSPICE
OSDI BSIM-CMG (4-source netlist on both sides; lifted source applied in the
netlist, shift-invariance never assumed analytically). Gate NRMSE ≤ 10 %.
The vs input had ZERO verification coverage before this.

## Canary A/B (the decisive evidence)

| tech | vs0/VDD | PRE-fix NRMSE % | PRE MRE % | PRE R² | POST NRMSE % | POST R² |
|------|---------|----------------|-----------|--------|--------------|---------|
| TSMC5 | 0.0 | 3.05 | 13.6 | 0.989 | 3.05 | 0.989 |
| TSMC5 | 0.1 | 24.31 | 200.9 | 0.178 | **2.72** | 0.990 |
| TSMC5 | 0.2 | 63.83 | 809.4 | −6.05 | **2.51** | 0.989 |
| TSMC7 | 0.1 | 10.07 | 81.0 | 0.891 | **4.42** | 0.979 |
| TSMC7 | 0.2 | 30.95 | 324.7 | −0.17 | **4.41** | 0.976 |
| TSMC12 | 0.1 | 19.00 | 158.4 | 0.534 | **0.05** | 1.000 |
| TSMC12 | 0.2 | 51.84 | 651.9 | −3.32 | **0.07** | 1.000 |
| TSMC16 | 0.1 | 18.91 | 155.3 | 0.541 | **0.06** | 1.000 |
| TSMC16 | 0.2 | 51.37 | 633.7 | −3.14 | **0.08** | 1.000 |

POST: **12/12 PASS**; lifted rows now as good as grounded controls (TSMC5 even
better — reduced effective Vds). Error direction pre-fix was Id OVER-prediction
(phantom-inflated Vgs), up to ~80× at low Vg. CPU and GPU runs agree to
displayed precision. Kill criterion (post-fix regression vs OSDI) NOT tripped.
Full curves: `lifted_source_sweep_{prefix,postfix}.{csv,md}`.

## Battery (post-fix, CPU, OMP_NUM_THREADS=1; logs in `s2_logs/`)

| Suite | Pre-P0 (documented) | Post-P0 | Verdict |
|-------|--------------------|---------| --------|
| inverter 8/8 | 8/8 | **8/8 PASS** — tran NRMSE bit-exact (1.62/1.09/1.41/1.45 %); VTC 1.13/3.90/1.47/1.53 % (trip-point scatter, ~20× amplification of sub-mV float noise; documented behaviour) | HELD |
| DC 55/55 | 55/55 | **55/55 PASS** | HELD |
| tran 64/64 | 64/64 | **64/64 PASS** | HELD |
| ring_osc | 3/4; TSMC7 8.97–8.98 %, 50.83 ps | **3/4, bit-identical** (2.97/8.98/3.01/2.88 %; TSMC7 50.83 ps) | HELD (predicted: all RO sources at rails) |
| opamp | 1/4 (T5 2.64 PASS; T7 30.7; T12 10.94; T16 flat) | **2/4** — **TSMC12 FLIPPED PASS (10.94 → 5.21 %)**; TSMC5 PASS but moved 2.64 → 9.78 % (pre-arbitrated, selected under buggy frame); TSMC7 changed failure mode 30.7 % → flat collapse (gain 0); TSMC16 still flat | **+1 GATE** |
| switchcap | 1/4 (T5 14.68; T12 8.33; T16 13.13 %) | **1/4** — T5 14.65, T7 3.06 PASS, T12 **10.29 (worse)**, T16 13.14; DN still overshoots Vin (T12 0.4866 → 0.5023) | ~UNCHANGED (frame was NOT the SC owner → R0 dump decides) |
| SRAM butterfly | 4/4 | **4/4** (SNM errs wobbled, gate = positive lobes) | HELD |
| SRAM force_ic | 0/8; attractor q≈0.87 / qb≈0.19–0.23 (24–30 % VDD) | **0/8** but attractor moved to q≈0.785–0.837 / **qb≈0.104–0.117 (13–18 % VDD)** — misses the 0.1·VDD band by ~25 mV on T12/16 | IMPROVED (halved); residual = P2+P3 co-owners as planned |

**Headline: 9/16 → 10/16** (TSMC12 opamp closed by the frame fix alone).

## Decision-table consequences (plan §2)

- "P0 flips opamp/SC cells outright" → TSMC12 opamp closed; **P4's opamp
  census shrinks to TSMC7 + TSMC16 (both flat-collapse mode, not near-miss)**.
- SC unchanged → frame was not the binding SC owner; ownership now rests on
  R0 (droop artifact + dump) and P2 (hold-freeze) / P5–P7 (charge model).
- force_ic 0/8 but attractor halved → P0+P2+P3 joint path confirmed viable;
  P2's reverse-conductance NR-instability cure is now the binding obstacle.
- TSMC5 opamp moved to 9.78 % (within gate) — re-frozen baseline (S8) must
  carry this so campaign vetoes use the corrected frame, not stale 2.64 %.
- TSMC7 opamp's failure mode changed (30.7 % → collapse): the corrected frame
  exposes the D5 derivative-fragility more, strengthening P4's case.

## Environment notes

- GPU 0 was saturated by an external job; first battery launch auto-selected
  CUDA and OOM'd. Battery re-run with `CUDA_VISIBLE_DEVICES=""` (the
  documented CPU harness env). `_get_nn_device()` has no env override —
  consider one if GPU contention recurs.
- Multi-GPU user ruling (2026-06-10) recorded in plan §3: campaign arms
  serialize, seeds fan out one-per-GPU (GPUs 1/2/3 free at audit time).

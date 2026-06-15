# V6.4.7 S11 (P3) — subthreshold-id arm — verdict KILL (force_ic), KEEP loss as infra

**Date:** 2026-06-15 · The ship-required SRAM `force_ic` arm. **Kill gate
(weak-inversion ratio ≥10× with VTC ≤5% → keep; else rewind, escalate to
S17/P9): NOT MET — force_ic stays 0/8 and the cell moves *away* from the rails.**
Headline unchanged at **14/16**; `force_ic` stays **0/8** (the production gatekeeper).

## What was built (default-off, recoverable infra — kept, not reverted)

- `SubthresholdIdLoss` (`bsimar/losses/bni_mae.py`, `--subthresh`): an
  **asinh-s2 (s2≈1e-9) sub-µA VALUE term** (Huber, sign-aware, masked
  `1e-12<|id_true|<1e-6`) that re-scales the subthreshold roll-off the global
  `s_id≈2.6e-5` crushes to ~0.01 % of normalized range (a 1 nA error is
  `asinh(1e-9/2.6e-5)≈4e-5`, ~zero base-loss mass — even though the regen-v2
  data HAS the rows, ~15 % of each cell below 1 µA). Plus a **sign-agnostic OFF
  ceiling hinge** `relu(asinh(|id_pred|/s2) − asinh(k·NFIN·1nA/s2))` on
  `|id_true|≤1e-10` rows — suppresses the hard-OFF over-prediction WITHOUT ever
  injecting current (not the D4 `Ioff_rail` floor). Wired into trainer + CLI;
  default path bit-unchanged (verified). Unit-tested (perfect-pred→0).
- `scripts/v6_4_7_s11_subvt_probe.py` — checkpoint-independent NN-vs-OSDI
  weak-inversion id-fidelity probe (the kill-gate leading indicator).
- `scripts/v6_4_7_s11_sram_gate.py` — combined arbiter: isolates a candidate
  (`{tech}_dn_medium_{dev}` production-name copy → Rule-19 local scope), runs
  `force_ic_probe` (N/8) + the subvt probe.
- `scripts/v6_4_7_s11_train.sh` / `_score.sh` — multi-GPU drivers.

## λ calibration (the gotcha)

The base val-MAE is **~0.001** (not the ~0.005 first assumed) and the raw
asinh-s2 term is **O(1)/row**, so λ controls a strong pull:

| λ | TSMC7 nmos val | vs control (0.00116) | note |
|---|---|---|---|
| 0.05 | 0.073 @ ep27 (killed) | ~12× WORSE | swamps the base fit |
| 0.15 | 0.20 @ ep14 (killed) | ~30× | far too aggressive |
| **0.008** | 0.00302 | ~2.6× | degrades base fit |
| **0.002** | **0.00160** | ~1.4× | operating point — inverter holds |

λ=0.002 chosen. 4-seed TSMC7 + 3-tech spread trained at λ=0.002, ceiling_w=1.

## Result 1 — the term WORKS on its target (weak-inversion fidelity)

Probe = median NN/OSDI |id| ratio on a fixed checkpoint-independent
subthreshold Id-Vgs grid (TSMC7 s42, matched A/B vs control-v2 s42):

| band | control-v2 | subthresh λ=0.002 | Δ |
|---|---|---|---|
| NMOS weak_lo [1e-9,1e-7] | ratio 1.84, \|log10\| 0.356 | ratio 1.14, \|log10\| 0.102 | **3.5× better** |
| NMOS weak_hi [1e-7,1e-6] | 0.97 | 0.91 | ~ |
| PMOS weak_lo | 0.90, \|log10\| 0.445 | 1.13, \|log10\| 0.090 | **5× better** |

So the subthreshold roll-off is now accurate to ~1.1–1.4× (was 0.9–1.8×). The
kill-gate metric improved, but **<10×** (3.5–5×).

## Result 2 — but it does NOT close force_ic, and moves the WRONG way

`force_ic` released-cell landing (0.1·VDD rail band; PASS needs both nodes within):

| tech / seed | control-v2 (state1 q/qb) | subthresh λ=0.002 | railed |
|---|---|---|---|
| TSMC7 s42 | inboard 0.749/0.121 (qb 46 mV out) | **symmetric 0.390/0.390** | 0/2 |
| TSMC7 s17 | symmetric 0.39/0.39 | symmetric 0.390/0.390 | 0/2 |
| TSMC7 s7 | — | symmetric 0.390/0.390 | 0/2 |
| TSMC7 s31 | — | inboard 0.750/0.122 (= control) | 0/2 |
| TSMC5 s42 | — | symmetric 0.328/0.328 | 0/2 |
| TSMC12 s42 | — | symmetric 0.415/0.415 | 0/2 |
| TSMC16 s42 | — | symmetric 0.411/0.411 | 0/2 |

**0/14 probes railed.** 6/7 (tech,seed) cells collapse to the **symmetric
metastable point q=qb=VDD/2** — strictly worse than control's near-railed
inboard attractor (TSMC7 s42 control had q=0.749 already AT the rail; only
qb=0.121 needed to drop 46 mV). Making the id surface more accurate/symmetric
**removed the asymmetry that kept the baseline partially railed.** The one
inboard-landing seed (s31) is identical to control (qb=0.122) — no improvement.

## Result 3 — protected gates hold (TSMC7 s42 scorer, vs control-v2 s42)

| gate | control-v2 | subthresh | verdict |
|---|---|---|---|
| inv_vtc_nrmse | 2.61 % | **2.96 %** | HOLD (≤5 %) |
| inv_tran_post | 1.21 % | 1.16 % | HOLD |
| ring_osc_err | 10.86 % | **7.88 %** | improved, still FAIL (>5 %; S12 corridor owns RO at 2.9 %) |
| opamp_gain_err | 99.99 % (flat) | 99.99 % (flat) | both collapsed = **v2-data retrain lottery, NOT P3** |
| switchcap | 1.76 % PASS | 1.64 % PASS | HOLD |

The subthreshold term is **gate-neutral-to-positive** on everything except its
own target. The opamp collapse is the documented v2-data fresh-retrain lottery
(control-v2 collapses identically) — not caused by P3.

## Verdict — KILL (force_ic) → escalate to S17/P9; KEEP the loss as infra

`force_ic` is **gain / NR-fixed-point owned**, not subthreshold-value owned —
the same value-surface-vs-fixed-point split as the opamp gain (S10) and RO
period (P0-C/P0-I). The cross-coupled SRAM pair needs enough trip gain to make
the symmetric metastable point *repelling*; a more accurate (more symmetric)
subthreshold id surface **reduces** that bistability, collapsing the released
cell to the symmetric point on 6/7 cells across all 4 techs. No subthreshold
VALUE variant (λ, ceiling_w, ceiling_k) addresses trip gain, so further P3
tuning is not pursued.

- **Model: NO checkpoint promoted.** The 14 `v6_4_7_s11sub_w0002_*` checkpoints
  are inert (don't match the parser resolver). Shipped model unchanged.
- **Code: KEEP `SubthresholdIdLoss` as default-off recoverable infra** (same
  disposition as `SobolevIdLoss` after S10) — it is a real, gate-neutral
  subthreshold-fidelity improvement (NMOS weak-band 1.84→1.14) that composes
  with any arm and could matter for off-state-leakage gates (e.g. the TSMC16 SC
  hold leak); not reverted, just not the force_ic lever.
- **force_ic remains 0/8, ship-required, OPEN → S17 = P9** (physics-anchored
  compose-at-inference: frozen MLP owns strong inversion + closed-form
  weak-inversion exponential owns OFF), behind its go/no-go fit gate, is the
  designated fallback now that S2 (P0 frame) + S7 (P2 reverse clamp) + S11 (P3
  subthreshold) have all failed to close it.

**Next per the serial chain: S13 = P8a (teacher-forced id supervision) — but its
RO target is already met by S12; the live ship-required gap is force_ic → S17/P9.**

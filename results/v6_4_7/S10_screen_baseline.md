# V6.4.7 S10 (P4) — fine-tune λ-screen baseline (control-v2 s17, tsmc7)

Warm-start source = control-v2 **s17** (the healthy tsmc7 seed; s42 is
opamp-collapsed). A/B reference for the screen. Scored 2026-06-14.

| metric | control-v2 s17 |
|---|---|
| opamp_gain_err | 10.46% (DN gain 180.5 vs NG 163.4 — gain too HIGH) |
| opamp_flat_flag | 0 (healthy) |
| ring_osc_period_err | 8.66% (best ctl-v2 seed; FAIL gate, recover-target) |
| inv_vtc_nrmse | 3.45% |
| inv_tran_post_nrmse | 1.18% |
| sc_charge_err / sc_pass | 1.60% / PASS |
| deriv gm_fwd (worst) | 137.3% |
| deriv **gds_fwd** (worst=PMOS) | **55.8%** (nmos 6.66%) |
| deriv gmb_fwd (worst) | 34.3% |
| offstate_id_excess_max | 2.91e-4 A |
| FD selfcheck | gm/gds/gmb <0.5% (chain correct) |

**Direction of the lever:** DN opamp gain (180.5) > NG (163.4). gain ≈ gm/gds.
The autograd gds is UNDER-predicted (PMOS gds_fwd 55.8% vs head ~1%), so
raising autograd gds toward OSDI LOWERS gain toward NG — the right direction
to pass the ≤10% gate. The Sobolev term targets exactly this.

**Screen kill gate (plan S10):** best config must cut TSMC7 opamp gain err
< ~15% with inverter held. (Note: control-v2 s17 is already 10.46% < 15%, so
the real bar is ≤10% pass + deriv fwd_inrail strictly < control-v2 + RO not
regressed beyond 8.66%.)

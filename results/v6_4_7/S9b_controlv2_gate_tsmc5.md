# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json

Scored 4 (tech,seed) pairs. Pass tol: RO≤5.0%, opamp≤10.0%, SC harness, inv VTC/tran≤5.0/5.0%.

| tech | cell | baseline | ctl-v2 best | best-seed | metric | verdict |
|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | pass | fail | s31 | 5.80 | REGRESSION |
| tsmc5 | opamp | pass | pass | s7 | 0.79 | OK |
| tsmc5 | switchcap | FAIL | fail | s7 | 11.52 | OK |
| tsmc5 | inverter | pass | pass | s7 | 0.93 | OK |

**Scorer headline (ro+opamp+sc, 12 cells): 1/3**

**Protected-gate regressions: 1** — tsmc5 ring_osc

> sram_butterfly (4) + force_ic checked separately via verify_complex_sram_snm.py on the selected mix.


## Raw per-(tech,seed) RESULT vectors

```json
{
 "tsmc5_s17": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_ctlv2_s17_tsmc5_nmos",
  "pmos": "v6_4_7_ctlv2_s17_tsmc5_pmos",
  "inv_vtc_nrmse": 1.23289638782982,
  "inv_vtc_maxerr_mv": 46.774340442538254,
  "inv_vtc_r2": 0.9993762456839361,
  "inv_vtc_dvtrip_mv": -0.4037221431166249,
  "inv_tran_post_nrmse": 1.2069744099888418,
  "inv_tran_post_maxerr_mv": 95.8238513812848,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 7.84637487875737,
  "ro_nrmse": 65.32663927568542,
  "ro_r2": -1.332934289392203,
  "ro_partial": false,
  "ro_dn_period_ps": 78.97515730804942,
  "sram_rail_snap_resid": 1.0034494086087316,
  "sram_q": 0.08709151658397968,
  "sram_qb": 0.6522421155956755,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.000734098749191836,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99954107869799,
  "opamp_vout_center": 6.525997074753193e-05,
  "sc_charge_err_pct": 11.693520472202025,
  "sc_droop_abs_mv": 0.12289296878692513,
  "sc_droop_pct_of_allow": 18.906610582603864,
  "sc_pass": 0
 },
 "tsmc5_s31": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_ctlv2_s31_tsmc5_nmos",
  "pmos": "v6_4_7_ctlv2_s31_tsmc5_pmos",
  "inv_vtc_nrmse": 1.5663659633737461,
  "inv_vtc_maxerr_mv": 98.55033326346307,
  "inv_vtc_r2": 0.9989931914885932,
  "inv_vtc_dvtrip_mv": -0.03571932782209242,
  "inv_tran_post_nrmse": 0.8620661428400702,
  "inv_tran_post_maxerr_mv": 30.765163902666647,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 5.796306483354654,
  "ro_nrmse": 74.43815760872255,
  "ro_r2": -2.0290964669837273,
  "ro_partial": false,
  "ro_dn_period_ps": 77.47390634619553,
  "sram_rail_snap_resid": 0.5043007360802267,
  "sram_q": 0.3273883246656126,
  "sram_qb": 0.3277954784521474,
  "opamp_flat_flag": 1,
  "opamp_gain": 3.258394575593541e-08,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99999997963016,
  "opamp_vout_center": -1.5615887922239986e-07,
  "sc_charge_err_pct": 11.940553525738247,
  "sc_droop_abs_mv": 0.00031319778431848633,
  "sc_droop_pct_of_allow": 0.048184274510536355,
  "sc_pass": 0
 },
 "tsmc5_s42": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_ctlv2_s42_tsmc5_nmos",
  "pmos": "v6_4_7_ctlv2_s42_tsmc5_pmos",
  "inv_vtc_nrmse": 1.101719253445569,
  "inv_vtc_maxerr_mv": 43.89064749624536,
  "inv_vtc_r2": 0.9995019163548303,
  "inv_vtc_dvtrip_mv": 0.2565594108534519,
  "inv_tran_post_nrmse": 1.1890047726810988,
  "inv_tran_post_maxerr_mv": 96.54699979068876,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 5.886868672934025,
  "ro_nrmse": 74.27824371303828,
  "ro_r2": -2.016095760767827,
  "ro_partial": false,
  "ro_dn_period_ps": 77.54022441368953,
  "sram_rail_snap_resid": 0.13410480499006702,
  "sram_q": 0.6497142473454978,
  "sram_qb": 0.08716812324354356,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.00023706846364379003,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99985179682146,
  "opamp_vout_center": 6.514225319292597e-05,
  "sc_charge_err_pct": 11.835808292459799,
  "sc_droop_abs_mv": 0.09973588319900362,
  "sc_droop_pct_of_allow": 15.343982030615939,
  "sc_pass": 0
 },
 "tsmc5_s7": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_ctlv2_s7_tsmc5_nmos",
  "pmos": "v6_4_7_ctlv2_s7_tsmc5_pmos",
  "inv_vtc_nrmse": 0.9331481280064401,
  "inv_vtc_maxerr_mv": 30.467822102287162,
  "inv_vtc_r2": 0.9996426764979994,
  "inv_vtc_dvtrip_mv": -0.12802707171710725,
  "inv_tran_post_nrmse": 1.218325230884503,
  "inv_tran_post_maxerr_mv": 95.92655908199032,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 7.211411449924382,
  "ro_nrmse": 68.88642073829833,
  "ro_r2": -1.5941143322368059,
  "ro_partial": false,
  "ro_dn_period_ps": 78.51017796374305,
  "sram_rail_snap_resid": 0.4970551293265224,
  "sram_q": 0.33430691887345765,
  "sram_qb": 0.32308583406223956,
  "opamp_flat_flag": 0,
  "opamp_gain": 161.218159316163,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 0.785415669892706,
  "opamp_vout_center": 6.49645820451868e-05,
  "sc_charge_err_pct": 11.520495924139345,
  "sc_droop_abs_mv": 0.00032375708730603137,
  "sc_droop_pct_of_allow": 0.04980878266246636,
  "sc_pass": 0
 }
}
```
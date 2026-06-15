# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json

Scored 4 (tech,seed) pairs. Pass tol: RO≤5.0%, opamp≤10.0%, SC harness, inv VTC/tran≤5.0/5.0%.

| tech | cell | baseline | ctl-v2 best | best-seed | metric | verdict |
|---|---|---|---|---|---|---|
| tsmc16 | ring_osc | pass | pass | s17 | 3.99 | OK |
| tsmc16 | opamp | FAIL | pass | s31 | 5.06 | NEW-PASS |
| tsmc16 | switchcap | FAIL | pass | s17 | 2.01 | NEW-PASS |
| tsmc16 | inverter | pass | pass | s31 | 1.33 | OK |

**Scorer headline (ro+opamp+sc, 12 cells): 3/3**

**Protected-gate regressions: 0** (none)

> sram_butterfly (4) + force_ic checked separately via verify_complex_sram_snm.py on the selected mix.


## Raw per-(tech,seed) RESULT vectors

```json
{
 "tsmc16_s17": {
  "tech": "TSMC16",
  "nmos": "v6_4_7_s12cor_w3_s17_tsmc16_nmos",
  "pmos": "v6_4_7_s12cor_w3_s17_tsmc16_pmos",
  "inv_vtc_nrmse": 1.9949175857362507,
  "inv_vtc_maxerr_mv": 145.54025054849328,
  "inv_vtc_r2": 0.9983551128018795,
  "inv_vtc_dvtrip_mv": -0.11705197547406954,
  "inv_tran_post_nrmse": 0.7452501575836554,
  "inv_tran_post_maxerr_mv": 34.298220599062624,
  "ro_ng_period_ps": 90.07615293130895,
  "ring_osc_period_err": 3.989145956698957,
  "ro_nrmse": 64.9778548662926,
  "ro_r2": -1.4679304152182358,
  "ro_partial": false,
  "ro_dn_period_ps": 93.66942214391824,
  "sram_rail_snap_resid": 0.14718800996661083,
  "sram_q": 0.8003049875513493,
  "sram_qb": 0.11775040797328867,
  "opamp_flat_flag": 0,
  "opamp_gain": 292.62243003973845,
  "opamp_ng_gain": 187.6616582499998,
  "opamp_gain_err": 55.930855971608025,
  "opamp_vout_center": -0.016483893743359693,
  "sc_charge_err_pct": 2.0064777705106964,
  "sc_droop_abs_mv": 0.0056558634785042194,
  "sc_droop_pct_of_allow": 0.7069829348130274,
  "sc_pass": 1
 },
 "tsmc16_s31": {
  "tech": "TSMC16",
  "nmos": "v6_4_7_s12cor_w3_s31_tsmc16_nmos",
  "pmos": "v6_4_7_s12cor_w3_s31_tsmc16_pmos",
  "inv_vtc_nrmse": 1.3282213166238617,
  "inv_vtc_maxerr_mv": 57.39281744328273,
  "inv_vtc_r2": 0.9992708328498529,
  "inv_vtc_dvtrip_mv": -0.02360419845498507,
  "inv_tran_post_nrmse": 0.7460199437690795,
  "inv_tran_post_maxerr_mv": 34.518441997604455,
  "ro_ng_period_ps": 90.07615293130895,
  "ring_osc_period_err": 4.033068208937846,
  "ro_nrmse": 65.3221082140877,
  "ro_r2": -1.4941499291245268,
  "ro_partial": false,
  "ro_dn_period_ps": 93.7089856190158,
  "sram_rail_snap_resid": 1.0001718859127906,
  "sram_q": 0.11709397608096132,
  "sram_qb": 0.8001375087302326,
  "opamp_flat_flag": 0,
  "opamp_gain": 197.15873608566534,
  "opamp_ng_gain": 187.6616582499998,
  "opamp_gain_err": 5.06074491946229,
  "opamp_vout_center": 0.00010236124850455965,
  "sc_charge_err_pct": 2.014555350424642,
  "sc_droop_abs_mv": 0.5199243888804883,
  "sc_droop_pct_of_allow": 64.99054861006104,
  "sc_pass": 1
 },
 "tsmc16_s42": {
  "tech": "TSMC16",
  "nmos": "v6_4_7_s12cor_w3_s42_tsmc16_nmos",
  "pmos": "v6_4_7_s12cor_w3_s42_tsmc16_pmos",
  "inv_vtc_nrmse": 1.6411004752556912,
  "inv_vtc_maxerr_mv": 118.20849536126022,
  "inv_vtc_r2": 0.9988868426825397,
  "inv_vtc_dvtrip_mv": 0.08900723755866924,
  "inv_tran_post_nrmse": 0.7464822313702612,
  "inv_tran_post_maxerr_mv": 34.47587110994421,
  "ro_ng_period_ps": 90.07615293130895,
  "ring_osc_period_err": 4.000219211703115,
  "ro_nrmse": 65.06529746249437,
  "ro_r2": -1.4745772166211242,
  "ro_partial": false,
  "ro_dn_period_ps": 93.67939650603026,
  "sram_rail_snap_resid": 0.14631317000594382,
  "sram_q": 0.799883310952915,
  "sram_qb": 0.11705053600475507,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.0008546489003041571,
  "opamp_ng_gain": 187.6616582499998,
  "opamp_gain_err": 99.99954457990606,
  "opamp_vout_center": 9.860283686659321e-05,
  "sc_charge_err_pct": 2.0076854073296584,
  "sc_droop_abs_mv": 0.2114841646396748,
  "sc_droop_pct_of_allow": 26.43552057995935,
  "sc_pass": 1
 },
 "tsmc16_s7": {
  "tech": "TSMC16",
  "nmos": "v6_4_7_s12cor_w3_s7_tsmc16_nmos",
  "pmos": "v6_4_7_s12cor_w3_s7_tsmc16_pmos",
  "inv_vtc_nrmse": 4.8394282605750165,
  "inv_vtc_maxerr_mv": 277.5375192595223,
  "inv_vtc_r2": 0.990320023218553,
  "inv_vtc_dvtrip_mv": -0.0490261101087075,
  "inv_tran_post_nrmse": 0.7443432652531514,
  "inv_tran_post_maxerr_mv": 34.49673429121602,
  "ro_ng_period_ps": 90.07615293130895,
  "ring_osc_period_err": 3.996106525708464,
  "ro_nrmse": 65.0285003863605,
  "ro_r2": -1.4717790596462184,
  "ro_partial": false,
  "ro_dn_period_ps": 93.67569195670413,
  "sram_rail_snap_resid": 0.5130467304085423,
  "sram_q": 0.4104371069103275,
  "sram_qb": 0.41043738432683385,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.0010639324899000928,
  "opamp_ng_gain": 187.6616582499998,
  "opamp_gain_err": 99.99943305814314,
  "opamp_vout_center": 9.886504982263103e-05,
  "sc_charge_err_pct": 2.017179281345087,
  "sc_droop_abs_mv": 1.1086366264361214,
  "sc_droop_pct_of_allow": 138.57957830451517,
  "sc_pass": 0
 }
}
```
# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json

Scored 4 (tech,seed) pairs. Pass tol: RO≤5.0%, opamp≤10.0%, SC harness, inv VTC/tran≤5.0/5.0%.

| tech | cell | baseline | ctl-v2 best | best-seed | metric | verdict |
|---|---|---|---|---|---|---|
| tsmc7 | ring_osc | FAIL | pass | s7 | 2.87 | NEW-PASS |
| tsmc7 | opamp | FAIL | fail | s7 | 96.63 | OK |
| tsmc7 | switchcap | pass | pass | s7 | 1.01 | OK |
| tsmc7 | inverter | pass | pass | s31 | 2.00 | OK |

**Scorer headline (ro+opamp+sc, 12 cells): 2/3**

**Protected-gate regressions: 0** (none)

> sram_butterfly (4) + force_ic checked separately via verify_complex_sram_snm.py on the selected mix.


## Raw per-(tech,seed) RESULT vectors

```json
{
 "tsmc7_s17": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_s12cor_w3_s17_tsmc7_nmos",
  "pmos": "v6_4_7_s12cor_w3_s17_tsmc7_pmos",
  "inv_vtc_nrmse": 3.8850238819003615,
  "inv_vtc_maxerr_mv": 282.61545294765955,
  "inv_vtc_r2": 0.9935931619080565,
  "inv_vtc_dvtrip_mv": 0.025585208139333737,
  "inv_tran_post_nrmse": 1.06823023093512,
  "inv_tran_post_maxerr_mv": 89.26844852081295,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 2.9065369750684984,
  "ro_nrmse": 75.6472827225439,
  "ro_r2": -2.198635839147181,
  "ro_partial": false,
  "ro_dn_period_ps": 47.997086946491216,
  "sram_rail_snap_resid": 0.5178100985064203,
  "sram_q": 0.3883576009889159,
  "sram_qb": 0.3883575738798153,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.007545805423979968,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 99.99538170223894,
  "opamp_vout_center": 0.00015314077400621908,
  "sc_charge_err_pct": 1.0164559997477043,
  "sc_droop_abs_mv": 0.0011510270277459433,
  "sc_droop_pct_of_allow": 0.15347027036612576,
  "sc_pass": 1
 },
 "tsmc7_s31": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_s12cor_w3_s31_tsmc7_nmos",
  "pmos": "v6_4_7_s12cor_w3_s31_tsmc7_pmos",
  "inv_vtc_nrmse": 1.9980324837340036,
  "inv_vtc_maxerr_mv": 137.47160196447538,
  "inv_vtc_r2": 0.9983054224338411,
  "inv_vtc_dvtrip_mv": 0.11141980721762756,
  "inv_tran_post_nrmse": 1.0722132332640182,
  "inv_tran_post_maxerr_mv": 89.66031196183921,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 2.917678360931247,
  "ro_nrmse": 75.68294555962684,
  "ro_r2": -2.201652452926975,
  "ro_partial": false,
  "ro_dn_period_ps": 48.002283448887255,
  "sram_rail_snap_resid": 0.5178042238935079,
  "sram_q": 0.38835347043035806,
  "sram_qb": 0.3883531679201309,
  "opamp_flat_flag": 0,
  "opamp_gain": 350.91202779527,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 114.77047727006622,
  "opamp_vout_center": 0.00016160547838325168,
  "sc_charge_err_pct": 1.021856788499472,
  "sc_droop_abs_mv": 0.31384292590352914,
  "sc_droop_pct_of_allow": 41.845723453803885,
  "sc_pass": 1
 },
 "tsmc7_s42": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_s12cor_w3_s42_tsmc7_nmos",
  "pmos": "v6_4_7_s12cor_w3_s42_tsmc7_pmos",
  "inv_vtc_nrmse": 3.0949638462797107,
  "inv_vtc_maxerr_mv": 153.55278026827756,
  "inv_vtc_r2": 0.9959339986009084,
  "inv_vtc_dvtrip_mv": 0.07657644829078825,
  "inv_tran_post_nrmse": 1.0784853152922285,
  "inv_tran_post_maxerr_mv": 88.92066601362392,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 2.887806035440947,
  "ro_nrmse": 75.59376248921147,
  "ro_r2": -2.194111389055415,
  "ro_partial": false,
  "ro_dn_period_ps": 47.98835056720648,
  "sram_rail_snap_resid": 0.16240683632046338,
  "sram_q": 0.7501642114890761,
  "sram_qb": 0.12180512724034753,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.012997527591524505,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 99.99204505692866,
  "opamp_vout_center": 0.00018534723438968446,
  "sc_charge_err_pct": 1.0282721918259372,
  "sc_droop_abs_mv": 0.09270949505546522,
  "sc_droop_pct_of_allow": 12.361266007395361,
  "sc_pass": 1
 },
 "tsmc7_s7": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_s12cor_w3_s7_tsmc7_nmos",
  "pmos": "v6_4_7_s12cor_w3_s7_tsmc7_pmos",
  "inv_vtc_nrmse": 3.1067701433026818,
  "inv_vtc_maxerr_mv": 200.5619620912621,
  "inv_vtc_r2": 0.995902918444011,
  "inv_vtc_dvtrip_mv": 0.1744683138811265,
  "inv_tran_post_nrmse": 1.0785069403943524,
  "inv_tran_post_maxerr_mv": 89.00771677851333,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 2.8733553726803964,
  "ro_nrmse": 75.55272206721152,
  "ro_r2": -2.1906441163397057,
  "ro_partial": false,
  "ro_dn_period_ps": 47.9816105705324,
  "sram_rail_snap_resid": 0.5179037005211486,
  "sram_q": 0.3884279447658284,
  "sram_qb": 0.3884277753908615,
  "opamp_flat_flag": 1,
  "opamp_gain": 5.498514318979817,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 96.63471625072776,
  "opamp_vout_center": 0.0001530384081678643,
  "sc_charge_err_pct": 1.0136331157263978,
  "sc_droop_abs_mv": 0.023000810787376924,
  "sc_droop_pct_of_allow": 3.0667747716502562,
  "sc_pass": 1
 }
}
```
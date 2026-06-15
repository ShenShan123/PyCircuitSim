# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json

Scored 4 (tech,seed) pairs. Pass tol: RO≤5.0%, opamp≤10.0%, SC harness, inv VTC/tran≤5.0/5.0%.

| tech | cell | baseline | ctl-v2 best | best-seed | metric | verdict |
|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | pass | pass | s42 | 4.57 | OK |
| tsmc5 | opamp | pass | fail | s7 | 100.00 | REGRESSION |
| tsmc5 | switchcap | FAIL | fail | s42 | 12.16 | OK |
| tsmc5 | inverter | pass | pass | s17 | 1.11 | OK |

**Scorer headline (ro+opamp+sc, 12 cells): 1/3**

**Protected-gate regressions: 1** — tsmc5 opamp

> sram_butterfly (4) + force_ic checked separately via verify_complex_sram_snm.py on the selected mix.


## Raw per-(tech,seed) RESULT vectors

```json
{
 "tsmc5_s17": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_s12cor_w3_s17_tsmc5_nmos",
  "pmos": "v6_4_7_s12cor_w3_s17_tsmc5_pmos",
  "inv_vtc_nrmse": 1.1107037408975162,
  "inv_vtc_maxerr_mv": 31.655139625986205,
  "inv_vtc_r2": 0.9994937595161467,
  "inv_vtc_dvtrip_mv": 0.10999904853242759,
  "inv_tran_post_nrmse": 0.7629072517289489,
  "inv_tran_post_maxerr_mv": 28.964102397202762,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 4.614130549372258,
  "ro_nrmse": 74.97056196837546,
  "ro_r2": -2.0725814615709983,
  "ro_partial": false,
  "ro_dn_period_ps": 76.60820705442973,
  "sram_rail_snap_resid": 0.496790411912549,
  "sram_q": 0.32708623225684313,
  "sram_qb": 0.32262208853130314,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.0006570940796411552,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99958921811147,
  "opamp_vout_center": 6.875011958665624e-05,
  "sc_charge_err_pct": 12.205143118263177,
  "sc_droop_abs_mv": 0.7638864199105977,
  "sc_droop_pct_of_allow": 117.52098767855348,
  "sc_pass": 0
 },
 "tsmc5_s31": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_s12cor_w3_s31_tsmc5_nmos",
  "pmos": "v6_4_7_s12cor_w3_s31_tsmc5_pmos",
  "inv_vtc_nrmse": 1.45316026735146,
  "inv_vtc_maxerr_mv": 55.92523290168838,
  "inv_vtc_r2": 0.9991334623581695,
  "inv_vtc_dvtrip_mv": -0.10830394074773375,
  "inv_tran_post_nrmse": 0.7558132384712106,
  "inv_tran_post_maxerr_mv": 28.98707134560802,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 4.601039650669777,
  "ro_nrmse": 74.94539327673105,
  "ro_r2": -2.0705187886376524,
  "ro_partial": false,
  "ro_dn_period_ps": 76.59862067950063,
  "sram_rail_snap_resid": 0.5047695360171228,
  "sram_q": 0.32810026622948774,
  "sram_qb": 0.32810019841112986,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.00024242857994456052,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99984844594861,
  "opamp_vout_center": 6.531242028202006e-05,
  "sc_charge_err_pct": 12.228938846731962,
  "sc_droop_abs_mv": 0.5422208908235637,
  "sc_droop_pct_of_allow": 83.41859858824057,
  "sc_pass": 0
 },
 "tsmc5_s42": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_s12cor_w3_s42_tsmc5_nmos",
  "pmos": "v6_4_7_s12cor_w3_s42_tsmc5_pmos",
  "inv_vtc_nrmse": 1.3245966440733268,
  "inv_vtc_maxerr_mv": 41.255242606719534,
  "inv_vtc_r2": 0.9992800079448102,
  "inv_vtc_dvtrip_mv": -0.31993766634152987,
  "inv_tran_post_nrmse": 0.7562199524178465,
  "inv_tran_post_maxerr_mv": 28.771075784372545,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 4.573183361811205,
  "ro_nrmse": 74.88262473044757,
  "ro_r2": -2.0653776776643813,
  "ro_partial": false,
  "ro_dn_period_ps": 76.57822171108748,
  "sram_rail_snap_resid": 0.5046796726770199,
  "sram_q": 0.32804167901741915,
  "sram_qb": 0.32804178724006294,
  "opamp_flat_flag": 1,
  "opamp_gain": 1.3979180904884928e-06,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99999912609252,
  "opamp_vout_center": 6.494824523997517e-05,
  "sc_charge_err_pct": 12.162752138292038,
  "sc_droop_abs_mv": 0.450151452770553,
  "sc_droop_pct_of_allow": 69.25406965700815,
  "sc_pass": 0
 },
 "tsmc5_s7": {
  "tech": "TSMC5",
  "nmos": "v6_4_7_s12cor_w3_s7_tsmc5_nmos",
  "pmos": "v6_4_7_s12cor_w3_s7_tsmc5_pmos",
  "inv_vtc_nrmse": 4.682032217955041,
  "inv_vtc_maxerr_mv": 323.83986245630757,
  "inv_vtc_r2": 0.9910044194024079,
  "inv_vtc_dvtrip_mv": -0.3936249486833354,
  "inv_tran_post_nrmse": 1.1472477486496993,
  "inv_tran_post_maxerr_mv": 94.188033426446,
  "ro_ng_period_ps": 73.22931104252189,
  "ring_osc_period_err": 4.634756945560431,
  "ro_nrmse": 75.00810337762128,
  "ro_r2": -2.0756594141833156,
  "ro_partial": false,
  "ro_dn_period_ps": 76.62331162225122,
  "sram_rail_snap_resid": 0.5065461734367822,
  "sram_q": 0.32663432964342287,
  "sram_qb": 0.32925501273390845,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.004939433082698588,
  "opamp_ng_gain": 159.96179431774985,
  "opamp_gain_err": 99.99691211698158,
  "opamp_vout_center": 7.37180848134288e-05,
  "sc_charge_err_pct": 12.202199214751005,
  "sc_droop_abs_mv": 0.2703794692977546,
  "sc_droop_pct_of_allow": 41.59684143042377,
  "sc_pass": 0
 }
}
```
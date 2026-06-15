# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json

Scored 4 (tech,seed) pairs. Pass tol: RO≤5.0%, opamp≤10.0%, SC harness, inv VTC/tran≤5.0/5.0%.

| tech | cell | baseline | ctl-v2 best | best-seed | metric | verdict |
|---|---|---|---|---|---|---|
| tsmc12 | ring_osc | pass | pass | s17 | 3.82 | OK |
| tsmc12 | opamp | pass | fail | s17 | 99.99 | REGRESSION |
| tsmc12 | switchcap | pass | pass | s17 | 2.54 | OK |
| tsmc12 | inverter | pass | pass | s31 | 1.89 | OK |

**Scorer headline (ro+opamp+sc, 12 cells): 2/3**

**Protected-gate regressions: 1** — tsmc12 opamp

> sram_butterfly (4) + force_ic checked separately via verify_complex_sram_snm.py on the selected mix.


## Raw per-(tech,seed) RESULT vectors

```json
{
 "tsmc12_s17": {
  "tech": "TSMC12",
  "nmos": "v6_4_7_s12cor_w3_s17_tsmc12_nmos",
  "pmos": "v6_4_7_s12cor_w3_s17_tsmc12_pmos",
  "inv_vtc_nrmse": 2.627516332451128,
  "inv_vtc_maxerr_mv": 188.96215491265798,
  "inv_vtc_r2": 0.9971485668398992,
  "inv_vtc_dvtrip_mv": -0.06127021840252045,
  "inv_tran_post_nrmse": 0.7946026891319831,
  "inv_tran_post_maxerr_mv": 39.24584787375671,
  "ro_ng_period_ps": 81.3982712612304,
  "ring_osc_period_err": 3.8163324493039004,
  "ro_nrmse": 67.15961518721947,
  "ro_r2": -1.5994194857073238,
  "ro_partial": false,
  "ro_dn_period_ps": 84.50469990054515,
  "sram_rail_snap_resid": 0.5181455735843493,
  "sram_q": 0.41451659867073865,
  "sram_qb": 0.41451645886747945,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.027263098524062775,
  "opamp_ng_gain": 188.43925089999982,
  "opamp_gain_err": 99.98553215511427,
  "opamp_vout_center": 0.00011333543911498672,
  "sc_charge_err_pct": 2.537047459194211,
  "sc_droop_abs_mv": 0.5094232358394124,
  "sc_droop_pct_of_allow": 63.67790447992655,
  "sc_pass": 1
 },
 "tsmc12_s31": {
  "tech": "TSMC12",
  "nmos": "v6_4_7_s12cor_w3_s31_tsmc12_nmos",
  "pmos": "v6_4_7_s12cor_w3_s31_tsmc12_pmos",
  "inv_vtc_nrmse": 1.8863121552685127,
  "inv_vtc_maxerr_mv": 132.17420910826115,
  "inv_vtc_r2": 0.9985303988810814,
  "inv_vtc_dvtrip_mv": -0.0540519597037048,
  "inv_tran_post_nrmse": 0.7950366766919453,
  "inv_tran_post_maxerr_mv": 39.53171064327224,
  "ro_ng_period_ps": 81.3982712612304,
  "ring_osc_period_err": 3.849545823344236,
  "ro_nrmse": 67.40835754061624,
  "ro_r2": -1.618710338227006,
  "ro_partial": false,
  "ro_dn_period_ps": 84.53173501284151,
  "sram_rail_snap_resid": 0.9930104154865929,
  "sram_q": 0.1300643415299753,
  "sram_qb": 0.7944083323892743,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.001631186581273255,
  "opamp_ng_gain": 188.43925089999982,
  "opamp_gain_err": 99.99913437005641,
  "opamp_vout_center": 0.00010806848151764354,
  "sc_charge_err_pct": 2.541825327150943,
  "sc_droop_abs_mv": 0.3513172170049317,
  "sc_droop_pct_of_allow": 43.91465212561646,
  "sc_pass": 1
 },
 "tsmc12_s42": {
  "tech": "TSMC12",
  "nmos": "v6_4_7_s12cor_w3_s42_tsmc12_nmos",
  "pmos": "v6_4_7_s12cor_w3_s42_tsmc12_pmos",
  "inv_vtc_nrmse": 5.979490519809941,
  "inv_vtc_maxerr_mv": 385.7011312195643,
  "inv_vtc_r2": 0.9852327131613684,
  "inv_vtc_dvtrip_mv": 0.06122425955334965,
  "inv_tran_post_nrmse": 0.7964983558678881,
  "inv_tran_post_maxerr_mv": 39.36492552160523,
  "ro_ng_period_ps": 81.3982712612304,
  "ring_osc_period_err": 3.829336646682486,
  "ro_nrmse": 67.25608997702072,
  "ro_r2": -1.6068929820760793,
  "ro_partial": false,
  "ro_dn_period_ps": 84.5152850924027,
  "sram_rail_snap_resid": 0.14532306293648253,
  "sram_q": 0.7996696158709762,
  "sram_qb": 0.11625845034918603,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.0003230593480645187,
  "opamp_ng_gain": 188.43925089999982,
  "opamp_gain_err": 99.99982856047957,
  "opamp_vout_center": 0.00010830761352479128,
  "sc_charge_err_pct": 2.543059559369815,
  "sc_droop_abs_mv": 0.14549830474153147,
  "sc_droop_pct_of_allow": 18.187288092691432,
  "sc_pass": 1
 },
 "tsmc12_s7": {
  "tech": "TSMC12",
  "nmos": "v6_4_7_s12cor_w3_s7_tsmc12_nmos",
  "pmos": "v6_4_7_s12cor_w3_s7_tsmc12_pmos",
  "inv_vtc_nrmse": 1.9952313440998082,
  "inv_vtc_maxerr_mv": 141.3153260776373,
  "inv_vtc_r2": 0.9983557840198438,
  "inv_vtc_dvtrip_mv": -0.011521301732819733,
  "inv_tran_post_nrmse": 0.7981883613171528,
  "inv_tran_post_maxerr_mv": 39.22783854439135,
  "ro_ng_period_ps": 81.3982712612304,
  "ring_osc_period_err": 3.835736312264859,
  "ro_nrmse": 67.30414338371416,
  "ro_r2": -1.6106194800173026,
  "ro_partial": false,
  "ro_dn_period_ps": 84.52049430955327,
  "sram_rail_snap_resid": 0.5179244691498356,
  "sram_q": 0.4145245956422341,
  "sram_qb": 0.4143395753198685,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.0006178856292711893,
  "opamp_ng_gain": 188.43925089999982,
  "opamp_gain_err": 99.99967210354195,
  "opamp_vout_center": 0.00010822541108922046,
  "sc_charge_err_pct": 2.540694495715555,
  "sc_droop_abs_mv": 2.681326037626275,
  "sc_droop_pct_of_allow": 335.16575470328434,
  "sc_pass": 0
 }
}
```
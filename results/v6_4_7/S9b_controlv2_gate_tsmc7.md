# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json

Scored 4 (tech,seed) pairs. Pass tol: RO≤5.0%, opamp≤10.0%, SC harness, inv VTC/tran≤5.0/5.0%.

| tech | cell | baseline | ctl-v2 best | best-seed | metric | verdict |
|---|---|---|---|---|---|---|
| tsmc7 | ring_osc | FAIL | fail | s17 | 8.66 | OK |
| tsmc7 | opamp | FAIL | fail | s17 | 10.46 | OK |
| tsmc7 | switchcap | pass | pass | s17 | 1.60 | OK |
| tsmc7 | inverter | pass | pass | s31 | 1.39 | OK |

**Scorer headline (ro+opamp+sc, 12 cells): 1/3**

**Protected-gate regressions: 0** (none)

> sram_butterfly (4) + force_ic checked separately via verify_complex_sram_snm.py on the selected mix.


## Raw per-(tech,seed) RESULT vectors

```json
{
 "tsmc7_s17": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_ctlv2_s17_tsmc7_nmos",
  "pmos": "v6_4_7_ctlv2_s17_tsmc7_pmos",
  "inv_vtc_nrmse": 3.449635848872838,
  "inv_vtc_maxerr_mv": 237.67785447010178,
  "inv_vtc_r2": 0.9949487034209017,
  "inv_vtc_dvtrip_mv": -0.09535770764429463,
  "inv_tran_post_nrmse": 1.180219853326736,
  "inv_tran_post_maxerr_mv": 88.25140193746833,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 8.658576471770568,
  "ro_nrmse": 62.29848747857682,
  "ro_r2": -1.1693675479978909,
  "ro_partial": false,
  "ro_dn_period_ps": 50.67992078735546,
  "sram_rail_snap_resid": 0.5204710696477854,
  "sram_q": 0.39035348458166164,
  "sram_qb": 0.3903533022358391,
  "opamp_flat_flag": 0,
  "opamp_gain": 180.48059828904826,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 10.460460634710888,
  "opamp_vout_center": 0.0001267352021055115,
  "sc_charge_err_pct": 1.601194944510172,
  "sc_droop_abs_mv": 0.000289787297769184,
  "sc_droop_pct_of_allow": 0.03863830636922454,
  "sc_pass": 1
 },
 "tsmc7_s31": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_ctlv2_s31_tsmc7_nmos",
  "pmos": "v6_4_7_ctlv2_s31_tsmc7_pmos",
  "inv_vtc_nrmse": 1.389703008554353,
  "inv_vtc_maxerr_mv": 84.34124234834972,
  "inv_vtc_r2": 0.9991802142614897,
  "inv_vtc_dvtrip_mv": -0.24488326757859946,
  "inv_tran_post_nrmse": 1.1164726083072993,
  "inv_tran_post_maxerr_mv": 87.89828501169961,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 8.81393473417133,
  "ro_nrmse": 62.04108903168984,
  "ro_r2": -1.1514782441860159,
  "ro_partial": false,
  "ro_dn_period_ps": 50.75238211243252,
  "sram_rail_snap_resid": 0.5202898120036962,
  "sram_q": 0.39021741089357537,
  "sram_qb": 0.3902173590027721,
  "opamp_flat_flag": 0,
  "opamp_gain": 186.83255397171067,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 14.348080452517836,
  "opamp_vout_center": 0.00015708027695212305,
  "sc_charge_err_pct": 1.645026167294598,
  "sc_droop_abs_mv": 0.23851821074977186,
  "sc_droop_pct_of_allow": 31.802428099969582,
  "sc_pass": 1
 },
 "tsmc7_s42": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_ctlv2_s42_tsmc7_nmos",
  "pmos": "v6_4_7_ctlv2_s42_tsmc7_pmos",
  "inv_vtc_nrmse": 2.4246184311034216,
  "inv_vtc_maxerr_mv": 130.5184369894349,
  "inv_vtc_r2": 0.9975045827795513,
  "inv_vtc_dvtrip_mv": -0.46299353771672713,
  "inv_tran_post_nrmse": 1.2055303327199776,
  "inv_tran_post_maxerr_mv": 87.827017028766,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 10.860100784805109,
  "ro_nrmse": 54.39195030145216,
  "ro_r2": -0.6536644738718969,
  "ro_partial": false,
  "ro_dn_period_ps": 51.70674334861931,
  "sram_rail_snap_resid": 0.1618442384077683,
  "sram_q": 0.749231789116133,
  "sram_qb": 0.12138317880582622,
  "opamp_flat_flag": 1,
  "opamp_gain": 0.01734725438202442,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 99.98938287146679,
  "opamp_vout_center": 0.00015219772678842525,
  "sc_charge_err_pct": 1.7551421748519018,
  "sc_droop_abs_mv": 0.14793909576732434,
  "sc_droop_pct_of_allow": 19.725212768976576,
  "sc_pass": 1
 },
 "tsmc7_s7": {
  "tech": "TSMC7",
  "nmos": "v6_4_7_ctlv2_s7_tsmc7_nmos",
  "pmos": "v6_4_7_ctlv2_s7_tsmc7_pmos",
  "inv_vtc_nrmse": 1.7330927000072085,
  "inv_vtc_maxerr_mv": 81.24107552979254,
  "inv_vtc_r2": 0.9987250300489187,
  "inv_vtc_dvtrip_mv": 0.0641150647665567,
  "inv_tran_post_nrmse": 0.9550135639056228,
  "inv_tran_post_maxerr_mv": 53.16915202840439,
  "ro_ng_period_ps": 46.64143635318292,
  "ring_osc_period_err": 10.071597954054477,
  "ro_nrmse": 57.450320929995755,
  "ro_r2": -0.8448584591832831,
  "ro_partial": false,
  "ro_dn_period_ps": 51.33897430267171,
  "sram_rail_snap_resid": 0.5210007792045562,
  "sram_q": 0.3907500800543917,
  "sram_qb": 0.39075058440341714,
  "opamp_flat_flag": 0,
  "opamp_gain": 362.3969317636894,
  "opamp_ng_gain": 163.38932252499984,
  "opamp_gain_err": 121.79964159422954,
  "opamp_vout_center": 0.00012673994606311796,
  "sc_charge_err_pct": 1.802078357687309,
  "sc_droop_abs_mv": 0.09538576910989027,
  "sc_droop_pct_of_allow": 12.718102547985369,
  "sc_pass": 1
 }
}
```
* Two-stage Miller opamp -- DirectNet LEVEL=73
* Benchmark 3b (docs/plans/2026-05-15-directnet-complex-circuits.md)
*
* Stage 1: NMOS diff pair (Mn1/Mn2) with PMOS current-mirror load
*          (Mp3/Mp4), NMOS tail current source (Mn5).
* Stage 2: common-source PMOS gain stage (Mp6) with NMOS current-source
*          load (Mn7).
* Cc: Miller compensation. CL: output load.
* Bias rails Vbn/Vbp set tail + 2nd-stage operating currents.
*
* The verify harness rewrites TECH=/VT=/VDD per technology and runs:
*   .op            -> operating point
*   .dc Vinp ...   -> DC transfer (open-loop gain, trip point)
* L_n=16n / L_p=20n / NFIN=2 match the tsmc*_dn_medium checkpoints.

Vdd vdd 0 0.80
Vbn vbn 0 0.36
Vbp vbp 0 0.44
Vinn inn 0 0.44
Vinp inp 0 0.44

* --- stage 1: diff pair + PMOS mirror load + NMOS tail ---
Mn1 n1   inp vtail 0   nmos_nn L=16n NFIN=2
Mn2 vo1i inn vtail 0   nmos_nn L=16n NFIN=2
Mp3 n1   n1  vdd   vdd pmos_nn L=20n NFIN=2
Mp4 vo1i n1  vdd   vdd pmos_nn L=20n NFIN=2
Mn5 vtail vbn 0    0   nmos_nn L=16n NFIN=2

* --- stage 2: CS PMOS gain stage + NMOS load ---
Mp6 vout vo1i vdd vdd pmos_nn L=20n NFIN=2
Mn7 vout vbn  0   0   nmos_nn L=16n NFIN=2

* --- compensation + load ---
Cc vo1i vout 20f
CL vout 0 50f

.model nmos_nn NMOS (LEVEL=73 TECH=tsmc12 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc12 VT=svt)

.dc Vinp 0.29 0.59 0.002

.end

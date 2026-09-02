* Two-stage Miller opamp — NN candidate
*
* Stage 1: NMOS diff pair (Mn1/Mn2) with PMOS current-mirror load
*          (Mp3/Mp4), NMOS tail current source (Mn5).
* Stage 2: common-source PMOS gain stage (Mp6) with NMOS current-source
*          load (Mn7).
* Cc: Miller compensation. CL: output load.
* Bias rails Vbn/Vbp set tail + 2nd-stage operating currents.
*
* The verify harness renders all declared tokens per technology and runs:
*   .op            -> operating point
*   .dc Vinp ...   -> DC transfer (open-loop gain, trip point)
* Default L_n=16n / L_p=20n / NFIN=2 is supplied by the catalog renderer.

Vdd vdd 0 <VDD>
Vbn vbn 0 <VBN>
Vbp vbp 0 <VBP>
Vinn inn 0 <VCM>
Vinp inp 0 <VCM>

Mn1 n1 inp vtail 0 nmos_nn L=<LN> NFIN=<NFN>
Mn2 vo1i inn vtail 0 nmos_nn L=<LN> NFIN=<NFN>
Mp3 n1 n1 vdd vdd pmos_nn L=<LP> NFIN=<NFP>
Mp4 vo1i n1 vdd vdd pmos_nn L=<LP> NFIN=<NFP>
Mn5 vtail vbn 0 0 nmos_nn L=<LN> NFIN=<NFN>
Mp6 vout vo1i vdd vdd pmos_nn L=<LP> NFIN=<NFP>
Mn7 vout vbn 0 0 nmos_nn L=<LN> NFIN=<NFN>
Cc vo1i vout <CC>
CL vout 0 <CL>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>

.end

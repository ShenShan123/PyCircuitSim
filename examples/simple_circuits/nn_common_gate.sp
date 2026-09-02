* Complementary common-gate stages — NN candidate
Vdd vdd 0 <VDD>
Vgn gn 0 <GATE_N>
Vgp gp 0 <GATE_P>
Vsn sn 0 0
Vsp sp 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Rn vdd dn 12k
Mn_cg dn gn sn bn nmos_nn L=<LN> NFIN=<NFN>
Rp dp 0 12k
Mp_cg dp gp sp bp pmos_nn L=<LP> NFIN=<NFP>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

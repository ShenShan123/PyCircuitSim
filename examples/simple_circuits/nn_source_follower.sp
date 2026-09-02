* Complementary source followers with explicit body terminals — NN candidate
Vddn vddn 0 <VDD>
Vddp vddp 0 <VDD>
Vgn gn 0 <HALF_VDD>
Vgp gp 0 <HALF_VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Mn_sf vddn gn sn bn nmos_nn L=<LN> NFIN=<NFN>
Rn sn 0 20k
Mp_sf 0 gp sp bp pmos_nn L=<LP> NFIN=<NFP>
Rp vddp sp 20k
.ic V(sn)=<FOLLOW_N_IC> V(sp)=<FOLLOW_P_IC>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

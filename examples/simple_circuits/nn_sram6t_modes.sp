* Full 6T SRAM hold/read/write modes — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Vwl wl 0 <WL_SPEC>
Vbl bl 0 <BL_SPEC>
Vblb blb 0 <BLB_SPEC>
Mp_l q qb vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_l q qb 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mp_r qb q vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_r qb q 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mn_al bl wl q bn nmos_nn L=<LN> NFIN=<NFN>
Mn_ar blb wl qb bn nmos_nn L=<LN> NFIN=<NFN>
Cq q 0 2f
Cqb qb 0 2f
.ic V(q)=<Q_IC> V(qb)=<QB_IC>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

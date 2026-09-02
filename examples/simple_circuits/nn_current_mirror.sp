* Complementary two-device current mirrors — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Irefn vdd nr <IBIAS>
Mn_ref nr nr 0 bn nmos_nn L=<LN> NFIN=<NFN>
Voutn on 0 0
Mn_out on nr 0 bn nmos_nn L=<LN> NFIN=<NFN>
Irefp pr 0 <IBIAS>
Mp_ref pr pr vdd bp pmos_nn L=<LP> NFIN=<NFP>
Voutp op 0 <VDD>
Mp_out op pr vdd bp pmos_nn L=<LP> NFIN=<NFP>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

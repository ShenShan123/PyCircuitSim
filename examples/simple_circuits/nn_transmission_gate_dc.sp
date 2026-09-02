* Bidirectional CMOS transmission gate — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Vphi phi 0 <VDD>
Vphib phib 0 0
Vinf inf 0 0
Mn_f outf phi inf bn nmos_nn L=<LN> NFIN=<NFN>
Mp_f outf phib inf bp pmos_nn L=<LP> NFIN=<NFP>
Rf outf 0 10k
Vrev outr 0 0
Mn_r inr phi outr bn nmos_nn L=<LN> NFIN=<NFN>
Mp_r inr phib outr bp pmos_nn L=<LP> NFIN=<NFP>
Rr inr 0 10k
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

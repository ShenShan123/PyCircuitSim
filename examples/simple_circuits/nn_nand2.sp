* Two-input static CMOS NAND — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Va a 0 <VA_SPEC>
Vb b 0 <VB_SPEC>
Mp_a out a vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mp_b out b vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_a out a nint bn nmos_nn L=<LN> NFIN=<NFN>
Mn_b nint b 0 bn nmos_nn L=<LN> NFIN=<NFN>
Cload out 0 5f
.ic V(out)=<VDD> V(nint)=0
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

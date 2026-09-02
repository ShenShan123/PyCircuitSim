* Two-input static CMOS NOR — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Va a 0 <VA_SPEC>
Vb b 0 <VB_SPEC>
Mp_a pint a vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mp_b out b pint bp pmos_nn L=<LP> NFIN=<NFP>
Mn_a out a 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mn_b out b 0 bn nmos_nn L=<LN> NFIN=<NFN>
Cload out 0 5f
.ic V(out)=<VDD> V(pint)=<VDD>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

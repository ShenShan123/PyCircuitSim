* Transmission-gate hold and feedthrough — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Vin vin 0 <HALF_VDD>
Vphi phi 0 PULSE <VDD> 0 1n 20p 20p 2n 4n
Vphib phib 0 PULSE 0 <VDD> 1n 20p 20p 2n 4n
Mn_tg hold phi vin bn nmos_nn L=<LN> NFIN=<NFN>
Mp_tg hold phib vin bp pmos_nn L=<LP> NFIN=<NFP>
Chold hold 0 100f
.ic V(hold)=<HALF_VDD>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

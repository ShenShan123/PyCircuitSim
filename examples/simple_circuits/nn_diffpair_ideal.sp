* Resistor-loaded NMOS differential pair with ideal tail — NN candidate
Vdd vdd 0 <VDD>
Vinp inp 0 DC=<VCM> AC=<AC_INP> 0
Vinn inn 0 DC=<VCM> AC=<AC_INN> 0
Vbn bn 0 <BODY_N>
Rn vdd outn 18k
Rp vdd outp 18k
Mn_l outn inp tail bn nmos_nn L=<LN> NFIN=<NFN>
Mn_r outp inn tail bn nmos_nn L=<LN> NFIN=<NFN>
Itail tail 0 <TAIL_CURRENT>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.temp <TEMP>
<ANALYSIS>
.end

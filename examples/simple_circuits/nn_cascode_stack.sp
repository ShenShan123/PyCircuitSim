* Complementary cascode compliance and AC gain branches — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Vbiasn biasn 0 <BIAS_N>
Vcasn casn 0 <CAS_N>
Voutn dn 0 0
Mn_lo nx biasn 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mn_hi dn casn nx bn nmos_nn L=<LN> NFIN=<NFN>
Vbiasp biasp 0 <BIAS_P>
Vcasp casp 0 <CAS_P>
Voutp dp 0 <VDD>
Mp_lo px biasp vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mp_hi dp casp px bp pmos_nn L=<LP> NFIN=<NFP>
Vinac inac 0 DC=<BIAS_N> AC=1 0
Rna vdd nac 20k
Mn_alo nax inac 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mn_ahi nac casn nax bn nmos_nn L=<LN> NFIN=<NFN>
Vipac ipac 0 DC=<BIAS_P> AC=1 0
Rpa pac 0 20k
Mp_alo pax ipac vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mp_ahi pac casp pax bp pmos_nn L=<LP> NFIN=<NFP>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

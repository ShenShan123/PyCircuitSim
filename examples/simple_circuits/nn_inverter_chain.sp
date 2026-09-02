* Open fanout-of-four inverter chain — NN candidate
Vdd vdd 0 <VDD>
Vbn bn 0 <BODY_N>
Vbp bp 0 <BODY_P>
Vin in 0 PULSE 0 <VDD> 0.5n 20p 20p 1n 2n
Mp_drv n1 in vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_drv n1 in 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mp_l0 o0 n1 vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_l0 o0 n1 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mp_l1 o1 n1 vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_l1 o1 n1 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mp_l2 o2 n1 vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_l2 o2 n1 0 bn nmos_nn L=<LN> NFIN=<NFN>
Mp_l3 o3 n1 vdd bp pmos_nn L=<LP> NFIN=<NFP>
Mn_l3 o3 n1 0 bn nmos_nn L=<LN> NFIN=<NFN>
C0 o0 0 2f
C1 o1 0 2f
C2 o2 0 2f
C3 o3 0 2f
.ic V(n1)=<VDD> V(o0)=0 V(o1)=0 V(o2)=0 V(o3)=0
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>
.end

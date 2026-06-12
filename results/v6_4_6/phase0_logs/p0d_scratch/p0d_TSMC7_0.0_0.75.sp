* 6T SRAM cell — DirectNet (TSMC7)
Vdd vdd 0 0.75
Vwl wl 0 0.75
Vbl bl 0 0.75
Vblb blb 0 0.75
.ic V(q)=0.0 V(qb)=0.75
Mpl qb q vdd vdd pmos_nn L=20n NFIN=2
Mnl qb q 0   0   nmos_nn L=16n NFIN=2
Mpr q qb vdd vdd pmos_nn L=20n NFIN=2
Mnr q qb 0   0   nmos_nn L=16n NFIN=2
Mal bl  wl q  0 nmos_nn L=16n NFIN=2
Mar blb wl qb 0 nmos_nn L=16n NFIN=2
.model nmos_nn NMOS (LEVEL=73 TECH=tsmc7 VT=ulvt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc7 VT=ulvt)
.op
.end

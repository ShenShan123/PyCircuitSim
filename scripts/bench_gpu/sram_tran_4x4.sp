* 4x4 6T SRAM array — DirectNet (16 cells, 96 devices)
Vdd vdd 0 0.75
Vwl0 wl0 0 PULSE 0 0.75 2n 0.1n 0.1n 5n 20n
Vwl1 wl1 0 0.0
Vwl2 wl2 0 0.0
Vwl3 wl3 0 0.0
Vbl0 bl0 0 PULSE 0.75 0 1n 0.1n 0.1n 8n 20n
Vblb0 blb0 0 0.75
Vbl1 bl1 0 0.75
Vblb1 blb1 0 0.75
Vbl2 bl2 0 0.75
Vblb2 blb2 0 0.75
Vbl3 bl3 0 0.75
Vblb3 blb3 0 0.75
X0_0l q0_0 qb0_0 vdd sraminv NF=4
X0_0r qb0_0 q0_0 vdd sraminv NF=4
Mal0_0 bl0  wl0 q0_0  0 nmos_nn L=16n NFIN=4
Mar0_0 blb0 wl0 qb0_0 0 nmos_nn L=16n NFIN=4
X0_1l q0_1 qb0_1 vdd sraminv NF=4
X0_1r qb0_1 q0_1 vdd sraminv NF=4
Mal0_1 bl1  wl0 q0_1  0 nmos_nn L=16n NFIN=4
Mar0_1 blb1 wl0 qb0_1 0 nmos_nn L=16n NFIN=4
X0_2l q0_2 qb0_2 vdd sraminv NF=4
X0_2r qb0_2 q0_2 vdd sraminv NF=4
Mal0_2 bl2  wl0 q0_2  0 nmos_nn L=16n NFIN=4
Mar0_2 blb2 wl0 qb0_2 0 nmos_nn L=16n NFIN=4
X0_3l q0_3 qb0_3 vdd sraminv NF=4
X0_3r qb0_3 q0_3 vdd sraminv NF=4
Mal0_3 bl3  wl0 q0_3  0 nmos_nn L=16n NFIN=4
Mar0_3 blb3 wl0 qb0_3 0 nmos_nn L=16n NFIN=4
X1_0l q1_0 qb1_0 vdd sraminv NF=4
X1_0r qb1_0 q1_0 vdd sraminv NF=4
Mal1_0 bl0  wl1 q1_0  0 nmos_nn L=16n NFIN=4
Mar1_0 blb0 wl1 qb1_0 0 nmos_nn L=16n NFIN=4
X1_1l q1_1 qb1_1 vdd sraminv NF=4
X1_1r qb1_1 q1_1 vdd sraminv NF=4
Mal1_1 bl1  wl1 q1_1  0 nmos_nn L=16n NFIN=4
Mar1_1 blb1 wl1 qb1_1 0 nmos_nn L=16n NFIN=4
X1_2l q1_2 qb1_2 vdd sraminv NF=4
X1_2r qb1_2 q1_2 vdd sraminv NF=4
Mal1_2 bl2  wl1 q1_2  0 nmos_nn L=16n NFIN=4
Mar1_2 blb2 wl1 qb1_2 0 nmos_nn L=16n NFIN=4
X1_3l q1_3 qb1_3 vdd sraminv NF=4
X1_3r qb1_3 q1_3 vdd sraminv NF=4
Mal1_3 bl3  wl1 q1_3  0 nmos_nn L=16n NFIN=4
Mar1_3 blb3 wl1 qb1_3 0 nmos_nn L=16n NFIN=4
X2_0l q2_0 qb2_0 vdd sraminv NF=4
X2_0r qb2_0 q2_0 vdd sraminv NF=4
Mal2_0 bl0  wl2 q2_0  0 nmos_nn L=16n NFIN=4
Mar2_0 blb0 wl2 qb2_0 0 nmos_nn L=16n NFIN=4
X2_1l q2_1 qb2_1 vdd sraminv NF=4
X2_1r qb2_1 q2_1 vdd sraminv NF=4
Mal2_1 bl1  wl2 q2_1  0 nmos_nn L=16n NFIN=4
Mar2_1 blb1 wl2 qb2_1 0 nmos_nn L=16n NFIN=4
X2_2l q2_2 qb2_2 vdd sraminv NF=4
X2_2r qb2_2 q2_2 vdd sraminv NF=4
Mal2_2 bl2  wl2 q2_2  0 nmos_nn L=16n NFIN=4
Mar2_2 blb2 wl2 qb2_2 0 nmos_nn L=16n NFIN=4
X2_3l q2_3 qb2_3 vdd sraminv NF=4
X2_3r qb2_3 q2_3 vdd sraminv NF=4
Mal2_3 bl3  wl2 q2_3  0 nmos_nn L=16n NFIN=4
Mar2_3 blb3 wl2 qb2_3 0 nmos_nn L=16n NFIN=4
X3_0l q3_0 qb3_0 vdd sraminv NF=4
X3_0r qb3_0 q3_0 vdd sraminv NF=4
Mal3_0 bl0  wl3 q3_0  0 nmos_nn L=16n NFIN=4
Mar3_0 blb0 wl3 qb3_0 0 nmos_nn L=16n NFIN=4
X3_1l q3_1 qb3_1 vdd sraminv NF=4
X3_1r qb3_1 q3_1 vdd sraminv NF=4
Mal3_1 bl1  wl3 q3_1  0 nmos_nn L=16n NFIN=4
Mar3_1 blb1 wl3 qb3_1 0 nmos_nn L=16n NFIN=4
X3_2l q3_2 qb3_2 vdd sraminv NF=4
X3_2r qb3_2 q3_2 vdd sraminv NF=4
Mal3_2 bl2  wl3 q3_2  0 nmos_nn L=16n NFIN=4
Mar3_2 blb2 wl3 qb3_2 0 nmos_nn L=16n NFIN=4
X3_3l q3_3 qb3_3 vdd sraminv NF=4
X3_3r qb3_3 q3_3 vdd sraminv NF=4
Mal3_3 bl3  wl3 q3_3  0 nmos_nn L=16n NFIN=4
Mar3_3 blb3 wl3 qb3_3 0 nmos_nn L=16n NFIN=4
.ic V(q0_0)=0.75 V(qb0_0)=0.0 V(q0_1)=0.75 V(qb0_1)=0.0 V(q0_2)=0.75 V(qb0_2)=0.0 V(q0_3)=0.75 V(qb0_3)=0.0 V(q1_0)=0.75 V(qb1_0)=0.0 V(q1_1)=0.75 V(qb1_1)=0.0 V(q1_2)=0.75 V(qb1_2)=0.0 V(q1_3)=0.75 V(qb1_3)=0.0 V(q2_0)=0.75 V(qb2_0)=0.0 V(q2_1)=0.75 V(qb2_1)=0.0 V(q2_2)=0.75 V(qb2_2)=0.0 V(q2_3)=0.75 V(qb2_3)=0.0 V(q3_0)=0.75 V(qb3_0)=0.0 V(q3_1)=0.75 V(qb3_1)=0.0 V(q3_2)=0.75 V(qb3_2)=0.0 V(q3_3)=0.75 V(qb3_3)=0.0
.subckt sraminv i o vdd NF=1
Mpl o i vdd vdd pmos_nn L=16n NFIN=NF
Mnl o i 0   0   nmos_nn L=16n NFIN=NF
.ends
.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc5 VT=lvt)
.tran 0.05n 10n
.end

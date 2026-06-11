* 5-stage CMOS ring oscillator -- BSIM-CMG LEVEL=72 control (V6.4.7 S6)
* TSMC7 VT=ulvt VDD=0.75 -- same circuit as verify_complex_ring_osc.py

Vdd vdd 0 0.75
.ic V(n1)=0.0 V(n2)=0.75 V(n3)=0.0 V(n4)=0.75 V(n5)=0.0

Mp1 n1 n5 vdd vdd pch_ulvt_mac L=20n NFIN=2 TFIN=6.0n
Mn1 n1 n5 0   0   nch_ulvt_mac L=16n NFIN=2 TFIN=6.0n
Cl1 n1 0 0.5f

Mp2 n2 n1 vdd vdd pch_ulvt_mac L=20n NFIN=2 TFIN=6.0n
Mn2 n2 n1 0   0   nch_ulvt_mac L=16n NFIN=2 TFIN=6.0n
Cl2 n2 0 0.5f

Mp3 n3 n2 vdd vdd pch_ulvt_mac L=20n NFIN=2 TFIN=6.0n
Mn3 n3 n2 0   0   nch_ulvt_mac L=16n NFIN=2 TFIN=6.0n
Cl3 n3 0 0.5f

Mp4 n4 n3 vdd vdd pch_ulvt_mac L=20n NFIN=2 TFIN=6.0n
Mn4 n4 n3 0   0   nch_ulvt_mac L=16n NFIN=2 TFIN=6.0n
Cl4 n4 0 0.5f

Mp5 n5 n4 vdd vdd pch_ulvt_mac L=20n NFIN=2 TFIN=6.0n
Mn5 n5 n4 0   0   nch_ulvt_mac L=16n NFIN=2 TFIN=6.0n
Cl5 n5 0 0.5f

.model nch_ulvt_mac NMOS (LEVEL=72)
.model pch_ulvt_mac PMOS (LEVEL=72)

.tran 2p 0.6000n

.end
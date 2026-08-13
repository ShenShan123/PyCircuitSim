* DirectNet CMOS Inverter DC Sweep (VTC) -- LEVEL=73, hierarchical (.subckt)
* The LEVEL=73 member of the inverter triplet; compare against
* bsimcmg_inverter_dc.sp (LEVEL=72 ground truth) and bsimar_inverter_dc.sp.
* TECH/VT are REQUIRED -- see the note in ../device/directnet_nmos_dc.sp.

* Power supply (tsmc5 rail)
Vdd vdd 0 0.65

* Input voltage
Vin in 0 0.0

* Inverter instance: ports (in, out, vdd)
Xinv in out vdd inv

.subckt inv i o vdd NF=10
Mp1 o i vdd vdd pmos_nn L=30n NFIN=NF
Mn1 o i 0 0 nmos_nn L=30n NFIN=NF
.ends

* Model definitions (LEVEL=73 DirectNet)
.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=svt)
.model pmos_nn PMOS (LEVEL=73 TECH=tsmc5 VT=svt)

* DC sweep: Vin from 0 to 0.65V
.dc Vin 0 0.65 0.01

.end

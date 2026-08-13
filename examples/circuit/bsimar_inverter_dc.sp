* BSIM-AR CMOS Inverter DC Sweep (VTC) -- LEVEL=74, hierarchical (.subckt)
* The LEVEL=74 member of the inverter triplet; compare against
* bsimcmg_inverter_dc.sp (LEVEL=72 ground truth) and directnet_inverter_dc.sp.
* AR inference is ~30-100x slower on CPU, so expect this sweep to take
* noticeably longer than the DirectNet one at the same step count.
* TECH/VT are REQUIRED -- see the note in ../device/directnet_nmos_dc.sp.

* Power supply (tsmc5 rail)
Vdd vdd 0 0.65

* Input voltage
Vin in 0 0.0

* Inverter instance: ports (in, out, vdd)
Xinv in out vdd inv

.subckt inv i o vdd NF=10
Mp1 o i vdd vdd pmos_ar L=30n NFIN=NF
Mn1 o i 0 0 nmos_ar L=30n NFIN=NF
.ends

* Model definitions (LEVEL=74 BSIM-AR Transformer)
.model nmos_ar NMOS (LEVEL=74 TECH=tsmc5 VT=svt)
.model pmos_ar PMOS (LEVEL=74 TECH=tsmc5 VT=svt)

* DC sweep: Vin from 0 to 0.65V
.dc Vin 0 0.65 0.01

.end

* BSIM-AR Transformer Inverter DC Sweep (VTC)
* Tests LEVEL=74 BSIM-AR model with inverter circuit

* Power supply
Vdd vdd 0 0.7

* Input voltage
Vin in 0 0.0

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 out in vdd vdd pmos_ar L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 out in 0 0 nmos_ar L=30n NFIN=10

* Model definitions (LEVEL=74 BSIM-AR Transformer)
.model nmos_ar NMOS (LEVEL=74 TECH=asap7 VT=rvt)
.model pmos_ar PMOS (LEVEL=74 TECH=asap7 VT=rvt)

* DC sweep: Vin from 0 to 0.7V
.dc Vin 0 0.7 0.01

.end

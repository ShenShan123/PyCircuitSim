* BSIM-AR Transformer Inverter DC Sweep (VTC) — hierarchical (.subckt) version
* Tests LEVEL=74 BSIM-AR model with inverter circuit

* Power supply
Vdd vdd 0 0.7

* Input voltage
Vin in 0 0.0

* Inverter instance: ports (in, out, vdd)
Xinv in out vdd inv

.subckt inv i o vdd NF=10
Mp1 o i vdd vdd pmos_ar L=30n NFIN=NF
Mn1 o i 0 0 nmos_ar L=30n NFIN=NF
.ends

* Model definitions (LEVEL=74 BSIM-AR Transformer)
.model nmos_ar NMOS (LEVEL=74 TECH=asap7 VT=rvt)
.model pmos_ar PMOS (LEVEL=74 TECH=asap7 VT=rvt)

* DC sweep: Vin from 0 to 0.7V
.dc Vin 0 0.7 0.01

.end

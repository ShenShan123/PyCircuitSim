* NN-based CMOS Inverter DC Sweep (VTC) — hierarchical (.subckt) version
* Tests LEVEL=73 NN model with inverter circuit

* Power supply
Vdd vdd 0 0.7

* Input voltage
Vin in 0 0.0

* Inverter instance: ports (in, out, vdd)
Xinv in out vdd inv

.subckt inv i o vdd NF=10
Mp1 o i vdd vdd pmos_nn L=30n NFIN=NF
Mn1 o i 0 0 nmos_nn L=30n NFIN=NF
.ends

* Model definitions (LEVEL=73 NN)
.model nmos_nn NMOS (LEVEL=73)
.model pmos_nn PMOS (LEVEL=73)

* DC sweep: Vin from 0 to 0.7V
.dc Vin 0 0.7 0.01

.end

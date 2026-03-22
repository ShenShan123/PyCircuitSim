* NN-based CMOS Inverter DC Sweep (VTC)
* Tests LEVEL=73 NN model with inverter circuit

* Power supply
Vdd vdd 0 0.7

* Input voltage
Vin in 0 0.0

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 out in vdd vdd pmos_nn L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 out in 0 0 nmos_nn L=30n NFIN=10

* Model definitions (LEVEL=73 NN)
.model nmos_nn NMOS (LEVEL=73)
.model pmos_nn PMOS (LEVEL=73)

* DC sweep: Vin from 0 to 0.7V
.dc Vin 0 0.7 0.01

.end

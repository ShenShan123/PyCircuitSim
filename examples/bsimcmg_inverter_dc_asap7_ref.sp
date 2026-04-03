* BSIM-CMG Inverter DC Sweep Reference (ASAP7 RVT, VDD=0.7V)

Vdd vdd 0 0.7
Vin in 0 0.0

Mp1 out in vdd vdd pmos1 L=7n NFIN=10
Mn1 out in 0 0 nmos1 L=7n NFIN=10

.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

.dc Vin 0 0.7 0.01

.end

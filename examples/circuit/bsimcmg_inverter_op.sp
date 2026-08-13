* BSIM-CMG CMOS Inverter DC Test — hierarchical (.subckt) version
* Test DC operating point with static input

* Power supply
Vdd 1 0 1.0

* Input voltage (DC)
Vin 2 0 0.0

* Inverter instance: ports are (in, out, vdd); ground is global
Xinv 2 3 1 inv

* Inverter cell: PMOS pull-up + NMOS pull-down, NFIN via parameter
.subckt inv i o vdd NF=10
Mp1 o i vdd vdd pmos1 L=30n NFIN=NF
Mn1 o i 0 0 nmos1 L=30n NFIN=NF
.ends

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

.end

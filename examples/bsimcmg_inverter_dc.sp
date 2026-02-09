* BSIM-CMG CMOS Inverter DC Test
* Test DC operating point with static input

* Power supply
Vdd 1 0 1.0

* Input voltage (DC)
Vin 2 0 0.0

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

.end

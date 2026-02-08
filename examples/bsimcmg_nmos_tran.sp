* BSIM-CMG NMOS Transient Test
* Simple circuit to verify transient analysis with BSIM-CMG

* Power supply
Vdd 1 0 1.0

* Input pulse: 0 -> 1V
Vin 2 0 PULSE 0 1.0 1n 0.1n 0.1n 2n 5n

* Load resistor (1kOhm)
Rload 1 3 1000

* NMOS (drain=load, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Model definition
.model nmos1 NMOS (LEVEL=72)

* Transient analysis: 0 to 5ns with 50ps steps
.tran 50p 5n

.end

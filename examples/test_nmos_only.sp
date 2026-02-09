* Simple NMOS test with resistor load
* Verify basic NMOS operation

* Power supply
Vdd 1 0 1.0

* Gate voltage (HIGH - NMOS should be ON)
Vg 2 0 1.0

* Load resistor (1k)
Rload 1 3 1000

* NMOS (drain=load, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Model definition
.model nmos1 NMOS (LEVEL=72)

.end

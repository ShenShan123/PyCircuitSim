* Simple PMOS test with resistor load
* Verify basic PMOS operation

* Power supply
Vdd 1 0 1.0

* Gate voltage (LOW - PMOS should be ON)
Vg 2 0 0.0

* Load resistor to ground (1k)
Rload 3 0 1000

* PMOS (source=Vdd, drain=load, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* Model definition
.model pmos1 PMOS (LEVEL=72)

.end

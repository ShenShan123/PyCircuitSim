* Test PMOS alone to verify current direction

Vdd 1 0 1.0
Vin 2 0 0.0

* PMOS only - should pull output high when gate is low
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* Load resistor
Rload 3 0 10k

.model pmos1 PMOS (LEVEL=72)

.op

.end

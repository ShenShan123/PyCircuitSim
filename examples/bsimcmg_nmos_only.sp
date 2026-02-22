* Test NMOS alone to verify current direction

Vdd 1 0 1.0
Vin 2 0 1.0

* NMOS only - should pull output low when gate is high
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Load resistor
Rload 3 0 10k

.model nmos1 NMOS (LEVEL=72)

.op

.end

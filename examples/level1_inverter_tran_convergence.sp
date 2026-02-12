* Level-1 CMOS Inverter to test convergence algorithms

* Power supply
Vdd 1 0 1.0

* Input pulse: 0 -> 1V, period=2ns
Vin 2 0 PULSE 0 1.0 0.5n 0.1n 0.1n 0.8n 2n

* PMOS (Level 1)
Mp1 3 2 1 1 pmos L=90n W=180n

* NMOS (Level 1)
Mn1 3 2 0 0 nmos L=90n W=180n

* Load capacitance
Cload 3 0 10e-15

* Transient analysis: 0 to 5ns with 10ps steps
.tran 10p 5n

.end

* BSIM-CMG CMOS Inverter Transient Simulation
* Testing with Vdd=1.5V and slower rise time for better convergence

* Power supply - 1.5V for even better PMOS operation
Vdd 1 0 1.5

* Input pulse: 0 -> 1.5V, period=10ns (slower)
* Rise/fall time = 0.5ns instead of 0.1ns
Vin 2 0 PULSE 0 1.5 1n 0.5n 0.5n 4n 10n

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Load capacitance (10fF)
Cload 3 0 10e-15

* Initial conditions to help DC convergence
.ic V(3)=1.5

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient analysis: 0 to 10ns with 1ps steps (finer resolution)
.tran 1p 10n

.end

* BSIM-CMG CMOS Inverter Transient Simulation
* Testing with Vdd=1.2V and very slow rise time for better convergence

* Power supply - 1.2V for better PMOS operation
Vdd 1 0 1.2

* Input pulse: 0 -> 1.2V, period=100ns (very slow)
* Rise/fall time = 5ns instead of 0.1ns
Vin 2 0 PULSE 0 1.2 10n 5n 5n 40n 100n

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Load capacitance (10fF)
Cload 3 0 10e-15

* Initial conditions to help DC convergence
.ic V(3)=1.2

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient analysis: 0 to 100ns with 100ps steps (larger timestep)
.tran 100p 100n

.end

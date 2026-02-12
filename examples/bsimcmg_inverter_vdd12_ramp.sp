* BSIM-CMG CMOS Inverter Transient Simulation
* Testing with Vdd=1.2V and very slow ramp input

* Power supply - 1.2V for better PMOS operation
Vdd 1 0 1.2

* Input: Very slow ramp from 0 to 1.2V over 5ns
Vin 2 0 0

* Use a PWL source for very slow transition
* Linear ramp: 0V at t=0, 1.2V at t=5ns
Vramp 2 0 PWL(0 0 5n 1.2)

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

* Transient analysis: 0 to 10ns with 10ps steps
.tran 10p 10n

.end

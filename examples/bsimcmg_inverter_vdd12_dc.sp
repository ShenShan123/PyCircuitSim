* BSIM-CMG CMOS Inverter DC Sweep Simulation
* Testing with Vdd=1.2V

* Power supply - 1.2V for better PMOS operation
Vdd 1 0 1.2

* Input DC source for sweep
Vin 2 0 0

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Small load capacitance (for transient tests)
*Cload 3 0 10e-15

* Initial conditions to help DC convergence
.ic V(3)=1.2

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* DC sweep: sweep Vin from 0V to 1.2V
.dc Vin 0 1.2 0.01

.end

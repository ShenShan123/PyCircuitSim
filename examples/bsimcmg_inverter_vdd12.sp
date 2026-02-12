* BSIM-CMG CMOS Inverter Transient Simulation
* Testing with Vdd=1.2V to overcome PMOS threshold voltage issue
* PMOS Vth ~0.7-0.8V, so Vdd=1.2V provides sufficient gate overdrive

* Power supply - 1.2V for better PMOS operation
Vdd 1 0 1.2

* Input pulse: 0 -> 1.2V, period=2ns
Vin 2 0 PULSE 0 1.2 0.5n 0.1n 0.1n 0.8n 2n

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

* Transient analysis: 0 to 5ns with 10ps steps
.tran 10p 5n

.end

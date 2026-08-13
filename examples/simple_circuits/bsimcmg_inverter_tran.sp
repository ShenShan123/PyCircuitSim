* BSIM-CMG CMOS Inverter Transient Simulation — hierarchical (.subckt) version
* Tests LEVEL=72 integration for both NMOS and PMOS

* Power supply
Vdd 1 0 1.0

* Input pulse: 0 -> 1V, period=2ns
Vin 2 0 PULSE 0 1.0 0.5n 0.1n 0.1n 0.8n 2n

* Inverter instance: ports (in, out, vdd); node 3 = output stays top-level
Xinv 2 3 1 inv

* Load capacitance (10fF)
Cload 3 0 10e-15

* Inverter cell. The .ic inside the body is remapped to the connected
* port node (here node 3) by the subckt expansion; VIC shows parameterized
* initial conditions.
.subckt inv i o vdd VIC=1.0
Mp1 o i vdd vdd pmos1 L=30n NFIN=10
Mn1 o i 0 0 nmos1 L=30n NFIN=10
.ic V(o)=VIC
.ends

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient analysis: 0 to 5ns with 10ps steps
.tran 10p 5n

.end

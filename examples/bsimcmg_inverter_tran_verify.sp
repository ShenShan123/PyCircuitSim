* BSIM-CMG Inverter Transient Verification — hierarchical (.subckt) version
* Matches NGSPICE test: Vdd=0.7V, L=30n, NFIN=10, Cload=10fF

* Power supply
Vdd 1 0 0.7

* Input pulse: 0 -> 0.7V
Vin 2 0 PULSE 0.0 0.7 5e-10 1e-10 1e-10 8e-10 2e-09

* Inverter instance: ports (in, out, vdd)
Xinv 2 3 1 inv

* Load capacitance
Cload 3 0 10f

* Initial condition: output starts high (node 3 is the inverter's out port)
.ic V(3)=0.7

.subckt inv i o vdd
Mp1 o i vdd vdd pmos1 L=30n NFIN=10
Mn1 o i 0 0 nmos1 L=30n NFIN=10
.ends

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient: 10ps step, 5ns total
.tran 1e-11 5e-09

.end

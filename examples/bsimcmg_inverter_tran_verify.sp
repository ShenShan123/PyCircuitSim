* BSIM-CMG Inverter Transient Verification
* Matches NGSPICE test: Vdd=0.7V, L=30n, NFIN=10, Cload=10fF

* Power supply
Vdd 1 0 0.7

* Input pulse: 0 -> 0.7V
Vin 2 0 PULSE 0.0 0.7 5e-10 1e-10 1e-10 8e-10 2e-09

* PMOS (drain=out, gate=in, source=Vdd, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L=30n NFIN=10

* Load capacitance
Cload 3 0 10f

* Initial condition: output starts high (PMOS on, NMOS off when Vin=0)
.ic V(3)=0.7

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient: 10ps step, 5ns total
.tran 1e-11 5e-09

.end

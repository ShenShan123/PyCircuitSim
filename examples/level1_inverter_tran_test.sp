* Level-1 CMOS Inverter Transient Simulation

* Power supply
Vdd 1 0 1.2

* Input pulse: 0 -> 1.2V, period=10ns
Vin 2 0 PULSE 0 1.2 1n 0.5n 0.5n 4n 10n

* PMOS (KP=-20e-6 VTO=-0.8)
Mp1 3 2 1 1 pmos1 L=1u W=10u

* NMOS (KP=20e-6 VTO=0.8)
Mn1 3 2 0 0 nmos1 L=1u W=10u

* Load capacitance (10fF)
Cload 3 0 10e-15

* Initial conditions
.ic V(3)=1.2

* Model definitions
.model nmos1 NMOS (LEVEL=1 KP=20e-6 VTO=0.8)
.model pmos1 PMOS (LEVEL=1 KP=-20e-6 VTO=-0.8)

* Transient analysis: 0 to 10ns with 10ps steps
.tran 10p 10n

.end

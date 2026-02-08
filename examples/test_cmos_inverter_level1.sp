* CMOS Inverter Test - Level 1 Models
* Tests NMOS and PMOS working together using direct model keywords

* Power supply
Vdd 1 0 3.3

* Input voltage (swept)
Vin 2 0 1.65

* PMOS transistor (drain=3, gate=2, source=1, bulk=1)
* Using default parameters: VTO=-0.7V, KP=-20uA/V^2
Mp1 3 2 1 1 PMOS L=1u W=20u

* NMOS transistor (drain=3, gate=2, source=0, bulk=0)
* Using default parameters: VTO=0.7V, KP=20uA/V^2
Mn1 3 2 0 0 NMOS L=1u W=10u

* DC sweep: Vin from 0 to 3.3V
.dc Vin 0 3.3 0.1

.end

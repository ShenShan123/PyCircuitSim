* Check NMOS CS DC operating point in NGSPICE

* DC Supply
Vdd 3 0 DC=3.3

* Input signal: DC bias only
Vin 1 0 DC=1.5

* Load resistor
Rl 3 2 3.3k

* NMOS transistor
Mn1 2 1 0 0 NMOS L=1u W=10u

* Model definition for Level 1 NMOS
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=50u LAMBDA=0.02)

* DC operating point
.op

* Output control
.control
run
print v(1) v(2) v(3)
print @mn1[id] @mn1[gm] @mn1[gds]
quit
.endc

.end

* RC Charging Circuit
* Demonstrates transient analysis of a capacitor charging through a resistor
* Time constant tau = R*C = 1k * 1n = 1us
* Capacitor will charge to ~63% of Vdd in 1 tau

V1 1 0 5
R1 1 2 1k
C1 2 0 1n

* Transient Analysis
* Time step: 100ns
* Stop time: 10us (10 time constants)
.tran 100n 10u

.end

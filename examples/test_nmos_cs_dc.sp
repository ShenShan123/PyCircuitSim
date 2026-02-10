* NMOS Common-Source Amplifier DC Operating Point Test
* Expected: V_drain = 2.743V, Id = 168.78uA (from NGSPICE)

* DC Supply
Vdd 3 0 DC=3.3

* Input DC bias
Vin 1 0 DC=1.5

* Load resistor (drain to Vdd)
Rl 3 2 3.3k

* NMOS transistor (drain=2, gate=1, source=0, bulk=0)
Mn1 2 1 0 0 NMOS L=1u W=10u

* Model definition for Level 1 NMOS
* IMPORTANT: LAMBDA=0.02 gives channel-length modulation (gds > 0)
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=50u LAMBDA=0.02)

* DC operating point
.op

.end

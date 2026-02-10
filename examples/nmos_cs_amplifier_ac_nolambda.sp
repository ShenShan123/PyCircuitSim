* NMOS Common-Source Amplifier - AC Analysis (NO LAMBDA)
* Level 1 MOSFET model without channel-length modulation

* DC Supply
Vdd 3 0 DC=3.3

* Input signal: DC bias + AC stimulus
Vin 1 0 DC=1.5 AC=0.01 0

* Load resistor (3.3kΩ)
Rl 3 2 3.3k

* NMOS transistor (drain=2, gate=1, source=0, bulk=0)
Mn1 2 1 0 0 NMOS L=1u W=10u

* Model definition for Level 1 NMOS (NO LAMBDA)
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=50u)

* AC analysis: decade sweep from 1 Hz to 10 MHz
.ac dec 10 1 10e6

.end

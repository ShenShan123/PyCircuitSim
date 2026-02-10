* NMOS Common-Source Amplifier - AC Analysis
* Level 1 MOSFET model
* Simple CS amplifier with resistive load

* DC Supply
Vdd 3 0 DC=3.3

* Input signal: DC bias + AC stimulus
* DC bias at 1.5V to put NMOS in saturation
Vin 1 0 DC=1.5 AC=0.01 0

* Load resistor (3.3kΩ for reasonable gain)
Rl 3 2 3.3k

* NMOS transistor (drain=2, gate=1, source=0, bulk=0)
* W/L = 10u/1u for moderate transconductance
Mn1 2 1 0 0 NMOS L=1u W=10u

* Model definition for Level 1 NMOS
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=50u LAMBDA=0.02)

* AC analysis: decade sweep from 1 Hz to 10 MHz
.ac dec 10 1 10e6

.end

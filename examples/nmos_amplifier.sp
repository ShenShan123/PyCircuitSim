* NMOS Common-Source Amplifier
* Demonstrates MOSFET biasing and DC operating point

Vdd 1 0 5
Vbias 2 0 2
M1 3 2 0 0 NMOS L=1u W=10u
Rd 1 3 1k

* DC Sweep Analysis
* Sweep gate bias voltage to see transistor turn-on
.dc Vbias 0 3 0.1

.end

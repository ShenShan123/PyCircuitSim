* CMOS Inverter
* Classic digital logic gate with MOSFETs
* Demonstrates non-linear DC sweep with transistor switching

Vdd 1 0 3.3
Vin 2 0 0
Mp1 3 2 1 1 PMOS L=1u W=20u
Mn1 3 2 0 0 NMOS L=1u W=10u
Rload 3 0 10000

* DC Sweep Analysis
* Sweep input voltage from 0V to 3.3V
* Output should switch from high to low around Vdd/2
.dc Vin 0 3.3 0.1

.end

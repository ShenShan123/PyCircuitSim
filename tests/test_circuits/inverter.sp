* CMOS Inverter Test Circuit
* This is a simple inverter with NMOS and PMOS transistors

Vdd vdd 0 3.3
Vin vin 0 0
Mp1 vout vin vdd vdd PMOS L=1u W=10u
Mn1 vout vin 0 0 NMOS L=1u W=5u
Cload vout 0 100p

.dc vin 0 3.3 0.1
.end

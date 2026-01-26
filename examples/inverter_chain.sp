* CMOS Inverter Chain (5 stages)
* Tests signal propagation through multiple logic stages

* Power supply
Vdd 1 0 3.3

* Input voltage
Vin 2 0 0

* Stage 1: Inverter
Mp1 1 2 3 1 PMOS L=0.18u W=0.5u
Mn1 0 2 3 0 NMOS L=0.18u W=0.25u

* Stage 2: Inverter
Mp2 1 3 4 1 PMOS L=0.18u W=0.5u
Mn2 0 3 4 0 NMOS L=0.18u W=0.25u

* Stage 3: Inverter
Mp3 1 4 5 1 PMOS L=0.18u W=0.5u
Mn3 0 4 5 0 NMOS L=0.18u W=0.25u

* Stage 4: Inverter
Mp4 1 5 6 1 PMOS L=0.18u W=0.5u
Mn4 0 5 6 0 NMOS L=0.18u W=0.25u

* Stage 5: Inverter
Mp5 1 6 7 1 PMOS L=0.18u W=0.5u
Mn5 0 6 7 0 NMOS L=0.18u W=0.25u

* DC Sweep: Sweep input voltage to see voltage transfer curve
.dc Vin 0 3.3 0.1

.end

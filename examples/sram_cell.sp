* 6T SRAM Cell
* Classic memory cell with cross-coupled inverters and access transistors

* Power supply
Vdd 1 0 3.3

* Storage node inverters (cross-coupled)
* Left inverter: M1 (PMOS), M2 (NMOS)
Mp1 1 2 3 1 PMOS L=0.18u W=0.5u
Mn1 0 2 3 0 NMOS L=0.18u W=0.25u

* Right inverter: M3 (PMOS), M4 (NMOS)
Mp2 1 3 2 1 PMOS L=0.18u W=0.5u
Mn2 0 3 2 0 NMOS L=0.18u W=0.25u

* Access transistors
* Bit lines
Vbl 4 0 1.65
Vnbl 5 0 1.65
* Word line
Vwl 6 0 0

Ma1 3 6 4 0 NMOS L=0.18u W=0.25u
Ma2 2 6 5 0 NMOS L=0.18u W=0.25u

* DC Sweep: Sweep word line voltage to simulate read access
.dc Vwl 0 3.3 0.1

.end

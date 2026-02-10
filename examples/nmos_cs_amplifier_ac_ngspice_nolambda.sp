* NMOS Common-Source Amplifier - AC Analysis for NGSPICE (NO LAMBDA)

* DC Supply
Vdd 3 0 DC=3.3

* Input signal: DC bias + AC stimulus
Vin 1 0 DC=1.5 AC=0.01

* Load resistor
Rl 3 2 3.3k

* NMOS transistor
Mn1 2 1 0 0 NMOS L=1u W=10u

* Model definition for Level 1 NMOS (NO LAMBDA)
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=50u)

* AC analysis
.ac dec 10 1 10e6

* Output control
.control
run
set hcopydevtype=ascii
print frequency vdb(2) vp(2) > nmos_cs_ngspice_nolambda.txt
quit
.endc

.end

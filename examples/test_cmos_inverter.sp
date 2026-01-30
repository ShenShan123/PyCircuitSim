* CMOS Inverter Test - DC Sweep
* Tests NMOS and PMOS working together in an inverter

* Power supply
Vdd 1 0 1.0

* Input voltage (swept)
Vin 2 0 0.5

* PMOS transistor (drain=3, gate=2, source=1, bulk=1)
Mp1 3 2 1 1 PMOS_VTL L=45n W=180n

* NMOS transistor (drain=3, gate=2, source=0, bulk=0)
Mn1 3 2 0 0 NMOS_VTL L=45n W=90n

* NMOS model
.model NMOS_VTL NMOS (LEVEL=54 VTH0=0.4 U0=100 TOX=1.8e-9 VSAT=1.5e5 K1=0.5)

* PMOS model
.model PMOS_VTL PMOS (LEVEL=54 VTH0=-0.4 U0=100 TOX=1.8e-9 VSAT=1.5e5 K1=0.5)

* DC sweep: Vin from 0 to 1V
.dc Vin 0 1 0.02

.end

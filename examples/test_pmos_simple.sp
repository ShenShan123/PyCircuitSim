* PMOS Simple Test Circuit - DC Sweep
* Tests PMOS in a simple load configuration

* Power supply
Vdd 1 0 1.0

* Input voltage (swept)
Vin 2 0 0.5

* PMOS transistor (drain=3, gate=2, source=1, bulk=1)
* Source tied to Vdd, bulk tied to Vdd
Mp1 3 2 1 1 PMOS_VTL L=45n W=180n

* Load resistor (10k)
Rload 3 0 10k

* PMOS model
.model PMOS_VTL PMOS (LEVEL=54 VTH0=-0.4 U0=100 TOX=1.8e-9 VSAT=1.5e5 K1=0.5)

* DC sweep: Vin from 0 to 1V
.dc Vin 0 1 0.02

.end

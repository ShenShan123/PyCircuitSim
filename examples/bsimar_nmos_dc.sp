* BSIM-AR NMOS DC Sweep
* Tests LEVEL=74 single NMOS device

* Supply
Vdd vdd 0 0.7
Vgs gate 0 0.0

* NMOS (drain=vdd, gate=gate, source=GND, bulk=GND)
Mn1 vdd gate 0 0 nmos_ar L=7n NFIN=10

* Model definition (LEVEL=74 BSIM-AR Transformer)
.model nmos_ar NMOS (LEVEL=74 TECH=asap7 VT=rvt)

* DC sweep: Vgs from 0 to 0.7V
.dc Vgs 0 0.7 0.01

.end

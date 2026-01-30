* Test NMOS with freePDK45 parameters for direct comparison
.include freePDK45nm_spice/freePDK45nm_TT.l

* Test circuit: NMOS with fixed biases
Vds drain 0 DC 0.1
Vgs gate 0 DC 0.5
Vbs bulk 0 DC 0.0

* NMOS instance: drain gate source bulk
Mn1 drain gate 0 bulk NMOS_VTL L=45n W=90n

* Analysis
.op

* Control
.end

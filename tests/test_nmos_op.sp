* Test NMOS - absolute path
.include /home/shenshan/NN_SPICE/freePDK45nm_spice/freePDK45nm_TT.l

Vds drain 0 DC 0.1
Vgs gate 0 DC 0.5
Vbs bulk 0 DC 0.0

Mn1 drain gate 0 bulk NMOS_VTL L=45n W=90n

.op

.altmode
alter vds = 0.1
dump all mn1
quit
.run

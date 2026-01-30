* Simple test - single bias point
.include /home/shenshan/NN_SPICE/freePDK45nm_spice/freePDK45nm_TT.l

Vgs 1 0 0.5
Vds 2 0 0.1
Mn1 2 1 0 0 NMOS_VTL L=45n W=90n

.op
.print dc i(Mn1)
.end

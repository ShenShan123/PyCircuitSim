* Single PMOS test
.include ../freePDK45nm_spice/freePDK45nm_TT.l
Mp1 2 1 0 0 PMOS_VTL L=45n W=180n
Vds 2 0 -0.1
Vgs 1 0 -0.5
.op
.print dc v(1) v(2) i(Vds)
.end

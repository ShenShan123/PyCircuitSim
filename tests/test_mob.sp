* Test mobility with different Vgs
.include freePDK45nm_spice/freePDK45nm_TT.l
Vdd 1 0 DC 1
Vgs 2 0 DC 0.5
Vds 3 0 DC 0.1
M1 3 2 0 0 NMOS_VTL L=45n W=90n

.op
.print dc v(2) v(3) i(M1)
.end

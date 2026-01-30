* test_nmos_L45_W90
.include ../freePDK45nm_spice/freePDK45nm_TT.l
Mn1 2 1 0 0 NMOS_VTL L=45n W=90n
Vds 2 0 0.5
Vgs 1 0 0.5
.dc Vds 0 1 0.02
.end

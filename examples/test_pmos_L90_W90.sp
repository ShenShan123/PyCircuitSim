* test_pmos_L90_W90
.include ../freePDK45nm_spice/freePDK45nm_TT.l
Mp1 2 1 0 0 PMOS_VTL L=90n W=90n
Vds 2 0 -0.5
Vgs 1 0 -0.5
.dc Vds 0 -1 -0.02
.end

* CMOS Inverter Test with freePDK45
.include ../freePDK45nm_spice/freePDK45nm_TT.l
Vdd 1 0 1.0
Vin 2 0 0.5
Mp1 3 2 1 1 PMOS_VTL L=45n W=180n
Mn1 3 2 0 0 NMOS_VTL L=45n W=90n
.dc Vin 0 1 0.02
.end

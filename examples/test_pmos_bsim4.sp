* PMOS BSIM4.5.0 Test - DC Sweep
* Test Id vs Vds at various Vgs values

.include freePDK45nm_spice/freePDK45nm_TT.l

* PMOS transistor (drain=2, gate=1, source=3, bulk=3)
Mp1 2 1 3 3 PMOS_VTL L=45n W=180n

* Source voltage (Vdd) - fixed at 1V
Vdd 3 0 1.0

* Drain voltage - sweep from 1V down to 0V (Vds = Vd - Vs goes from 0 to -1V)
Vd 2 3 0.5

* Gate voltage - fixed at 0.5V relative to source (Vgs = -0.5V)
Vg 1 3 0.5

* DC sweep: Drain from 1V to 0V (Vds from 0 to -1V)
.dc Vd 1 0 -0.02

* Print currents and voltages
.print dc V(1) V(2) V(3) i(Vd) i(Vg) i(Vdd)

.end

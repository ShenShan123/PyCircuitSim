* NMOS BSIM4.5.0 Test - DC Sweep
* Test Id vs Vds at various Vgs values

.include freePDK45nm_spice/freePDK45nm_TT.l

* NMOS transistor (drain=2, gate=1, source=0, bulk=0)
Mn1 2 1 0 0 NMOS_VTL L=45n W=90n

* Drain voltage source
Vds 2 0 0.5

* Gate voltage source
Vgs 1 0 0.5

* DC sweep: Vds from 0 to 1V
.dc Vds 0 1 0.02

* Print currents and voltages
.print dc V(1) V(2) i(Vds)

.end

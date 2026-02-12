* Level-1 CMOS Inverter Transient Test - NGSPICE Reference
* This netlist is identical to level1_inverter_tran.sp for validation

* Power supply
Vdd vdd 0 1.0

* Input pulse (0V to 1V, 100ps rise/fall, 1ns pulse width, 2ns period)
Vin in 0 PULSE 0 1.0 0 100p 100p 1n 2n

* NMOS Model (Level 1) - VTO=0.5V, KP=100uA/V^2
.model NMOS_LEVEL1 nmos (VTO=0.5 KP=100e-6)

* PMOS Model (Level 1) - VTO=-0.5V, KP=-40uA/V^2
.model PMOS_LEVEL1 pmos (VTO=-0.5 KP=-40e-6)

* CMOS Inverter (node 3 is output)
* Terminal order: drain gate source bulk
Mn1 out in 0 0 NMOS_LEVEL1 W=1u L=180n
Mp1 out in vdd vdd PMOS_LEVEL1 W=2u L=180n

* Load capacitor (10fF)
Cload out 0 10f

* Initial condition
.ic V(out)=0.0

* Transient analysis (50ps timestep, 5ns duration)
.tran 50p 5n

* Output
.print tran V(in) V(out)
.end

* 5-Transistor Operational Amplifier
* Simple differential amplifier with active load

* Power supply
Vdd 1 0 3.3
Vss 8 0 0

* Input voltages (differential inputs)
Vin_plus 2 0 1.65
Vin_minus 3 0 1.65

* Current source for bias
Ibias 1 4 10u

* Differential pair (NMOS input transistors)
M1 5 2 4 8 NMOS L=1u W=10u
M2 6 3 4 8 NMOS L=1u W=10u

* Active load (PMOS current mirror)
M3 5 5 1 1 PMOS L=1u W=20u
M4 6 5 1 1 PMOS L=1u W=20u

* Output is at node 6 (drain of M2 and M4)

* DC Sweep: Sweep differential input voltage
* Note: Need to sweep Vin_plus while keeping Vin_minus fixed
* This requires sweeping two sources - for simplicity, sweep Vin_plus
.dc Vin_plus 0 3.3 0.1

.end

* Odd-stage CMOS ring oscillator — NN candidate
*
* Five inverter stages in a loop; n5 feeds back to n1. Alternating .ic seeds
* the latch out of its (unstable) DC operating point so oscillation starts.
* The verify harness renders all declared tokens per technology and resolves
* the selected NN-family checkpoint.
*
* Stage count, load, geometry and initial conditions are explicit tokens.

Vdd vdd 0 <VDD>
<RING_STAGES>
.ic <RING_IC>
.model nmos_nn NMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<NVT>)
.model pmos_nn PMOS (LEVEL=<LEVEL> TECH=<TECH> VT=<PVT>)
.temp <TEMP>
<ANALYSIS>

.end

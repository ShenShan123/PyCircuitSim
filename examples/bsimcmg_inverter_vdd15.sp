* BSIM-CMG CMOS Inverter Transient Simulation using ASAP7 PDK model
* Testing with higher Vdd (1.5V) to overcome PMOS threshold voltage issue
.include /home/shenshan/NN_SPICE/PyCMG/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm

* Power supply - HIGHER VOLTAGE to overcome PMOS threshold
Vdd 1 0 1.5

* Input pulse: 0 -> 1.5V, period=2ns
Vin 2 0 PULSE 0 1.5 0.5n 0.1n 0.1n 0.8n 2n

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos_lvt L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos_lvt L=30n NFIN=10

* Load capacitance (10fF)
Cload 3 0 10e-15

* Initial conditions to help DC convergence
.ic V(3)=1.5

* Transient analysis: 0 to 5ns with 10ps steps
.tran 10p 5n

.end

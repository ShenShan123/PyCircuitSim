* BSIM-CMG CMOS Inverter with SLVT (lower Vth) and higher VDD
.include /home/shenshan/NN_SPICE/PyCMG/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm

* Higher VDD (2V) for better margin over Vth
Vdd 1 0 2.0

* Input pulse: 0 -> 2V, period=2ns
Vin 2 0 PULSE 0 2.0 0.5n 0.1n 0.1n 0.8n 2n

* PMOS SLVT (lower Vth ~0.3V)
Mp1 3 2 1 1 pmos_slvt L=30n NFIN=10

* NMOS SLVT (lower Vth)
Mn1 3 2 0 0 nmos_slvt L=30n NFIN=10

* Load capacitance (10fF)
Cload 3 0 10e-15

* Transient analysis: 0 to 5ns with 10ps steps
.tran 10p 5n

.end

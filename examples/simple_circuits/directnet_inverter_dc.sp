* NN CMOS inverter voltage-transfer curve
Vdd vdd 0 <VDD>
Vin in 0 0
Mn out in 0 0 nmos_nn L=<LN> NFIN=<NFN>
Mp out in vdd vdd pmos_nn L=<LP> NFIN=<NFP>
.model nmos_nn NMOS (<NMOS_PARAMS>)
.model pmos_nn PMOS (<PMOS_PARAMS>)
.temp <TEMP>
<ANALYSIS>
.end

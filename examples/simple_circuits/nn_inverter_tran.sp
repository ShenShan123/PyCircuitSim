* NN CMOS inverter pulse response
Vdd vdd 0 <VDD>
Vin in 0 PULSE 0 <VDD> <TD> <TR> <TF> <PW> <PER>
Mn out in 0 0 nmos_nn L=<LN> NFIN=<NFN>
Mp out in vdd vdd pmos_nn L=<LP> NFIN=<NFP>
Cload out 0 <CLOAD>
.ic V(out)=<VDD>
.model nmos_nn NMOS (<NMOS_PARAMS>)
.model pmos_nn PMOS (<PMOS_PARAMS>)
.temp <TEMP>
<ANALYSIS>
.end

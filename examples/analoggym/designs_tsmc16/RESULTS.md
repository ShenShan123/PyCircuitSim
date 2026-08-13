# Results -- AnalogGym on TSMC16 (BSIM-CMG via PyCMG)

Every number below comes from running the decks in this tree under ngspice-45.2 with the
`bsimcmg.osdi` binary from `PyCircuitSim/external_compact_models/PyCMG`. Nothing is copied from
the AnalogGym paper or from the sky130 port.


## Coverage

| category | designs | fully passing | partial |
|---|--:|--:|--:|
| amplifier | 7 | 7 | 0 |
| ldo | 3 | 3 | 0 |
| sensing_front_end | 6 | 6 | 0 |
| voltage_reference | 1 | 1 | 0 |
| charge_pump | 1 | 1 | 0 |
| **total** | **18** | **18** | **0** |


## Amplifier

| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) | PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Alfio_RAFFC_Pin_3 | 102.1 | 491 kHz | 55.2 | 168 µW | -47.4 | -45.9 | -62.1 | 1.48 mV | 0.071 | 0.139 | 10/10 |
| Fan_SMC_Pin_3 | 113.0 | 1.17 MHz | 59.2 | 5.35 µW | -36.9 | -36.8 | -78.3 | 777 µV | 0.072 | 0.283 | 10/10 |
| Leung_NMCNR_Pin_3 | 125.2 | 518 kHz | 153.8 | 10.3 µW | -75.8 | -77.8 | -89.8 | 152 µV | 0.013 | 0.025 | 10/10 |
| Peng_IAC_Pin_3 | 114.1 | 2.51 MHz | 59.3 | 178 µW | -46.3 | -45.3 | -64.2 | 451 µV | 0.230 | 0.781 | 10/10 |
| Qu2017_AZC_Pin_3 | 123.2 | 4.03 MHz | 86.1 | 99.5 µW | -33.8 | -31.7 | -45.1 | 1.29 mV | 0.236 | 1.265 | 10/10 |
| Qu_LEC_Pin_3 | 126.3 | 2.57 MHz | 88.2 | 13.1 µW | -95.0 | -94.3 | -88.8 | -38.9 µV | 0.172 | 0.743 | 10/10 |
| Song_DACFC_Pin_3 | 111.4 | 616 kHz | 55.8 | 21.8 µW | -69.7 | -54.4 | -56.0 | -1.9 mV | 0.059 | 0.293 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Basic_LDO | 599 mV | 601 mV | 0.0320 | 0.0565 | 44.1 mW | 68.2 | 1.14 MHz | 125.2 | -38.0 | 24.9 mV | 9/9 |
| ldo_1 | 638 mV | 639 mV | 0.0059 | 0.0155 | 44 mW | 60.3 | 140 kHz | 89.6 | -70.6 | 177 mV | 9/9 |
| ldo_2 | 571 mV | 571 mV | 0.0220 | 0.0002 | 44.2 mW | 58.1 | 22.4 MHz | 163.1 | -41.0 | 28.4 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| front_end_11_6T_schematic | 61.5 mV | 2.55 mV/C | 149 mV | 58.2 mV | 198 mV | 4/4 |
| front_end_25_6T_schematic | 99 mV | 422 µV/C | 60 mV | 88.9 mV | 131 mV | 4/4 |
| ptat_1 | 109 mV | 369 µV/C | 53.4 mV | 100 mV | 137 mV | 4/4 |
| ptat_2 | 142 mV | 3.9 mV/C | 227 mV | 131 mV | 344 mV | 4/4 |
| ptat_4 | 110 mV | 458 µV/C | 55.2 mV | 100 mV | 145 mV | 4/4 |

The amplifier shipped in this category runs its own AC bench (gain, GBW, true phase margin, power):

| design | A_v (dB) | GBW | PM (deg) | P | pass |
|---|--:|--:|--:|--:|:--:|
| SMCNR_SE_2st_AMP | 77.8 | 649 kHz | 121.1 | 2.11 µW | 4/4 |


## Voltage Reference

| design | output | Vref @25 C | spread | TC (ppm/C) | pass |
|---|---|--:|--:|--:|:--:|
| dual_output_subthreshold_vref | vref1 | 210 mV | 11.7 mV | 335 | 3/3 |
|  | vref2 | 395 mV | 4.73 mV | 72 |  |


## Charge Pump

| measurement | value |
|---|--:|
| source (up) current, average | 9.03 µA |
| sink (down) current, average | 9.03 µA |
| up/down mismatch | 0.00 % |
| up current range | 4.41 µA ... 12.3 µA |
| down current range | 7.97 µA ... 11 µA |



## Audit criteria

`!` means that one or more requested analyses did not complete, so the design
is partial even if all available numeric gates pass. Amplifiers include gain,
GBW, true (unwrapped) phase margin, power, offset, CMRR, both PSRR polarities,
temperature drift, and both slew directions. LDOs include regulation, loop
gain/GBW/true phase margin, power, line and load regulation, both-load PSRR,
and load-step excursion. Sensors require in-rail output that rises
monotonically over every solved sweep point, 0.3-6 mV/C sensitivity, and a
staircase-free characteristic (local slope and single-step share gates in
tools/sfe.py); references require every output in range with <=500 ppm/C; the
charge pump requires both current directions, 2-200 uA magnitude, and <=5%
mismatch. Simulations use the 0.8 V core rail for TSMC16.

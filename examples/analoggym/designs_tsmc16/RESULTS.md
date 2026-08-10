# Results -- AnalogGym on TSMC16 (BSIM-CMG via PyCMG)

Every number below comes from running the decks in this tree under ngspice-45.2 with the
`bsimcmg.osdi` binary from `PyCircuitSim/external_compact_models/PyCMG`. Nothing is copied from
the AnalogGym paper or from the sky130 port.


## Coverage

| category | designs | fully passing | partial |
|---|--:|--:|--:|
| amplifier | 17 | 17 | 0 |
| ldo | 5 | 5 | 0 |
| sensing_front_end | 13 | 13 | 0 |
| voltage_reference | 2 | 2 | 0 |
| charge_pump | 1 | 1 | 0 |
| **total** | **38** | **38** | **0** |


## Amplifier

| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) | PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Alfio_RAFFC_Pin_3 | 102.1 | 491 kHz | 55.2 | 168 µW | -47.4 | -45.9 | -62.1 | 1.48 mV | 0.071 | 0.139 | 10/10 |
| Fan_SMC_Pin_3 | 113.0 | 1.17 MHz | 59.2 | 5.35 µW | -36.9 | -36.8 | -78.3 | 777 µV | 0.072 | 0.283 | 10/10 |
| HoiLee_AFFC_Pin_3 | 99.7 | 586 kHz | 55.3 | 6.31 µW | -35.6 | -35.7 | -82.2 | 1.5 mV | 0.794 | 0.033 | 10/10 |
| Leung_DFCFC1_Pin_3 | 101.4 | 521 kHz | 63.9 | 51.9 µW | -70.3 | -63.5 | -68.6 | 795 µV | 0.111 | 0.104 | 10/10 |
| Leung_DFCFC2_Pin_3 | 108.7 | 750 kHz | 61.0 | 5.49 µW | -47.7 | -47.0 | -69.0 | 1.4 mV | 0.019 | 0.226 | 10/10 |
| Leung_NMCF_Pin_3 | 104.5 | 516 kHz | 58.8 | 49.6 µW | -37.6 | -37.6 | -97.9 | 466 µV | 0.109 | 0.094 | 10/10 |
| Leung_NMCNR_Pin_3 | 125.2 | 518 kHz | 153.8 | 10.3 µW | -75.8 | -77.8 | -89.8 | 152 µV | 0.013 | 0.025 | 10/10 |
| Peng_ACBC_Pin_3 | 113.2 | 1.34 MHz | 60.0 | 16.4 µW | -37.1 | -37.0 | -77.6 | 867 µV | 0.112 | 0.408 | 10/10 |
| Peng_IAC_Pin_3 | 114.1 | 2.51 MHz | 59.3 | 178 µW | -46.3 | -45.3 | -64.2 | 451 µV | 0.230 | 0.781 | 10/10 |
| Peng_TCFC_Pin_3 | 126.5 | 618 kHz | 86.4 | 29.2 µW | -38.1 | -37.8 | -68.0 | -1.33 mV | 0.074 | 0.358 | 10/10 |
| Qu2017_AZC_Pin_3 | 123.2 | 4.03 MHz | 86.1 | 99.5 µW | -33.8 | -31.7 | -45.1 | 1.29 mV | 0.236 | 1.265 | 10/10 |
| Qu_LEC_Pin_3 | 126.3 | 2.57 MHz | 88.2 | 13.1 µW | -95.0 | -94.3 | -88.8 | -38.9 µV | 0.172 | 0.743 | 10/10 |
| Ramos_PFC_Pin_3 | 124.2 | 502 kHz | 175.8 | 33.7 µW | -41.9 | -41.6 | -70.4 | 256 µV | 0.055 | 0.163 | 10/10 |
| Sau_CFCC_Pin_3 | 99.0 | 1.07 MHz | 67.4 | 2.88 µW | -36.3 | -36.2 | -74.3 | 941 µV | 0.208 | 0.177 | 10/10 |
| Song_DACFC_Pin_3 | 111.4 | 616 kHz | 55.8 | 21.8 µW | -69.7 | -54.4 | -56.0 | -1.9 mV | 0.059 | 0.293 | 10/10 |
| Tan_CLIA_Pin_3 | 123.9 | 6.91 MHz | 57.9 | 21.2 µW | -51.6 | -50.0 | -65.2 | 925 µV | 1.091 | 0.217 | 10/10 |
| Yan_AZ_Pin_3 | 130.2 | 449 kHz | 112.2 | 41.6 µW | -45.0 | -38.2 | -43.4 | 281 µV | 0.029 | 0.162 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Basic_LDO | 599 mV | 601 mV | 0.0320 | 0.0565 | 44.1 mW | 68.2 | 1.14 MHz | 125.2 | -38.0 | 24.9 mV | 9/9 |
| ldo_1 | 638 mV | 639 mV | 0.0059 | 0.0155 | 44 mW | 60.3 | 140 kHz | 89.6 | -70.6 | 177 mV | 9/9 |
| ldo_2 | 571 mV | 571 mV | 0.0220 | 0.0002 | 44.2 mW | 58.1 | 22.4 MHz | 163.1 | -41.0 | 28.4 mV | 9/9 |
| ldo_folded_cascode | 606 mV | 616 mV | 0.0352 | 0.3362 | 45.1 mW | 47.9 | 20.7 MHz | 48.9 | -38.0 | 10.4 mV | 9/9 |
| ldo_simple | 604 mV | 606 mV | 0.0887 | 0.0680 | 44 mW | 53.1 | 2.03 MHz | 52.1 | -35.5 | 49.6 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| PTAT_65_classic1 | 518 mV | 393 µV/C | 54.6 mV | 509 mV | 548 mV | 4/4 |
| PTAT_CLASSIC | 452 mV | 672 µV/C | 75.1 mV | 435 mV | 497 mV | 4/4 |
| PTAT_SENSOR | 217 mV | 561 µV/C | 81.8 mV | 205 mV | 262 mV | 4/4 |
| front_end_11_6T_schematic | 61.5 mV | 2.55 mV/C | 149 mV | 58.2 mV | 198 mV | 4/4 |
| front_end_25_6T_schematic | 99 mV | 422 µV/C | 60 mV | 88.9 mV | 131 mV | 4/4 |
| front_end_31_3T_schematic | 238 mV | 700 µV/C | 102 mV | 224 mV | 295 mV | 4/4 |
| front_end_42_2_2015_REF_schematic | 124 mV | 410 µV/C | 58.4 mV | 114 mV | 155 mV | 4/4 |
| ptat_1 | 109 mV | 369 µV/C | 53.4 mV | 100 mV | 137 mV | 4/4 |
| ptat_2 | 142 mV | 3.9 mV/C | 227 mV | 131 mV | 344 mV | 4/4 |
| ptat_3 | 143 mV | 555 µV/C | 82.5 mV | 130 mV | 188 mV | 4/4 |
| ptat_4 | 110 mV | 458 µV/C | 55.2 mV | 100 mV | 145 mV | 4/4 |
| ptat_6 | 142 mV | 3.9 mV/C | 227 mV | 131 mV | 344 mV | 4/4 |

The amplifier shipped in this category runs its own AC bench (gain, GBW, true phase margin, power):

| design | A_v (dB) | GBW | PM (deg) | P | pass |
|---|--:|--:|--:|--:|:--:|
| SMCNR_SE_2st_AMP | 77.8 | 649 kHz | 121.1 | 2.11 µW | 4/4 |


## Voltage Reference

| design | output | Vref @25 C | spread | TC (ppm/C) | pass |
|---|---|--:|--:|--:|:--:|
| dual_output_subthreshold_vref | vref1 | 210 mV | 11.7 mV | 335 | 3/3 |
|  | vref2 | 395 mV | 4.73 mV | 72 |  |
| three_output_vref | vref1 | 210 mV | 11.7 mV | 335 | 3/3 |
|  | vref2 | 395 mV | 4.73 mV | 72 |  |
|  | vref3 | 197 mV | 2.36 mV | 72 |  |


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

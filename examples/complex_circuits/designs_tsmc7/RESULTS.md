# Results -- AnalogGym on TSMC7 (BSIM-CMG via PyCMG)

Every number below comes from running the decks in this tree under ngspice-45.2 with the
`bsimcmg.osdi` binary from `PyCircuitSim/external_compact_models/bsim_cmg`. Nothing is copied from
the AnalogGym paper or from the sky130 port.


## Coverage

| category | designs | fully passing | partial |
|---|--:|--:|--:|
| amplifier | 5 | 5 | 0 |
| ldo | 2 | 2 | 0 |
| sensing_front_end | 3 | 3 | 0 |
| voltage_reference | 1 | 1 | 0 |
| charge_pump | 1 | 1 | 0 |
| **total** | **12** | **12** | **0** |


## Amplifier

| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) | PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Fan_SMC_Pin_3 | 115.9 | 1.36 MHz | 58.0 | 17.7 µW | -58.9 | -58.3 | -81.6 | 54.3 µV | 0.047 | 0.386 | 10/10 |
| Leung_NMCNR_Pin_3 | 127.8 | 475 kHz | 156.9 | 11 µW | -84.1 | -81.1 | -91.9 | 78.5 µV | 0.017 | 0.018 | 10/10 |
| Peng_IAC_Pin_3 | 117.1 | 500 kHz | 55.1 | 34.9 µW | -52.8 | -53.0 | -87.8 | 57.4 µV | 0.060 | 0.164 | 10/10 |
| Qu2017_AZC_Pin_3 | 126.0 | 1.49 MHz | 103.5 | 28.7 µW | -33.3 | -32.5 | -52.2 | 857 µV | 0.100 | 0.463 | 10/10 |
| Song_DACFC_Pin_3 | 113.7 | 744 kHz | 68.2 | 20.6 µW | -84.4 | -59.1 | -59.6 | -747 µV | 0.063 | 0.255 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| ldo_1 | 565 mV | 565 mV | 0.0268 | 0.0105 | 41.3 mW | 81.2 | 2.97 MHz | 72.5 | -33.8 | 109 mV | 9/9 |
| ldo_2 | 570 mV | 570 mV | 0.0477 | 0.0050 | 41.4 mW | 60.2 | 15 MHz | 158.3 | -38.5 | 56.8 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| front_end_25_6T_schematic | 60.8 mV | 338 µV/C | 47.4 mV | 52.6 mV | 86.5 mV | 4/4 |
| ptat_1 | 119 mV | 3.52 mV/C | 526 mV | 114 mV | 567 mV | 4/4 |
| ptat_4 | 197 mV | 4.87 mV/C | 497 mV | 192 mV | 684 mV | 4/4 |


## Voltage Reference

| design | output | Vref @25 C | spread | TC (ppm/C) | pass |
|---|---|--:|--:|--:|:--:|
| dual_output_subthreshold_vref | vref1 | 197 mV | 9.07 mV | 280 | 3/3 |
|  | vref2 | 274 mV | 9.32 mV | 207 |  |


## Charge Pump

| measurement | value |
|---|--:|
| source (up) current, average | 9.23 µA |
| sink (down) current, average | 9.22 µA |
| up/down mismatch | 0.01 % |
| up current range | 5.89 µA ... 10.3 µA |
| down current range | 8.99 µA ... 9.91 µA |



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
mismatch. Simulations use the 0.75 V core rail for TSMC7.

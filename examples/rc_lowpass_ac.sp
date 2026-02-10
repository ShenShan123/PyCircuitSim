* RC Low-Pass Filter - AC Analysis Test
* Simple first-order RC low-pass filter
* Cutoff frequency: fc = 1/(2*pi*R*C) = 1/(2*pi*1k*100n) = 1.59 kHz
* Expected -3dB at fc, -20dB/decade roll-off

* Input voltage source with AC stimulus
Vin 1 0 DC=0 AC=1 0

* RC filter components
R1 1 2 1k
C1 2 0 100n

* AC analysis: decade sweep from 100 Hz to 100 kHz
.ac dec 10 100 100k

.end

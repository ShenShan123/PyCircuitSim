* RC Low-Pass Filter - AC Analysis Test for NGSPICE
* Simple first-order RC low-pass filter
* Cutoff frequency: fc = 1/(2*pi*R*C) = 1/(2*pi*1k*100n) = 1.59 kHz

* Input voltage source with AC stimulus
Vin 1 0 DC=0 AC=1

* RC filter components
R1 1 2 1k
C1 2 0 100n

* AC analysis: decade sweep from 100 Hz to 100 kHz
.ac dec 10 100 100k

* Output control
.control
run
set hcopydevtype=ascii
print frequency vdb(2) vp(2) > rc_lowpass_ngspice.txt
quit
.endc

.end

* RC Low-Pass Filter - Transient Analysis
* Two-stage RC filter demonstrating step response

V1 1 0 5
R1 1 2 1k
C1 2 0 1n
R2 2 3 1k
C2 3 0 1n

* Transient Analysis
* Simulate 20 microseconds to see step response
.tran 200n 20u

* To run DC sweep instead, comment out .tran and uncomment:
*.dc V1 0 10 0.5

.end

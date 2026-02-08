#!/bin/bash
cd /home/shenshan/NN_SPICE
python main.py test_simple.sp 2>&1 | tee /tmp/test_output.txt
grep "C DEBUG" /tmp/test_output.txt

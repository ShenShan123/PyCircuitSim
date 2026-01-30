#!/usr/bin/env python3
"""Test BSIM4 directly via Python ctypes"""

import sys
sys.path.insert(0, '/home/shenshan/NN_SPICE')

from ctypes import *

# Load the library
lib = CDLL('/home/shenshan/NN_SPICE/pycircuitsim/models/bsim4v5/bridge/libbsim4.so')

# Define structures (simplified, matching the C structures)
class BSIM4_Model(Structure):
    _fields_ = [("type", c_int)]

class BSIM4_Instance(Structure):
    _fields_ = [("L", c_double), ("W", c_double)]

class BSIM4_Internal(Structure):
    _fields_ = [("Ids", c_double), ("Vth", c_double), ("Vgsteff", c_double)]

# Set up function prototypes
lib.BSIM4_InitModel_45nm_NMOS.restype = None
lib.BSIM4_InitModel_45nm_NMOS.argtypes = [POINTER(BSIM4_Model)]

lib.BSIM4_InitInstance.restype = None
lib.BSIM4_InitInstance.argtypes = [POINTER(BSIM4_Instance), c_double, c_double]

lib.BSIM4_Evaluate.restype = c_int
lib.BSIM4_Evaluate.argtypes = [POINTER(BSIM4_Model), POINTER(BSIM4_Instance),
                                c_double, c_double, c_double, POINTER(BSIM4_Internal)]

# Initialize
model = BSIM4_Model()
instance = BSIM4_Instance()
internal = BSIM4_Internal()

lib.BSIM4_InitModel_45nm_NMOS(byref(model))
lib.BSIM4_InitInstance(byref(instance), 45e-9, 90e-9)

# Evaluate
Vds = 0.1
Vgs = 0.5
Vbs = 0.0

ret = lib.BSIM4_Evaluate(byref(model), byref(instance), Vds, Vgs, Vbs, byref(internal))

print(f"Vgs = {Vgs} V")
print(f"Vth = {internal.Vth:.6f} V")
print(f"Vgsteff = {internal.Vgsteff:.6f} V")
print(f"Ids = {internal.Ids:.6e} A = {internal.Ids*1e6:.3f} µA")

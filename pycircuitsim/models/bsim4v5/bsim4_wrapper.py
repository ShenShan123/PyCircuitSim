"""
BSIM4 Python Wrapper

This module provides a Python interface to the BSIM4 compact model
using ctypes to call the C implementation.

Author: PyCircuitSim Team
"""

import ctypes
import os
import numpy as np
from pathlib import Path

# Determine the library path
_lib_path = Path(__file__).parent / "bridge" / "libbsim4.so"

# Load the shared library
try:
    _lib = ctypes.CDLL(str(_lib_path))
except OSError as e:
    raise ImportError(
        f"Failed to load BSIM4 library from {_lib_path}: {e}\n"
        "Please compile the library first using: cd bridge && make"
    )

# Define ctypes structures

class BSIM4_Instance(ctypes.Structure):
    """BSIM4 instance structure (device geometry and instance parameters)"""
    _fields_ = [
        ("L", ctypes.c_double),
        ("W", ctypes.c_double),
        ("drainArea", ctypes.c_double),
        ("sourceArea", ctypes.c_double),
        ("drainSquares", ctypes.c_double),
        ("sourceSquares", ctypes.c_double),
        ("drainPerimeter", ctypes.c_double),
        ("sourcePerimeter", ctypes.c_double),
        ("sa", ctypes.c_double),
        ("sb", ctypes.c_double),
        ("sd", ctypes.c_double),
        ("nf", ctypes.c_double),
        ("off", ctypes.c_int),
        ("vds", ctypes.c_double),
        ("vgs", ctypes.c_double),
        ("vbs", ctypes.c_double),
        ("Vgsteff", ctypes.c_double),
        ("vdsat", ctypes.c_double),
        ("vth", ctypes.c_double),
    ]


class BSIM4_Model(ctypes.Structure):
    """BSIM4 model structure (technology parameters)"""
    _fields_ = [
        ("type", ctypes.c_int),
        ("mobMod", ctypes.c_int),
        ("capMod", ctypes.c_int),
        ("dioMod", ctypes.c_int),
        ("trnqsMod", ctypes.c_int),
        ("acnqsMod", ctypes.c_int),
        ("fnoiMod", ctypes.c_int),
        ("tnoiMod", ctypes.c_int),
        ("rdsMod", ctypes.c_int),
        ("rbodyMod", ctypes.c_int),
        ("rgateMod", ctypes.c_int),
        ("perMod", ctypes.c_int),
        ("geoMod", ctypes.c_int),
        ("igcMod", ctypes.c_int),
        ("igbMod", ctypes.c_int),
        ("tempMod", ctypes.c_int),
        ("paramChk", ctypes.c_int),
        ("tox", ctypes.c_double),
        ("toxp", ctypes.c_double),
        ("toxm", ctypes.c_double),
        ("dtox", ctypes.c_double),
        ("epsrox", ctypes.c_double),
        ("coxe", ctypes.c_double),
        ("cdsc", ctypes.c_double),
        ("cdscb", ctypes.c_double),
        ("cdscd", ctypes.c_double),
        ("cit", ctypes.c_double),
        ("nfactor", ctypes.c_double),
        ("xj", ctypes.c_double),
        ("vsat", ctypes.c_double),
        ("at", ctypes.c_double),
        ("mstar", ctypes.c_double),
        ("a0", ctypes.c_double),
        ("ags", ctypes.c_double),
        ("a1", ctypes.c_double),
        ("a2", ctypes.c_double),
        ("keta", ctypes.c_double),
        ("nsub", ctypes.c_double),
        ("ndep", ctypes.c_double),
        ("nsd", ctypes.c_double),
        ("phin", ctypes.c_double),
        ("ngate", ctypes.c_double),
        ("gamma1", ctypes.c_double),
        ("gamma2", ctypes.c_double),
        ("vbx", ctypes.c_double),
        ("vbm", ctypes.c_double),
        ("xt", ctypes.c_double),
        ("k1", ctypes.c_double),
        ("kt1", ctypes.c_double),
        ("kt1l", ctypes.c_double),
        ("kt2", ctypes.c_double),
        ("k2", ctypes.c_double),
        ("k3", ctypes.c_double),
        ("k3b", ctypes.c_double),
        ("w0", ctypes.c_double),
        ("dvtp0", ctypes.c_double),
        ("dvtp1", ctypes.c_double),
        ("lpe0", ctypes.c_double),
        ("lpeb", ctypes.c_double),
        ("litl", ctypes.c_double),
        ("dvt0", ctypes.c_double),
        ("dvt1", ctypes.c_double),
        ("dvt2", ctypes.c_double),
        ("dvt0w", ctypes.c_double),
        ("dvt1w", ctypes.c_double),
        ("dvt2w", ctypes.c_double),
        ("drout", ctypes.c_double),
        ("dsub", ctypes.c_double),
        ("vth0", ctypes.c_double),
        ("eu", ctypes.c_double),
        ("ua", ctypes.c_double),
        ("ua1", ctypes.c_double),
        ("ub", ctypes.c_double),
        ("ub1", ctypes.c_double),
        ("uc", ctypes.c_double),
        ("uc1", ctypes.c_double),
        ("u0", ctypes.c_double),
        ("ute", ctypes.c_double),
        ("voff", ctypes.c_double),
        ("minv", ctypes.c_double),
        ("voffl", ctypes.c_double),
        ("voffcvbn", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("rdsw", ctypes.c_double),
        ("rdswmin", ctypes.c_double),
        ("rdwmin", ctypes.c_double),
        ("rswmin", ctypes.c_double),
        ("rsw", ctypes.c_double),
        ("rdw", ctypes.c_double),
        ("prwg", ctypes.c_double),
        ("prwb", ctypes.c_double),
        ("prt", ctypes.c_double),
        ("eta0", ctypes.c_double),
        ("etab", ctypes.c_double),
        ("pclm", ctypes.c_double),
        ("pdibl1", ctypes.c_double),
        ("pdibl2", ctypes.c_double),
        ("pdiblb", ctypes.c_double),
        ("fprout", ctypes.c_double),
        ("pdits", ctypes.c_double),
        ("pditsd", ctypes.c_double),
        ("pditsl", ctypes.c_double),
        ("pscbe1", ctypes.c_double),
        ("pscbe2", ctypes.c_double),
        ("pvag", ctypes.c_double),
        ("wr", ctypes.c_double),
        ("dwg", ctypes.c_double),
        ("dwb", ctypes.c_double),
        ("b0", ctypes.c_double),
        ("b1", ctypes.c_double),
        ("alpha0", ctypes.c_double),
        ("alpha1", ctypes.c_double),
        ("beta0", ctypes.c_double),
        ("agidl", ctypes.c_double),
        ("bgidl", ctypes.c_double),
        ("cgidl", ctypes.c_double),
        ("egidl", ctypes.c_double),
        ("aigc", ctypes.c_double),
        ("bigc", ctypes.c_double),
        ("cigc", ctypes.c_double),
        ("aigsd", ctypes.c_double),
        ("bigsd", ctypes.c_double),
        ("cigsd", ctypes.c_double),
        ("aigbacc", ctypes.c_double),
        ("bigbacc", ctypes.c_double),
        ("cigbacc", ctypes.c_double),
        ("aigbinv", ctypes.c_double),
        ("bigbinv", ctypes.c_double),
        ("cigbinv", ctypes.c_double),
        ("nigc", ctypes.c_double),
        ("nigbacc", ctypes.c_double),
        ("nigbinv", ctypes.c_double),
        ("ntox", ctypes.c_double),
        ("eigbinv", ctypes.c_double),
        ("pigcd", ctypes.c_double),
        ("poxedge", ctypes.c_double),
        ("toxref", ctypes.c_double),
        ("ijthdfwd", ctypes.c_double),
        ("ijthsfwd", ctypes.c_double),
        ("ijthdrev", ctypes.c_double),
        ("ijthsrev", ctypes.c_double),
        ("xjbvd", ctypes.c_double),
        ("xjbvs", ctypes.c_double),
        ("bvd", ctypes.c_double),
        ("bvs", ctypes.c_double),
        ("jtss", ctypes.c_double),
        ("jtsd", ctypes.c_double),
        ("jtssws", ctypes.c_double),
        ("jtsswd", ctypes.c_double),
        ("jtsswgs", ctypes.c_double),
        ("jtsswgd", ctypes.c_double),
        ("njts", ctypes.c_double),
        ("njtssw", ctypes.c_double),
        ("njtsswg", ctypes.c_double),
        ("xtss", ctypes.c_double),
        ("xtsd", ctypes.c_double),
        ("xtssws", ctypes.c_double),
        ("xtsswd", ctypes.c_double),
        ("xtsswgs", ctypes.c_double),
        ("xtsswgd", ctypes.c_double),
        ("tnjts", ctypes.c_double),
        ("tnjtssw", ctypes.c_double),
        ("tnjtsswg", ctypes.c_double),
        ("vtss", ctypes.c_double),
        ("vtsd", ctypes.c_double),
        ("vtssws", ctypes.c_double),
        ("vtsswd", ctypes.c_double),
        ("vtsswgs", ctypes.c_double),
        ("vtsswgd", ctypes.c_double),
        ("cgsl", ctypes.c_double),
        ("cgdl", ctypes.c_double),
        ("ckappas", ctypes.c_double),
        ("ckappad", ctypes.c_double),
        ("cf", ctypes.c_double),
        ("vfbcv", ctypes.c_double),
        ("clc", ctypes.c_double),
        ("cle", ctypes.c_double),
        ("dwc", ctypes.c_double),
        ("dlc", ctypes.c_double),
        ("xw", ctypes.c_double),
        ("xl", ctypes.c_double),
        ("dlcig", ctypes.c_double),
        ("dwj", ctypes.c_double),
        ("noff", ctypes.c_double),
        ("voffcv", ctypes.c_double),
        ("acde", ctypes.c_double),
        ("moin", ctypes.c_double),
        ("tcj", ctypes.c_double),
        ("tcjsw", ctypes.c_double),
        ("tcjswg", ctypes.c_double),
        ("tpb", ctypes.c_double),
        ("tpbsw", ctypes.c_double),
        ("tpbswg", ctypes.c_double),
        ("dmcg", ctypes.c_double),
        ("dmci", ctypes.c_double),
        ("dmdg", ctypes.c_double),
        ("dmcgt", ctypes.c_double),
        ("xgw", ctypes.c_double),
        ("xgl", ctypes.c_double),
        ("rshg", ctypes.c_double),
        ("ngcon", ctypes.c_double),
        ("temp", ctypes.c_double),
        ("tnom", ctypes.c_double),
        ("vfb", ctypes.c_double),
        ("gbmin", ctypes.c_double),
        ("Xdep0", ctypes.c_double),
        ("cdep0", ctypes.c_double),
        ("voffcbn", ctypes.c_double),
        ("sqrtPhi", ctypes.c_double),
    ]


class BSIM4_Output(ctypes.Structure):
    """BSIM4 output structure (currents, conductances, charges)"""
    _fields_ = [
        ("Id", ctypes.c_double),
        ("Ib", ctypes.c_double),
        ("Ig", ctypes.c_double),
        ("Is", ctypes.c_double),
        ("Gm", ctypes.c_double),
        ("Gds", ctypes.c_double),
        ("Gmbs", ctypes.c_double),
        ("Ggb", ctypes.c_double),
        ("Gbd", ctypes.c_double),
        ("Gbs", ctypes.c_double),
        ("Qg", ctypes.c_double),
        ("Qb", ctypes.c_double),
        ("Qd", ctypes.c_double),
        ("Qs", ctypes.c_double),
        ("Cgg", ctypes.c_double),
        ("Cgd", ctypes.c_double),
        ("Cgs", ctypes.c_double),
        ("Cgb", ctypes.c_double),
        ("Vth", ctypes.c_double),
        ("Vgsteff", ctypes.c_double),
        ("error", ctypes.c_int),
    ]


# Define function prototypes

_lib.BSIM4_Evaluate.argtypes = [
    ctypes.POINTER(BSIM4_Model),
    ctypes.POINTER(BSIM4_Instance),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(BSIM4_Output),
]
_lib.BSIM4_Evaluate.restype = ctypes.c_int

_lib.BSIM4_InitModel_45nm_NMOS.argtypes = [ctypes.POINTER(BSIM4_Model)]
_lib.BSIM4_InitModel_45nm_NMOS.restype = None

_lib.BSIM4_InitModel_45nm_PMOS.argtypes = [ctypes.POINTER(BSIM4_Model)]
_lib.BSIM4_InitModel_45nm_PMOS.restype = None

_lib.BSIM4_InitInstance.argtypes = [
    ctypes.POINTER(BSIM4_Instance),
    ctypes.c_double,
    ctypes.c_double,
]
_lib.BSIM4_InitInstance.restype = None

_lib.BSIM4_SetParam.argtypes = [
    ctypes.POINTER(BSIM4_Model),
    ctypes.c_char_p,
    ctypes.c_double,
]
_lib.BSIM4_SetParam.restype = ctypes.c_int

_lib.BSIM4_GetParam.argtypes = [
    ctypes.POINTER(BSIM4_Model),
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_double),
]
_lib.BSIM4_GetParam.restype = ctypes.c_int


class BSIM4Device:
    """
    BSIM4 MOSFET Device

    This class provides a Python interface to the BSIM4 compact model.

    Example:
        >>> model = BSIM4Model("nmos")
        >>> model.set_param("VTH0", 0.4)
        >>> model.set_param("U0", 100.0)  # cm^2/V-s
        >>>
        >>> device = BSIM4Device(model, L=45e-9, W=1e-6)
        >>> output = device.evaluate(Vds=1.0, Vgs=1.0, Vbs=0.0)
        >>> print(f"Id = {output.Id}, Gm = {output.Gm}")
    """

    def __init__(self, model, L, W, nf=1.0):
        """
        Initialize a BSIM4 device

        Args:
            model: BSIM4Model instance
            L: Channel length (m)
            W: Channel width (m)
            nf: Number of fingers (default 1.0)
        """
        self._model = model
        self._instance = BSIM4_Instance()
        _lib.BSIM4_InitInstance(ctypes.byref(self._instance), L, W)
        self._instance.nf = nf

    def evaluate(self, Vds, Vgs, Vbs=0.0):
        """
        Evaluate the BSIM4 model at a given bias point

        Args:
            Vds: Drain-source voltage (V)
            Vgs: Gate-source voltage (V)
            Vbs: Bulk-source voltage (V, default 0.0)

        Returns:
            BSIM4Output object containing currents and conductances
        """
        output = BSIM4_Output()
        ret = _lib.BSIM4_Evaluate(
            ctypes.byref(self._model._model),
            ctypes.byref(self._instance),
            Vds,
            Vgs,
            Vbs,
            ctypes.byref(output),
        )

        if ret != 0:
            raise RuntimeError(f"BSIM4 evaluation failed with error code {ret}")

        return BSIM4Output(output)


class BSIM4Model:
    """
    BSIM4 Model Parameters

    This class holds the technology parameters for a BSIM4 model.
    """

    def __init__(self, device_type="nmos", technology="45nm"):
        """
        Initialize a BSIM4 model

        Args:
            device_type: "nmos" or "pmos"
            technology: Technology node (default "45nm")
        """
        self._model = BSIM4_Model()

        if device_type.lower() == "nmos":
            _lib.BSIM4_InitModel_45nm_NMOS(ctypes.byref(self._model))
        elif device_type.lower() == "pmos":
            _lib.BSIM4_InitModel_45nm_PMOS(ctypes.byref(self._model))
        else:
            raise ValueError(f"Unknown device type: {device_type}")

    def set_param(self, name, value):
        """
        Set a model parameter

        Args:
            name: Parameter name (e.g., "VTH0", "U0", "TOX")
            value: Parameter value

        Returns:
            True if successful, False if parameter not found
        """
        ret = _lib.BSIM4_SetParam(ctypes.byref(self._model), name.encode(), value)
        return ret == 0

    def get_param(self, name):
        """
        Get a model parameter value

        Args:
            name: Parameter name

        Returns:
            Parameter value, or None if not found
        """
        value = ctypes.c_double()
        ret = _lib.BSIM4_GetParam(ctypes.byref(self._model), name.encode(), ctypes.byref(value))
        if ret == 0:
            return value.value
        return None


class BSIM4Output:
    """
    BSIM4 Model Output

    This class holds the output from a BSIM4 model evaluation.
    """

    def __init__(self, output_struct):
        self.Id = output_struct.Id
        self.Ib = output_struct.Ib
        self.Ig = output_struct.Ig
        self.Is = output_struct.Is
        self.Gm = output_struct.Gm
        self.Gds = output_struct.Gds
        self.Gmbs = output_struct.Gmbs
        self.Ggb = output_struct.Ggb
        self.Gbd = output_struct.Gbd
        self.Gbs = output_struct.Gbs
        self.Qg = output_struct.Qg
        self.Qb = output_struct.Qb
        self.Qd = output_struct.Qd
        self.Qs = output_struct.Qs
        self.Cgg = output_struct.Cgg
        self.Cgd = output_struct.Cgd
        self.Cgs = output_struct.Cgs
        self.Cgb = output_struct.Cgb
        self.Vth = output_struct.Vth
        self.Vgsteff = output_struct.Vgsteff
        self.error = output_struct.error

    def __repr__(self):
        return (
            f"BSIM4Output(Id={self.Id:.6e}, Gm={self.Gm:.6e}, Gds={self.Gds:.6e}, "
            f"Gmbs={self.Gmbs:.6e})"
        )

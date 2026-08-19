"""
OSDI constants, ctypes structure definitions, and function type declarations.

This module contains all low-level OSDI interface definitions used by the
PyCMG ctypes-based OSDI host. It has no dependencies on other pycmg modules.
"""

from __future__ import annotations

import ctypes

# ---------------------------------------------------------------------------
# OSDI constants (from osdi_0_3.h)
# ---------------------------------------------------------------------------
PARA_TY_MASK = 3
PARA_TY_REAL = 0
PARA_TY_INT = 1
PARA_TY_STR = 2
PARA_KIND_MASK = 3 << 30
PARA_KIND_MODEL = 0 << 30
PARA_KIND_INST = 1 << 30
PARA_KIND_OPVAR = 2 << 30

ACCESS_FLAG_READ = 0
ACCESS_FLAG_SET = 1
ACCESS_FLAG_INSTANCE = 4

CALC_RESIST_RESIDUAL = 1
CALC_REACT_RESIDUAL = 2
CALC_RESIST_JACOBIAN = 4
CALC_REACT_JACOBIAN = 8
CALC_NOISE = 16
CALC_OP = 32
CALC_RESIST_LIM_RHS = 64
CALC_REACT_LIM_RHS = 128
ENABLE_LIM = 256
INIT_LIM = 512
ANALYSIS_NOISE = 1024
ANALYSIS_DC = 2048
ANALYSIS_AC = 4096
ANALYSIS_TRAN = 8192
ANALYSIS_IC = 16384
ANALYSIS_STATIC = 32768
ANALYSIS_NODESET = 65536

EVAL_RET_FLAG_LIM = 1
EVAL_RET_FLAG_FATAL = 2
EVAL_RET_FLAG_FINISH = 4
EVAL_RET_FLAG_STOP = 8

LOG_LVL_MASK = 7

INIT_ERR_OUT_OF_BOUNDS = 1

UINT32_MAX = 0xFFFFFFFF

_INSTANCE_NAME = ctypes.c_char_p(b"osdi_host")


# ---------------------------------------------------------------------------
# OSDI ctypes structure definitions
# ---------------------------------------------------------------------------

class OsdiLimFunction(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("num_args", ctypes.c_uint32),
        ("func_ptr", ctypes.c_void_p),
    ]


class OsdiSimParas(ctypes.Structure):
    _fields_ = [
        ("names", ctypes.POINTER(ctypes.c_char_p)),
        ("vals", ctypes.POINTER(ctypes.c_double)),
        ("names_str", ctypes.POINTER(ctypes.c_char_p)),
        ("vals_str", ctypes.POINTER(ctypes.c_char_p)),
    ]


class OsdiSimInfo(ctypes.Structure):
    _fields_ = [
        ("paras", OsdiSimParas),
        ("abstime", ctypes.c_double),
        ("prev_solve", ctypes.POINTER(ctypes.c_double)),
        ("prev_state", ctypes.POINTER(ctypes.c_double)),
        ("next_state", ctypes.POINTER(ctypes.c_double)),
        ("flags", ctypes.c_uint32),
    ]


class OsdiInitErrorPayload(ctypes.Union):
    _fields_ = [("parameter_id", ctypes.c_uint32)]


class OsdiInitError(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint32),
        ("payload", OsdiInitErrorPayload),
    ]


class OsdiInitInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("num_errors", ctypes.c_uint32),
        ("errors", ctypes.POINTER(OsdiInitError)),
    ]


class OsdiNodePair(ctypes.Structure):
    _fields_ = [
        ("node_1", ctypes.c_uint32),
        ("node_2", ctypes.c_uint32),
    ]


class OsdiJacobianEntry(ctypes.Structure):
    _fields_ = [
        ("nodes", OsdiNodePair),
        ("react_ptr_off", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class OsdiNode(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("units", ctypes.c_char_p),
        ("residual_units", ctypes.c_char_p),
        ("resist_residual_off", ctypes.c_uint32),
        ("react_residual_off", ctypes.c_uint32),
        ("resist_limit_rhs_off", ctypes.c_uint32),
        ("react_limit_rhs_off", ctypes.c_uint32),
        ("is_flow", ctypes.c_bool),
    ]


class OsdiParamOpvar(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.POINTER(ctypes.c_char_p)),
        ("num_alias", ctypes.c_uint32),
        ("description", ctypes.c_char_p),
        ("units", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("len", ctypes.c_uint32),
    ]


class OsdiNoiseSource(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("nodes", OsdiNodePair),
    ]


# ---------------------------------------------------------------------------
# OSDI function type declarations
# ---------------------------------------------------------------------------

ACCESS_FUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_uint32)
SETUP_MODEL_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.POINTER(OsdiSimParas),
                                    ctypes.POINTER(OsdiInitInfo))
SETUP_INSTANCE_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_double,
                                       ctypes.c_uint32, ctypes.POINTER(OsdiSimParas),
                                       ctypes.POINTER(OsdiInitInfo))
EVAL_FUNC = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.POINTER(OsdiSimInfo))
LOAD_NOISE_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_double, ctypes.POINTER(ctypes.c_double))
LOAD_RESIDUAL_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_double))
LOAD_SPICE_RHS_DC_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double))
LOAD_SPICE_RHS_TRAN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_double),
                                            ctypes.POINTER(ctypes.c_double),
                                            ctypes.c_double)
LOAD_JACOBIAN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
LOAD_JACOBIAN_REACT_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_double)
LOAD_JACOBIAN_TRAN_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_double)


# ---------------------------------------------------------------------------
# OSDI descriptor -- main interface struct
# ---------------------------------------------------------------------------

class OsdiDescriptor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("num_nodes", ctypes.c_uint32),
        ("num_terminals", ctypes.c_uint32),
        ("nodes", ctypes.POINTER(OsdiNode)),
        ("num_jacobian_entries", ctypes.c_uint32),
        ("jacobian_entries", ctypes.POINTER(OsdiJacobianEntry)),
        ("num_collapsible", ctypes.c_uint32),
        ("collapsible", ctypes.POINTER(OsdiNodePair)),
        ("collapsed_offset", ctypes.c_uint32),
        ("noise_sources", ctypes.POINTER(OsdiNoiseSource)),
        ("num_noise_src", ctypes.c_uint32),
        ("num_params", ctypes.c_uint32),
        ("num_instance_params", ctypes.c_uint32),
        ("num_opvars", ctypes.c_uint32),
        ("param_opvar", ctypes.POINTER(OsdiParamOpvar)),
        ("node_mapping_offset", ctypes.c_uint32),
        ("jacobian_ptr_resist_offset", ctypes.c_uint32),
        ("num_states", ctypes.c_uint32),
        ("state_idx_off", ctypes.c_uint32),
        ("bound_step_offset", ctypes.c_uint32),
        ("instance_size", ctypes.c_uint32),
        ("model_size", ctypes.c_uint32),
        ("access", ACCESS_FUNC),
        ("setup_model", SETUP_MODEL_FUNC),
        ("setup_instance", SETUP_INSTANCE_FUNC),
        ("eval", EVAL_FUNC),
        ("load_noise", LOAD_NOISE_FUNC),
        ("load_residual_resist", LOAD_RESIDUAL_FUNC),
        ("load_residual_react", LOAD_RESIDUAL_FUNC),
        ("load_limit_rhs_resist", LOAD_RESIDUAL_FUNC),
        ("load_limit_rhs_react", LOAD_RESIDUAL_FUNC),
        ("load_spice_rhs_dc", LOAD_SPICE_RHS_DC_FUNC),
        ("load_spice_rhs_tran", LOAD_SPICE_RHS_TRAN_FUNC),
        ("load_jacobian_resist", LOAD_JACOBIAN_FUNC),
        ("load_jacobian_react", LOAD_JACOBIAN_REACT_FUNC),
        ("load_jacobian_tran", LOAD_JACOBIAN_TRAN_FUNC),
    ]

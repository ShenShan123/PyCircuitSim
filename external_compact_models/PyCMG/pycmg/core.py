"""
OSDI library loader, model/instance wrappers, and simulation state management.

This module provides the low-level OSDI interface classes:
- OsdiLibrary: Loads compiled .osdi files
- OsdiModel: Model parameter storage
- OsdiSimulation: Simulation state (node voltages, Jacobian, residuals)
- OsdiInstance: Device instance with eval/solve methods
- apply_param: Parameter application to model/instance
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from .osdi_types import (
    ACCESS_FLAG_INSTANCE,
    ACCESS_FLAG_SET,
    ANALYSIS_DC,
    ANALYSIS_IC,
    ANALYSIS_STATIC,
    ANALYSIS_TRAN,
    CALC_OP,
    CALC_REACT_JACOBIAN,
    CALC_REACT_LIM_RHS,
    CALC_REACT_RESIDUAL,
    CALC_RESIST_JACOBIAN,
    CALC_RESIST_LIM_RHS,
    CALC_RESIST_RESIDUAL,
    ENABLE_LIM,
    EVAL_RET_FLAG_FATAL,
    INIT_ERR_OUT_OF_BOUNDS,
    INIT_LIM,
    LOG_LVL_MASK,
    PARA_KIND_INST,
    PARA_KIND_MASK,
    PARA_TY_INT,
    PARA_TY_MASK,
    PARA_TY_REAL,
    UINT32_MAX,
    OsdiDescriptor,
    OsdiInitInfo,
    OsdiLimFunction,
    OsdiSimInfo,
    OsdiSimParas,
    _INSTANCE_NAME,
)


# ---------------------------------------------------------------------------
# Memory management helpers
# ---------------------------------------------------------------------------

def _load_libc() -> Optional[ctypes.CDLL]:
    path = ctypes.util.find_library("c")
    if not path:
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


_LIBC = _load_libc()


class AlignedBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.ptr = ctypes.c_void_p()
        self._buffer: Optional[ctypes.Array] = None
        self._use_libc = False
        if size <= 0:
            return
        alignment = ctypes.alignment(ctypes.c_longdouble)
        if _LIBC is not None and hasattr(_LIBC, "posix_memalign"):
            mem = ctypes.c_void_p()
            res = _LIBC.posix_memalign(ctypes.byref(mem), alignment, size)
            if res == 0 and mem.value is not None:
                ctypes.memset(mem, 0, size)
                self.ptr = mem
                self._use_libc = True
                return
        self._buffer = ctypes.create_string_buffer(size)
        self.ptr = ctypes.cast(self._buffer, ctypes.c_void_p)

    def close(self) -> None:
        if self._use_libc and self.ptr.value:
            if _LIBC is not None and hasattr(_LIBC, "free"):
                _LIBC.free(self.ptr)
            self.ptr = ctypes.c_void_p()
            self._use_libc = False
        self._buffer = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# OSDI callback implementations (limiting, logging)
# ---------------------------------------------------------------------------

@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_bool, ctypes.POINTER(ctypes.c_bool),
                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
def _pnjlim(init: bool,
            check: ctypes.POINTER(ctypes.c_bool),
            vnew: float,
            vold: float,
            vt: float,
            vcrit: float) -> float:
    triggered = False
    if init:
        vnew = vcrit
        triggered = True
    elif vnew > vcrit and abs(vnew - vold) > 2.0 * vt:
        if vold > 0.0:
            arg = (vnew - vold) / vt
            if arg > 0.0:
                vnew = vold + vt * math.log(arg + 1.0)
            else:
                vnew = vcrit
        else:
            if vnew > 0:
                vnew = vt * math.log(vnew / vt)
            else:
                vnew = vcrit
        triggered = True
    if check:
        check[0] = triggered
    return vnew


@ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32)
def _osdi_log(handle: ctypes.c_void_p, msg: ctypes.c_char_p, lvl: int) -> None:
    instance = b"osdi"
    if handle:
        try:
            instance = ctypes.cast(handle, ctypes.c_char_p).value or instance
        except Exception:
            pass
    text = msg.decode("utf-8", errors="replace") if msg else ""
    level = lvl & LOG_LVL_MASK
    sys.stderr.write(f"osdi[{instance.decode('utf-8', errors='replace')}] lvl={level} {text}\n")


# ---------------------------------------------------------------------------
# Init result checker
# ---------------------------------------------------------------------------

def _check_init_result(desc: Optional[OsdiDescriptor], info: OsdiInitInfo) -> None:
    def _cleanup() -> None:
        if info.num_errors != 0 and info.errors:
            if _LIBC is not None and hasattr(_LIBC, "free"):
                _LIBC.free(info.errors)
    if info.flags & EVAL_RET_FLAG_FATAL:
        _cleanup()
        raise RuntimeError("OSDI fatal error reported during setup")
    if info.num_errors == 0:
        _cleanup()
        return
    message = []
    fatal = False
    if not info.errors:
        sys.stderr.write(f"OSDI init: {info.num_errors} errors reported but error array is null\n")
        return
    for i in range(info.num_errors):
        err = info.errors[i]
        if err.code == INIT_ERR_OUT_OF_BOUNDS:
            param_id = err.payload.parameter_id
            name = "unknown"
            if desc is not None and desc.param_opvar and param_id < desc.num_params:
                param = desc.param_opvar[param_id]
                if param.name and param.name[0]:
                    name = param.name[0].decode("utf-8", errors="replace")
            message.append(f"parameter out of bounds: {name}")
        else:
            fatal = True
            message.append("unknown OSDI init error")
    _cleanup()
    if message:
        msg = "; ".join(message)
        if fatal:
            raise RuntimeError(msg)
        sys.stderr.write(f"OSDI init warning: {msg}\n")


# ---------------------------------------------------------------------------
# OSDI library loader and model/instance wrappers
# ---------------------------------------------------------------------------

class OsdiLibrary:
    def __init__(self, path: str) -> None:
        mode = getattr(ctypes, "RTLD_LOCAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        self._lib = ctypes.CDLL(path, mode=mode)
        self._check_version()
        self._count = self._load_descriptor_count()
        self._descriptors = self._load_descriptors(self._count)
        self._install_log()
        self._install_lim()

    def _check_version(self) -> None:
        try:
            major = ctypes.c_uint32.in_dll(self._lib, "OSDI_VERSION_MAJOR").value
            minor = ctypes.c_uint32.in_dll(self._lib, "OSDI_VERSION_MINOR").value
        except ValueError as exc:
            raise RuntimeError("missing OSDI version symbols") from exc
        if major != 0 or minor != 3:
            raise RuntimeError("unsupported OSDI version")

    def _load_descriptor_count(self) -> int:
        return ctypes.c_uint32.in_dll(self._lib, "OSDI_NUM_DESCRIPTORS").value

    def _load_descriptors(self, count: int) -> ctypes.Array:
        if count <= 0:
            raise RuntimeError("no OSDI descriptors")
        array_type = OsdiDescriptor * count
        try:
            return array_type.in_dll(self._lib, "OSDI_DESCRIPTORS")
        except ValueError:
            ptr = ctypes.c_void_p.in_dll(self._lib, "OSDI_DESCRIPTORS")
            if not ptr.value:
                raise RuntimeError("missing OSDI descriptors")
            return (OsdiDescriptor * count).from_address(ptr.value)

    def _install_log(self) -> None:
        try:
            log_var = ctypes.c_void_p.in_dll(self._lib, "osdi_log")
        except ValueError:
            return
        log_var.value = ctypes.cast(_osdi_log, ctypes.c_void_p).value

    def _install_lim(self) -> None:
        try:
            table_ptr = ctypes.POINTER(ctypes.POINTER(OsdiLimFunction)).in_dll(self._lib, "OSDI_LIM_TABLE")
            len_val = ctypes.c_uint32.in_dll(self._lib, "OSDI_LIM_TABLE_LEN").value
        except ValueError:
            return
        if not table_ptr or not table_ptr[0]:
            return
        table = table_ptr[0]
        for i in range(len_val):
            entry = table[i]
            if entry.name and entry.name.decode("utf-8", errors="replace") == "pnjlim":
                table[i].func_ptr = ctypes.cast(_pnjlim, ctypes.c_void_p).value

    def descriptor(self, index: int = 0) -> Optional[OsdiDescriptor]:
        if index < 0 or index >= self._count:
            return None
        return self._descriptors[index]

    def descriptor_by_name(self, name: str) -> Optional[OsdiDescriptor]:
        for i in range(self._count):
            desc = self._descriptors[i]
            if desc.name and name == desc.name.decode("utf-8", errors="replace"):
                return desc
        return None


class OsdiModel:
    def __init__(self, desc: OsdiDescriptor) -> None:
        if not desc:
            raise RuntimeError("descriptor is null")
        self._desc = desc
        self._buf = AlignedBuffer(int(desc.model_size))

    def data(self) -> ctypes.c_void_p:
        return self._buf.ptr

    @property
    def descriptor(self) -> OsdiDescriptor:
        return self._desc

    def process_params(self) -> None:
        sim_params = OsdiSimParas()
        info = OsdiInitInfo()
        self._desc.setup_model(_INSTANCE_NAME, self._buf.ptr, ctypes.byref(sim_params), ctypes.byref(info))
        _check_init_result(self._desc, info)

    def set_param(self, name: str, value: float) -> None:
        desc = self._desc
        if not desc.param_opvar:
            raise RuntimeError("descriptor has no parameter metadata")
        for i in range(desc.num_params):
            param = desc.param_opvar[i]
            param_name = param.name[0].decode("utf-8", errors="replace") if param.name else ""
            if param_name and name == param_name:
                ptr = desc.access(None, self._buf.ptr, i, ACCESS_FLAG_SET)
                if not ptr:
                    raise RuntimeError("invalid parameter access")
                ty = param.flags & PARA_TY_MASK
                if ty == PARA_TY_INT:
                    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[0] = int(value)
                elif ty == PARA_TY_REAL:
                    ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0] = float(value)
                else:
                    raise RuntimeError(f"string parameter not supported: {name}")
                return
        raise RuntimeError(f"parameter not found: {name}")


# ---------------------------------------------------------------------------
# Simulation state management
# ---------------------------------------------------------------------------

class OsdiSimulation:
    def __init__(self) -> None:
        self.node_names: List[str] = ["gnd"]
        self.node_index: Dict[str, int] = {"gnd": 0}
        self.terminal_indices: List[int] = []
        self.internal_indices: List[int] = []
        self.residual_resist = self._make_array(1)
        self.residual_react = self._make_array(1)
        self.rhs_tran = self._make_array(1)
        self.solve = self._make_array(1)
        self.prev_solve = self._make_array(1)
        self.has_prev_solve = False
        self.jacobian_info: List[Tuple[int, int]] = [(0, 0)]
        self.jacobian_index: Dict[int, int] = {0: 0}
        self.jacobian_resist = self._make_array(1)
        self.jacobian_react = self._make_array(1)
        self.state_prev = self._make_array(0)
        self.state_next = self._make_array(0)
        self.noise_dense = self._make_array(0)
        self.sim_param_names: List[str] = []
        self.sim_param_vals: List[float] = []

    def _make_array(self, size: int) -> ctypes.Array:
        return (ctypes.c_double * max(size, 0))()

    def _resize_vectors(self, new_size: int) -> None:
        self.residual_resist = self._resize_array(self.residual_resist, new_size)
        self.residual_react = self._resize_array(self.residual_react, new_size)
        self.rhs_tran = self._resize_array(self.rhs_tran, new_size)
        self.solve = self._resize_array(self.solve, new_size)
        self.prev_solve = self._resize_array(self.prev_solve, new_size)

    @staticmethod
    def _resize_array(arr: ctypes.Array, new_size: int) -> ctypes.Array:
        new_arr = (ctypes.c_double * new_size)()
        for i in range(min(len(arr), new_size)):
            new_arr[i] = arr[i]
        return new_arr

    def register_node(self, name: str) -> int:
        if name in self.node_index:
            return self.node_index[name]
        idx = len(self.node_names)
        self.node_names.append(name)
        self.node_index[name] = idx
        self._resize_vectors(len(self.node_names))
        return idx

    def copy_solve_to_prev(self) -> None:
        for i in range(len(self.solve)):
            self.prev_solve[i] = self.solve[i]

    def register_jacobian_entry(self, row: int, col: int) -> None:
        if row == 0 or col == 0:
            return
        key = (row << 32) | col
        if key in self.jacobian_index:
            return
        idx = len(self.jacobian_info)
        self.jacobian_info.append((row, col))
        self.jacobian_index[key] = idx

    def get_jacobian_entry(self, row: int, col: int) -> int:
        if row == 0 or col == 0:
            return 0
        key = (row << 32) | col
        return self.jacobian_index.get(key, 0)

    def build_jacobian(self) -> None:
        size = len(self.jacobian_info)
        if not hasattr(self, 'jacobian_resist') or len(self.jacobian_resist) != size:
            self.jacobian_resist = self._make_array(size)
        else:
            ctypes.memset(ctypes.addressof(self.jacobian_resist), 0,
                          ctypes.sizeof(self.jacobian_resist))
        if not hasattr(self, 'jacobian_react') or len(self.jacobian_react) != size:
            self.jacobian_react = self._make_array(size)
        else:
            ctypes.memset(ctypes.addressof(self.jacobian_react), 0,
                          ctypes.sizeof(self.jacobian_react))

    def clear(self) -> None:
        for i in range(len(self.residual_resist)):
            self.residual_resist[i] = 0.0
            self.residual_react[i] = 0.0
            self.rhs_tran[i] = 0.0
        for i in range(len(self.jacobian_resist)):
            self.jacobian_resist[i] = 0.0
            self.jacobian_react[i] = 0.0

    def set_voltage(self, node: str, voltage: float) -> None:
        if node not in self.node_index:
            raise RuntimeError(f"unknown node: {node}")
        self.solve[self.node_index[node]] = voltage

    def set_sim_param(self, name: str, value: float) -> None:
        for i, key in enumerate(self.sim_param_names):
            if key == name:
                self.sim_param_vals[i] = value
                return
        self.sim_param_names.append(name)
        self.sim_param_vals.append(value)

    def build_sim_paras(self) -> Tuple[OsdiSimParas, ctypes.Array, ctypes.Array, ctypes.Array, ctypes.Array]:
        names = [ctypes.c_char_p(n.encode("utf-8")) for n in self.sim_param_names]
        vals = [ctypes.c_double(v) for v in self.sim_param_vals]
        names.append(ctypes.c_char_p(None))
        vals.append(ctypes.c_double(0.0))
        names_arr = (ctypes.c_char_p * len(names))(*names)
        vals_arr = (ctypes.c_double * len(vals))(*vals)
        names_str_arr = (ctypes.c_char_p * 1)(None)
        vals_str_arr = (ctypes.c_char_p * 1)(None)
        sim_params = OsdiSimParas(names=names_arr, vals=vals_arr,
                                  names_str=names_str_arr,
                                  vals_str=vals_str_arr)
        return sim_params, names_arr, vals_arr, names_str_arr, vals_str_arr


class OsdiInstance:
    def __init__(self, desc: OsdiDescriptor) -> None:
        if not desc:
            raise RuntimeError("descriptor is null")
        self._desc = desc
        self._buf = AlignedBuffer(int(desc.instance_size))

    def data(self) -> ctypes.c_void_p:
        return self._buf.ptr

    @property
    def descriptor(self) -> OsdiDescriptor:
        return self._desc

    def _ptr_at(self, offset: int, ctype: ctypes._SimpleCData) -> ctypes.POINTER:
        base = self._buf.ptr.value or 0
        addr = base + offset
        return ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctype))

    def process_params(self, model: OsdiModel, connected_terminals: int, temperature: float) -> List[int]:
        sim_params = OsdiSimParas()
        info = OsdiInitInfo()
        self._desc.setup_instance(_INSTANCE_NAME, self._buf.ptr, model.data(),
                                  temperature, connected_terminals,
                                  ctypes.byref(sim_params), ctypes.byref(info))
        _check_init_result(self._desc, info)
        return self._collapse_nodes(connected_terminals)

    def bind_simulation(self, sim: OsdiSimulation, model: OsdiModel,
                        connected_terminals: int, temperature: float) -> None:
        internal_nodes = self.process_params(model, connected_terminals, temperature)
        nodes = self._desc.nodes
        terminal_indices: List[int] = []
        for i in range(connected_terminals):
            node_name = nodes[i].name.decode("utf-8", errors="replace") if nodes[i].name else ""
            terminal_indices.append(sim.register_node(node_name))
        sim.terminal_indices = terminal_indices

        internal_indices: List[int] = []
        for idx in internal_nodes:
            node_name = nodes[idx].name.decode("utf-8", errors="replace") if nodes[idx].name else ""
            internal_indices.append(sim.register_node(node_name))
        sim.internal_indices = internal_indices

        mapping_ptr = self._ptr_at(self._desc.node_mapping_offset, ctypes.c_uint32)
        for i in range(self._desc.num_nodes):
            idx = mapping_ptr[i]
            if idx < len(terminal_indices):
                mapping_ptr[i] = terminal_indices[idx]
            elif idx == UINT32_MAX:
                mapping_ptr[i] = 0
            else:
                internal_idx = idx - len(terminal_indices)
                if internal_idx >= len(internal_indices):
                    mapping_ptr[i] = 0
                else:
                    mapping_ptr[i] = internal_indices[internal_idx]

        for i in range(self._desc.num_jacobian_entries):
            entry = self._desc.jacobian_entries[i]
            row = mapping_ptr[entry.nodes.node_1]
            col = mapping_ptr[entry.nodes.node_2]
            sim.register_jacobian_entry(int(row), int(col))
        sim.build_jacobian()

        ptr_resist = self._ptr_at(self._desc.jacobian_ptr_resist_offset,
                                  ctypes.POINTER(ctypes.c_double))
        ptr_resist = ctypes.cast(ptr_resist, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
        for i in range(self._desc.num_jacobian_entries):
            entry = self._desc.jacobian_entries[i]
            row = mapping_ptr[entry.nodes.node_1]
            col = mapping_ptr[entry.nodes.node_2]
            idx = sim.get_jacobian_entry(int(row), int(col))
            elem_ptr = ctypes.cast(ctypes.byref(sim.jacobian_resist, idx * ctypes.sizeof(ctypes.c_double)),
                                   ctypes.POINTER(ctypes.c_double))
            ptr_resist[i] = elem_ptr
            if entry.react_ptr_off != UINT32_MAX:
                react_ptr_loc = self._ptr_at(entry.react_ptr_off, ctypes.POINTER(ctypes.c_double))
                react_ptr_loc = ctypes.cast(react_ptr_loc, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
                react_ptr = ctypes.cast(ctypes.byref(sim.jacobian_react, idx * ctypes.sizeof(ctypes.c_double)),
                                        ctypes.POINTER(ctypes.c_double))
                react_ptr_loc[0] = react_ptr

        sim.state_prev = sim._make_array(self._desc.num_states)
        sim.state_next = sim._make_array(self._desc.num_states)
        sim.noise_dense = sim._make_array(self._desc.num_noise_src)

    def eval(self, model: OsdiModel, sim: OsdiSimulation, flags: int) -> int:
        return self.eval_with_time(model, sim, flags, 0.0)

    def eval_with_time(self, model: OsdiModel, sim: OsdiSimulation, flags: int, abstime: float) -> int:
        sim_params, names_arr, vals_arr, names_str_arr, vals_str_arr = sim.build_sim_paras()
        sim_info = OsdiSimInfo()
        sim_info.paras = sim_params
        sim_info.abstime = abstime
        sim_info.prev_solve = sim.prev_solve if sim.has_prev_solve else sim.solve
        sim_info.prev_state = sim.state_prev
        sim_info.next_state = sim.state_next
        sim_info.flags = flags
        # Keep arrays alive while eval runs (store on sim to survive across GC cycles).
        sim._keep_alive = (names_arr, vals_arr, names_str_arr, vals_str_arr)
        return self._desc.eval(_INSTANCE_NAME, self._buf.ptr, model.data(), ctypes.byref(sim_info))

    def load_residuals(self, model: OsdiModel, sim: OsdiSimulation) -> None:
        self._desc.load_residual_resist(self._buf.ptr, model.data(), sim.residual_resist)
        self._desc.load_limit_rhs_resist(self._buf.ptr, model.data(), sim.residual_resist)
        self._desc.load_residual_react(self._buf.ptr, model.data(), sim.residual_react)
        self._desc.load_limit_rhs_react(self._buf.ptr, model.data(), sim.residual_react)

    def load_jacobian(self, model: OsdiModel, sim: OsdiSimulation) -> None:
        self._desc.load_jacobian_resist(self._buf.ptr, model.data())
        self._desc.load_jacobian_react(self._buf.ptr, model.data(), 1.0)

    def load_spice_rhs_dc(self, model: OsdiModel, sim: OsdiSimulation) -> None:
        self._desc.load_spice_rhs_dc(self._buf.ptr, model.data(), sim.residual_resist, sim.solve)

    def load_spice_rhs_tran(self, model: OsdiModel, sim: OsdiSimulation, alpha: float) -> None:
        self._desc.load_spice_rhs_tran(self._buf.ptr, model.data(), sim.rhs_tran, sim.prev_solve, alpha)

    def load_jacobian_tran(self, model: OsdiModel, sim: OsdiSimulation, alpha: float) -> None:
        self._desc.load_jacobian_tran(self._buf.ptr, model.data(), alpha)

    def _nr_step(self, sim: OsdiSimulation, residual_vec: List[float]) -> bool:
        """Single Newton-Raphson step: build system, solve, clamp update.

        Returns True if the linear solve succeeded and node voltages were
        updated, False on singular Jacobian.
        """
        internal_pos = {idx: i for i, idx in enumerate(sim.internal_indices)}
        n = len(sim.internal_indices)
        a = np.zeros((n, n), dtype=float)
        b = np.zeros(n, dtype=float)
        for i, idx in enumerate(sim.internal_indices):
            b[i] = -residual_vec[idx]
        for k, (row, col) in enumerate(sim.jacobian_info):
            if row in internal_pos and col in internal_pos:
                a[internal_pos[row], internal_pos[col]] = sim.jacobian_resist[k]
        try:
            delta = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return False
        for i, idx in enumerate(sim.internal_indices):
            update = max(-0.2, min(0.2, float(delta[i])))
            sim.solve[idx] = sim.solve[idx] + update
        return True

    def solve_internal_nodes(self, model: OsdiModel, sim: OsdiSimulation,
                             max_iter: int, tol: float) -> bool:
        if not sim.internal_indices:
            return True
        for _ in range(max_iter):
            flags = (ANALYSIS_DC | ANALYSIS_STATIC | CALC_RESIST_RESIDUAL |
                     CALC_RESIST_JACOBIAN | CALC_RESIST_LIM_RHS |
                     ENABLE_LIM | INIT_LIM)
            self.eval(model, sim, flags)
            sim.clear()
            self.load_residuals(model, sim)
            self.load_jacobian(model, sim)
            norm = max(abs(sim.residual_resist[idx]) for idx in sim.internal_indices)
            if norm < tol:
                return True
            if not self._nr_step(sim, list(sim.residual_resist)):
                return False
        return False

    def solve_internal_nodes_tran(self, model: OsdiModel, sim: OsdiSimulation,
                                  abstime: float, alpha: float,
                                  max_iter: int, tol: float) -> bool:
        if not sim.internal_indices:
            return True
        for _ in range(max_iter):
            flags = (ANALYSIS_TRAN | CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN |
                     CALC_RESIST_LIM_RHS | CALC_REACT_RESIDUAL |
                     CALC_REACT_JACOBIAN | CALC_REACT_LIM_RHS |
                     ENABLE_LIM | INIT_LIM)
            self.eval_with_time(model, sim, flags, abstime)
            sim.clear()
            self.load_residuals(model, sim)
            self.load_jacobian_tran(model, sim, alpha)
            self.load_spice_rhs_tran(model, sim, alpha)
            total_residual = [
                sim.residual_resist[i] + alpha * sim.residual_react[i] - sim.rhs_tran[i]
                for i in range(len(sim.residual_resist))
            ]
            norm = max(abs(total_residual[idx]) for idx in sim.internal_indices)
            if norm < tol:
                return True
            if not self._nr_step(sim, total_residual):
                return False
        return False

    def _collapse_nodes(self, connected_terminals: int) -> List[int]:
        back_map: List[int] = list(range(connected_terminals, int(self._desc.num_nodes)))
        mapping_ptr = self._ptr_at(self._desc.node_mapping_offset, ctypes.c_uint32)
        for i in range(self._desc.num_nodes):
            mapping_ptr[i] = i
        collapsed_ptr = self._ptr_at(self._desc.collapsed_offset, ctypes.c_bool)
        for i in range(self._desc.num_collapsible):
            if not collapsed_ptr[i]:
                continue
            candidate = self._desc.collapsible[i]
            from_node = candidate.node_1
            to_node = candidate.node_2
            mapped_from = mapping_ptr[from_node]
            collapse_to_gnd = (to_node == UINT32_MAX)
            mapped_to = UINT32_MAX if collapse_to_gnd else mapping_ptr[to_node]
            if not collapse_to_gnd and mapped_to == UINT32_MAX:
                collapse_to_gnd = True
            if mapped_from < connected_terminals and (collapse_to_gnd or mapped_to < connected_terminals):
                continue
            if not collapse_to_gnd and mapped_from < mapped_to:
                mapped_from, mapped_to = mapped_to, mapped_from
            for j in range(self._desc.num_nodes):
                val = mapping_ptr[j]
                if val == mapped_from:
                    mapping_ptr[j] = mapped_to
                elif val > mapped_from and val != UINT32_MAX:
                    mapping_ptr[j] = val - 1
            if mapped_from >= connected_terminals:
                remove_idx = mapped_from - connected_terminals
                if 0 <= remove_idx < len(back_map):
                    back_map.pop(remove_idx)
        return back_map


# ---------------------------------------------------------------------------
# Parameter application
# ---------------------------------------------------------------------------

def apply_param(desc: OsdiDescriptor,
                inst: Optional[OsdiInstance],
                model: Optional[OsdiModel],
                name: str,
                value: float,
                from_modelcard: bool) -> bool:
    applied = False
    for i in range(desc.num_params):
        param = desc.param_opvar[i]
        param_name = param.name[0].decode("utf-8", errors="replace") if param.name else ""
        if not param_name or name.lower() != param_name.lower():
            continue
        ptr = None
        if (param.flags & PARA_KIND_MASK) == PARA_KIND_INST:
            if inst is None:
                return False
            ptr = desc.access(inst.data(), model.data() if model else None, i,
                              ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE)
        else:
            if model is None:
                return False
            ptr = desc.access(None, model.data(), i, ACCESS_FLAG_SET)
        if not ptr:
            raise RuntimeError(f"invalid parameter access for {name}")
        ty = param.flags & PARA_TY_MASK
        if ty == PARA_TY_INT:
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))[0] = int(value)
            applied = True
        elif ty == PARA_TY_REAL:
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0] = float(value)
            applied = True
        else:
            if not from_modelcard:
                raise RuntimeError(f"string parameter not supported: {name}")
            applied = True
        break
    if not applied and not from_modelcard:
        raise RuntimeError(f"parameter not found: {name}")
    return applied

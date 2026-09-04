"""Authoritative inventory for NN compact-model simple-circuit evaluation.

This module is deliberately stdlib-only.  Campaign enumeration, coverage,
reporting, CLIs, and tests all import the same immutable case descriptions
without importing Torch, a simulator, or a PDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


SIMPLE_V1 = "simple-v1"
SIMPLE_V2 = "simple-v2"
QUALIFICATION = "qualification"
DIAGNOSTIC = "diagnostic"

_PROFILE_HEADLINES: Dict[str, str] = {
    "common_source_ac": "bandwidth_error_pct",
    "common_source_floating_ac": "bulk_response_max_error_v",
    "inverter_vtc": "trip_shift_v",
    "inverter_energy": "energy_error_pct",
    "active_load_op": "output_error_v",
    "active_load_ac": "bandwidth_error_pct",
    "source_follower": "gain_error_pct",
    "gain": "gain_error_pct",
    "opamp": "gain_error_pct",
    "ring_osc": "period_error_pct",
    "current_mirror": "ratio_error_pct",
    "mirror_iref": "iref_worst_ratio_error_pct",
    "inverter_chain": "delay_error_pct",
    "transmission_gate": "ron_error_pct",
    "hold_droop": "droop_error_v",
    "diffpair": "diff_gain_error_pct",
    "diffpair_diff_ac": "diff_gain_error_pct",
    "diffpair_cm_ac": "cm_gain_error_pct",
    "cascode": "output_resistance_error_pct",
    "logic_vtc": "trip_shift_v",
    "logic_tran": "delay_error_pct",
    "sram_hold": "hold_margin_error_v",
    "sram_read": "read_disturb_error_v",
    "sram_write": "write_time_error_pct",
    "sram_write_margin": "write_trip_error_v",
    "sram_snm": "nrmse_pct",
    "opamp_diff_ac": "diff_gain_error_pct",
    "opamp_cm_ac": "cm_gain_error_pct",
    "opamp_supply_ac": "supply_gain_error_pct",
    "switchcap": "charge_error_vdd_pct",
    "switchcap_multicycle": "cycle_drift_error_v",
    "ring_supply": "period_error_pct",
    "diode_load": "diode_drop_worst_error_v",
    "bias_op": "bias_node_error_v",
    "bias_fanout_op": "bias_node_error_v",
    "self_bias_cell": "bias_current_error_pct",
    "self_bias_cascode": "output_resistance_error_pct",
    "mos_reference": "line_sensitivity_error_pct",
    "unity_gain": "follow_error_v",
    "settling": "settling_error_pct",
    "line_regulation": "line_regulation_error_pct",
    "load_regulation": "load_droop_error_v",
    "closed_loop_ac": "bandwidth_error_pct",
    "ldo_psrr_ac": "psrr_error_db",
    "ldo_output_impedance_ac": "output_impedance_error_pct",
}


@dataclass(frozen=True)
class AnalysisSpec:
    """One experiment rendered twice from a canonical circuit template."""

    name: str
    kind: str
    card: str
    signals: Tuple[str, ...]
    metric_profile: str = "trace"
    phase_align: bool = False
    template_substitutions: Tuple[Tuple[str, str], ...] = ()
    device_kinds: Tuple[str, ...] = ("nmos", "pmos")

    def substitutions(self) -> Dict[str, str]:
        """Return experiment-specific values for the shared template."""
        return dict(self.template_substitutions)

    @property
    def headline_metric(self) -> str:
        """Metric that answers this analysis's primary diagnostic question."""
        return _PROFILE_HEADLINES.get(self.metric_profile, "nrmse_pct")


@dataclass(frozen=True)
class DeviceRoleSpec:
    """One independently bindable geometry/VT role in a circuit template."""

    name: str
    token: str
    polarity: str
    instances: Tuple[str, ...]
    nfin_delta: int = 0
    length_m: Optional[float] = None
    vt: str = ""

    def __post_init__(self) -> None:
        if self.polarity not in {"nmos", "pmos"}:
            raise ValueError(f"unknown role polarity {self.polarity!r}")
        if not self.name or not self.token or not self.instances:
            raise ValueError("device roles require a name, token, and instances")


@dataclass(frozen=True)
class CircuitCase:
    """A simple topology and the evidence contract attached to it."""

    case_id: str
    label: str
    template: str
    score_version: str
    role: str
    analyses: Tuple[AnalysisSpec, ...]
    #: Difficulty tier owning ``template`` under ``circuit_templates/``.  This is what
    #: the case demands of a compact model, not what it is used for, so the
    #: tier is declared here and verified against the file's location rather
    #: than inferred from it.
    tier: str = ""
    device_kinds: Tuple[str, ...] = ("nmos", "pmos")
    device_roles: Tuple[DeviceRoleSpec, ...] = ()
    omp_threads: Tuple[int, ...] = (1,)
    required_metrics: Tuple[str, ...] = (
        "mre_pct", "r2", "nrmse_pct", "max_err",
    )
    report_key: str = ""
    suite_id: str = ""
    training_use: str = "held_out"
    gate_metric: str = ""
    gate_condition: str = ""
    report_label: str = ""
    report_gate_text: str = ""
    #: Quantities formed by combining two or more of this case's analyses.
    #: A rejection ratio is the motivating example: CMRR needs the
    #: differential and common-mode gains, which no single analysis produces.
    derived_metrics: Tuple[str, ...] = ()
    control_nrmse_limit_pct: float = 1.0

    @property
    def campaign_suite(self) -> str:
        if self.suite_id:
            return self.suite_id
        return f"verify_circuit_topologies__{self.case_id}"

    @property
    def result_key(self) -> str:
        return self.report_key or self.case_id


def _dc(
    name: str,
    card: str,
    signals: Tuple[str, ...],
    profile: str = "trace",
    *,
    subs: Tuple[Tuple[str, str], ...] = (),
    device_kinds: Tuple[str, ...] = ("nmos", "pmos"),
) -> AnalysisSpec:
    return AnalysisSpec(
        name=name, kind="dc", card=card, signals=signals,
        metric_profile=profile, template_substitutions=subs,
        device_kinds=device_kinds,
    )


def _tran(
    name: str,
    card: str,
    signals: Tuple[str, ...],
    profile: str = "transient",
    *,
    phase_align: bool = False,
    subs: Tuple[Tuple[str, str], ...] = (),
    device_kinds: Tuple[str, ...] = ("nmos", "pmos"),
) -> AnalysisSpec:
    return AnalysisSpec(
        name=name, kind="tran", card=card, signals=signals,
        metric_profile=profile, phase_align=phase_align,
        template_substitutions=subs, device_kinds=device_kinds,
    )


def _ac(
    name: str,
    signals: Tuple[str, ...],
    profile: str = "ac",
    *,
    subs: Tuple[Tuple[str, str], ...] = (),
    card: str = "ac dec 12 1e3 1e11",
    device_kinds: Tuple[str, ...] = ("nmos", "pmos"),
) -> AnalysisSpec:
    """One AC experiment.

    The default decade grid suits resistively loaded stages.  A high-gain
    Miller stage needs a lower start frequency, because its dominant pole can
    sit below 1 kHz and a grid that begins above it never samples the DC gain
    the rejection ratios are formed from.
    """
    return AnalysisSpec(
        name=name, kind="ac", card=card, signals=signals,
        metric_profile=profile, template_substitutions=subs,
        device_kinds=device_kinds,
    )


def _op(
    name: str,
    signals: Tuple[str, ...],
    profile: str = "trace",
    *,
    subs: Tuple[Tuple[str, str], ...] = (),
    device_kinds: Tuple[str, ...] = ("nmos", "pmos"),
) -> AnalysisSpec:
    return AnalysisSpec(
        name=name,
        kind="op",
        card="op",
        signals=signals,
        metric_profile=profile,
        template_substitutions=subs,
        device_kinds=device_kinds,
    )


#: AC grid for high-gain stages: low enough to sample the DC gain of a
#: Miller-compensated opamp, wide enough to keep the unity-gain crossing.
_OPAMP_AC_CARD = "ac dec 10 10 1e11"


_V1_CASES: Tuple[CircuitCase, ...] = (
    CircuitCase(
        "ring_osc", "5-stage ring oscillator",
        "ring_oscillator.spice.tmpl",
        SIMPLE_V1, QUALIFICATION,
        (_tran("oscillation", "tran 2p 1.2n uic", ("v(n5)",),
               "ring_osc", phase_align=True),),
        tier="L3_blocks",
        omp_threads=(1, 2, 4), report_key="ring_osc",
        suite_id="verify_circuit_ring_osc",
        training_use="qualification_only",
        gate_metric="period error %",
        gate_condition="≤5 %",
        report_label="Ring oscillator",
        report_gate_text="period error %, gate ≤5 %",
    ),
    CircuitCase(
        "opamp", "two-stage Miller opamp",
        "opamp_miller.spice.tmpl",
        SIMPLE_V1, QUALIFICATION,
        (_dc("transfer", "dc Vinp <OPAMP_LO> <OPAMP_HI> 0.002",
             ("v(vout)",), "opamp"),),
        tier="L3_blocks",
        omp_threads=(1, 2, 4), report_key="opamp",
        suite_id="verify_circuit_opamp",
        training_use="qualification_only",
        gate_metric="open-loop gain error %",
        gate_condition="≤10 %",
        report_label="Two-stage Miller opamp (DC)",
        report_gate_text="open-loop gain error %, gate ≤10 %",
    ),
    CircuitCase(
        "sram_snm", "6T SRAM read-SNM half-cell",
        "sram_snm_half_cell.spice.tmpl",
        SIMPLE_V1, QUALIFICATION,
        (_dc("read_snm", "dc Vq 0 <VDD> 0.005", ("v(qb)",),
             "sram_snm"),),
        tier="L2_stages",
        report_key="sram_snm", suite_id="verify_circuit_sram_snm",
        training_use="qualification_only",
        gate_metric="worst lobe NRMSE %",
        gate_condition="≤10 % and every lobe positive",
        report_label="6T SRAM read SNM",
        report_gate_text=(
            "worst lobe NRMSE %, gate ≤10 % and all lobes positive"
        ),
    ),
    CircuitCase(
        "switchcap", "switched-capacitor sample/hold",
        "switched_capacitor.spice.tmpl",
        SIMPLE_V1, QUALIFICATION,
        (_tran("sample_hold", "tran 5p 12n uic", ("v(vsamp)",),
               "switchcap"),),
        tier="L3_blocks",
        report_key="switchcap", suite_id="verify_circuit_switchcap",
        training_use="qualification_only",
        gate_metric="charge error % of VDD",
        gate_condition=(
            "≤5 % and droop error within max(10 % of reference, 0.1 % VDD)"
        ),
        report_label="Switched-capacitor cell",
        report_gate_text="charge error % of VDD, gate ≤5 %",
    ),
)


_V2_CASES: Tuple[CircuitCase, ...] = (
    CircuitCase(
        "common_source_nn", "common-source terminal and floating-bulk AC",
        "common_source.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _ac(
                "nmos_fixed", ("v(out)",), "common_source_ac",
                subs=(
                    ("INPUT_SPEC", "DC=<GATE_N> AC=1 0"),
                    ("LOAD_NETWORK", "Rd vdd out <CS_LOAD>"),
                    ("BULK_NETWORK", ""),
                    ("DEVICE_PREFIX", "<N_PREFIX>n"),
                    ("SOURCE_NODE", "0"), ("BULK_NODE", "0"),
                    ("DEVICE", "<N_DEVICE>"),
                ),
                device_kinds=("nmos",),
            ),
            _ac(
                "pmos_fixed", ("v(out)",), "common_source_ac",
                subs=(
                    ("INPUT_SPEC", "DC=<GATE_P> AC=1 0"),
                    ("LOAD_NETWORK", "Rd out 0 <CS_LOAD>"),
                    ("BULK_NETWORK", ""),
                    ("DEVICE_PREFIX", "<P_PREFIX>p"),
                    ("SOURCE_NODE", "vdd"), ("BULK_NODE", "vdd"),
                    ("DEVICE", "<P_DEVICE>"),
                ),
                device_kinds=("pmos",),
            ),
            _ac(
                "nmos_floating", ("v(out)", "v(vb)"),
                "common_source_floating_ac",
                subs=(
                    ("INPUT_SPEC", "DC=<GATE_N> AC=1 0"),
                    ("LOAD_NETWORK", "Rd vdd out <CS_LOAD>"),
                    ("BULK_NETWORK", "Rb vb 0 <BULK_LOAD>"),
                    ("DEVICE_PREFIX", "<N_PREFIX>n"),
                    ("SOURCE_NODE", "0"), ("BULK_NODE", "vb"),
                    ("DEVICE", "<N_DEVICE>"),
                ),
                device_kinds=("nmos",),
            ),
            _ac(
                "pmos_floating", ("v(out)", "v(vb)"),
                "common_source_floating_ac",
                subs=(
                    ("INPUT_SPEC", "DC=<GATE_P> AC=1 0"),
                    ("LOAD_NETWORK", "Rd out 0 <CS_LOAD>"),
                    ("BULK_NETWORK", "Rb vb vdd <BULK_LOAD>"),
                    ("DEVICE_PREFIX", "<P_PREFIX>p"),
                    ("SOURCE_NODE", "vdd"), ("BULK_NODE", "vb"),
                    ("DEVICE", "<P_DEVICE>"),
                ),
                device_kinds=("pmos",),
            ),
        ),
        tier="L1_primitives",
        device_kinds=(),
        required_metrics=(
            "mre_pct", "r2", "nrmse_pct", "max_err",
            "gain_error_pct", "bandwidth_error_pct",
            "bulk_response_max_error_v",
        ),
    ),
    CircuitCase(
        "inverter_energy", "inverter VTC, switching energy, and leakage",
        "inverter.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc(
                "vtc", "dc Vin 0 <VDD> 0.005",
                ("v(out)", "i(Vdd)"), "inverter_vtc",
                subs=(
                    ("INPUT_SPEC", "0"),
                    ("OUTPUT_LOAD", "Cload out 0 <LOGIC_LOAD>"),
                    ("INITIAL_CONDITION", ""),
                ),
            ),
            _tran(
                "switching", "tran 2p 4n uic",
                ("v(in)", "v(out)", "i(Vdd)"), "inverter_energy",
                subs=(
                    ("INPUT_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                     "<INPUT_RISE> <INPUT_FALL> <INPUT_WIDTH> "
                     "<INPUT_PERIOD> <PULSE_CLOSE>"),
                    ("OUTPUT_LOAD", "Cload out 0 <LOGIC_LOAD>"),
                    ("INITIAL_CONDITION", ".ic V(out)=<VDD>"),
                ),
            ),
        ),
        tier="L2_stages",
        device_kinds=(),
        required_metrics=(
            "mre_pct", "r2", "nrmse_pct", "max_err", "trip_shift_v",
            "leakage_error_a", "delay_error_pct", "energy_error_pct",
        ),
    ),
    CircuitCase(
        "source_follower", "complementary source followers with body effect",
        "source_follower.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos", "dc Vgn 0 <VDD> 0.005", ("v(sn)", "i(Vddn)"),
                "source_follower", device_kinds=("nmos",)),
            _dc("pmos", "dc Vgp <VDD> 0 -0.005", ("v(sp)", "i(Vddp)"),
                "source_follower", device_kinds=("pmos",)),
        ),
        tier="L1_primitives",
    ),
    CircuitCase(
        "common_gate", "complementary common-gate stages",
        "common_gate.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos", "dc Vsn 0 <HALF_VDD> 0.005",
                ("v(dn)", "i(Vsn)"), "gain", device_kinds=("nmos",)),
            _dc("pmos", "dc Vsp <VDD> <HALF_VDD> -0.005",
                ("v(dp)", "i(Vsp)"), "gain", device_kinds=("pmos",)),
        ),
        tier="L1_primitives",
    ),
    CircuitCase(
        "current_mirror", "NMOS and PMOS two-device current mirrors",
        "current_mirror.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos", "dc Voutn 0 <VDD> 0.005",
                ("i(Voutn)",), "current_mirror", device_kinds=("nmos",)),
            _dc("pmos", "dc Voutp <VDD> 0 -0.005",
                ("i(Voutp)",), "current_mirror", device_kinds=("pmos",)),
            # Mirror accuracy versus bias current.  The compliance sweeps above
            # score one operating current; this one walks the reference from
            # deep subthreshold up through strong inversion, where the ratio
            # error of a fitted model is largest and least constrained.
            _dc("nmos_iref", "dc Irefn <IREF_LO> <IREF_HI> <IREF_STEP>",
                ("i(Voutn)",), "mirror_iref",
                subs=(("MIRROR_OUT_N", "<MID_RAIL>"),),
                device_kinds=("nmos",)),
            _dc("pmos_iref", "dc Irefp <IREF_LO> <IREF_HI> <IREF_STEP>",
                ("i(Voutp)",), "mirror_iref",
                subs=(("MIRROR_OUT_P", "<MID_RAIL>"),),
                device_kinds=("pmos",)),
        ),
        tier="L2_stages",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "ratio_error_pct", "output_resistance_error_pct",
                          "iref_ratio_error_pct", "iref_worst_ratio_error_pct"),
    ),
    CircuitCase(
        "inverter_chain", "open fanout-of-four inverter chain",
        "inverter_chain.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (_tran("fo4", "tran 2p 4n uic", ("v(in)", "v(n1)", "v(o0)"),
               "inverter_chain", phase_align=True),),
        tier="L2_stages",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "delay_error_pct", "rise_fall_error_pct",
                          "amplitude_error_pct",
                          "phase_aligned_nrmse_pct"),
    ),
    CircuitCase(
        "transmission_gate_dc", "bidirectional transmission gate",
        "transmission_gate_dc.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("forward", "dc Vinf 0 <VDD> 0.005",
                ("v(outf)", "i(Vinf)"), "transmission_gate"),
            _dc("reverse", "dc Vrev 0 <VDD> 0.005",
                ("v(inr)", "i(Vrev)"), "transmission_gate"),
        ),
        tier="L2_stages",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "ron_error_pct"),
    ),
    CircuitCase(
        "transmission_gate_hold", "transmission-gate hold and feedthrough",
        "transmission_gate_hold.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (_tran("hold", "tran 2p 4n uic", ("v(hold)", "v(phi)"),
               "hold_droop"),),
        tier="L3_blocks",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "droop_error_v", "feedthrough_error_v"),
    ),
    CircuitCase(
        "diffpair_ideal", "resistor-loaded differential pair, ideal tail",
        "diffpair_ideal.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("steering", "dc Vinp <DIFF_LO> <DIFF_HI> 0.001",
                ("v(outn)", "v(outp)"), "diffpair",
                device_kinds=("nmos",)),
            _ac("differential_ac", ("v(outn)", "v(outp)"),
                "diffpair_diff_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "0")),
                device_kinds=("nmos",)),
            _ac("common_mode_ac", ("v(outn)", "v(outp)"),
                "diffpair_cm_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "1")),
                device_kinds=("nmos",)),
        ),
        tier="L2_stages",
        device_kinds=("nmos",),
        derived_metrics=("cmrr",),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "diff_gain_error_pct", "cm_gain_error_pct",
                          "cmrr_db_error"),
    ),
    CircuitCase(
        "diffpair_active", "resistor-loaded differential pair, active tail",
        "diffpair_active.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("steering", "dc Vinp <DIFF_LO> <DIFF_HI> 0.001",
                ("v(outn)", "v(outp)"), "diffpair",
                device_kinds=("nmos",)),
            _ac("differential_ac", ("v(outn)", "v(outp)"),
                "diffpair_diff_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "0")),
                device_kinds=("nmos",)),
            _ac("common_mode_ac", ("v(outn)", "v(outp)"),
                "diffpair_cm_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "1")),
                device_kinds=("nmos",)),
        ),
        tier="L2_stages",
        device_kinds=("nmos",),
        derived_metrics=("cmrr",),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "diff_gain_error_pct", "cm_gain_error_pct",
                          "cmrr_db_error"),
    ),
    CircuitCase(
        "diffpair_active_load", "differential stages with active mirror loads",
        "diffpair_active_load.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _op(
                "nmos_steering", ("v(nout)", "v(nmirror)"),
                "active_load_op",
                subs=(("ACTIVE_LOAD_STAGE", "<N_ACTIVE_LOAD_STAGE>"),
                      ("N_INP_DC", "<N_STEER_INP>"),
                      ("N_INN_DC", "<VCM>"),
                      ("N_AC_INP", "0"), ("N_AC_INN", "0"),
                      ("P_AC_INP", "0"), ("P_AC_INN", "0")),
            ),
            _op(
                "pmos_steering", ("v(pout)", "v(pmirror)"),
                "active_load_op",
                subs=(("ACTIVE_LOAD_STAGE", "<P_ACTIVE_LOAD_STAGE>"),
                      ("P_INP_DC", "<P_STEER_INP>"),
                      ("P_INN_DC", "<P_VCM>"),
                      ("N_AC_INP", "0"), ("N_AC_INN", "0"),
                      ("P_AC_INP", "0"), ("P_AC_INN", "0")),
            ),
            _ac(
                "nmos_ac", ("v(nout)",), "active_load_ac",
                subs=(("ACTIVE_LOAD_STAGE", "<N_ACTIVE_LOAD_STAGE>"),
                      ("N_AC_INP", "1"), ("N_AC_INN", "0"),
                      ("P_AC_INP", "0"), ("P_AC_INN", "0")),
            ),
            _ac(
                "pmos_ac", ("v(pout)",), "active_load_ac",
                subs=(("ACTIVE_LOAD_STAGE", "<P_ACTIVE_LOAD_STAGE>"),
                      ("N_AC_INP", "0"), ("N_AC_INN", "0"),
                      ("P_AC_INP", "1"), ("P_AC_INN", "0")),
            ),
        ),
        tier="L2_stages",
        device_roles=(
            DeviceRoleSpec(
                "n_input", "N_INPUT_DEVICE", "nmos",
                ("n_in_l", "n_in_r"), nfin_delta=1,
            ),
            DeviceRoleSpec(
                "p_load", "P_LOAD_DEVICE", "pmos",
                ("p_load_d", "p_load_o"),
            ),
            DeviceRoleSpec(
                "p_input", "P_INPUT_DEVICE", "pmos",
                ("p_in_l", "p_in_r"), nfin_delta=1,
            ),
            DeviceRoleSpec(
                "n_load", "N_LOAD_DEVICE", "nmos",
                ("n_load_d", "n_load_o"),
            ),
        ),
        required_metrics=(
            "mre_pct", "r2", "nrmse_pct", "max_err", "output_error_v",
            "internal_node_nrmse_pct", "bandwidth_error_pct",
        ),
    ),
    CircuitCase(
        "cascode_stack", "complementary cascode/current-source stacks",
        "cascode_stack.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos_compliance", "dc Voutn 0 <VDD> 0.005",
                ("i(Voutn)", "v(nx)"), "cascode",
                device_kinds=("nmos",)),
            _dc("pmos_compliance", "dc Voutp <VDD> 0 -0.005",
                ("i(Voutp)", "v(px)"), "cascode",
                device_kinds=("pmos",)),
            _ac("gain", ("v(nac)", "v(pac)"), "cascode_ac"),
        ),
        tier="L2_stages",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "output_resistance_error_pct"),
    ),
    CircuitCase(
        "nand2", "two-input CMOS NAND",
        "nand2.spice.tmpl", SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("input_a", "dc Va 0 <VDD> 0.005",
                ("v(out)", "v(nint)"), "logic_vtc",
                subs=(("VA_SPEC", "0"), ("VB_SPEC", "<VDD>"))),
            _dc("input_b", "dc Vb 0 <VDD> 0.005",
                ("v(out)", "v(nint)"), "logic_vtc",
                subs=(("VA_SPEC", "<VDD>"), ("VB_SPEC", "0"))),
            _tran("transient", "tran 2p 4n uic",
                  ("v(a)", "v(out)", "v(nint)"), "logic_tran",
                  subs=(("VA_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                                    "<INPUT_RISE> <INPUT_FALL> <INPUT_WIDTH> "
                                    "<INPUT_PERIOD> <PULSE_CLOSE>"),
                        ("VB_SPEC", "<VDD>"))),
        ),
        tier="L2_stages",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "trip_shift_v", "delay_error_pct",
                          "internal_node_nrmse_pct"),
    ),
    CircuitCase(
        "nor2", "two-input CMOS NOR",
        "nor2.spice.tmpl", SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("input_a", "dc Va 0 <VDD> 0.005",
                ("v(out)", "v(pint)"), "logic_vtc",
                subs=(("VA_SPEC", "0"), ("VB_SPEC", "0"))),
            _dc("input_b", "dc Vb 0 <VDD> 0.005",
                ("v(out)", "v(pint)"), "logic_vtc",
                subs=(("VA_SPEC", "0"), ("VB_SPEC", "0"))),
            _tran("transient", "tran 2p 4n uic",
                  ("v(a)", "v(out)", "v(pint)"), "logic_tran",
                  subs=(("VA_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                                    "<INPUT_RISE> <INPUT_FALL> <INPUT_WIDTH> "
                                    "<INPUT_PERIOD> <PULSE_CLOSE>"),
                        ("VB_SPEC", "0"))),
        ),
        tier="L2_stages",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "trip_shift_v", "delay_error_pct",
                          "internal_node_nrmse_pct"),
    ),
    CircuitCase(
        "sram6t_modes", "full 6T SRAM hold/read/write modes",
        "sram6t_modes.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _tran("hold", "tran 2p 3n uic", ("v(q)", "v(qb)"), "sram_hold",
                  subs=(("WL_SPEC", "0"), ("BL_SPEC", "<VDD>"),
                        ("BLB_SPEC", "<VDD>"))),
            _tran("hold_state0", "tran 2p 3n uic",
                  ("v(q)", "v(qb)"), "sram_hold",
                  subs=(("WL_SPEC", "0"), ("BL_SPEC", "<VDD>"),
                        ("BLB_SPEC", "<VDD>"), ("Q_IC", "0"),
                        ("QB_IC", "<VDD>"))),
            _tran("read", "tran 2p 3n uic", ("v(q)", "v(qb)"), "sram_read",
                  subs=(("WL_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                                    "<INPUT_RISE> <INPUT_FALL> <SRAM_WL_WIDTH> "
                                    "<SRAM_WL_PERIOD> <PULSE_CLOSE>"),
                        ("BL_SPEC", "<VDD>"), ("BLB_SPEC", "<VDD>"))),
            _tran("read_state0", "tran 2p 3n uic",
                  ("v(q)", "v(qb)"), "sram_read",
                  subs=(("WL_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                                    "<INPUT_RISE> <INPUT_FALL> <SRAM_WL_WIDTH> "
                                    "<SRAM_WL_PERIOD> <PULSE_CLOSE>"),
                        ("BL_SPEC", "<VDD>"), ("BLB_SPEC", "<VDD>"),
                        ("Q_IC", "0"), ("QB_IC", "<VDD>"))),
            _tran("write", "tran 2p 3n uic", ("v(q)", "v(qb)"), "sram_write",
                  subs=(("WL_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                                    "<INPUT_RISE> <INPUT_FALL> <SRAM_WL_WIDTH> "
                                    "<SRAM_WL_PERIOD> <PULSE_CLOSE>"),
                        ("BL_SPEC", "0"), ("BLB_SPEC", "<VDD>"))),
            _tran("write_state0", "tran 2p 3n uic",
                  ("v(qb)", "v(q)"), "sram_write",
                  subs=(("WL_SPEC", "<PULSE_OPEN> 0 <VDD> <INPUT_DELAY> "
                                    "<INPUT_RISE> <INPUT_FALL> <SRAM_WL_WIDTH> "
                                    "<SRAM_WL_PERIOD> <PULSE_CLOSE>"),
                        ("BL_SPEC", "<VDD>"), ("BLB_SPEC", "0"),
                        ("Q_IC", "0"), ("QB_IC", "<VDD>"))),
            # Write margin: hold the wordline on and walk the bitline down
            # until the cell flips.  The trip point is a property of the
            # latch's regenerative loop, so it moves with the model's
            # cross-coupled gain rather than with any single device curve.
            _dc("write_margin", "dc Vbl <VDD> 0 -0.005",
                ("v(q)", "v(qb)"), "sram_write_margin",
                subs=(("WL_SPEC", "<VDD>"), ("BL_SPEC", "<VDD>"),
                      ("BLB_SPEC", "<VDD>"))),
            _dc("write_margin_state0", "dc Vblb <VDD> 0 -0.005",
                ("v(qb)", "v(q)"), "sram_write_margin",
                subs=(("WL_SPEC", "<VDD>"), ("BL_SPEC", "<VDD>"),
                      ("BLB_SPEC", "<VDD>"), ("Q_IC", "0"),
                      ("QB_IC", "<VDD>"))),
        ),
        tier="L3_blocks",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "hold_margin_error_v", "read_disturb_error_v",
                          "write_time_error_pct", "write_final_error_v",
                          "retention", "write_trip_error_v"),
    ),
    CircuitCase(
        "opamp_rejection", "Miller opamp common-mode and supply rejection",
        "opamp_miller.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _ac("differential_ac", ("v(vout)",), "opamp_diff_ac",
                card=_OPAMP_AC_CARD,
                subs=(("VINP_SPEC", "DC=<VCM> AC=1 0"),
                      ("VINN_SPEC", "DC=<VCM> AC=0 0"),
                      ("VDD_SPEC", "DC=<VDD> AC=0 0"))),
            _ac("common_mode_ac", ("v(vout)",), "opamp_cm_ac",
                card=_OPAMP_AC_CARD,
                subs=(("VINP_SPEC", "DC=<VCM> AC=1 0"),
                      ("VINN_SPEC", "DC=<VCM> AC=1 0"),
                      ("VDD_SPEC", "DC=<VDD> AC=0 0"))),
            _ac("supply_ac", ("v(vout)",), "opamp_supply_ac",
                card=_OPAMP_AC_CARD,
                subs=(("VINP_SPEC", "DC=<VCM> AC=0 0"),
                      ("VINN_SPEC", "DC=<VCM> AC=0 0"),
                      ("VDD_SPEC", "DC=<VDD> AC=1 0"))),
        ),
        tier="L3_blocks",
        # The opamp ties every bulk to a rail, so there is no body terminal
        # for the body-bias corner to move.
        device_kinds=(),
        derived_metrics=("cmrr", "psrr"),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "cmrr_db_error", "psrr_db_error"),
    ),
    CircuitCase(
        "switchcap_multicycle", "switched-capacitor accumulated charge error",
        "switched_capacitor.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        # Ten clock periods rather than the scored cell's three.  A per-cycle
        # charge error that is inside threshold on the first sample can still
        # accumulate; a three-cycle window cannot distinguish the two.
        (_tran("accumulate", "tran 5p 40n uic",
               ("v(vsamp)", "v(phi)"), "switchcap_multicycle"),),
        tier="L3_blocks",
        device_kinds=(),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "cycle_drift_error_v", "final_sample_error_v"),
    ),
    CircuitCase(
        "ring_osc_supply", "ring oscillator period and supply current",
        "ring_oscillator.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        # The scored ring cell measures one period at nominal supply.  This
        # diagnostic adds the dynamic supply current — a charge-model
        # observable no other case reports — and is meant to be run across the
        # vdd_low/vdd_high corners, where the published ring failures live.
        (_tran("oscillation", "tran 2p 3n uic", ("v(n5)", "i(Vdd)"),
               "ring_supply", phase_align=True),),
        tier="L3_blocks",
        device_kinds=(),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "period_error_pct", "supply_current_error_pct"),
    ),

    CircuitCase(
        "diode_load", "resistively fed diode-connected devices",
        "diode_load.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        # A supply ramp over the smallest circuit whose operating current is
        # not imposed by the deck.  This is the bottom rung of the self-bias
        # ladder: if it fails, nothing above it can be read as a composition
        # effect.
        (
            _op("op_nominal", ("v(nr)", "v(np)"), "diode_load"),
            _dc("supply_ramp", "dc Vdd 0 <VDD> 0.005",
                ("v(nr)", "v(np)"), "diode_load"),
            _dc("supply_down", "dc Vdd <VDD> 0 -0.005",
                ("v(nr)", "v(np)"), "diode_load"),
            _dc("load_low", "dc Vdd 0 <VDD> 0.005",
                ("v(nr)", "v(np)"), "diode_load",
                subs=(("DIODE_RLOAD", "20k"),)),
            _dc("load_high", "dc Vdd 0 <VDD> 0.005",
                ("v(nr)", "v(np)"), "diode_load",
                subs=(("DIODE_RLOAD", "2e6"),)),
        ),
        tier="L1_primitives",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "diode_drop_error_v"),
    ),

    # -- Tier A: the model determines the operating point -----------------
    # Every case above biases its devices from ideal sources, so the fixed
    # point is pinned by the deck.  These three remove that support: the
    # solution is whatever the compact model says it is, which is the
    # mechanism the V7.6.4 closure loop isolated and could not reach.
    CircuitCase(
        "bias_tree_fanout", "generated bias rail with scalable mirror fanout",
        "bias_tree_fanout.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        tuple(
            _op(
                f"fanout_{count}", ("v(nbias)", "i(Vdd)"),
                "bias_fanout_op",
                subs=(("BIAS_BRANCHES", f"<BIAS_BRANCHES_{count}>"),
                      ("BIAS_FANOUT_IC", f"<BIAS_FANOUT_IC_{count}>"),
                      ("BIAS_CURRENT_SPEC", "<IBIAS>")),
                device_kinds=("nmos",),
            )
            for count in (2, 4, 8, 16)
        ),
        tier="L3_blocks",
        device_kinds=("nmos",),
        required_metrics=(
            "mre_pct", "r2", "nrmse_pct", "max_err",
            "bias_node_error_v", "supply_current_error_pct",
        ),
    ),
    CircuitCase(
        "beta_multiplier", "constant-gm self-biased current reference",
        "beta_multiplier.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        # A supply ramp is the start-up trajectory: the cell has a degenerate
        # zero-current solution, so this sweep asks whether the candidate
        # leaves it at the same supply the reference does.
        (
            _op("op_nominal", ("v(na)", "v(nb)", "i(Vdd)"), "bias_op"),
            _dc("supply_ramp", "dc Vdd 0 <VDD> 0.005",
                ("v(na)", "v(nb)", "i(Vdd)"), "self_bias_cell"),
            _dc("supply_down", "dc Vdd <VDD> 0 -0.005",
                ("v(na)", "v(nb)", "i(Vdd)"), "trace"),
        ),
        tier="L3_blocks",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "bias_current_error_pct", "startup_vdd_error_v",
                          "bias_node_error_v"),
    ),
    CircuitCase(
        "self_biased_cascode", "cascode with an internally generated rail",
        "self_biased_cascode.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (_dc("nmos_compliance", "dc Voutn 0 <VDD> 0.005",
             ("i(Voutn)", "v(nx)", "v(nc)", "v(nb)"), "self_bias_cascode",
             device_kinds=("nmos",)),),
        tier="L3_blocks",
        device_kinds=("nmos",),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "output_resistance_error_pct", "bias_node_error_v"),
    ),
    CircuitCase(
        "mos_ratio_reference", "complementary MOS-referenced voltages",
        "diode_load.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (_dc("supply_ramp", "dc Vdd 0 <VDD> 0.005",
             ("v(nr)", "v(np)"), "mos_reference",
             subs=(("DIODE_RLOAD", "<REF_RBIAS>"),)),),
        tier="L1_primitives",
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "vref_error_v", "line_sensitivity_error_pct"),
    ),

    # -- Tier B: closed negative-feedback systems --------------------------
    # Each of these runs at least one transient WITHOUT `uic`, so the
    # integration starts from a computed DC operating point instead of from a
    # supplied `.ic`.  Every other transient in the catalog skips that path.
    CircuitCase(
        "unity_gain_buffer", "Miller opamp closed in unity gain",
        "unity_gain_buffer.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("transfer", "dc Vin <BUF_LO> <BUF_HI> 0.005",
                ("v(vout)", "v(vin)"), "unity_gain"),
            _tran("settling", "tran <BUF_TSTEP> <BUF_TSTOP>",
                  ("v(vout)", "v(vin)"), "settling",
                  subs=(("BUFFER_IN", "<BUFFER_STEP>"),)),
            _ac("closed_loop_ac", ("v(vout)",), "closed_loop_ac",
                subs=(("BUFFER_IN", "DC=<MID_RAIL> AC=1 0"),),
                card="ac dec 10 10 1e11"),
        ),
        tier="L4_systems",
        device_kinds=(),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "follow_error_v", "closed_loop_gain_error_pct",
                          "settling_error_pct", "overshoot_error_v",
                          "bandwidth_error_pct", "peaking_error_db"),
    ),
    CircuitCase(
        "ota_5t_buffer", "5T OTA in unity gain with on-chip bias",
        "ota_5t_buffer.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("transfer", "dc Vin <BUF_LO> <BUF_HI> 0.005",
                ("v(vout)", "v(vin)"), "unity_gain"),
            _tran("settling", "tran <BUF_TSTEP> <BUF_TSTOP>",
                  ("v(vout)", "v(vin)"), "settling",
                  subs=(("BUFFER_IN", "<BUFFER_STEP>"),)),
            _ac("closed_loop_ac", ("v(vout)",), "closed_loop_ac",
                subs=(("BUFFER_IN", "DC=<MID_RAIL> AC=1 0"),),
                card="ac dec 10 10 1e11"),
        ),
        tier="L4_systems",
        device_kinds=(),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "follow_error_v", "closed_loop_gain_error_pct",
                          "settling_error_pct", "overshoot_error_v",
                          "bandwidth_error_pct", "peaking_error_db"),
    ),
    CircuitCase(
        "multistage_buffer_12t",
        "12T self-biased Miller amplifier with buffered unity feedback",
        "multistage_buffer_12t.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _tran("settling", "tran <BUF_TSTEP> <BUF_TSTOP>",
                  ("v(vout)", "v(vin)"), "settling",
                  subs=(("BUFFER_IN", "<BUFFER_STEP>"),)),
            _ac("closed_loop_ac", ("v(vout)",), "closed_loop_ac",
                subs=(("BUFFER_IN", "DC=<MID_RAIL> AC=1 0"),),
                card="ac dec 10 10 1e11"),
        ),
        tier="L4_systems",
        device_kinds=(),
        required_metrics=(
            "mre_pct", "r2", "nrmse_pct", "max_err",
            "settling_error_pct", "overshoot_error_v",
            "bandwidth_error_pct", "peaking_error_db",
        ),
    ),
    CircuitCase(
        "ldo_regulator", "LDO: 5T error amp, PMOS pass device, RC load",
        "ldo_regulator.spice.tmpl",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("line_regulation", "dc Vdd <LDO_VDD_LO> <VDD> 0.005",
                ("v(vout)", "v(nfb)"), "line_regulation",
                subs=(("LDO_LOAD_SPEC", "<LDO_LOAD_DC>"),)),
            _tran("load_step", "tran <LDO_TSTEP> <LDO_TSTOP>",
                  ("v(vout)",), "load_regulation",
                  subs=(("LDO_LOAD_SPEC", "<LDO_LOAD_STEP>"),)),
            _ac("supply_ac", ("v(vout)",), "ldo_psrr_ac",
                subs=(("LDO_VDD_SPEC", "DC=<VDD> AC=1 0"),
                      ("LDO_LOAD_SPEC", "DC=<LDO_LOAD_DC> AC=0 0")),
                card="ac dec 10 10 1e10"),
            _ac("output_impedance_ac", ("v(vout)",),
                "ldo_output_impedance_ac",
                subs=(("LDO_VDD_SPEC", "DC=<VDD> AC=0 0"),
                      ("LDO_LOAD_SPEC", "DC=<LDO_LOAD_DC> AC=1 0")),
                card="ac dec 10 10 1e10"),
        ),
        tier="L4_systems",
        device_kinds=(),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "line_regulation_error_pct", "vout_error_v",
                          "load_droop_error_v", "recovery_error_v",
                          "psrr_error_db", "output_impedance_error_pct"),
    ),
)


CASES: Tuple[CircuitCase, ...] = _V1_CASES + _V2_CASES
_BY_ID: Dict[str, CircuitCase] = {case.case_id: case for case in CASES}

if len(_BY_ID) != len(CASES):  # fail at import: duplicate IDs corrupt evidence
    raise RuntimeError("duplicate simple-circuit case_id in catalog")


def cases(
    *,
    score_version: Optional[str] = None,
    role: Optional[str] = None,
) -> Tuple[CircuitCase, ...]:
    """Return catalog cases filtered by score version and/or role."""
    return tuple(
        case for case in CASES
        if (score_version is None or case.score_version == score_version)
        and (role is None or case.role == role)
    )


def get_case(case_id: str) -> CircuitCase:
    """Resolve one case ID or raise with the complete accepted vocabulary."""
    try:
        return _BY_ID[case_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown simple-circuit case {case_id!r}; "
            f"available: {sorted(_BY_ID)}"
        ) from exc

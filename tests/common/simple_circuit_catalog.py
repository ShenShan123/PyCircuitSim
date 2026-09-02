"""Authoritative inventory for NN compact-model simple-circuit evaluation.

This module is deliberately stdlib-only.  Campaign enumeration, coverage,
reporting, CLIs, and tests all import the same immutable case descriptions
without importing Torch, a simulator, or a PDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


SIMPLE_V1 = "simple-v1"
SIMPLE_V2 = "simple-v2"
QUALIFICATION = "qualification"
DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class AnalysisSpec:
    """One experiment rendered from a candidate/reference deck pair."""

    name: str
    kind: str
    candidate_card: str
    reference_card: str
    signals: Tuple[str, ...]
    metric_profile: str = "trace"
    phase_align: bool = False
    candidate_substitutions: Tuple[Tuple[str, str], ...] = ()
    reference_substitutions: Tuple[Tuple[str, str], ...] = ()

    def substitutions(self, reference: bool) -> Dict[str, str]:
        values = (self.reference_substitutions if reference
                  else self.candidate_substitutions)
        return dict(values)


@dataclass(frozen=True)
class CircuitCase:
    """A simple topology and the evidence contract attached to it."""

    case_id: str
    label: str
    candidate_deck: str
    reference_deck: str
    score_version: str
    role: str
    analyses: Tuple[AnalysisSpec, ...]
    device_kinds: Tuple[str, ...] = ("nmos", "pmos")
    omp_threads: Tuple[int, ...] = (1,)
    required_metrics: Tuple[str, ...] = (
        "mre_pct", "r2", "nrmse_pct", "max_err",
    )
    report_key: str = ""
    suite_id: str = ""
    legacy_suite_id: Optional[str] = None
    training_use: str = "held_out"
    gate_metric: str = ""
    gate_condition: str = ""
    report_label: str = ""
    report_gate_text: str = ""

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
) -> AnalysisSpec:
    return AnalysisSpec(
        name=name, kind="dc", candidate_card=f".{card}",
        reference_card=card, signals=signals, metric_profile=profile,
        candidate_substitutions=subs, reference_substitutions=subs,
    )


def _tran(
    name: str,
    card: str,
    signals: Tuple[str, ...],
    profile: str = "transient",
    *,
    phase_align: bool = False,
    candidate_subs: Tuple[Tuple[str, str], ...] = (),
    reference_subs: Tuple[Tuple[str, str], ...] = (),
) -> AnalysisSpec:
    return AnalysisSpec(
        name=name, kind="tran", candidate_card=f".{card}",
        reference_card=card, signals=signals, metric_profile=profile,
        phase_align=phase_align,
        candidate_substitutions=candidate_subs,
        reference_substitutions=reference_subs,
    )


def _ac(
    name: str,
    signals: Tuple[str, ...],
    profile: str = "ac",
    *,
    subs: Tuple[Tuple[str, str], ...] = (),
) -> AnalysisSpec:
    card = "ac dec 12 1e3 1e11"
    return AnalysisSpec(
        name=name, kind="ac", candidate_card=f".{card}",
        reference_card=card, signals=signals, metric_profile=profile,
        candidate_substitutions=subs, reference_substitutions=subs,
    )


_V1_CASES: Tuple[CircuitCase, ...] = (
    CircuitCase(
        "ring_osc", "5-stage ring oscillator",
        "directnet_ring_osc_tran.sp", "bsimcmg_ring_osc_tran.cir",
        SIMPLE_V1, QUALIFICATION,
        (_tran("oscillation", "tran 2p 1.2n uic", ("v(n5)",),
               "ring_osc", phase_align=True),),
        omp_threads=(1, 2, 4), report_key="ring_osc",
        suite_id="verify_circuit_ring_osc",
        legacy_suite_id="verify_complex_ring_osc",
        training_use="qualification_only",
        gate_metric="period error %",
        gate_condition="≤5 %",
        report_label="Ring oscillator",
        report_gate_text="period error %, gate ≤5 %",
    ),
    CircuitCase(
        "opamp", "two-stage Miller opamp",
        "directnet_opamp_miller_dc.sp", "bsimcmg_opamp_miller_dc.cir",
        SIMPLE_V1, QUALIFICATION,
        (_dc("transfer", "dc Vinp <OPAMP_LO> <OPAMP_HI> 0.002",
             ("v(vout)",), "opamp"),),
        omp_threads=(1, 2, 4), report_key="opamp",
        suite_id="verify_circuit_opamp",
        legacy_suite_id="verify_complex_opamp",
        training_use="qualification_only",
        gate_metric="open-loop gain error %",
        gate_condition="≤10 %",
        report_label="Two-stage Miller opamp (DC)",
        report_gate_text="open-loop gain error %, gate ≤10 %",
    ),
    CircuitCase(
        "sram_snm", "6T SRAM read-SNM half-cell",
        "directnet_sram_snm_dc.sp", "bsimcmg_sram_snm_dc.cir",
        SIMPLE_V1, QUALIFICATION,
        (_dc("read_snm", "dc Vq 0 <VDD> 0.005", ("v(qb)",),
             "sram_snm"),),
        report_key="sram_snm", suite_id="verify_circuit_sram_snm",
        legacy_suite_id="verify_complex_sram_snm",
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
        "directnet_switchcap_tran.sp", "bsimcmg_switchcap_tran.cir",
        SIMPLE_V1, QUALIFICATION,
        (_tran("sample_hold", "tran 5p 12n uic", ("v(vsamp)",),
               "switchcap"),),
        report_key="switchcap", suite_id="verify_circuit_switchcap",
        legacy_suite_id="verify_complex_switchcap",
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
        "source_follower", "complementary source followers with body effect",
        "nn_source_follower.sp", "bsimcmg_source_follower.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos", "dc Vgn 0 <VDD> 0.005", ("v(sn)", "i(Vddn)"),
                "source_follower"),
            _dc("pmos", "dc Vgp <VDD> 0 -0.005", ("v(sp)", "i(Vddp)"),
                "source_follower"),
        ),
    ),
    CircuitCase(
        "common_gate", "complementary common-gate stages",
        "nn_common_gate.sp", "bsimcmg_common_gate.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos", "dc Vsn 0 <HALF_VDD> 0.005",
                ("v(dn)", "i(Vsn)"), "gain"),
            _dc("pmos", "dc Vsp <VDD> <HALF_VDD> -0.005",
                ("v(dp)", "i(Vsp)"), "gain"),
        ),
    ),
    CircuitCase(
        "current_mirror", "NMOS and PMOS two-device current mirrors",
        "nn_current_mirror.sp", "bsimcmg_current_mirror.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos", "dc Voutn 0 <VDD> 0.005",
                ("i(Voutn)",), "current_mirror"),
            _dc("pmos", "dc Voutp <VDD> 0 -0.005",
                ("i(Voutp)",), "current_mirror"),
        ),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "ratio_error_pct", "output_resistance_error_pct"),
    ),
    CircuitCase(
        "inverter_chain", "open fanout-of-four inverter chain",
        "nn_inverter_chain.sp", "bsimcmg_inverter_chain.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (_tran("fo4", "tran 2p 4n uic", ("v(in)", "v(n1)", "v(o0)"),
               "inverter_chain", phase_align=True),),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "delay_error_pct", "rise_fall_error_pct",
                          "amplitude_error_pct",
                          "phase_aligned_nrmse_pct"),
    ),
    CircuitCase(
        "transmission_gate_dc", "bidirectional transmission gate",
        "nn_transmission_gate_dc.sp", "bsimcmg_transmission_gate_dc.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("forward", "dc Vinf 0 <VDD> 0.005",
                ("v(outf)", "i(Vinf)"), "transmission_gate"),
            _dc("reverse", "dc Vrev 0 <VDD> 0.005",
                ("v(inr)", "i(Vrev)"), "transmission_gate"),
        ),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "ron_error_pct"),
    ),
    CircuitCase(
        "transmission_gate_hold", "transmission-gate hold and feedthrough",
        "nn_transmission_gate_hold.sp", "bsimcmg_transmission_gate_hold.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (_tran("hold", "tran 2p 4n uic", ("v(hold)", "v(phi)"),
               "hold_droop"),),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "droop_error_v", "feedthrough_error_v"),
    ),
    CircuitCase(
        "diffpair_ideal", "resistor-loaded differential pair, ideal tail",
        "nn_diffpair_ideal.sp", "bsimcmg_diffpair_ideal.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("steering", "dc Vinp <DIFF_LO> <DIFF_HI> 0.001",
                ("v(outn)", "v(outp)"), "diffpair"),
            _ac("differential_ac", ("v(outn)", "v(outp)"),
                "diffpair_diff_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "0"))),
            _ac("common_mode_ac", ("v(outn)", "v(outp)"),
                "diffpair_cm_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "1"))),
        ),
        device_kinds=("nmos",),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "diff_gain_error_pct", "cm_gain_error_pct"),
    ),
    CircuitCase(
        "diffpair_active", "resistor-loaded differential pair, active tail",
        "nn_diffpair_active.sp", "bsimcmg_diffpair_active.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("steering", "dc Vinp <DIFF_LO> <DIFF_HI> 0.001",
                ("v(outn)", "v(outp)"), "diffpair"),
            _ac("differential_ac", ("v(outn)", "v(outp)"),
                "diffpair_diff_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "0"))),
            _ac("common_mode_ac", ("v(outn)", "v(outp)"),
                "diffpair_cm_ac",
                subs=(("AC_INP", "1"), ("AC_INN", "1"))),
        ),
        device_kinds=("nmos",),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "diff_gain_error_pct", "cm_gain_error_pct"),
    ),
    CircuitCase(
        "cascode_stack", "complementary cascode/current-source stacks",
        "nn_cascode_stack.sp", "bsimcmg_cascode_stack.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("nmos_compliance", "dc Voutn 0 <VDD> 0.005",
                ("i(Voutn)", "v(nx)"), "cascode"),
            _dc("pmos_compliance", "dc Voutp <VDD> 0 -0.005",
                ("i(Voutp)", "v(px)"), "cascode"),
            _ac("gain", ("v(nac)", "v(pac)"), "cascode_ac"),
        ),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "output_resistance_error_pct"),
    ),
    CircuitCase(
        "nand2", "two-input CMOS NAND",
        "nn_nand2.sp", "bsimcmg_nand2.cir", SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("input_a", "dc Va 0 <VDD> 0.005",
                ("v(out)", "v(nint)"), "logic_vtc",
                subs=(("VA_SPEC", "0"), ("VB_SPEC", "<VDD>"))),
            _dc("input_b", "dc Vb 0 <VDD> 0.005",
                ("v(out)", "v(nint)"), "logic_vtc",
                subs=(("VA_SPEC", "<VDD>"), ("VB_SPEC", "0"))),
            _tran("transient", "tran 2p 4n uic",
                  ("v(a)", "v(out)", "v(nint)"), "logic_tran",
                  candidate_subs=(("VA_SPEC", "PULSE 0 <VDD> 0.5n 20p 20p 1n 2n"),
                                  ("VB_SPEC", "<VDD>")),
                  reference_subs=(("VA_SPEC", "PULSE(0 <VDD> 0.5n 20p 20p 1n 2n)"),
                                  ("VB_SPEC", "<VDD>"))),
        ),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "trip_shift_v", "delay_error_pct",
                          "internal_node_nrmse_pct"),
    ),
    CircuitCase(
        "nor2", "two-input CMOS NOR",
        "nn_nor2.sp", "bsimcmg_nor2.cir", SIMPLE_V2, DIAGNOSTIC,
        (
            _dc("input_a", "dc Va 0 <VDD> 0.005",
                ("v(out)", "v(pint)"), "logic_vtc",
                subs=(("VA_SPEC", "0"), ("VB_SPEC", "0"))),
            _dc("input_b", "dc Vb 0 <VDD> 0.005",
                ("v(out)", "v(pint)"), "logic_vtc",
                subs=(("VA_SPEC", "0"), ("VB_SPEC", "0"))),
            _tran("transient", "tran 2p 4n uic",
                  ("v(a)", "v(out)", "v(pint)"), "logic_tran",
                  candidate_subs=(("VA_SPEC", "PULSE 0 <VDD> 0.5n 20p 20p 1n 2n"),
                                  ("VB_SPEC", "0")),
                  reference_subs=(("VA_SPEC", "PULSE(0 <VDD> 0.5n 20p 20p 1n 2n)"),
                                  ("VB_SPEC", "0"))),
        ),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "trip_shift_v", "delay_error_pct",
                          "internal_node_nrmse_pct"),
    ),
    CircuitCase(
        "sram6t_modes", "full 6T SRAM hold/read/write modes",
        "nn_sram6t_modes.sp", "bsimcmg_sram6t_modes.cir",
        SIMPLE_V2, DIAGNOSTIC,
        (
            _tran("hold", "tran 2p 3n uic", ("v(q)", "v(qb)"), "sram_hold",
                  candidate_subs=(("WL_SPEC", "0"), ("BL_SPEC", "<VDD>"),
                                  ("BLB_SPEC", "<VDD>")),
                  reference_subs=(("WL_SPEC", "0"), ("BL_SPEC", "<VDD>"),
                                  ("BLB_SPEC", "<VDD>"))),
            _tran("read", "tran 2p 3n uic", ("v(q)", "v(qb)"), "sram_read",
                  candidate_subs=(("WL_SPEC", "PULSE 0 <VDD> 0.5n 20p 20p 1.5n 3n"),
                                  ("BL_SPEC", "<VDD>"), ("BLB_SPEC", "<VDD>")),
                  reference_subs=(("WL_SPEC", "PULSE(0 <VDD> 0.5n 20p 20p 1.5n 3n)"),
                                  ("BL_SPEC", "<VDD>"), ("BLB_SPEC", "<VDD>"))),
            _tran("write", "tran 2p 3n uic", ("v(q)", "v(qb)"), "sram_write",
                  candidate_subs=(("WL_SPEC", "PULSE 0 <VDD> 0.5n 20p 20p 1.5n 3n"),
                                  ("BL_SPEC", "0"), ("BLB_SPEC", "<VDD>")),
                  reference_subs=(("WL_SPEC", "PULSE(0 <VDD> 0.5n 20p 20p 1.5n 3n)"),
                                  ("BL_SPEC", "0"), ("BLB_SPEC", "<VDD>"))),
        ),
        required_metrics=("mre_pct", "r2", "nrmse_pct", "max_err",
                          "hold_margin_error_v", "read_disturb_error_v",
                          "write_time_error_pct", "write_final_error_v",
                          "retention"),
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


def suite_cases(score_version: str) -> Tuple[CircuitCase, ...]:
    """Cases belonging to one versioned campaign contract."""
    return cases(score_version=score_version)


def suite_ids(score_version: str) -> Tuple[str, ...]:
    """Campaign suite identifiers for one versioned circuit contract."""
    return tuple(case.campaign_suite for case in suite_cases(score_version))


def iter_legacy_aliases() -> Iterable[Tuple[str, str]]:
    """Yield persisted historical suite ID -> current suite ID mappings."""
    for case in CASES:
        if case.legacy_suite_id:
            yield case.legacy_suite_id, case.campaign_suite

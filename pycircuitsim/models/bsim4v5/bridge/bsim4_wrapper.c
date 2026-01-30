/*
 * BSIM4.5.0 Standalone Wrapper Implementation
 *
 * This file provides a standalone interface to the BSIM4.5.0 model
 * extracted from the original UC Berkeley implementation.
 *
 * Phase 1: Core I-V characteristics (Vth, mobility, velocity saturation)
 *
 * Author: PyCircuitSim Team
 */

#include "bsim4_standalone.h"
#include "bsim4_iv_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Physical constants */
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define K_BOLTZMANN 1.380649e-23
#define Q_CHARGE 1.60219e-19

/*
 * Main evaluation function - Uses BSIM4.5.0 core I-V model
 */
int BSIM4_Evaluate(
    const BSIM4_Model *model,
    const BSIM4_Instance *instance,
    double Vds,
    double Vgs,
    double Vbs,
    BSIM4_Output *output)
{
    BSIM4_Internal internal;
    int ret;

    /* Initialize output to zero */
    memset(output, 0, sizeof(BSIM4_Output));

    /* Call the BSIM4.5.0 core I-V model */
    ret = bsim4_iv_evaluate(model, instance, Vds, Vgs, Vbs, &internal);

    if (ret != 0) {
        output->error = ret;
        return ret;
    }

    /* Copy results from internal state to output */
    output->Id = internal.Ids;
    output->Gm = internal.Gm;
    output->Gds = internal.Gds;
    output->Gmbs = internal.Gmbs;

    /* Calculate other currents (simplified for now) */
    output->Ib = 0.0;
    output->Ig = 0.0;
    output->Is = -(output->Id + output->Ib);

    /* Small-signal parameters (to be added in later phases) */
    output->Ggb = 0.0;
    output->Gbd = 0.0;
    output->Gbs = 0.0;

    /* Charges (simplified - to be added in Phase 11) */
    double Cox = model->epsrox * EPS0 / model->tox;
    double C_ox = Cox * instance->L * instance->W;
    output->Qg = C_ox * Vgs;
    output->Qb = 0.0;
    output->Qd = 0.0;
    output->Qs = 0.0;

    /* Capacitances (simplified - to be added in Phase 11) */
    output->Cgg = C_ox;
    output->Cgd = 0.0;
    output->Cgs = 0.0;
    output->Cgb = 0.0;

    /* Operating point info (for debugging) */
    output->Vth = internal.Vth;
    output->Vgsteff = internal.Vgsteff;

    output->error = 0;

    return 0;
}

/*
 * Initialize model with 45nm NMOS parameters
 * Based on PTM 45nm technology
 */
void BSIM4_InitModel_45nm_NMOS(BSIM4_Model *model)
{
    memset(model, 0, sizeof(BSIM4_Model));

    model->type = 1;  /* NMOS */

    /* 45nm technology parameters from freePDK45nm_TT.l (NMOS_VTL) */
    model->tox = 1.14e-9;      /* Gate oxide thickness (m) - freePDK45 value! */
    model->toxp = 1.0e-9;      /* Physical oxide thickness (m) */
    model->toxm = 1.14e-9;     /* Oxide thickness at which mob is measured (m) */
    model->dtox = 0.14e-9;     /* Tox difference (m) */
    model->epsrox = 3.9;     /* Oxide dielectric constant */
    model->coxe = model->epsrox * EPS0 / model->tox;  /* Oxide capacitance per unit area (F/m^2) */

    /* Threshold voltage parameters - from freePDK45 NMOS_VTL */
    model->vth0 = 0.322;      /* Nominal Vth (V) - freePDK45 value! */
    model->k1 = 0.4;          /* First-order body effect */

    /* Temperature coefficients for Vth */
    model->kt1 = -0.002;      /* Primary Vth temp coefficient (V-m/K) - typical for NMOS */
    model->kt1l = 0.0;        /* Length dependence of kt1 */
    model->kt2 = -0.01;       /* Secondary Vth temp coefficient (V/K) */
    model->k2 = 0.0;          /* Second-order body effect */
    model->k3 = 0.0;          /* Narrow width effect */
    model->k3b = 0.0;

    /* Short-channel effects - from freePDK45 */
    model->dvt0 = 1.0;        /* Short-channel Vth shift coefficient */
    model->dvt1 = 2.0;        /* Short-channel effect exponent */
    model->dvt2 = 0.0;        /* Additional short-channel effect */
    model->dvt0w = 0.0;       /* Width effect on dvt0 */
    model->dvt1w = 0.0;       /* Width effect on dvt1 */
    model->dvt2w = 0.0;       /* Width effect on dvt2 */
    model->dsub = 0.1;        /* DIBL coefficient */
    model->dvtp0 = 1.0e-10;  /* Vth shift due to reverse bias */
    model->dvtp1 = 0.1;       /* Vth shift coefficient */
    model->lpe0 = 0.0;        /* Lateral non-uniform doping effect */
    model->lpeb = 0.0;        /* Lateral non-uniform doping coefficient */
    model->litl = 0.0;        /* Lateral non-uniform doping length */

    /* Mobility parameters - from freePDK45 */
    model->u0 = 0.045;        /* Low-field mobility (m^2/V-s) - freePDK45 value! */
    model->ute = -1.5;       /* Temperature coefficient of u0 */
    model->ua = 6.0e-10;      /* Mobility degradation - vertical field (m/V) */
    model->ua1 = 4.31e-9;     /* Gate-bias dependence of ua */
    model->ub = 1.2e-18;      /* Mobility degradation - vertical field squared (m^2/V^2) */
    model->ub1 = 7.61e-18;    /* Gate-bias dependence of ub */
    model->uc = 0.0;         /* Mobility degradation - lateral field (1/V) */
    model->uc1 = -5.6e-11;   /* Gate-bias dependence of uc */
    model->eu = 2.0;         /* Power law mobility exponent for mobMod=2 */

    /* Saturation velocity - from freePDK45 */
    model->vsat = 148000.0;   /* Saturation velocity (m/s) */
    model->at = -0.8;        /* Temperature exponent of vsat (vsat ~ T^at) */
    model->mstar = 1.0;      /* Subthreshold parameter */

    /* Subthreshold swing */
    model->nfactor = 2.1;    /* Subthreshold swing coefficient */
    model->nsub = 1.0e17;     /* Substrate doping (cm^-3) - placeholder */
    model->ndep = 3.4e18;     /* Channel doping (cm^-3) - freePDK45 value! */
    model->nsd = 2.0e20;      /* S/D doping (cm^-3) */
    model->phin = 0.0;       /* Surface potential (V) - freePDK45 uses 0 */

    /* Gate parameters */
    model->ngate = 3.0e20;    /* Poly gate doping concentration (cm^-3) - freePDK45 value! */
    model->vfb = -0.55;       /* Flat-band voltage (V) */

    /* Vth parameters */
    model->voff = 0.0;        /* Offset voltage in subthreshold region - set to 0 for debugging */
    model->minv = 0.05;       /* Gate-source voltage for capacitance model */
    model->delta = 0.01;      /* Vth effective width effect */

    /* Parasitic resistance - from freePDK45 */
    model->rdsw = 155.0;      /* Sheet resistance of S/D diffusion (ohms·µm) - freePDK45 value! */
    model->rdswmin = 0.0;     /* Minimum rdsw */
    model->rdwmin = 0.0;      /* Minimum rdw */
    model->rswmin = 0.0;      /* Minimum rsw */
    model->rsw = 80.0;        /* Source resistance per width (ohms·µm) - freePDK45 value! */
    model->rdw = 80.0;        /* Drain resistance per width (ohms·µm) - freePDK45 value! */
    model->prwg = 0.0;       /* Gate bias effect on rdsw */
    model->prwb = 0.0;       /* Body bias effect on rdsw */
    model->prt = 0.0;        /* Temperature coefficient of rdsw */

    /* Subthreshold and DIBL */
    model->eta0 = 0.006;      /* DIBL coefficient - freePDK45 value! */
    model->etab = 0.0;
    model->pclm = 0.02;       /* Channel length modulation coefficient - freePDK45! */
    model->pdibl1 = 0.001;   /* DIBL coefficient - freePDK45! */
    model->pdibl2 = 0.001;   /* DIBL coefficient - freePDK45! */
    model->pdiblb = -0.005;  /* Body effect on pdibl - freePDK45! */
    model->fprout = 0.2;     /* Field-induced drain mobility degradation factor - freePDK45! */
    model->pdits = 0.08;     /* DITS coefficient - freePDK45! */
    model->pditsd = 0.23;    /* DITS drain voltage coefficient - freePDK45! */
    model->pditsl = 2300000; /* DITS length coefficient - freePDK45! */

    /* Geometry effects */
    model->a0 = 1.0;         /* Bulk charge effect coefficient */
    model->ags = 0.0;        /* Gate bias coefficient of abulk */
    model->a1 = 0.0;         /* First non-saturation factor */
    model->a2 = 1.0;         /* Second non-saturation factor */
    model->keta = 0.04;       /* Non-uniform depletion width effect - freePDK45! */
    model->b0 = 0.0;         /* Abulk geometry effect */
    model->b1 = 0.0;         /* Abulk geometry effect */
    model->dwg = 0.0;        /* Gate bias effect on effective width */
    model->dwb = 0.0;        /* Body bias effect on effective width */
    model->w0 = 2.5e-6;       /* Width effect - freePDK45! */

    /* Gate current and tunneling */
    model->agidl = 0.0002;   /* GIDL parameter - freePDK45! */
    model->bgidl = 2.1e-9;    /* GIDL parameter - freePDK45! */
    model->cgidl = 0.0002;   /* GIDL parameter - freePDK45! */
    model->egidl = 0.8;      /* GIDL parameter - freePDK45! */
    model->aigc = 0.02;      /* Gate-to-channel Igc parameter - freePDK45! */
    model->bigc = 0.0027;    /* Gate-to-channel Igc parameter - freePDK45! */
    model->cigc = 0.002;      /* Gate-to-channel Igc parameter - freePDK45! */
    model->aigsd = 0.02;     /* Gate-to-S/D Igsd parameter - freePDK45! */
    model->bigsd = 0.0027;    /* Gate-to-S/D Igsd parameter - freePDK45! */
    model->cigsd = 0.002;    /* Gate-to-S/D Igsd parameter - freePDK45! */
    model->aigbacc = 0.012;  /* Igbs parameter for accumulation - freePDK45! */
    model->bigbacc = 0.0028;  /* Igbs parameter for accumulation - freePDK45! */
    model->cigbacc = 0.002;   /* Igbs parameter for accumulation - freePDK45! */
    model->aigbinv = 0.014;   /* Igbs parameter for inversion - freePDK45! */
    model->bigbinv = 0.004;   /* Igbs parameter for inversion - freePDK45! */
    model->cigbinv = 0.004;   /* Igbs parameter for inversion - freePDK45! */
    model->nigbacc = 1.0;    /* Emission coefficient for accumulation */
    model->nigbinv = 3.0;    /* Emission coefficient for inversion */
    model->nigc = 1.0;       /* Emission coefficient for gate */
    model->ntox = 1.0;       /* Tunneling mass exponent */
    model->eigbinv = 1.1;    /* Igbs energy parameter in inversion */
    model->poxedge = 1.0;    /* Oxide edge gate current parameter */
    model->toxref = 1.14e-9; /* Tox at which Igc model is measured */

    /* Diode and junction parameters */
    model->ijthdfwd = 0.01;  /* Forward diode current for noise - freePDK45! */
    model->ijthsfwd = 0.01;  /* Forward diode current for noise - freePDK45! */
    model->ijthdrev = 0.001; /* Reverse diode current for noise - freePDK45! */
    model->ijthsrev = 0.001; /* Reverse diode current for noise - freePDK45! */
    model->xjbvd = 1.0;      /* BVD grading coefficient */
    model->xjbvs = 1.0;      /* BVS grading coefficient */
    model->bvd = 10.0;       /* Drain junction breakdown voltage */
    model->bvs = 10.0;       /* Source junction breakdown voltage */

    /* SCBE (Substrate Current Body Effect) - from freePDK45 */
    model->pscbe1 = 8.14e-8;  /* SCBE coefficient 1 - freePDK45! */
    model->pscbe2 = 1.0e-7;   /* SCBE coefficient 2 - freePDK45! */
    model->pvag = 1.0e-20;   /* Gate dependence of SCBE */

    /* Additional effects */
    model->pvag = 1.0e-20;  /* Gate dependence of SCBE */
    model->alpha0 = 0.074;   /* Impact ionization coefficient 1 - freePDK45! */
    model->alpha1 = 0.005;   /* Impact ionization coefficient 2 - freePDK45! */
    model->beta0 = 30.0;      /* Impact ionization coefficient 3 - freePDK45! */

    /* Junction depth */
    model->xj = 1.98e-8;      /* Junction depth (m) - freePDK45 value! */

    /* Overlap capacitance */
    model->cgsl = 1.1e-10;   /* Source-gate overlap capacitance per width */
    model->cgdl = 2.653e-10; /* Drain-gate overlap capacitance per width */
    model->ckappas = 0.03;   /* Source-bias coefficient for overlap capacitance */
    model->ckappad = 0.03;   /* Drain-bias coefficient for overlap capacitance */
    model->cf = 0.0;         /* Fringing field capacitance per width */
    model->vfbcv = 0.0;      /* Flat-band voltage for CV model */

    /* Geometry parameters */
    model->clc = 0.0;        /* Vdsat depletion capacitance */
    model->cle = 0.0;        /* Vdsat depletion capacitance length coefficient */
    model->dwc = 0.0;        /* Width correction */
    model->dlc = 0.0;        /* Length correction */
    model->xw = 0.0;         /* Width offset */
    model->xl = -20e-9;      /* Length offset - freePDK45 value! */
    model->dlcig = 0.0;      /* Length reduction for Igc model */
    model->dwj = 0.0;        /* Width reduction for junction diode */
    model->noff = 0.9;       /* Voffcv coefficient - freePDK45! */
    model->voffcv = 0.02;    /* Offset voltage in CV model - freePDK45! */
    model->acde = 1.0;       /* Accumulation capacitance coefficient - freePDK45! */
    model->moin = 15.0;      /* Gate insulator thickness coefficient - freePDK45! */
    model->tcj = 0.001;      /* Temperature coefficient of cj */
    model->tcjsw = 0.001;    /* Temperature coefficient of cjsw */
    model->tcjswg = 0.001;   /* Temperature coefficient of cjswg */
    model->tpb = 0.005;      /* Temperature coefficient of pb */
    model->tpbsw = 0.005;    /* Temperature coefficient of pbsw */
    model->tpbswg = 0.001;   /* Temperature coefficient of pbswg */

    /* Gate resistance */
    model->dmcg = 0.0;       /* Distance between gate contact and channel */
    model->dmci = 0.0;       /* Distance between gate contacts */
    model->dmdg = 0.0;       /* Distance between gate contact and drain */
    model->dmcgt = 0.0;      /* Temperature coefficient of dmcg */
    model->xgw = 0.0;        /* Gate electrode width */
    model->xgl = 0.0;        /* Gate electrode length */
    model->rshg = 0.4;       /* Gate sheet resistance */
    model->ngcon = 1.0;      /* Number of gate contacts */

    /* Model selectors - from freePDK45 */
    model->mobMod = 0;       /* Mobility model - freePDK45 uses 0! */
    model->capMod = 2;       /* Capacitance model */
    model->dioMod = 1;       /* Diode model */
    model->trnqsMod = 0;    /* Transient NQS model */
    model->acnqsMod = 0;    /* AC NQS model */
    model->fnoiMod = 1;      /* Flicker noise model */
    model->tnoiMod = 0;      /* Thermal noise model */
    model->rdsMod = 0;       /* RDS model */
    model->rbodyMod = 1;     /* Body resistance model */
    model->rgateMod = 1;     /* Gate resistance model */
    model->perMod = 1;       /* Perimeter model */
    model->geoMod = 1;       /* Geometry model */
    model->igcMod = 1;       /* Gate current model */
    model->igbMod = 1;       /* Gate-body current model */
    model->tempMod = 0;      /* Temperature model */
    model->paramChk = 1;     /* Parameter checking flag */

    /* Temperature - must be set before calculating sqrtPhi! */
    model->temp = 300.0;      /* Device temperature (K) */
    model->tnom = 300.0;      /* Nominal temperature (K) - 27°C = 300 K */

    /* Depletion depth at zero bias - Calculate from ndep */
    if (model->phin > 0.0) {
        model->sqrtPhi = sqrt(model->phin);
    } else {
        /* Calculate Phis from ndep when phin = 0 */
        double ni = 1.45e16;  /* m^-3 */
        double Vtm = K_BOLTZMANN * model->temp / Q_CHARGE;
        double ndep_si = model->ndep * 1e6;  /* cm^-3 to m^-3 */
        double Phis = 2.0 * Vtm * log(ndep_si / ni);
        model->sqrtPhi = sqrt(Phis);
    }

    /* Calculate Xdep0 = sqrt(2*eps_si*Phis/(q*ndep)) */
    double eps_si = 11.7 * EPS0;
    double Phis = model->sqrtPhi * model->sqrtPhi;
    double ndep_si = model->ndep * 1e6;  /* cm^-3 to m^-3 */
    model->Xdep0 = sqrt(2.0 * eps_si * Phis / (Q_CHARGE * ndep_si));

    /* Calculate cdep0 = eps_si / Xdep0 (depletion capacitance per unit area) */
    model->cdep0 = eps_si / model->Xdep0;

    model->voffcbn = 0.0;    /* Voff for capacitance model (same as voffcvbn) */

    /* Additional parameters from freePDK45 */
    model->cdsc = 0.0;       /* Drain/source depletion capacitance */
    model->cdscb = 0.0;      /* Body-bias coefficient of cdsc */
    model->cdscd = 0.0;      /* DIBL coefficient of cdsc */
    model->cit = 0.0;        /* Interface trap capacitance */
    model->drout = 0.5;      /* DIBL output resistance coefficient - freePDK45! */
    model->wr = 1.0;         /* Width dependence of rdsw - freePDK45! */
}

/*
 * Initialize model with 45nm PMOS parameters
 * Based on freePDK45 PMOS_VTL model
 */
void BSIM4_InitModel_45nm_PMOS(BSIM4_Model *model)
{
    memset(model, 0, sizeof(BSIM4_Model));

    /* Device type */
    model->type = -1;  /* PMOS */

    /* Model selectors - from freePDK45 */
    model->mobMod = 0;       /* freePDK45 uses mobMod=0 */
    model->capMod = 2;       /* Capacitance model */
    model->dioMod = 1;       /* Diode model */
    model->trnqsMod = 0;    /* Transient NQS model */
    model->acnqsMod = 0;    /* AC NQS model */
    model->fnoiMod = 1;      /* Flicker noise model */
    model->tnoiMod = 0;      /* Thermal noise model */
    model->rdsMod = 0;       /* RDS model */
    model->rbodyMod = 1;     /* Body resistance model */
    model->rgateMod = 1;     /* Gate resistance model */
    model->perMod = 1;       /* Perimeter model */
    model->geoMod = 1;       /* Geometry model */
    model->igcMod = 1;       /* Gate current model */
    model->igbMod = 1;       /* Gate-body current model */
    model->tempMod = 0;      /* Temperature model */
    model->paramChk = 1;     /* Parameter checking flag */

    /* Oxide parameters - freePDK45 values */
    model->tox = 1.26e-9;      /* freePDK45 PMOS value! */
    model->toxp = 1.0e-9;       /* freePDK45 value! */
    model->toxm = 1.26e-9;      /* freePDK45 value! */
    model->dtox = 2.6e-10;      /* freePDK45 value! */
    model->epsrox = 3.9;
    model->toxref = 1.3e-9;     /* freePDK45 value! */

    /* Geometry parameters - freePDK45 values */
    model->xl = -20e-9;         /* freePDK45 value! */

    /* Threshold voltage parameters - freePDK45 values */
    model->vth0 = 0.3021;       /* freePDK45 value! (stored as positive, type=-1 makes it negative) */
    model->k1 = 0.4;            /* freePDK45 value! */
    model->k2 = -0.01;          /* freePDK45 value! */

    /* Temperature coefficients for Vth */
    model->kt1 = -0.002;        /* Primary Vth temp coefficient (V-m/K) - typical for PMOS */
    model->kt1l = 0.0;          /* Length dependence of kt1 */
    model->kt2 = -0.01;         /* Secondary Vth temp coefficient (V/K) */
    model->k3 = 0.0;
    model->k3b = 0.0;
    model->w0 = 2.5e-6;         /* freePDK45 value! */
    model->dvt0 = 1.0;          /* freePDK45 value! */
    model->dvt1 = 2.0;          /* freePDK45 value! */
    model->dvt2 = -0.032;       /* freePDK45 value! */
    model->dvt0w = 0.0;
    model->dvt1w = 0.0;
    model->dvt2w = 0.0;
    model->dvtp0 = 1e-11;
    model->dvtp1 = 0.05;
    model->nfactor = 2.22;      /* freePDK45 value! */
    model->voff = -0.126;       /* freePDK45 value! */

    /* Substrate and doping parameters - freePDK45 values */
    model->nsub = 1.0e17;       /* Substrate doping (cm^-3) - placeholder */
    model->ngate = 2.0e20;      /* freePDK45 value! */
    model->ndep = 2.44e18;      /* freePDK45 value! */
    model->nsd = 2.0e20;        /* freePDK45 value! */
    model->xj = 1.98e-8;        /* freePDK45 value! */

    /* Mobility parameters - freePDK45 values */
    model->u0 = 0.02;           /* freePDK45 PMOS value! */
    model->ute = -1.5;          /* Temperature exponent of u0 */
    model->ua = 2.0e-9;         /* freePDK45 PMOS value! */
    model->ub = 5.0e-19;        /* freePDK45 PMOS value! */
    model->uc = 0.0;            /* freePDK45 value! */
    model->eu = 0.0;
    model->vsat = 69000.0;      /* freePDK45 PMOS value! */
    model->at = -0.8;           /* Temperature exponent of vsat (vsat ~ T^at) */
    model->a0 = 1.0;
    model->ags = 1.0e-20;
    model->a1 = 0.0;
    model->a2 = 1.0;
    model->keta = -0.047;       /* freePDK45 value! */

    /* Saturation parameters - freePDK45 values */
    model->delta = 0.01;        /* freePDK45 value! */

    /* DIBL parameters - freePDK45 values */
    model->eta0 = 0.0055;       /* freePDK45 PMOS value! */
    model->etab = 0.0;

    /* CLM parameters - freePDK45 values */
    model->pclm = 0.12;         /* freePDK45 PMOS value! */

    /* SCBE parameters - freePDK45 values */
    model->pscbe1 = 8.14e-8;    /* freePDK45 value! */
    model->pscbe2 = 9.58e-7;    /* freePDK45 PMOS value! */

    /* Resistance parameters - freePDK45 values */
    model->rsw = 75.0;          /* freePDK45 PMOS value! */
    model->rdw = 75.0;          /* freePDK45 PMOS value! */
    model->rdsw = 155.0;        /* freePDK45 value! */
    model->rdswmin = 0.0;
    model->rdwmin = 0.0;
    model->rswmin = 0.0;

    /* Other parameters */
    model->cdsc = 0.0;
    model->cdscb = 0.0;
    model->cdscd = 0.0;
    model->cit = 0.0;
    model->vfb = 0.55;          /* freePDK45 value! */

    /* DIBL width effect */
    model->pdibl1 = 0.001;      /* freePDK45 value! */
    model->pdibl2 = 0.001;      /* freePDK45 value! */
    model->pdiblb = 3.4e-8;     /* freePDK45 value! */
    model->drout = 0.56;        /* freePDK45 value! */

    /* Temperature */
    model->temp = 300.0;        /* Device temperature (K) */
    model->tnom = 300.0;        /* Nominal temperature (K) - 27°C = 300 K */

    /* Depletion depth at zero bias */
    model->Xdep0 = 0.0;         /* Will be calculated from ndep */
    model->cdep0 = 0.0;         /* Depletion capacitance at zero bias (F/m^2) */
    model->sqrtPhi = 0.0;       /* sqrt(Phi) for geometry calculations */
    model->voffcbn = 0.0;       /* Voff for capacitance model (same as voffcvbn) */
}

/*
 * Initialize instance
 */
void BSIM4_InitInstance(BSIM4_Instance *instance, double L, double W)
{
    memset(instance, 0, sizeof(BSIM4_Instance));

    instance->L = L;
    instance->W = W;
    instance->drainArea = 0.0;
    instance->sourceArea = 0.0;
    instance->drainSquares = 1.0;
    instance->sourceSquares = 1.0;
    instance->drainPerimeter = 0.0;
    instance->sourcePerimeter = 0.0;

    /* Stress effect */
    instance->sa = 0.0;
    instance->sb = 0.0;
    instance->sd = 0.0;

    instance->nf = 1.0;
    instance->off = 0;
}

/*
 * Set parameter by name
 * Extended to support freePDK45 and other PDK libraries
 */
int BSIM4_SetParam(BSIM4_Model *model, const char *param_name, double value)
{
    /* Instance parameters (not model parameters) */
    if (strcmp(param_name, "L") == 0) return -1;
    if (strcmp(param_name, "W") == 0) return -1;

    /* Oxide parameters */
    if (strcmp(param_name, "TOX") == 0 || strcmp(param_name, "TOXE") == 0) {
        model->tox = value;
        return 0;
    }
    if (strcmp(param_name, "TOXP") == 0) { model->toxp = value; return 0; }
    if (strcmp(param_name, "TOXM") == 0) { model->toxm = value; return 0; }
    if (strcmp(param_name, "DTOX") == 0) { model->dtox = value; return 0; }
    if (strcmp(param_name, "EPSROX") == 0) { model->epsrox = value; return 0; }
    if (strcmp(param_name, "TOXREF") == 0) { model->toxref = value; return 0; }

    /* Threshold voltage parameters */
    if (strcmp(param_name, "VTH0") == 0) {
        /* freePDK45 stores vth0 with sign included (-0.3 for PMOS, +0.3 for NMOS)
         * But our code uses type field to handle sign, so take absolute value
         * The Vth calculation does: Vth = type * vth0
         * So vth0 should always be positive in the model structure */
        model->vth0 = fabs(value);
        return 0;
    }
    if (strcmp(param_name, "K1") == 0) { model->k1 = value; return 0; }
    if (strcmp(param_name, "K2") == 0) { model->k2 = value; return 0; }
    if (strcmp(param_name, "K3") == 0) { model->k3 = value; return 0; }
    if (strcmp(param_name, "K3B") == 0) { model->k3b = value; return 0; }
    if (strcmp(param_name, "W0") == 0) { model->w0 = value; return 0; }
    if (strcmp(param_name, "DVT0") == 0) { model->dvt0 = value; return 0; }
    if (strcmp(param_name, "DVT1") == 0) { model->dvt1 = value; return 0; }
    if (strcmp(param_name, "DVT2") == 0) { model->dvt2 = value; return 0; }
    if (strcmp(param_name, "DVT0W") == 0) { model->dvt0w = value; return 0; }
    if (strcmp(param_name, "DVT1W") == 0) { model->dvt1w = value; return 0; }
    if (strcmp(param_name, "DVT2W") == 0) { model->dvt2w = value; return 0; }
    if (strcmp(param_name, "DVTP0") == 0) { model->dvtp0 = value; return 0; }
    if (strcmp(param_name, "DVTP1") == 0) { model->dvtp1 = value; return 0; }

    /* Mobility parameters */
    if (strcmp(param_name, "U0") == 0) {
        model->u0 = value;  /* Parser already handles unit conversion */
        return 0;
    }
    if (strcmp(param_name, "UA") == 0) { model->ua = value; return 0; }
    if (strcmp(param_name, "UA1") == 0) { model->ua1 = value; return 0; }
    if (strcmp(param_name, "UB") == 0) { model->ub = value; return 0; }
    if (strcmp(param_name, "UB1") == 0) { model->ub1 = value; return 0; }
    if (strcmp(param_name, "UC") == 0) { model->uc = value; return 0; }
    if (strcmp(param_name, "UC1") == 0) { model->uc1 = value; return 0; }
    if (strcmp(param_name, "UTE") == 0) { model->ute = value; return 0; }
    if (strcmp(param_name, "EU") == 0) { model->eu = value; return 0; }
    if (strcmp(param_name, "VSAT") == 0) { model->vsat = value; return 0; }
    if (strcmp(param_name, "AT") == 0) { model->at = value; return 0; }
    if (strcmp(param_name, "A0") == 0) { model->a0 = value; return 0; }
    if (strcmp(param_name, "AGS") == 0) { model->ags = value; return 0; }
    if (strcmp(param_name, "A1") == 0) { model->a1 = value; return 0; }
    if (strcmp(param_name, "A2") == 0) { model->a2 = value; return 0; }
    if (strcmp(param_name, "KETA") == 0) { model->keta = value; return 0; }
    if (strcmp(param_name, "ETA0") == 0) { model->eta0 = value; return 0; }
    if (strcmp(param_name, "ETAB") == 0) { model->etab = value; return 0; }

    /* Substrate and doping parameters */
    if (strcmp(param_name, "NSUB") == 0) { model->nsub = value; return 0; }
    if (strcmp(param_name, "NDEP") == 0) { model->ndep = value; return 0; }
    if (strcmp(param_name, "NSD") == 0) { model->nsd = value; return 0; }
    if (strcmp(param_name, "PHIN") == 0) { model->phin = value; return 0; }
    if (strcmp(param_name, "NGATE") == 0) { model->ngate = value; return 0; }
    if (strcmp(param_name, "GAMMA1") == 0) { model->gamma1 = value; return 0; }
    if (strcmp(param_name, "GAMMA2") == 0) { model->gamma2 = value; return 0; }
    if (strcmp(param_name, "VBX") == 0) { model->vbx = value; return 0; }
    if (strcmp(param_name, "VBM") == 0) { model->vbm = value; return 0; }
    if (strcmp(param_name, "XT") == 0) { model->xt = value; return 0; }
    if (strcmp(param_name, "XJ") == 0) { model->xj = value; return 0; }
    if (strcmp(param_name, "CDSC") == 0) { model->cdsc = value; return 0; }
    if (strcmp(param_name, "CDSCB") == 0) { model->cdscb = value; return 0; }
    if (strcmp(param_name, "CDSCD") == 0) { model->cdscd = value; return 0; }
    if (strcmp(param_name, "CIT") == 0) { model->cit = value; return 0; }

    /* Subthreshold parameters */
    if (strcmp(param_name, "NFACTOR") == 0) { model->nfactor = value; return 0; }
    if (strcmp(param_name, "MSTAR") == 0) { model->mstar = value; return 0; }
    if (strcmp(param_name, "VOFF") == 0) { model->voff = value; return 0; }
    if (strcmp(param_name, "voff") == 0) { model->voff = value; return 0; }
    if (strcmp(param_name, "VOFFL") == 0) { model->voffl = value; return 0; }
    if (strcmp(param_name, "MINV") == 0) { model->minv = value; return 0; }
    if (strcmp(param_name, "VOFFCVBN") == 0) { model->voffcvbn = value; model->voffcbn = value; return 0; }
    if (strcmp(param_name, "VOFFCBN") == 0) { model->voffcbn = value; model->voffcvbn = value; return 0; }
    if (strcmp(param_name, "CDEP0") == 0) { model->cdep0 = value; return 0; }
    if (strcmp(param_name, "XDEP0") == 0) { model->Xdep0 = value; return 0; }
    if (strcmp(param_name, "SQRTPhi") == 0) { model->sqrtPhi = value; return 0; }
    if (strcmp(param_name, "LPE0") == 0) { model->lpe0 = value; return 0; }
    if (strcmp(param_name, "LPEB") == 0) { model->lpeb = value; return 0; }
    if (strcmp(param_name, "DSUB") == 0) { model->dsub = value; return 0; }

    /* Parasitic resistance parameters */
    if (strcmp(param_name, "RDSW") == 0) { model->rdsw = value; return 0; }
    if (strcmp(param_name, "RDSWMIN") == 0) { model->rdswmin = value; return 0; }
    if (strcmp(param_name, "RDWMIN") == 0) { model->rdwmin = value; return 0; }
    if (strcmp(param_name, "RSWMIN") == 0) { model->rswmin = value; return 0; }
    if (strcmp(param_name, "RSW") == 0) { model->rsw = value; return 0; }
    if (strcmp(param_name, "RDW") == 0) { model->rdw = value; return 0; }
    if (strcmp(param_name, "rswmin") == 0) { model->rswmin = value; return 0; }
    if (strcmp(param_name, "rsw") == 0) { model->rsw = value; return 0; }
    if (strcmp(param_name, "rdw") == 0) { model->rdw = value; return 0; }
    if (strcmp(param_name, "PRWG") == 0) { model->prwg = value; return 0; }
    if (strcmp(param_name, "prwg") == 0) { model->prwg = value; return 0; }
    if (strcmp(param_name, "PRWB") == 0) { model->prwb = value; return 0; }
    if (strcmp(param_name, "PRT") == 0) { model->prt = value; return 0; }
    if (strcmp(param_name, "WR") == 0) { model->wr = value; return 0; }

    /* DIBL and CLM parameters */
    if (strcmp(param_name, "PCLM") == 0) { model->pclm = value; return 0; }
    if (strcmp(param_name, "PDIBL1") == 0) { model->pdibl1 = value; return 0; }
    if (strcmp(param_name, "PDIBL2") == 0) { model->pdibl2 = value; return 0; }
    if (strcmp(param_name, "PDIBLB") == 0) { model->pdiblb = value; return 0; }
    if (strcmp(param_name, "DROUT") == 0) { model->drout = value; return 0; }
    if (strcmp(param_name, "FPROUT") == 0) { model->fprout = value; return 0; }
    if (strcmp(param_name, "PDITS") == 0) { model->pdits = value; return 0; }
    if (strcmp(param_name, "PDITSD") == 0) { model->pditsd = value; return 0; }
    if (strcmp(param_name, "PDITSL") == 0) { model->pditsl = value; return 0; }

    /* SCBE parameters */
    if (strcmp(param_name, "PSCBE1") == 0) { model->pscbe1 = value; return 0; }
    if (strcmp(param_name, "PSCBE2") == 0) { model->pscbe2 = value; return 0; }
    if (strcmp(param_name, "PVAG") == 0) { model->pvag = value; return 0; }

    /* Geometry parameters */
    if (strcmp(param_name, "DELTA") == 0) { model->delta = value; return 0; }
    if (strcmp(param_name, "DWC") == 0) { model->dwc = value; return 0; }
    if (strcmp(param_name, "DLC") == 0) { model->dlc = value; return 0; }
    if (strcmp(param_name, "XW") == 0) { model->xw = value; return 0; }
    if (strcmp(param_name, "XL") == 0) { model->xl = value; return 0; }
    if (strcmp(param_name, "DWG") == 0) { model->dwg = value; return 0; }
    if (strcmp(param_name, "DWB") == 0) { model->dwb = value; return 0; }
    /* Note: LINT and WINT are instance parameters, not model parameters */
    /* They are handled separately in instance initialization */
    if (strcmp(param_name, "WINT") == 0) { return 0; }  /* Accept but ignore */
    if (strcmp(param_name, "LINT") == 0) { return 0; }  /* Accept but ignore */
    if (strcmp(param_name, "DWJ") == 0) { model->dwj = value; return 0; }
    if (strcmp(param_name, "B0") == 0) { model->b0 = value; return 0; }
    if (strcmp(param_name, "B1") == 0) { model->b1 = value; return 0; }
    if (strcmp(param_name, "LITL") == 0) { model->litl = value; return 0; }

    /* Temperature parameters */
    if (strcmp(param_name, "KT1") == 0) { model->kt1 = value; return 0; }
    if (strcmp(param_name, "KT1L") == 0) { model->kt1l = value; return 0; }
    if (strcmp(param_name, "KT2") == 0) { model->kt2 = value; return 0; }
    if (strcmp(param_name, "TEMP") == 0) { model->temp = value; return 0; }
    if (strcmp(param_name, "TNOM") == 0) { model->tnom = value; return 0; }

    /* Impact ionization parameters */
    if (strcmp(param_name, "ALPHA0") == 0) { model->alpha0 = value; return 0; }
    if (strcmp(param_name, "ALPHA1") == 0) { model->alpha1 = value; return 0; }
    if (strcmp(param_name, "BETA0") == 0) { model->beta0 = value; return 0; }

    /* Gate current parameters */
    if (strcmp(param_name, "AGIDL") == 0) { model->agidl = value; return 0; }
    if (strcmp(param_name, "BGIDL") == 0) { model->bgidl = value; return 0; }
    if (strcmp(param_name, "CGIDL") == 0) { model->cgidl = value; return 0; }
    if (strcmp(param_name, "EGIDL") == 0) { model->egidl = value; return 0; }
    if (strcmp(param_name, "AIGC") == 0) { model->aigc = value; return 0; }
    if (strcmp(param_name, "BIGC") == 0) { model->bigc = value; return 0; }
    if (strcmp(param_name, "CIGC") == 0) { model->cigc = value; return 0; }
    if (strcmp(param_name, "AIGSD") == 0) { model->aigsd = value; return 0; }
    if (strcmp(param_name, "BIGSD") == 0) { model->bigsd = value; return 0; }
    if (strcmp(param_name, "CIGSD") == 0) { model->cigsd = value; return 0; }
    if (strcmp(param_name, "AIGBACC") == 0) { model->aigbacc = value; return 0; }
    if (strcmp(param_name, "BIGBACC") == 0) { model->bigbacc = value; return 0; }
    if (strcmp(param_name, "CIGBACC") == 0) { model->cigbacc = value; return 0; }
    if (strcmp(param_name, "AIGBINV") == 0) { model->aigbinv = value; return 0; }
    if (strcmp(param_name, "BIGBINV") == 0) { model->bigbinv = value; return 0; }
    if (strcmp(param_name, "CIGBINV") == 0) { model->cigbinv = value; return 0; }
    if (strcmp(param_name, "NIGC") == 0) { model->nigc = value; return 0; }
    if (strcmp(param_name, "NIGBACC") == 0) { model->nigbacc = value; return 0; }
    if (strcmp(param_name, "NIGBINV") == 0) { model->nigbinv = value; return 0; }
    if (strcmp(param_name, "NTOX") == 0) { model->ntox = value; return 0; }
    if (strcmp(param_name, "EIGBINV") == 0) { model->eigbinv = value; return 0; }
    if (strcmp(param_name, "PIGCD") == 0) { model->pigcd = value; return 0; }
    if (strcmp(param_name, "POXEDGE") == 0) { model->poxedge = value; return 0; }
    if (strcmp(param_name, "DLCIG") == 0) { model->dlcig = value; return 0; }

    /* Diode parameters */
    if (strcmp(param_name, "JTSS") == 0) { model->jtss = value; return 0; }
    if (strcmp(param_name, "JTSD") == 0) { model->jtsd = value; return 0; }
    if (strcmp(param_name, "JTSSWS") == 0) { model->jtssws = value; return 0; }
    if (strcmp(param_name, "JTSSWD") == 0) { model->jtsswd = value; return 0; }
    if (strcmp(param_name, "JTSSWGS") == 0) { model->jtsswgs = value; return 0; }
    if (strcmp(param_name, "JTSSWGD") == 0) { model->jtsswgd = value; return 0; }
    if (strcmp(param_name, "NJTS") == 0) { model->njts = value; return 0; }
    if (strcmp(param_name, "NJTSSW") == 0) { model->njtssw = value; return 0; }
    if (strcmp(param_name, "NJTSSWG") == 0) { model->njtsswg = value; return 0; }
    if (strcmp(param_name, "XTSS") == 0) { model->xtss = value; return 0; }
    if (strcmp(param_name, "XTSD") == 0) { model->xtsd = value; return 0; }
    if (strcmp(param_name, "XTSSWS") == 0) { model->xtssws = value; return 0; }
    if (strcmp(param_name, "XTSSWD") == 0) { model->xtsswd = value; return 0; }
    if (strcmp(param_name, "XTSSWGS") == 0) { model->xtsswgs = value; return 0; }
    if (strcmp(param_name, "XTSSWGD") == 0) { model->xtsswgd = value; return 0; }
    if (strcmp(param_name, "TNJTS") == 0) { model->tnjts = value; return 0; }
    if (strcmp(param_name, "TNJTSSW") == 0) { model->tnjtssw = value; return 0; }
    if (strcmp(param_name, "TNJTSSWG") == 0) { model->tnjtsswg = value; return 0; }
    if (strcmp(param_name, "VTSS") == 0) { model->vtss = value; return 0; }
    if (strcmp(param_name, "VTSD") == 0) { model->vtsd = value; return 0; }
    if (strcmp(param_name, "VTSSWS") == 0) { model->vtssws = value; return 0; }
    if (strcmp(param_name, "VTSSWD") == 0) { model->vtsswd = value; return 0; }
    if (strcmp(param_name, "VTSSWGS") == 0) { model->vtsswgs = value; return 0; }
    if (strcmp(param_name, "VTSSWGD") == 0) { model->vtsswgd = value; return 0; }
    if (strcmp(param_name, "BVD") == 0) { model->bvd = value; return 0; }
    if (strcmp(param_name, "BVS") == 0) { model->bvs = value; return 0; }
    if (strcmp(param_name, "XJBVD") == 0) { model->xjbvd = value; return 0; }
    if (strcmp(param_name, "XJBVS") == 0) { model->xjbvs = value; return 0; }

    /* Overlap capacitance parameters */
    if (strcmp(param_name, "CGSL") == 0) { model->cgsl = value; return 0; }
    if (strcmp(param_name, "CGDL") == 0) { model->cgdl = value; return 0; }
    if (strcmp(param_name, "CKAPPAS") == 0) { model->ckappas = value; return 0; }
    if (strcmp(param_name, "CKAPPAD") == 0) { model->ckappad = value; return 0; }
    if (strcmp(param_name, "CF") == 0) { model->cf = value; return 0; }
    if (strcmp(param_name, "CLC") == 0) { model->clc = value; return 0; }
    if (strcmp(param_name, "CLE") == 0) { model->cle = value; return 0; }
    if (strcmp(param_name, "VFBCV") == 0) { model->vfbcv = value; return 0; }
    if (strcmp(param_name, "ACDE") == 0) { model->acde = value; return 0; }
    if (strcmp(param_name, "MOIN") == 0) { model->moin = value; return 0; }
    if (strcmp(param_name, "NOFF") == 0) { model->noff = value; return 0; }
    if (strcmp(param_name, "VOFFCV") == 0) { model->voffcv = value; model->voffcbn = value; return 0; }
    if (strcmp(param_name, "voffcv") == 0) { model->voffcv = value; model->voffcbn = value; return 0; }

    /* Gate resistance parameters */
    if (strcmp(param_name, "DMCG") == 0) { model->dmcg = value; return 0; }
    if (strcmp(param_name, "DMCI") == 0) { model->dmci = value; return 0; }
    if (strcmp(param_name, "DMDG") == 0) { model->dmdg = value; return 0; }
    if (strcmp(param_name, "DMCGT") == 0) { model->dmcgt = value; return 0; }
    if (strcmp(param_name, "XGW") == 0) { model->xgw = value; return 0; }
    if (strcmp(param_name, "XGL") == 0) { model->xgl = value; return 0; }
    if (strcmp(param_name, "RSHG") == 0) { model->rshg = value; return 0; }
    if (strcmp(param_name, "NGCON") == 0) { model->ngcon = value; return 0; }

    /* Temperature coefficients */
    if (strcmp(param_name, "TCJ") == 0) { model->tcj = value; return 0; }
    if (strcmp(param_name, "TCJSW") == 0) { model->tcjsw = value; return 0; }
    if (strcmp(param_name, "TCJSWG") == 0) { model->tcjswg = value; return 0; }
    if (strcmp(param_name, "TPB") == 0) { model->tpb = value; return 0; }
    if (strcmp(param_name, "TPBSW") == 0) { model->tpbsw = value; return 0; }
    if (strcmp(param_name, "TPBSWG") == 0) { model->tpbswg = value; return 0; }

    /* Other parameters */
    if (strcmp(param_name, "VFB") == 0) { model->vfb = value; return 0; }
    if (strcmp(param_name, "GBMIN") == 0) { model->gbmin = value; return 0; }
    if (strcmp(param_name, "IJTHDFWD") == 0) { model->ijthdfwd = value; return 0; }
    if (strcmp(param_name, "IJTHSFWD") == 0) { model->ijthsfwd = value; return 0; }
    if (strcmp(param_name, "IJTHDREV") == 0) { model->ijthdrev = value; return 0; }
    if (strcmp(param_name, "IJTHSREV") == 0) { model->ijthsrev = value; return 0; }

    return -1;  /* Unknown parameter - ignore for now */
}

/*
 * Get parameter by name
 */
int BSIM4_GetParam(const BSIM4_Model *model, const char *param_name, double *value)
{
    if (strcmp(param_name, "TOX") == 0 || strcmp(param_name, "TOXE") == 0) {
        *value = model->tox;
        return 0;
    }
    if (strcmp(param_name, "VTH0") == 0) { *value = model->vth0; return 0; }
    if (strcmp(param_name, "U0") == 0) {
        *value = model->u0 * 1e4;  /* Convert m^2/V-s to cm^2/V-s */
        return 0;
    }
    if (strcmp(param_name, "VSAT") == 0) { *value = model->vsat; return 0; }
    if (strcmp(param_name, "K1") == 0) { *value = model->k1; return 0; }
    if (strcmp(param_name, "K2") == 0) { *value = model->k2; return 0; }
    if (strcmp(param_name, "K3") == 0) { *value = model->k3; return 0; }
    if (strcmp(param_name, "ETA0") == 0) { *value = model->eta0; return 0; }
    if (strcmp(param_name, "ETAB") == 0) { *value = model->etab; return 0; }
    if (strcmp(param_name, "DSUB") == 0) { *value = model->dsub; return 0; }
    if (strcmp(param_name, "NSUB") == 0) { *value = model->nsub; return 0; }
    if (strcmp(param_name, "NDEP") == 0) { *value = model->ndep; return 0; }
    if (strcmp(param_name, "XJ") == 0) { *value = model->xj; return 0; }
    if (strcmp(param_name, "UA") == 0) { *value = model->ua; return 0; }
    if (strcmp(param_name, "UB") == 0) { *value = model->ub; return 0; }
    if (strcmp(param_name, "UC") == 0) { *value = model->uc; return 0; }
    if (strcmp(param_name, "PCLM") == 0) { *value = model->pclm; return 0; }
    if (strcmp(param_name, "RDSW") == 0) { *value = model->rdsw; return 0; }
    if (strcmp(param_name, "DELTA") == 0) { *value = model->delta; return 0; }
    if (strcmp(param_name, "VOFF") == 0) { *value = model->voff; return 0; }
    if (strcmp(param_name, "NFACTOR") == 0) { *value = model->nfactor; return 0; }
    if (strcmp(param_name, "MSTAR") == 0) { *value = model->mstar; return 0; }
    if (strcmp(param_name, "VOFFCVBN") == 0) { *value = model->voffcvbn; return 0; }
    if (strcmp(param_name, "VOFFCBN") == 0) { *value = model->voffcbn; return 0; }
    if (strcmp(param_name, "CDEP0") == 0) { *value = model->cdep0; return 0; }
    if (strcmp(param_name, "XDEP0") == 0) { *value = model->Xdep0; return 0; }
    if (strcmp(param_name, "SQRTPhi") == 0) { *value = model->sqrtPhi; return 0; }
    if (strcmp(param_name, "COXE") == 0) { *value = model->coxe; return 0; }

    return -1;  /* Unknown parameter */
}

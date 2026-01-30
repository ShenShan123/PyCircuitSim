/*
 * BSIM4.5.0 Core I-V Model
 *
 * This file contains the fundamental I-V equations extracted from
 * the original BSIM4.5.0 implementation (src/b4ld.c).
 *
 * Phase 1: Basic I-V Characteristics (without advanced effects)
 * - Threshold voltage (Vth)
 * - Mobility degradation (ueff)
 * - Gate overdrive (Vgsteff)
 * - Saturation voltage (Vdsat)
 * - Effective drain voltage (Vdseff)
 * - Core drain current (Ids)
 *
 * Author: PyCircuitSim Team
 */

#include "bsim4_iv_core.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Physical constants */
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10
#define Q_CHARGE 1.60219e-19
#define K_BOLTZMANN 1.380649e-23

/* Maximum/minimum values for numerical stability */
#define MAX_EXP 1e38
#define MIN_EXP 1e-38
#define EXP_THRESHOLD 88.0
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define CHARGE (Q_CHARGE * 1e6)  /* Convert to SI units */

/*
 * Poly Gate Depletion Effect
 * Original: b4ld.c lines 4709-4739
 *
 * Calculates effective gate voltage accounting for poly-Si gate depletion
 * This reduces the effective gate voltage at high Vgs
 */
static void bsim4_poly_depletion(
    const BSIM4_Model *m,
    double Vgs,
    double *Vgs_eff_out,
    double *dVgs_eff_dVg_out)
{
    double phi = 1.12;  /* Bandgap of Si + 2*phi_F for poly gate */
    double Vgs_eff = Vgs;
    double dVgs_eff_dVg = 1.0;

    /* Poly depletion is active when:
     * 1. ngate is in reasonable range (1e18 to 1e25 cm^-3)
     * 2. Vgs > phi (gate voltage exceeds barrier)
     */
    if ((m->ngate > 1.0e18) && (m->ngate < 1.0e25) && (Vgs > phi)) {
        double T1 = CHARGE * EPSSI * m->ngate / (m->coxe * m->coxe);
        double T8 = Vgs - phi;
        double T4 = sqrt(1.0 + 2.0 * T8 / T1);
        double T2 = 2.0 * T8 / (T4 + 1.0);
        double T3 = 0.5 * T2 * T2 / T1;  /* Vpoly */
        double T7 = 1.12 - T3 - 0.05;
        double T6 = sqrt(T7 * T7 + 0.224);
        double T5 = 1.12 - 0.5 * (T7 + T6);

        Vgs_eff = Vgs - T5;
        dVgs_eff_dVg = 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6);
    }

    *Vgs_eff_out = Vgs_eff;
    *dVgs_eff_dVg_out = dVgs_eff_dVg;
}

/*
 * Calculate effective gate-source voltage with smoothing function
 * Original: b4ld.c lines ~1088-1220
 *
 * This implements the full BSIM4 Vgsteff smoothing function which includes:
 * 1. 'n' parameter calculation (subthreshold swing)
 * 2. T10 calculation using mstar
 * 3. T9 calculation using voffcbn
 * 4. Final Vgsteff = T10 / T9
 */
static void bsim4_calc_vgs_eff(
    const BSIM4_States *s,
    const BSIM4_Model *m,
    BSIM4_Internal *i,
    double Vgs,
    double Vds,
    double Vbs)
{
    double Vgst, Vgsteff;
    double n, dn_dVb, dn_dVd;
    double Vtm, Phis, sqrtPhis, Xdep;
    double T0, T1, T2, T3, T4, T5, T9, T10, T11;
    double ExpVgst;
    double dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb;
    double dT10_dVg, dT10_dVd, dT10_dVb;
    double dT9_dVg, dT9_dVd, dT9_dVb;
    double dVth_dVb, dVth_dVd;
    double Vth;
    (void)s;  /* Suppress unused parameter warning */

    /* Get Vth and its derivatives */
    Vth = i->Vth;
    dVth_dVb = i->dVth_dVb;
    dVth_dVd = i->dVth_dVd;

    /* Calculate thermal voltage */
    Vtm = i->Vtm;

    /* Calculate surface potential */
    if (m->phin > 0.0) {
        Phis = m->phin;
    } else {
        double ni = 1.45e16;  /* m^-3 */
        double nsub_si = m->nsub * 1e6;
        Phis = 2.0 * Vtm * log(nsub_si / ni);
    }
    sqrtPhis = sqrt(Phis);

    /* Calculate depletion depth Xdep */
    /* Xdep = Xdep0 * sqrtPhis / sqrtPhi */
    if (m->sqrtPhi > 0.0) {
        Xdep = m->Xdep0 * sqrtPhis / m->sqrtPhi;
    } else {
        Xdep = m->Xdep0;  /* Simplified */
    }

    /* Calculate Vgst (gate-source overdrive before smoothing) */
    Vgst = Vgs - Vth;

    /* Calculate 'n' parameter (subthreshold swing) */
    /* Original: b4ld.c lines 1088-1109 */
    T1 = EPSSI / Xdep;
    T2 = m->nfactor * T1;
    T3 = m->cdsc + m->cdscb * Vbs + m->cdscd * Vds;
    T4 = (T2 + T3 * 1.0 + m->cit) / m->coxe;  /* Using Theta0 = 1.0 for now */

    if (T4 >= -0.5) {
        n = 1.0 + T4;
        dn_dVb = 0.0;  /* Simplified */
        dn_dVd = 0.0;  /* Simplified */
    } else {
        T0 = 1.0 / (3.0 + 8.0 * T4);
        n = (1.0 + 3.0 * T4) * T0;
        T0 *= T0;
        dn_dVb = 0.0;  /* Simplified */
        dn_dVd = 0.0;  /* Simplified */
    }

    /* Calculate T10 using mstar */
    /* Original: b4ld.c lines 1160-1185 */
    T0 = n * Vtm;
    T1 = m->mstar * Vgst;
    T2 = T1 / T0;
    if (T2 > EXP_THRESHOLD) {
        T10 = T1;
        dT10_dVg = m->mstar;
        dT10_dVd = -dVth_dVd * m->mstar;
        dT10_dVb = -dVth_dVb * m->mstar;
    } else if (T2 < -EXP_THRESHOLD) {
        T10 = Vtm * log(1.0 + MIN_EXP);
        T10 *= n;
        dT10_dVg = 0.0;
        dT10_dVd = T10 * dn_dVd;
        dT10_dVb = T10 * dn_dVb;
    } else {
        ExpVgst = exp(T2);
        T3 = Vtm * log(1.0 + ExpVgst);
        T10 = n * T3;
        dT10_dVg = m->mstar * ExpVgst / (1.0 + ExpVgst);
        dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n);
        dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n);
    }

    /* Calculate T9 using voffcbn */
    /* Original: b4ld.c lines 1187-1213 */
    T1 = m->voffcbn - (1.0 - m->mstar) * Vgst;
    T2 = T1 / T0;
    if (T2 < -EXP_THRESHOLD) {
        T3 = m->coxe * MIN_EXP / m->cdep0;
        T9 = m->mstar + T3 * n;
        dT9_dVg = 0.0;
        dT9_dVd = dn_dVd * T3;
        dT9_dVb = dn_dVb * T3;
    } else if (T2 > EXP_THRESHOLD) {
        T3 = m->coxe * MAX_EXP / m->cdep0;
        T9 = m->mstar + T3 * n;
        dT9_dVg = 0.0;
        dT9_dVd = dn_dVd * T3;
        dT9_dVb = dn_dVb * T3;
    } else {
        ExpVgst = exp(T2);
        T3 = m->coxe / m->cdep0;
        T4 = T3 * ExpVgst;
        T5 = T1 * T4 / T0;
        T9 = m->mstar + n * T4;
        dT9_dVg = T3 * (m->mstar - 1.0) * ExpVgst / Vtm;
        dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb;
        dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd;
    }

    /* Calculate Vgsteff = T10 / T9 */
    /* Original: b4ld.c lines 1215-1219 */
    T11 = T9 * T9;
    Vgsteff = T10 / T9;
    dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11;
    dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11;
    dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11;

    /* Store results */
    i->Vgsteff = Vgsteff;
    i->dVgsteff_dVg = dVgsteff_dVg;
    i->dVgsteff_dVd = dVgsteff_dVd;
    i->dVgsteff_dVb = dVgsteff_dVb;
}

/*
 * Calculate threshold voltage
 * Original: b4ld.c lines ~988-1135
 */
static void bsim4_calc_vth(
    const BSIM4_Model *m,
    double Vds,
    double Vbs,
    double Weff,
    double Leff_in,
    double *Vth_out,
    double *dVth_dVb_out,
    double *dVth_dVd_out)
{
    double Vth, Phis, sqrtPhisb;
    double Vbseff;
    double DIBL_Sft, dDIBL_Sft_dVd, dDIBL_Sft_dVb;
    double lt1, Theta0, dTheta0_dVb, Delt_vth, dDelt_vth_dVb;
    double Vth_NarrowW;
    double K1ox, k2ox;
    double dVth_dVb, dVth_dVd;

    /* Calculate surface potential (2*phi_F) */
    /* If phin is provided, use it; otherwise calculate from NSUB */
    if (m->phin > 0.0) {
        Phis = m->phin;
    } else {
        /* Calculate 2*phi_F from NSUB (substrate doping) */
        /* 2*phi_F = 2 * (kT/q) * ln(NSUB/ni) */
        /* where ni (intrinsic carrier concentration) = 1.45e10 cm^-3 for Si at 300K */
        double ni = 1.45e16;  /* m^-3, converted from 1.45e10 cm^-3 */
        double Vtm = K_BOLTZMANN * 300.0 / Q_CHARGE;  /* Thermal voltage at 300K */
        double nsub_si = m->nsub * 1e6;  /* Convert cm^-3 to m^-3 */
        Phis = 2.0 * Vtm * log(nsub_si / ni);
    }

    /* Calculate effective body voltage */
    Vbseff = Vbs;
    if (Vbseff > 0.0) Vbseff = 0.0;

    /* Calculate sqrt terms for body effect */
    sqrtPhisb = sqrt(Phis - Vbseff);
    double sqrtPhis0 = sqrt(Phis);  /* sqrt(Phis) at Vbs=0 */

    /* K1ox is the body effect coefficient */
    K1ox = m->type * m->k1;

    /* Calculate K1ox (body effect coefficient) */
    /* For PMOS, body effect works in opposite direction, so multiply by type */
    K1ox = m->type * m->k1;

    /* Calculate k2ox (simplified) */
    k2ox = m->k2;

        /* Phase 3: Calculate proper short-channel effect parameters (DIBL) */
    /* Based on BSIM4.5.0 technical manual */

    /* For 45nm technology, estimate characteristic length from geometry */
    /* TODO: Use proper BSIM4 parameters (ll, lln) when available */
    /* Using larger Lchar to reduce DIBL effect for now */
    double Lchar = Leff_in;  /* Using L_in as characteristic length minimizes DIBL */

    /* Calculate Vds dependence for DIBL (Theta0) */
    /* Using Leff_in for the exponent calculation */
    double Leff_dibl = Leff_in;  /* Actual effective length */

    /* Prevent numerical overflow in exponential */
    double exponent = -Vds / (2.0 * Lchar);
    if (exponent < -50.0) {
        Theta0 = 0.0;
    } else if (exponent > 50.0) {
        Theta0 = 1.0;
    } else {
        Theta0 = exp(exponent);
    }

    /* Calculate short-channel effect coefficient */
    /* For 45nm L=45nm, the short-channel effect is significant */
    /* Using (L / (L + 2*Lchar))^dvt1 as the scaling factor */
    double L_ratio = Leff_dibl / (Leff_dibl + 2.0 * Lchar);
    double sceff = pow(L_ratio, m->dvt1);

    /* Calculate DIBL shift */
    /* BSIM4 formula: Delta_vth = DVT0 * (1 - Theta0) * sceff */
    /* Where Theta0 = exp(-Vds / Lchar) gives Vds dependence (drain-induced) */
    /* And sceff = (L / (L + 2*Lchar))^DVT1 gives L dependence (short-channel) */
    /* NOTE: Original code had * Phis which was incorrect - removed */
    /* TEMPORARY: Disable DIBL to get baseline Vth */
    Delt_vth = 0.0;  /* m->dvt0 * (1.0 - Theta0) * sceff; */

    /* Clamp DIBL shift to reasonable range */
    double max_dibl = 0.55 * Phis;  /* Maximum DIBL shift is 55% of Phis */
    if (Delt_vth > max_dibl) Delt_vth = max_dibl;
    if (Delt_vth < 0.0) Delt_vth = 0.0;

    /* Calculate narrow width effect */
    Vth_NarrowW = m->tox * (Phis - Vbseff) / Weff;

    /* Calculate Vth with DIBL */
    DIBL_Sft = Delt_vth;
    dDIBL_Sft_dVd = 0.0;  /* TODO: Calculate derivative in later phase */
    dDIBL_Sft_dVb = 0.0;

    /* Calculate final threshold voltage */
    /* Body effect: K1ox * (sqrtPhisb - sqrtPhis0) gives change from Vbs */
    /* When Vbs=0: sqrtPhisb = sqrtPhis0, so body effect = 0 */
    Vth = m->type * m->vth0 + K1ox * (sqrtPhisb - sqrtPhis0) - k2ox * Vbseff
          - Delt_vth - Vth_NarrowW;

    /* Calculate derivatives */
    dVth_dVb = -K1ox * (1.0 / (2.0 * sqrtPhisb)) - k2ox;  /* Simplified */
    dVth_dVd = 0.0;  /* TODO: Add proper derivative */

    *Vth_out = Vth;
    if (dVth_dVb_out) *dVth_dVb_out = dVth_dVb;
    if (dVth_dVd_out) *dVth_dVd_out = dVth_dVd;
}

/*
 * Calculate effective mobility
 * Original: b4ld.c lines ~1220-1350
 */
static void bsim4_calc_ueff(
    const BSIM4_Model *m,
    double Vgs,
    double Vth,
    double Vds,
    double Vbs,
    double *ueff_out,
    double *dueff_dVg_out,
    double *dueff_dVd_out,
    double *dueff_dVb_out)
{
    double ueff, u0;
    double T0, T1, T2, T3, T4, T5, T9;
    double ua, ub, uc, eu;
    double Denomi, dDenomi_dVg, dDenomi_dVd, dDenomi_dVb;
    double Vgsteff, Vbseff, Vtm;

    /* Get mobility parameters */
    u0 = m->u0;
    ua = m->ua;
    ub = m->ub;
    uc = m->uc;
    eu = m->eu;

    /* Calculate effective gate overdrive and body bias */
    Vgsteff = Vgs - Vth;
    if (Vgsteff < 0.0) Vgsteff = 0.0;

    Vbseff = Vbs;
    if (Vbseff > 0.0) Vbseff = 0.0;

    Vtm = K_BOLTZMANN * m->temp / Q_CHARGE;

    /* Full BSIM4 mobility model based on mobMod */
    if (m->mobMod == 0) {
        /* Mobility Model 0: Basic model */
        T0 = Vgsteff + Vth + Vth;  /* 2*Vth + Vgsteff */
        T2 = ua + uc * Vbseff;
        T3 = T0 / m->tox;
        T5 = T3 * (T2 + ub * T3);

        dDenomi_dVg = (T2 + 2.0 * ub * T3) / m->tox;
        dDenomi_dVd = dDenomi_dVg * 2.0 * 0.0;  /* Simplified: dVth/dVd = 0 */
        dDenomi_dVb = dDenomi_dVg * 0.0 + uc * T3;  /* Simplified */
    } else if (m->mobMod == 1) {
        /* Mobility Model 1: Enhanced model with uc dependence */
        T0 = Vgsteff + Vth + Vth;
        T2 = 1.0 + uc * Vbseff;
        T3 = T0 / m->tox;
        T4 = T3 * (ua + ub * T3);
        T5 = T4 * T2;

        dDenomi_dVg = (ua + 2.0 * ub * T3) * T2 / m->tox;
        dDenomi_dVd = dDenomi_dVg * 2.0 * 0.0;
        dDenomi_dVb = dDenomi_dVg * 0.0 + uc * T4;
    } else {
        /* Mobility Model 2: Power law model with eu */
        double vtfbphi1 = 1.12 - 0.7;  /* Simplified */
        T0 = (Vgsteff + vtfbphi1) / m->tox;
        T1 = exp(eu * log(T0));
        T2 = ua + uc * Vbseff;
        T5 = T1 * T2;

        dDenomi_dVg = T2 * (T1 * eu / T0 / m->tox);
        dDenomi_dVd = 0.0;
        dDenomi_dVb = T1 * uc;
    }

    /* Clamp denominator for numerical stability */
    if (T5 >= -0.8) {
        Denomi = 1.0 + T5;
    } else {
        T9 = 1.0 / (7.0 + 10.0 * T5);
        Denomi = (0.6 + T5) * T9;
        T9 *= T9;
        dDenomi_dVg *= T9;
        dDenomi_dVd *= T9;
        dDenomi_dVb *= T9;
    }

    /* Calculate effective mobility */
    ueff = u0 / Denomi;
    T9 = -ueff / Denomi;

    /* Calculate derivatives */
    if (dueff_dVg_out) {
        *dueff_dVg_out = T9 * dDenomi_dVg;
    }
    if (dueff_dVd_out) {
        *dueff_dVd_out = T9 * dDenomi_dVd;
    }
    if (dueff_dVb_out) {
        *dueff_dVb_out = T9 * dDenomi_dVb;
    }

    *ueff_out = ueff;
}

/*
 * Calculate saturation voltage
 * Original: b4ld.c lines ~1407-1466
 */
static void bsim4_calc_vdsat(
    const BSIM4_Model *m,
    double Vgsteff,
    double ueff,
    double Leff,
    double Weff,
    double Rds,
    double *Vdsat_out,
    double *dVdsat_dVg_out,
    double *dVdsat_dVd_out,
    double *dVdsat_dVb_out)
{
    double Vdsat, EsatL, T0, T1, T2, T3;
    double Cox, Abulk, Vgst2Vtm;
    double dVdsat_dVg, dVdsat_dVd, dVdsat_dVb;

    /* Calculate oxide capacitance */
    Cox = m->epsrox * EPS0 / m->tox;

    /* Calculate Abulk (bulk charge effect) */
    Abulk = 1.0 + m->k1 / (2.0 * sqrt(m->phin + 0.4));
    if (Abulk < 0.01) Abulk = 0.01;

    /* Calculate Vgst2Vtm */
    /* Vgst2Vtm = Vgsteff + 2*Vtm (thermal voltage) */
    {
        double Vtm = K_BOLTZMANN * 300.0 / Q_CHARGE;  /* ~0.0259V at 300K */
        Vgst2Vtm = Vgsteff + 2.0 * Vtm;
        if (Vgst2Vtm < 0.0) Vgst2Vtm = 0.0;
    }

    /* Calculate EsatL (saturation field * length) */
    EsatL = 2.0 * m->vsat / ueff;

    /* Calculate Vdsat (simplified for Phase 1) */
    if (Rds == 0.0) {
        /* No source/drain resistance */
        T0 = Abulk * EsatL + Vgst2Vtm;
        Vdsat = Vgst2Vtm * EsatL / T0;

        dVdsat_dVg = EsatL * Vgst2Vtm / (T0 * T0);
        dVdsat_dVd = 0.0;
        dVdsat_dVb = 0.0;
    } else {
        /* With source/drain resistance - simplified approximation */
        Vdsat = Vgst2Vtm * EsatL / (Abulk * EsatL + Vgst2Vtm);
        dVdsat_dVg = EsatL * Vgst2Vtm / pow(Abulk * EsatL + Vgst2Vtm, 2);
        dVdsat_dVd = 0.0;
        dVdsat_dVb = 0.0;
    }

    *Vdsat_out = Vdsat;
    if (dVdsat_dVg_out) *dVdsat_dVg_out = dVdsat_dVg;
    if (dVdsat_dVd_out) *dVdsat_dVd_out = dVdsat_dVd;
    if (dVdsat_dVb_out) *dVdsat_dVb_out = dVdsat_dVb;
}

/*
 * Calculate effective drain voltage (smooth Vdseff)
 * Original: b4ld.c lines ~1469-1508
 */
static void bsim4_calc_vdseff(
    const BSIM4_Model *m,
    double Vds,
    double Vdsat,
    double *Vdseff_out,
    double *dVdseff_dVg_out,
    double *dVdseff_dVd_out,
    double *dVdseff_dVb_out)
{
    double Vdseff, delta;
    double T1, T2, T9;
    double dVdseff_dVg, dVdseff_dVd, dVdseff_dVb;

    delta = m->delta;

    /* Calculate smooth Vdseff using hyperbolic transition */
    T1 = Vdsat - Vds - delta;
    T2 = sqrt(T1 * T1 + 4.0 * delta * Vdsat);

    if (T1 >= 0.0) {
        Vdseff = Vdsat - 0.5 * (T1 + T2);
    } else {
        T9 = 2.0 * delta;
        double T4 = T9 / (T2 - T1);
        double T5 = 1.0 - T4;
        Vdseff = Vdsat * T5;
    }

    /* Clamp Vdseff */
    if (Vdseff > Vds) Vdseff = Vds;
    if (Vdseff < 0.0) Vdseff = 0.0;

    *Vdseff_out = Vdseff;
    if (dVdseff_dVg_out) *dVdseff_dVg_out = 0.0;
    if (dVdseff_dVd_out) *dVdseff_dVd_out = 0.0;
    if (dVdseff_dVb_out) *dVdseff_dVb_out = 0.0;
}

/*
 * Calculate core drain current (Idl) without advanced effects
 * Original: b4ld.c lines ~1608-1633
 */
static void bsim4_calc_idl(
    const BSIM4_Model *m,
    const BSIM4_States *s,
    BSIM4_Internal *i,
    double Vds,
    double Vbs)
{
    double ueff, Cox, Coxeff, CoxeffWovL;
    double beta, fgche1, fgche2, gche;
    double T0, T1, Abulk, Abulk0, Vgst2Vtm;
    double EsatL, Rds, dEsatL_dVg;
    double Vgsteff, Vdseff, Vdsat, Vth, Leff, Weff, Xdep;

    /* Get internal values */
    Vgsteff = i->Vgsteff;
    Vdseff = i->Vdseff;
    Vdsat = i->Vdsat;
    Vth = i->Vth;
    Leff = i->Leff;
    Weff = i->Weff;

    /* Calculate mobility */
    bsim4_calc_ueff(m, s->Vgs, Vth, Vds, s->Vbs, &ueff,
                    &i->dueff_dVg, &i->dueff_dVd, &i->dueff_dVb);

    /* Calculate oxide capacitance */
    Cox = m->epsrox * EPS0 / m->tox;
    Coxeff = Cox;  /* Simplified, no quantum effects in Phase 1 */
    CoxeffWovL = Coxeff * Weff / Leff;

    /* Calculate transconductance factor */
    beta = ueff * CoxeffWovL;

    /* Calculate saturation field */
    EsatL = 2.0 * m->vsat / ueff;
    dEsatL_dVg = -2.0 * m->vsat / (ueff * ueff) * i->dueff_dVg;

    /* Calculate Abulk (bulk charge effect) - BSIM4 style */
    /* First calculate depletion depth */
    {
        double Phis, sqrtPhis;

        /* Calculate surface potential */
        if (m->phin > 0.0) {
            Phis = m->phin;
        } else {
            /* Calculate 2*phi_F from NSUB */
            double ni = 1.45e16;  /* m^-3 */
            double Vtm = K_BOLTZMANN * 300.0 / Q_CHARGE;
            double nsub_si = m->nsub * 1e6;
            Phis = 2.0 * Vtm * log(nsub_si / ni);
        }

        sqrtPhis = sqrt(Phis);

        /* Calculate depletion depth Xdep */
        /* Xdep = sqrt(2*eps_si*Phis/(q*Nsub)) */
        /* Simplified: use sqrt(Phis) as proxy */
        Xdep = sqrtPhis / sqrt(m->phin + 0.4);  /* Normalized */

        /* Calculate Abulk0 (basic bulk charge effect) */
        T0 = 1.0 / sqrt(1.0 + 2.0 * Leff / Xdep);
        Abulk0 = 1.0 + m->k1 / (2.0 * sqrtPhis) * T0;

        /* Add geometry-dependent effects */
        if (m->a0 > 0.0) {
            double T1, T2, T5, T6, T7, T8;
            T1 = Leff / (Leff + 2.0 * Xdep);
            T6 = T1 * T1;
            T7 = T1 * T6;
            T2 = m->a0 * T7 + m->b0 / (Weff + m->b1);
            Abulk0 = 1.0 + T1 * T2;

            /* Add ags Vgsteff dependence */
            if (m->ags > 0.0) {
                T8 = m->ags * m->a0 * T7;
                Abulk = Abulk0 - T8 * Vgsteff;
            } else {
                Abulk = Abulk0;
            }
        } else {
            Abulk = Abulk0;
        }

        /* Clamp Abulk */
        if (Abulk < 0.1) {
            T1 = 1.0 / (3.0 - 20.0 * Abulk);
            Abulk = (0.2 - Abulk) * T1;
        }
        if (Abulk < 0.01) Abulk = 0.01;
    }

    /* Calculate Vgst2Vtm */
    /* Vgst2Vtm = Vgsteff + 2*Vtm (thermal voltage) */
    {
        double Vtm = K_BOLTZMANN * 300.0 / Q_CHARGE;  /* ~0.0259V at 300K */
        Vgst2Vtm = Vgsteff + 2.0 * Vtm;
        if (Vgst2Vtm < 0.0) Vgst2Vtm = 0.0;
    }

    /* Calculate source/drain resistance - BSIM4 style */
    /* rsw and rdw are in ohms·µm, Weff needs to be in µm */
    {
        double Weff_µm = Weff * 1e6;  /* Convert m to µm */
        /* Total source + drain resistance */
        Rds = (m->rsw + m->rdw) / Weff_µm;
    }

    /* Calculate fgche1 and fgche2 (field factors) - BSIM4 style */
    /* Original: fgche1 = Vgsteff * (1.0 - 0.5 * Vdseff * Abulk / Vgst2Vtm) */
    T0 = 1.0 - 0.5 * Vdseff * Abulk / Vgst2Vtm;
    fgche1 = Vgsteff * T0;
    fgche2 = 1.0 + Vdseff / EsatL;

    /* Calculate channel conductance */
    gche = beta * fgche1 / fgche2;

    /* Calculate drain current with Rds - BSIM4 style
     * Idl = (gche * Vds) / (1 + gche * Rds)
     * NOTE: gche is already d(Id)/d(Vds), so Id = gche * Vds
     */
    T1 = 1.0 + gche * Rds;
    i->Idl = (gche * Vds) / T1;  /* Multiply by Vds to get actual current */

    /* Calculate derivatives (simplified for Phase 1) */
    i->dIdl_dVg = beta / fgche2 * (T0 - Vgsteff * 0.5 * Abulk / Vgst2Vtm * (1.0 - Vdseff / EsatL * dEsatL_dVg / Abulk));
    i->dIdl_dVd = 0.0;
    i->dIdl_dVb = 0.0;

    i->gche = gche;
    i->beta = beta;
    i->EsatL = EsatL;
}

/*
 * Main entry point: Calculate BSIM4.5.0 core I-V characteristics
 *
 * This function computes the drain current and conductances at a given
 * bias point using the BSIM4.5.0 model equations.
 *
 * Inputs:
 *   m  - Model parameters
 *   s  - State (voltages)
 *   Vds, Vgs, Vbs - Terminal voltages
 *
 * Outputs:
 *   i  - Internal state (currents, conductances)
 */
int bsim4_iv_evaluate(
    const BSIM4_Model *m,
    const BSIM4_Instance *inst,
    double Vds,
    double Vgs,
    double Vbs,
    BSIM4_Internal *i)
{
    BSIM4_States s;
    double type = m->type;
    double ueff;
    int ret;

    /* Initialize internal state */
    memset(i, 0, sizeof(BSIM4_Internal));
    s.Vgs = Vgs;
    s.Vds = Vds;
    s.Vbs = Vbs;

    /* Calculate effective geometry */
    /* TODO: Need to understand the correct BSIM4 geometry formula */
    /* For now, use drawn dimensions without xl/xw correction */
    /* Note: The xl parameter interpretation needs verification */
    /* Calculate effective length and width with geometry corrections */
    i->Leff = inst->L - m->xl;  /* Apply length offset correction */
    i->Weff = inst->W - m->xw;  /* Apply width offset correction */

    /* Calculate thermal voltage */
    i->Vtm = K_BOLTZMANN * m->temp / Q_CHARGE;

    /* Calculate threshold voltage */
    double dVth_dVb, dVth_dVd;
    bsim4_calc_vth(m, Vds, Vbs, i->Weff, i->Leff,
                   &i->Vth, &dVth_dVb, &dVth_dVd);
    i->dVth_dVb = dVth_dVb;
    i->dVth_dVd = dVth_dVd;

    /* Apply poly depletion effect to get effective gate voltage */
    double Vgs_eff, dVgs_eff_dVg;
    bsim4_poly_depletion(m, Vgs, &Vgs_eff, &dVgs_eff_dVg);

    /* Calculate Vgsteff - use simple calculation for I-V model */
    /* Note: Use effective gate voltage from poly depletion */
    i->Vgsteff = Vgs_eff - i->Vth;
    if (i->Vgsteff < 0.0) i->Vgsteff = 0.0;
    i->dVgsteff_dVg = dVgs_eff_dVg;
    i->dVgsteff_dVd = 0.0;
    i->dVgsteff_dVb = 0.0;

    /* Calculate mobility with full model (including Vbs dependence) */
    bsim4_calc_ueff(m, Vgs_eff, i->Vth, Vds, Vbs, &ueff,
                    &i->dueff_dVg, &i->dueff_dVd, &i->dueff_dVb);

    /* Calculate saturation voltage */
    /* Rds using rsw and rdw (in ohms·µm) */
    {
        double Weff_µm = i->Weff * 1e6;
        double Rds = (m->rsw + m->rdw) / Weff_µm;
        bsim4_calc_vdsat(m, i->Vgsteff, ueff, i->Leff, i->Weff, Rds,
                         &i->Vdsat, &i->dVdsat_dVg, &i->dVdsat_dVd, &i->dVdsat_dVb);
    }

    /* Calculate effective drain voltage */
    bsim4_calc_vdseff(m, Vds, i->Vdsat,
                      &i->Vdseff, &i->dVdseff_dVg, &i->dVdseff_dVd, &i->dVdseff_dVb);

    /* Calculate core drain current */
    bsim4_calc_idl(m, &s, i, Vds, Vbs);

    /* Apply device type sign */
    if (type < 0) {
        i->Idl = -i->Idl;
        i->dIdl_dVg = -i->dIdl_dVg;
    }

    /* ============================================================
     * SECOND-ORDER EFFECTS: DIBL, CLM, SCBE
     * ============================================================ */

    /* Effect 1: Add DIBL to Ids (Vds-dependent current increase)
     * This accounts for drain-induced barrier lowering
     * Modeled as a multiplicative factor based on Vds
     */
    double VADIBL, dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb;
    double Idsa, dIdsa_dVg, dIdsa_dVd, dIdsa_dVb;

    /* Calculate VADIBL characteristic voltage for DIBL
     * This is similar to the VASCBE calculation but for DIBL effect
     */
    if (m->dvt1 > 0.0) {
        /* DIBL is active - calculate VADIBL */
        /* Using simplified model: VADIBL proportional to 1/Vds at low Vds */
        /* Full BSIM4 has much more complex calculation */
        double Vgst2Vtm = fabs(i->Vgsteff) - 2.0 * i->Vtm;
        if (Vgst2Vtm < 0.0) Vgst2Vtm = 0.0;

        /* VADIBL increases with Vds (drain-induced effect) */
        /* Simplified: VADIBL = (Vgst2Vtm - const) / scaling */
        double VADIBL_const = Vgst2Vtm * 0.5;
        double VADIBL_scaling = 0.1;  /* 1/(m^-1) characteristic */

        if (Vds != 0.0) {
            VADIBL = VADIBL_const / (fabs(Vds) + VADIBL_scaling);
        } else {
            VADIBL = 0.0;
        }

        /* Derivatives (simplified) */
        dVADIBL_dVg = 0.5 / (fabs(Vds) + VADIBL_scaling);  /* Approximate */
        dVADIBL_dVd = -VADIBL_const * VADIBL_scaling / (Vds * Vds + VADIBL_scaling * fabs(Vds));
        dVADIBL_dVb = 0.0;  /* Simplified */
    } else {
        VADIBL = 0.0;
        dVADIBL_dVg = 0.0;
        dVADIBL_dVd = 0.0;
        dVADIBL_dVb = 0.0;
    }

    /* Apply DIBL to Ids: Idsa = Idl * (1 + diffVds / VADIBL) */
    double diffVds = fabs(Vds);
    double DIBL_factor = 1.0 + (VADIBL > 1e-10 ? diffVds / VADIBL : 0.0);

    Idsa = i->Idl * DIBL_factor;
    dIdsa_dVg = i->dIdl_dVg * DIBL_factor;
    dIdsa_dVd = i->dIdl_dVd * DIBL_factor;
    if (VADIBL > 1e-10) {
        dIdsa_dVd += i->Idl * diffVds * dVADIBL_dVd / VADIBL;
    }
    dIdsa_dVb = i->dIdl_dVb * DIBL_factor;

    /* Effect 2: Add CLM (Channel Length Modulation) to Ids
     * CLM accounts for channel length reduction at high Vds
     * Modeled using Abulk factor and log(Va/Vasat) relationship
     */
    double Abulk, dAbulk_dVg, dAbulk_dVd, dAbulk_dVb;
    double Idclm, dIdclm_dVg, dIdclm_dVd, dIdclm_dVb;

    /* Calculate Abulk (bulk charge effect coefficient)
     * This accounts for charge sharing affecting channel length
     */
    if (m->a0 > 0.0 || m->b0 > 0.0 || m->b1 > 0.0) {
        /* Geometry-dependent Abulk calculation */
        /* Based on BSIM4.5.0 equations */

        /* Calculate depletion width */
        double Xdep;  /* Depletion depth */
        if (m->Xdep0 > 0.0) {
            Xdep = m->Xdep0;
        } else {
            /* Calculate from ndep if not specified */
            double ni = 1.45e16;  /* m^-3 */
            double Vtm = i->Vtm;
            double nsub_si = m->ndep * 1e6;  /* cm^-3 to m^-3 */
            double Phis = 2.0 * Vtm * log(nsub_si / ni);
            double Phis_max = Phis;
            /* Depletion depth: sqrt(2*eps_si*Phis/(q*Ndep)) */
            double eps_si = 11.7 * EPS0;  /* Silicon permittivity */
            double q = Q_CHARGE;
            /* Xdep = sqrt(2*eps_si*Phis_max/(q*m->ndep*1e6)) */
            Xdep = sqrt(2.0 * eps_si * Phis_max / (q * m->ndep * 1e6));
        }

        /* Calculate junction depth parameter */
        double sqrt_xj_Xdep = sqrt(m->xj * Xdep);

        /* Calculate Leff including lateral diffusion */
        double Leff_lateral = i->Leff + 2.0 * sqrt_xj_Xdep;

        /* Calculate geometry factors */
        double T5 = i->Leff / Leff_lateral;
        double T6 = T5 * T5;
        double T7 = T5 * T6;

        /* Calculate body effect term */
        double K1ox = m->type * m->k1;
        double sqrtPhisb;
        {
            double Phis;
            if (m->phin > 0.0) {
                Phis = m->phin;
            } else {
                double ni = 1.45e16;
                double Vtm = i->Vtm;
                double nsub_si = m->nsub * 1e6;
                Phis = 2.0 * Vtm * log(nsub_si / ni);
            }
            double Vbseff = Vbs;
            if (Vbseff > 0.0) Vbseff = 0.0;
            sqrtPhisb = sqrt(Phis - Vbseff);
        }

        /* Calculate T1 and T2 terms */
        double T1_geom = 0.5 * K1ox * sqrtPhisb;
        double T2_body = (0.5 * K1ox * sqrtPhisb) - m->type * m->k2 - m->k3b * 0.0;  /* k3b * Vth_NarrowW */

        /* Calculate T2 from geometry */
        double tmp2_geom = m->a0 * T5;
        double tmp3_geom = i->Weff + m->b1;  /* Weff + b1 */
        double tmp4_geom = m->b0 / tmp3_geom;
        double T2_geom = tmp2_geom + tmp4_geom;

        /* Calculate Abulk0 (bias-independent) */
        double Abulk0 = 1.0 + T1_geom * T2_geom;

        /* Add gate bias dependence (if ags is specified) */
        double dAbulk_dVg = -T1_geom * (m->ags * m->a0 * T7);

        /* Calculate final Abulk with Vgsteff dependence */
        Abulk = Abulk0 + dAbulk_dVg * i->Vgsteff;

        /* Clamp Abulk to avoid numerical issues */
        if (Abulk0 < 0.1) {
            double T9 = 1.0 / (3.0 - 20.0 * Abulk0);
            Abulk0 = (0.2 - Abulk0) * T9;
        }
        if (Abulk < 0.1) {
            double T9 = 1.0 / (3.0 - 20.0 * Abulk);
            Abulk = (0.2 - Abulk) * T9;
            dAbulk_dVg *= T9 * T9;
        }

        /* Apply body effect coefficient (keta) if specified */
        if (m->keta != 0.0) {
            double T2_keta = m->keta * Vbs;  /* Reuse Vbseff calculation */
            if (T2_keta >= -0.9) {
                double T0_keta = 1.0 / (1.0 + T2_keta);
                Abulk *= T0_keta;
                Abulk0 *= T0_keta;
                dAbulk_dVg *= T0_keta;
                dAbulk_dVb = dAbulk_dVb * T0_keta + Abulk0 * (-m->keta * T0_keta * T0_keta);
            } else {
                /* For very large negative keta*Vbs */
                double T1_keta = 1.0 / (0.8 + T2_keta);
                double T0_keta = (17.0 + 20.0 * T2_keta) * T1_keta;
                double dketa_dVb = -m->keta * T1_keta * T1_keta;
                Abulk *= T0_keta;
                Abulk0 *= T0_keta;
                dAbulk_dVg *= T0_keta;
                dAbulk_dVb = dAbulk_dVb * T0_keta + Abulk0 * dketa_dVb;
            }
        }
    } else {
        /* No CLM specified */
        Abulk = 1.0;
        dAbulk_dVg = 0.0;
        dAbulk_dVd = 0.0;
        dAbulk_dVb = 0.0;
    }

    /* Apply CLM using log(Va/Vasat) relationship
     * The CLM effect is: Id *= (1 + log(Va/Vasat) / Abulk)
     * where Va is actual drain voltage and Vasat is saturation voltage
     * Note: We use Vds (not Vdseff) because Vdseff is clamped to Vdsat
     */
    if (fabs(i->Vdsat) > 1e-6 && Abulk > 0.1) {
        /* Use actual Vds for CLM (not clamped Vdseff) */
        double Va = fabs(Vds);
        double Vasat = i->Vdsat;

        /* Check if in saturation */
        if (Va <= Vasat) {
            /* Linear region - no CLM effect */
            Idclm = Idsa;
            dIdclm_dVg = dIdsa_dVg;
            dIdclm_dVd = dIdsa_dVd;
            dIdclm_dVb = dIdsa_dVb;
        } else {
            /* Saturation region - apply CLM */
            double Cclm = Abulk;  /* CLM coefficient */

            /* Calculate log(Va/Vasat) */
            double ratio = Va / Vasat;
            if (ratio < 1.0) ratio = 1.0;  /* Clamp for numerical stability */

            double log_ratio = log(ratio);
            double CLM_factor = 1.0 + log_ratio / Cclm;

            /* Apply CLM */
            Idclm = Idsa * CLM_factor;

            /* Derivatives (simplified - Vds derivative gives CLM contribution to Gds) */
            dIdclm_dVg = dIdsa_dVg * CLM_factor;
            dIdclm_dVd = dIdsa_dVd * CLM_factor + Idsa / (Cclm * Va);
            dIdclm_dVb = dIdsa_dVb * CLM_factor;
        }
    } else {
        /* No CLM or Vasat too small */
        Idclm = Idsa;
        dIdclm_dVg = dIdsa_dVg;
        dIdclm_dVd = dIdsa_dVd;
        dIdclm_dVb = dIdsa_dVb;
    }

    /* Effect 3: Add SCBE (Substrate Current Body Effect) to Ids
     * This accounts for impact ionization at high drain fields
     * Modeled as additional current factor based on VASCBE
     */
    double VASCBE, dVASCBE_dVg, dVASCBE_dVd, dVASCBE_dVb;
    double Id_final, dId_final_dVg, dId_final_dVd, dId_final_dVb;

    if (m->pscbe2 > 0.0 && m->pscbe1 > 0.0) {
        /* SCBE is active */
        double litl_val;
        if (m->litl > 0.0) {
            litl_val = m->litl;
        } else {
            /* If litl not specified, use fraction of Leff */
            litl_val = 0.1 * i->Leff;  /* 10% of Leff as lateral length */
        }

        /* Calculate VASCBE (critical voltage for substrate current) */
        double diffVds_SCBE = fabs(Vds);

        if (diffVds_SCBE > m->pscbe1 * litl_val / 50.0) {
            /* Normal SCBE operation */
            double T0_SCBE = m->pscbe1 * litl_val / diffVds_SCBE;
            VASCBE = i->Leff * exp(T0_SCBE) / m->pscbe2;
        } else {
            /* Very low Vds - set VASCBE to large value to minimize effect */
            VASCBE = 1e10;  /* Effectively infinite */
        }

        /* Apply SCBE to Ids: Id = Idclm * (1 + diffVds / VASCBE) */
        double SCBE_factor = 1.0;
        if (VASCBE > 1.0 && VASCBE < 1e10) {
            SCBE_factor = 1.0 + diffVds_SCBE / VASCBE;
        }

        Id_final = Idclm * SCBE_factor;
        dId_final_dVg = dIdclm_dVg * SCBE_factor;
        dId_final_dVd = dIdclm_dVd * SCBE_factor + Idclm * diffVds_SCBE / VASCBE;
        dId_final_dVb = dIdclm_dVb * SCBE_factor;

    } else {
        /* SCBE not specified or disabled */
        Id_final = Idclm;
        dId_final_dVg = dIdclm_dVg;
        dId_final_dVd = dIdclm_dVd;
        dId_final_dVb = dIdclm_dVb;
        VASCBE = 0.0;
        dVASCBE_dVg = 0.0;
        dVASCBE_dVd = 0.0;
        dVASCBE_dVb = 0.0;
    }

    /* Final Ids with all three effects */
    i->Ids = Id_final;
    i->Gm = dId_final_dVg;
    i->Gds = dId_final_dVd;
    i->Gmbs = dId_final_dVb;

    /* Store Abulk for debugging/output */
    i->Abulk = Abulk;
    i->VASCBE = VASCBE;

    return 0;
}

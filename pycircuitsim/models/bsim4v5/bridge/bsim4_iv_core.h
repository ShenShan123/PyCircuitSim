/*
 * BSIM4.5.0 Core I-V Model Header
 *
 * Defines the interface for the BSIM4.5.0 I-V calculations
 *
 * Author: PyCircuitSim Team
 */

#ifndef BSIM4_IV_CORE_H
#define BSIM4_IV_CORE_H

#include "bsim4_standalone.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Internal state structure for BSIM4.5.0 calculations
 * This holds all intermediate variables during evaluation
 */
typedef struct {
    /* Effective geometry */
    double Leff;
    double Weff;

    /* Thermal voltage */
    double Vtm;

    /* Voltage calculations */
    double Vth;           /* Threshold voltage */
    double Vgsteff;       /* Effective gate overdrive */
    double Vdsat;         /* Saturation voltage */
    double Vdseff;        /* Effective drain voltage */

    /* Mobility */
    double ueff;          /* Effective mobility */
    double dueff_dVg;     /* d(ueff)/d(Vgs) */
    double dueff_dVd;     /* d(ueff)/d(Vds) */
    double dueff_dVb;     /* d(ueff)/d(Vbs) */

    /* Transconductance parameters */
    double beta;          /* Transconductance factor */
    double gche;          /* Channel conductance */
    double EsatL;         /* Saturation field * length */

    /* Currents and conductances */
    double Idl;           /* Linear/saturation current (before DIBL/CLM/SCBE) */
    double Ids;           /* Final drain current */
    double Gm;            /* Transconductance d(Id)/d(Vgs) */
    double Gds;           /* Output conductance d(Id)/d(Vds) */
    double Gmbs;          /* Bulk transconductance d(Id)/d(Vbs) */

    /* Derivatives */
    double dIdl_dVg;      /* d(Idl)/d(Vgs) */
    double dIdl_dVd;      /* d(Idl)/d(Vds) */
    double dIdl_dVb;      /* d(Idl)/d(Vbs) */

    double dVth_dVb;      /* d(Vth)/d(Vbs) */
    double dVth_dVd;      /* d(Vth)/d(Vds) */

    double dVgsteff_dVg;  /* d(Vgsteff)/d(Vgs) */
    double dVgsteff_dVd;  /* d(Vgsteff)/d(Vds) */
    double dVgsteff_dVb;  /* d(Vgsteff)/d(Vbs) */

    double dVdsat_dVg;    /* d(Vdsat)/d(Vgs) */
    double dVdsat_dVd;    /* d(Vdsat)/d(Vds) */
    double dVdsat_dVb;    /* d(Vdsat)/d(Vbs) */

    double dVdseff_dVg;   /* d(Vdseff)/d(Vgs) */
    double dVdseff_dVd;   /* d(Vdseff)/d(Vds) */
    double dVdseff_dVb;   /* d(Vdseff)/d(Vbs) */

    /* Second-order effects parameters */
    double Abulk;         /* Bulk charge effect coefficient for CLM */
    double VASCBE;        /* Critical voltage for SCBE */

} BSIM4_Internal;

/*
 * Input state structure
 */
typedef struct {
    double Vgs;
    double Vds;
    double Vbs;
} BSIM4_States;

/*
 * Main evaluation function - Phase 1 (Basic I-V)
 *
 * Calculates the drain current and conductances at a given bias point
 * using the core BSIM4.5.0 model equations (without DIBL, CLM, SCBE).
 *
 * Args:
 *   m     - Model parameters (technology parameters)
 *   inst  - Instance parameters (geometry)
 *   Vds   - Drain-source voltage (V)
 *   Vgs   - Gate-source voltage (V)
 *   Vbs   - Bulk-source voltage (V)
 *   i     - Output: Internal state including currents and conductances
 *
 * Returns:
 *   0 on success, error code on failure
 */
int bsim4_iv_evaluate(
    const BSIM4_Model *m,
    const BSIM4_Instance *inst,
    double Vds,
    double Vgs,
    double Vbs,
    BSIM4_Internal *i
);

#ifdef __cplusplus
}
#endif

#endif /* BSIM4_IV_CORE_H */

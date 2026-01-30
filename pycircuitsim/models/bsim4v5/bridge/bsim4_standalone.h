/*
 * BSIM4 Standalone Interface
 *
 * This header provides a standalone interface to the BSIM4 model
 * extracted from the original SPICE-3f5 implementation.
 *
 * Author: PyCircuitSim Team
 */

#ifndef BSIM4_STANDALONE_H
#define BSIM4_STANDALONE_H

#ifdef __cplusplus
extern "C" {
#endif

/* BSIM4 Instance Structure (standalone version) */
typedef struct {
    /* Geometry */
    double L;
    double W;
    double drainArea;
    double sourceArea;
    double drainSquares;
    double sourceSquares;
    double drainPerimeter;
    double sourcePerimeter;

    /* Stress effect instance parameters */
    double sa;
    double sb;
    double sd;

    /* Instance parameters */
    double nf;  /* Number of fingers */
    int off;    /* Initial condition flag */

    /* Internal node voltages (for debugging) */
    double vds;
    double vgs;
    double vbs;

    /* Operating point output */
    double Vgsteff;     /* Effective gate overdrive voltage */
    double vdsat;       /* Saturation voltage */
    double vth;         /* Threshold voltage */

} BSIM4_Instance;

/* BSIM4 Model Structure (standalone version) */
typedef struct {
    /* Device Type */
    int type;  /* 1 = NMOS, -1 = PMOS */

    /* Model selectors */
    int mobMod;    /* Mobility model */
    int capMod;    /* Capacitance model */
    int dioMod;    /* Diode model */
    int trnqsMod;  /* Transient NQS model */
    int acnqsMod;  /* AC NQS model */
    int fnoiMod;   /* Flicker noise model */
    int tnoiMod;   /* Thermal noise model */
    int rdsMod;    /* RDS model */
    int rbodyMod;  /* Body resistance model */
    int rgateMod;  /* Gate resistance model */
    int perMod;    /* Perimeter model */
    int geoMod;    /* Geometry model */
    int igcMod;    /* Gate current model */
    int igbMod;    /* Gate-body current model */
    int tempMod;   /* Temperature model */
    int paramChk;  /* Parameter checking flag */

    /* Oxide thickness and related */
    double tox;     /* Gate oxide thickness (m) */
    double toxp;    /* Physical oxide thickness (m) */
    double toxm;    /* Oxide thickness at which mob is measured (m) */
    double dtox;    /* Tox difference */
    double epsrox;  /* Oxide dielectric constant */
    double coxe;      /* Oxide capacitance per unit area (F/m^2) */

    /* Substrate and doping */
    double cdsc;      /* Drain/source depletion capacitance */
    double cdscb;     /* Body-bias coefficient of cdsc */
    double cdscd;     /* DIBL coefficient of cdsc */
    double cit;       /* Interface trap capacitance */
    double nfactor;   /* Subthreshold swing factor */
    double xj;        /* Junction depth (m) */
    double vsat;      /* Saturation velocity (m/s) */
    double at;        /* Temperature coefficient of vsat */
    double mstar;     /* Subthreshold parameter */
    double a0;        /* Bulk charge effect coefficient */
    double ags;       /* Gate bias coefficient of abulk */
    double a1;        /* First non-saturation factor */
    double a2;        /* Second non-saturation factor */
    double keta;      /* Non-uniform depletion width effect */
    double nsub;      /* Substrate doping concentration */
    double ndep;      /* Channel doping concentration */
    double nsd;       /* S/D doping concentration */
    double phin;      /* Surface potential */
    double ngate;     /* Poly gate doping concentration */

    /* Threshold voltage */
    double gamma1;    /* Body effect coefficient 1 */
    double gamma2;    /* Body effect coefficient 2 */
    double vbx;       /* Vbx */
    double vbm;       /* Vbm */
    double xt;        /* Doping depth */
    double k1;        /* First-order body effect */
    double kt1;       /* Vth temperature coefficient */
    double kt1l;      /* Length dependence of kt1 */
    double kt2;       /* Vth temperature coefficient */
    double k2;        /* Second-order body effect */
    double k3;        /* Third-order body effect */
    double k3b;       /* Body effect coefficient */
    double w0;        /* Width effect */
    double dvtp0;     /* Vth shift due to reverse bias */
    double dvtp1;     /* Vth shift coefficient */
    double lpe0;      /* Lateral non-uniform doping effect */
    double lpeb;      /* Lateral non-uniform doping coefficient */
    double litl;      /* Lateral non-uniform doping length */
    double dvt0;      /* Short-channel effect coefficient 0 */
    double dvt1;      /* Short-channel effect coefficient 1 */
    double dvt2;      /* Short-channel effect coefficient 2 */
    double dvt0w;     /* Width effect on dvt0 */
    double dvt1w;     /* Width effect on dvt1 */
    double dvt2w;     /* Width effect on dvt2 */
    double drout;     /* DIBL output resistance coefficient */
    double dsub;      /* DIBL coefficient */
    double vth0;      /* Threshold voltage at Vbs=0 */

    /* Mobility */
    double eu;        /* Mobility degradation coefficient */
    double ua;        /* Mobility degradation coefficient */
    double ua1;       /* Gate-bias dependence of ua */
    double ub;        /* Mobility degradation coefficient */
    double ub1;       /* Gate-bias dependence of ub */
    double uc;        /* Mobility degradation coefficient */
    double uc1;       /* Gate-bias dependence of uc */
    double u0;        /* Low-field mobility */
    double ute;       /* Temperature coefficient of u0 */
    double voff;      /* Offset voltage in subthreshold region */
    double minv;      /* Gate-source voltage for capacitance model */
    double voffl;     /* voff length dependence */
    double voffcvbn;  /* Offset voltage for capacitance model */
    double delta;     /* Vth effective width effect */

    /* Parasitic resistance */
    double rdsw;      /* Sheet resistance of S/D diffusion */
    double rdswmin;   /* Minimum rdsw */
    double rdwmin;    /* Minimum rdw */
    double rswmin;    /* Minimum rsw */
    double rsw;       /* Source resistance per width */
    double rdw;       /* Drain resistance per width */
    double prwg;      /* Gate bias effect on rdsw */
    double prwb;      /* Body bias effect on rdsw */
    double prt;       /* Temperature coefficient of rdsw */

    /* Subthreshold and DIBL */
    double eta0;      /* DIBL coefficient */
    double etab;      /* Body bias coefficient of DIBL */
    double pclm;      /* Channel length modulation coefficient */
    double pdibl1;    /* DIBL coefficient */
    double pdibl2;    /* DIBL coefficient */
    double pdiblb;    /* Body effect on pdibl */
    double fprout;    /* Field-induced drain mobility degradation factor */
    double pdits;     /* DITS coefficient */
    double pditsd;    /* DITS drain voltage coefficient */
    double pditsl;    /* DITS length coefficient */
    double pscbe1;    /* SCBE coefficient 1 */
    double pscbe2;    /* SCBE coefficient 2 */
    double pvag;      /* Gate dependence of SCBE */
    double wr;        /* Width dependence of rdsw */
    double dwg;       /* Gate bias effect on effective width */
    double dwb;       /* Body bias effect on effective width */
    double b0;        /* Bulk charge effect coefficient for abulk */
    double b1;        /* Bulk charge effect coefficient for abulk */

    /* Velocity saturation */
    double alpha0;    /* Impact ionization coefficient 1 */
    double alpha1;    /* Impact ionization coefficient 2 */
    double beta0;     /* Impact ionization coefficient 3 */

    /* Gate current and tunneling */
    double agidl;     /* GIDL parameter */
    double bgidl;     /* GIDL parameter */
    double cgidl;     /* GIDL parameter */
    double egidl;     /* GIDL parameter */
    double aigc;      /* Gate-to-channel Igc parameter */
    double bigc;      /* Gate-to-channel Igc parameter */
    double cigc;      /* Gate-to-channel Igc parameter */
    double aigsd;     /* Gate-to-S/D Igsd parameter */
    double bigsd;     /* Gate-to-S/D Igsd parameter */
    double cigsd;     /* Gate-to-S/D Igsd parameter */
    double aigbacc;   /* Igbs parameter for accumulation */
    double bigbacc;   /* Igbs parameter for accumulation */
    double cigbacc;   /* Igbs parameter for accumulation */
    double aigbinv;   /* Igbs parameter for inversion */
    double bigbinv;   /* Igbs parameter for inversion */
    double cigbinv;   /* Igbs parameter for inversion */
    double nigc;      /* Igc emission coefficient */
    double nigbacc;   /* Igbs emission coefficient in accumulation */
    double nigbinv;   /* Igbs emission coefficient in inversion */
    double ntox;      /* Tunneling mass exponent */
    double eigbinv;   /* Igbs energy parameter in inversion */
    double pigcd;     /* Gate-to-drain Igc parameter */
    double poxedge;   /* Oxide edge gate current parameter */
    double toxref;    /* Tox at which Igc model is measured */
    double ijthdfwd;  /* Forward diode current for noise */
    double ijthsfwd;  /* Forward diode current for noise */
    double ijthdrev;  /* Reverse diode current for noise */
    double ijthsrev;  /* Reverse diode current for noise */
    double xjbvd;     /* BVD grading coefficient */
    double xjbvs;     /* BVS grading coefficient */
    double bvd;       /* Drain junction breakdown voltage */
    double bvs;       /* Source junction breakdown voltage */

    /* Diode parameters */
    double jtss;      /* Source/drain bottom saturation current density */
    double jtsd;      /* Drain junction bottom saturation current density */
    double jtssws;    /* Source sidewall saturation current density */
    double jtsswd;    /* Drain sidewall saturation current density */
    double jtsswgs;   /* Source-gate sidewall saturation current density */
    double jtsswgd;   /* Drain-gate sidewall saturation current density */
    double njts;      /* Bottom emission coefficient */
    double njtssw;    /* Sidewall emission coefficient */
    double njtsswg;   /* Gate sidewall emission coefficient */
    double xtss;      /* Bottom depletion grading coefficient */
    double xtsd;      /* Drain bottom depletion grading coefficient */
    double xtssws;    /* Source sidewall depletion grading coefficient */
    double xtsswd;    /* Drain sidewall depletion grading coefficient */
    double xtsswgs;   /* Source-gate sidewall depletion grading coefficient */
    double xtsswgd;   /* Drain-gate sidewall depletion grading coefficient */
    double tnjts;     /* Temperature coefficient of jtss */
    double tnjtssw;   /* Temperature coefficient of jtssws */
    double tnjtsswg;  /* Temperature coefficient of jtsswgs */
    double vtss;      /* Temperature dependence of jts */
    double vtsd;      /* Temperature dependence of jtsd */
    double vtssws;    /* Temperature dependence of jtssws */
    double vtsswd;    /* Temperature dependence of jtsswd */
    double vtsswgs;   /* Temperature dependence of jtsswgs */
    double vtsswgd;   /* Temperature dependence of jtsswgd */

    /* Overlap capacitance and resistance */
    double cgsl;      /* Source-gate overlap capacitance per width */
    double cgdl;      /* Drain-gate overlap capacitance per width */
    double ckappas;   /* Source-bias coefficient for overlap capacitance */
    double ckappad;   /* Drain-bias coefficient for overlap capacitance */
    double cf;        /* Fringing field capacitance per width */
    double vfbcv;     /* Flat-band voltage for CV model */
    double clc;       /* Vdsat depletion capacitance */
    double cle;       /* Vdsat depletion capacitance length coefficient */
    double dwc;       /* Width correction */
    double dlc;       /* Length correction */
    double xw;        /* Width offset */
    double xl;        /* Length offset */
    double dlcig;     /* Length reduction for Igc model */
    double dwj;       /* Width reduction for junction diode */
    double noff;      /* Voffcv coefficient */
    double voffcv;    /* Offset voltage in CV model */
    double acde;      /* Accumulation capacitance coefficient */
    double moin;      /* Gate insulator thickness coefficient */
    double tcj;       /* Temperature coefficient of cj */
    double tcjsw;     /* Temperature coefficient of cjsw */
    double tcjswg;    /* Temperature coefficient of cjswg */
    double tpb;       /* Temperature coefficient of pb */
    double tpbsw;     /* Temperature coefficient of pbsw */
    double tpbswg;    /* Temperature coefficient of pbswg */

    /* Gate resistance */
    double dmcg;      /* Distance between gate contact and channel */
    double dmci;      /* Distance between gate contacts */
    double dmdg;      /* Distance between gate contact and drain */
    double dmcgt;     /* Temperature coefficient of dmcg */
    double xgw;       /* Gate electrode width */
    double xgl;       /* Gate electrode length */
    double rshg;      /* Gate sheet resistance */
    double ngcon;     /* Number of gate contacts */

    /* Length dependence parameters (omitted for brevity - would be prefixed with 'l') */

    /* Temperature */
    double temp;      /* Device temperature (K) */
    double tnom;      /* Nominal temperature (K) */
    double vfb;       /* Flat-band voltage */
    double gbmin;     /* Minimum gate conductance */

    /* Additional parameters for Vgsteff calculation */
    double Xdep0;     /* Depletion depth at zero bias (m) */
    double cdep0;     /* Depletion capacitance at zero bias (F/m^2) */
    double voffcbn;   /* Voff for capacitance model (same as voffcvbn) */
    double sqrtPhi;   /* sqrt(Phi) for geometry calculations */

} BSIM4_Model;

/* Output structure for model evaluation */
typedef struct {
    double Id;     /* Drain current (A) */
    double Ib;     /* Bulk current (A) */
    double Ig;     /* Gate current (A) */
    double Is;     /* Source current (A) */

    double Gm;     /* Transconductance dId/dVgs (S) */
    double Gds;    /* Output conductance dId/dVds (S) */
    double Gmbs;   /* Bulk transconductance dId/dVbs (S) */
    double Ggb;    /* Gate-gate conductance (S) */

    double Gbd;    /* Bulk-drain conductance (S) */
    double Gbs;    /* Bulk-source conductance (S) */

    /* Charge for capacitance calculation */
    double Qg;     /* Gate charge (C) */
    double Qb;     /* Bulk charge (C) */
    double Qd;     /* Drain charge (C) */
    double Qs;     /* Source charge (C) */

    /* Capacitances */
    double Cgg;    /* Gate-gate capacitance (F) */
    double Cgd;    /* Gate-drain capacitance (F) */
    double Cgs;    /* Gate-source capacitance (F) */
    double Cgb;    /* Gate-bulk capacitance (F) */

    /* Operating point info (for debugging) */
    double Vth;    /* Threshold voltage (V) */
    double Vgsteff;/* Effective gate overdrive (V) */

    int error;     /* Error flag (0 = success) */
} BSIM4_Output;

/* Main evaluation function */
int BSIM4_Evaluate(
    const BSIM4_Model *model,
    const BSIM4_Instance *instance,
    double Vds,
    double Vgs,
    double Vbs,
    BSIM4_Output *output
);

/* Initialize model with default parameters for a given technology node */
void BSIM4_InitModel_45nm_NMOS(BSIM4_Model *model);
void BSIM4_InitModel_45nm_PMOS(BSIM4_Model *model);

/* Initialize instance with default parameters */
void BSIM4_InitInstance(BSIM4_Instance *instance, double L, double W);

/* Set model parameter by name */
int BSIM4_SetParam(BSIM4_Model *model, const char *param_name, double value);

/* Get model parameter by name */
int BSIM4_GetParam(const BSIM4_Model *model, const char *param_name, double *value);

#ifdef __cplusplus
}
#endif

#endif /* BSIM4_STANDALONE_H */

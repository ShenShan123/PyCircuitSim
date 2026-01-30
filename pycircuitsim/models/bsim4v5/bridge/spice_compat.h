/*
 * SPICE Compatibility Layer for BSIM4
 *
 * Minimal SPICE-3f5 compatibility structures to allow the BSIM4 model
 * to run standalone without a full SPICE simulator.
 */

#ifndef SPICE_COMPAT_H
#define SPICE_COMPAT_H

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SPICE type definitions */
typedef int IFuid;  /* Unique ID for names */
typedef int IFvalue;

/* Generic model and instance pointers */
typedef void *GENmodel;
typedef void *GENinstance;

/* Minimal circuit structure */
typedef struct {
    double *CKTstate0;     /* State vector at current iteration */
    double *CKTstate1;     /* State vector at previous iteration */
    double CKTreltol;      /* Relative tolerance */
    double CKTabstol;      /* Absolute tolerance */
    double CKTgmin;        /* Minimum conductance */
    double CKTvoltTol;     /* Voltage tolerance */
    double CKTtemp;        /* Circuit temperature */
} CKTcircuit;

/* Complex number for AC analysis */
typedef struct {
    double real;
    double imag;
} complex_t;

/* Noise state structure */
typedef struct {
    double *noise;  /* Noise output */
} NOISEAN;

/* Maximum values for expressions */
#define MAX_EXP 1e38
#define MIN_EXP 1e-38
#define EXP_THRESHOLD 88.0

/* Math functions */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

/* Standard fabs is available */
#define FABS fabs

/* Temperature constants */
#define Kbolt 1.3806226e-23
#define Charge_q 1.6021918e-19
#define EPS0 8.85418e-12
#define EPSSI 1.03594e-10

/* Macros from SPICE */
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define SQRT(x) sqrt(x)
#define POW(x,y) pow(x,y)
#define EXP(x) exp(x)
#define LOG(x) log(x)
#define LOG10(x) log10(x)

#ifdef __cplusplus
}
#endif

#endif /* SPICE_COMPAT_H */

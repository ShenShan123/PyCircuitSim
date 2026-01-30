/*
 * const.h - Physical constants for SPICE compatibility
 */

#ifndef CONST_H
#define CONST_H

/* Physical constants */
#define Kbolt 1.3806226e-23        /* Boltzmann constant (J/K) */
#define Charge_q 1.6021918e-19     /* Electron charge (C) */
#define EPS0 8.85418e-12           /* Permittivity of free space (F/m) */
#define EPSSI 1.03594e-10          /* Permittivity of silicon (F/m) */

/* Math constants */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Expression limits */
#define MAX_EXP 1e38
#define MIN_EXP 1e-38
#define EXP_THRESHOLD 88.0

#endif /* CONST_H */

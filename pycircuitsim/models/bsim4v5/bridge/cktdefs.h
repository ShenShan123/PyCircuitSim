/*
 * cktdefs.h - Circuit structure definitions for SPICE compatibility
 */

#ifndef CKTDEFS_H
#define CKTDEFS_H

#include "gendefs.h"

/* Circuit structure */
typedef struct CKTcircuit {
    double *CKTstate0;      /* State vector at current iteration */
    double *CKTstate1;      /* State vector at previous iteration */
    double *CKTrhs;         /* Right-hand side vector */
    double **CKTmatrix;     /* MNA matrix */
    int CKTnumStates;       /* Number of states */
    double CKTreltol;       /* Relative tolerance */
    double CKTabstol;       /* Absolute tolerance */
    double CKTvoltTol;      /* Voltage tolerance */
    double CKTgmin;         /* Minimum conductance */
    double CKTtemp;         /* Circuit temperature (K) */
    double CKTnomTemp;      /* Nominal temperature (K) */
    int CKTstep;            /* Current iteration */
    int CKTtranDumpBlock;   /* Transient analysis flag */
    double CKTtime;         /* Current time */
} CKTcircuit;

#endif /* CKTDEFS_H */

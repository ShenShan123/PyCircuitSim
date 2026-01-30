/*
 * gendefs.h - Generic type definitions for SPICE compatibility
 */

#ifndef GENEDEFS_H
#define GENEDEFS_H

/* Generic model and instance types */
typedef void *GENmodel;
typedef void *GENinstance;

/* Boolean type */
#ifndef BOOL
#define BOOL int
#define TRUE 1
#define FALSE 0
#endif

/* Error codes */
#define OK 0
#define E_BADPARM -1

#endif /* GENEDEFS_H */

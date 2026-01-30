/*
 * ifsim.h - Interface definitions for SPICE compatibility
 */

#ifndef IFSIM_H
#define IFSIM_H

/* Unique ID type */
typedef int IFuid;

/* Value types */
typedef int IFvalue;

/* Parameter types */
typedef enum {
    IF_REAL,
    IF_INTEGER,
    IF_STRING,
    IF_FLAG,
    IF_COMPLEX
} IFtype;

#endif /* IFSIM_H */

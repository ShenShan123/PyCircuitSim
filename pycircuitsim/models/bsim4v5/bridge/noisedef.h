/*
 * noisedef.h - Noise analysis definitions for SPICE compatibility
 */

#ifndef NOISEDEF_H
#define NOISEDEF_H

/* Noise analysis structure */
typedef struct {
    double *noise;     /* Noise output array */
    int noizenodes;    /* Number of noise nodes */
} NOISEAN;

#endif /* NOISEDEF_H */

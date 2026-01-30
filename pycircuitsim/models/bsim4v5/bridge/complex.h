/*
 * complex.h - Complex number support for SPICE compatibility
 */

#ifndef COMPLEX_H
#define COMPLEX_H

/* Complex number type */
typedef struct {
    double real;
    double imag;
} complex_t;

/* Complex number operations */
#define COMPLEX(x, y) ((complex_t){(x), (y)})

#endif /* COMPLEX_H */

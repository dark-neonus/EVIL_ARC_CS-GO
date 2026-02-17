#ifndef INTEGRAL_2D_H
#define INTEGRAL_2D_H

/*
 * integral_2d.h — Interface for 2D numerical integration
 *
 * Both CPU and CUDA implementations share this header.
 * The integral is computed over a 2D domain (circle or triangle)
 * using a rectangular grid approximation: ∫∫_D f(x,y) dx dy ≈ Σ f(xi,yj)·Δx·Δy
 * where the sum runs only over grid points inside the domain D.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Supported domain shapes */
typedef enum {
    SHAPE_CIRCLE,
    SHAPE_TRIANGLE
} Shape;

/* Parameters for 2D integration */
typedef struct {
    Shape shape;
    double radius;       /* circle radius or triangle side length              */
    double center_x;     /* center of the domain (x)                           */
    double center_y;     /* center of the domain (y)                           */
    int    nx;           /* number of grid points along x                      */
    int    ny;           /* number of grid points along y                      */
    double x_min;        /* bounding box left                                  */
    double x_max;        /* bounding box right                                 */
    double y_min;        /* bounding box bottom                                */
    double y_max;        /* bounding box top                                   */
} IntegralParams2D;

/* CPU version — returns the integral value */
double integrate_2d_cpu(IntegralParams2D params);

/* CUDA version — returns the integral value (internally launches kernels) */
double integrate_2d_cuda(IntegralParams2D params);

#ifdef __cplusplus
}
#endif

#endif /* INTEGRAL_2D_H */

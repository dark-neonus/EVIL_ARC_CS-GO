#ifndef MATH_OPS_H
#define MATH_OPS_H

/*
 * math_ops.h — Vector and matrix operations interface
 *
 * Provides both CPU and CUDA routines for:
 *   - vector addition, scaling, dot product, norm
 *   - matrix addition, scalar multiplication
 * These are building blocks for the DFT iterative solver.
 */

#include <stddef.h>  /* for size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Vector operations (CPU) ──────────────────────────────────────────────── */

/* c[i] = a[i] + b[i] */
void vec_add_cpu(const double *a, const double *b, double *c, size_t n);

/* c[i] = alpha * a[i] */
void vec_scale_cpu(const double *a, double alpha, double *c, size_t n);

/* returns Σ a[i]*b[i] */
double vec_dot_cpu(const double *a, const double *b, size_t n);

/* returns sqrt(Σ a[i]²) */
double vec_norm_cpu(const double *a, size_t n);

/* ── Vector operations (CUDA) ─────────────────────────────────────────────── */

void vec_add_cuda(const double *a, const double *b, double *c, size_t n);
void vec_scale_cuda(const double *a, double alpha, double *c, size_t n);
double vec_dot_cuda(const double *a, const double *b, size_t n);
double vec_norm_cuda(const double *a, size_t n);

/* ── Matrix operations (CPU) — matrices stored as flat row-major arrays ─── */

/* C[i][j] = A[i][j] + B[i][j],  total elements = rows * cols */
void mat_add_cpu(const double *A, const double *B, double *C, size_t rows, size_t cols);

/* C[i][j] = alpha * A[i][j] */
void mat_scale_cpu(const double *A, double alpha, double *C, size_t rows, size_t cols);

/* ── Matrix operations (CUDA) ─────────────────────────────────────────────── */

void mat_add_cuda(const double *A, const double *B, double *C, size_t rows, size_t cols);
void mat_scale_cuda(const double *A, double alpha, double *C, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif /* MATH_OPS_H */

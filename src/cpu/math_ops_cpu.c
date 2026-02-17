/*
 * math_ops_cpu.c — CPU implementations of vector and matrix operations
 *
 * These are straightforward sequential loops.  They serve as:
 *   - reference implementations to verify CUDA results,
 *   - fallback when no GPU is available.
 *
 * Matrices are stored in **row-major** flat arrays:
 *   element (i,j) of an (rows × cols) matrix = A[i * cols + j].
 */

#include <math.h>
#include <stddef.h>
#include "../include/math_ops.h"

/* ═══════════ Vector operations ═══════════════════════════════════════════ */

/* Element-wise addition: c[i] = a[i] + b[i] */
void vec_add_cpu(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/* Scalar multiplication: c[i] = alpha * a[i] */
void vec_scale_cpu(const double *a, double alpha, double *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = alpha * a[i];
    }
}

/* Dot product: returns Σ a[i]*b[i] */
double vec_dot_cpu(const double *a, const double *b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Euclidean norm: returns sqrt(Σ a[i]²) */
double vec_norm_cpu(const double *a, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * a[i];
    }
    return sqrt(sum);
}

/* ═══════════ Matrix operations ═══════════════════════════════════════════ */

/* Element-wise matrix addition: C = A + B */
void mat_add_cpu(const double *A, const double *B, double *C,
                 size_t rows, size_t cols) {
    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) {
        C[i] = A[i] + B[i];
    }
}

/* Matrix scalar multiplication: C = alpha * A */
void mat_scale_cpu(const double *A, double alpha, double *C,
                   size_t rows, size_t cols) {
    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) {
        C[i] = alpha * A[i];
    }
}

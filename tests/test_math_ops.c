/*
 * test_math_ops.c — Validates vector/matrix operations (CPU vs CUDA)
 *
 * Creates small test vectors/matrices, runs both CPU and CUDA versions,
 * and compares results.
 *
 * Usage:  ./test_math_ops [size]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/math_ops.h"

#define DEFAULT_SIZE 1000

/* Fill array with values: a[i] = start + i * step */
static void fill_array(double *a, size_t n, double start, double step) {
    for (size_t i = 0; i < n; i++) {
        a[i] = start + i * step;
    }
}

/* Compare two arrays, return max absolute difference */
static double max_diff(const double *a, const double *b, size_t n) {
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs(a[i] - b[i]);
        if (diff > d) d = diff;
    }
    return d;
}

int main(int argc, char **argv) {
    size_t n = DEFAULT_SIZE;
    if (argc > 1) n = (size_t)atoi(argv[1]);

    printf("=== Math Operations Test (n = %zu) ===\n\n", n);

    /* ── Allocate arrays ──────────────────────────────────────────────── */
    double *a   = (double *)malloc(n * sizeof(double));
    double *b   = (double *)malloc(n * sizeof(double));
    double *c_cpu = (double *)malloc(n * sizeof(double));
    double *c_gpu = (double *)malloc(n * sizeof(double));

    fill_array(a, n, 1.0, 0.1);   /* a = [1.0, 1.1, 1.2, ...] */
    fill_array(b, n, 0.5, 0.05);  /* b = [0.5, 0.55, 0.6, ...] */

    /* ── Test: vector addition ────────────────────────────────────────── */
    vec_add_cpu(a, b, c_cpu, n);
    vec_add_cuda(a, b, c_gpu, n);
    printf("[vec_add]   max diff = %.2e  %s\n",
           max_diff(c_cpu, c_gpu, n),
           max_diff(c_cpu, c_gpu, n) < 1e-10 ? "PASS" : "FAIL");

    /* ── Test: vector scaling ─────────────────────────────────────────── */
    vec_scale_cpu(a, 2.5, c_cpu, n);
    vec_scale_cuda(a, 2.5, c_gpu, n);
    printf("[vec_scale] max diff = %.2e  %s\n",
           max_diff(c_cpu, c_gpu, n),
           max_diff(c_cpu, c_gpu, n) < 1e-10 ? "PASS" : "FAIL");

    /* ── Test: dot product ────────────────────────────────────────────── */
    double dot_cpu = vec_dot_cpu(a, b, n);
    double dot_gpu = vec_dot_cuda(a, b, n);
    printf("[vec_dot]   diff     = %.2e  %s\n",
           fabs(dot_cpu - dot_gpu),
           fabs(dot_cpu - dot_gpu) < 1e-6 ? "PASS" : "FAIL");

    /* ── Test: vector norm ────────────────────────────────────────────── */
    double norm_cpu = vec_norm_cpu(a, n);
    double norm_gpu = vec_norm_cuda(a, n);
    printf("[vec_norm]  diff     = %.2e  %s\n",
           fabs(norm_cpu - norm_gpu),
           fabs(norm_cpu - norm_gpu) < 1e-6 ? "PASS" : "FAIL");

    /* ── Test: matrix addition (reuse arrays as 10×(n/10) matrices) ──── */
    size_t rows = 10, cols = n / 10;
    if (cols > 0) {
        mat_add_cpu(a, b, c_cpu, rows, cols);
        mat_add_cuda(a, b, c_gpu, rows, cols);
        printf("[mat_add]   max diff = %.2e  %s\n",
               max_diff(c_cpu, c_gpu, rows * cols),
               max_diff(c_cpu, c_gpu, rows * cols) < 1e-10 ? "PASS" : "FAIL");

        mat_scale_cpu(a, 3.14, c_cpu, rows, cols);
        mat_scale_cuda(a, 3.14, c_gpu, rows, cols);
        printf("[mat_scale] max diff = %.2e  %s\n",
               max_diff(c_cpu, c_gpu, rows * cols),
               max_diff(c_cpu, c_gpu, rows * cols) < 1e-10 ? "PASS" : "FAIL");
    }

    free(a); free(b); free(c_cpu); free(c_gpu);
    printf("\nDone.\n");
    return 0;
}

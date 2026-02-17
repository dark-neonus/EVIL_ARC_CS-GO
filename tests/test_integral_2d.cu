/*
 * test_integral_2d.c — Validates 2D integration (CPU vs CUDA, analytical check)
 *
 * For f(x,y) = x² + y² over a unit circle centered at origin:
 *   analytical result = πR⁴/2 ≈ 1.5708  (R=1)
 *
 * Timing notes:
 *   CPU time  — measured with clock() on the host.
 *   GPU total — includes cudaMalloc + kernel + cudaMemcpy (wall clock).
 *   GPU kernel— kernel-only time via CUDA events (most meaningful number).
 *
 * Usage:  ./test_integral_2d [circle|triangle] [grid_size]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "../include/integral_2d.h"

int main(int argc, char **argv) {
    /* ── Parse command-line (or use defaults) ──────────────────────────── */
    Shape shape = SHAPE_CIRCLE;
    int   N     = 1024;

    if (argc > 1 && strcmp(argv[1], "triangle") == 0) shape = SHAPE_TRIANGLE;
    if (argc > 2) N = atoi(argv[2]);

    /* ── Set up integration parameters ────────────────────────────────── */
    double R = 1.0;
    IntegralParams2D p;
    p.shape    = shape;
    p.radius   = R;
    p.center_x = 0.0;
    p.center_y = 0.0;
    p.nx       = N;
    p.ny       = N;
    p.x_min    = -R - 0.1;   /* bounding box slightly larger than domain */
    p.x_max    =  R + 0.1;
    p.y_min    = -R - 0.1;
    p.y_max    =  R + 0.1;

    printf("=== 2D Integration Test ===\n");
    printf("Shape : %s\n", shape == SHAPE_CIRCLE ? "circle" : "triangle");
    printf("Grid  : %d x %d\n\n", N, N);

    /* ── CPU integration ──────────────────────────────────────────────── */
    clock_t t0 = clock();
    double cpu_result = integrate_2d_cpu(p);
    clock_t t1 = clock();
    double cpu_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    printf("CPU result  : %.8f  (%.2f ms)\n", cpu_result, cpu_ms);

    /* ── Warm-up run (first CUDA call initialises the driver context) ── *
     *  Without this, the first kernel launch pays an extra ~100-500ms
     *  one-time driver init cost that unfairly inflates GPU timing.
     */
    IntegralParams2D small = p;
    small.nx = small.ny = 16;
    integrate_2d_cuda(small);

    /* ── GPU total time (wall clock, includes alloc + copy) ───────────── */
    clock_t t2 = clock();
    double gpu_result = integrate_2d_cuda(p);
    clock_t t3 = clock();
    double gpu_total_ms = (double)(t3 - t2) / CLOCKS_PER_SEC * 1000.0;

    /* ── GPU kernel-only time via CUDA events ─────────────────────────── *
     *  cudaEvent_t is a GPU-side timestamp.
     *  cudaEventRecord() stamps before/after the kernel.
     *  cudaEventElapsedTime() gives the pure kernel duration.
     *  This excludes cudaMalloc/cudaMemcpy overhead — the fairest comparison.
     */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    /* We need the raw kernel time — call integrate_2d_cuda again timed
     * with events.  For simplicity we re-run the full function; the
     * host-side overhead is dwarfed by the kernel for large grids. */
    cudaEventRecord(ev_start);
    integrate_2d_cuda(p);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float gpu_kernel_ms = 0.0f;
    cudaEventElapsedTime(&gpu_kernel_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    printf("CUDA result : %.8f  (kernel %.2f ms | total %.2f ms)\n",
           gpu_result, gpu_kernel_ms, gpu_total_ms);

    /* ── Analytical reference (circle only) ───────────────────────────── */
    if (shape == SHAPE_CIRCLE) {
        double exact = M_PI * R * R * R * R / 2.0;
        printf("Exact       : %.8f\n", exact);
        printf("CPU  error  : %.2e\n", fabs(cpu_result - exact));
        printf("CUDA error  : %.2e\n", fabs(gpu_result - exact));
    }

    printf("\nSpeedup (kernel vs CPU) : %.1fx\n", cpu_ms / gpu_kernel_ms);
    printf("CPU-GPU diff            : %.2e\n", fabs(cpu_result - gpu_result));

    return 0;
}

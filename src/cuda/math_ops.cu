/*
 * math_ops.cu — CUDA implementations of vector and matrix operations
 *
 * CONCEPT:
 *   Every element is independent → perfectly parallel.
 *   We map one thread to one element: thread i processes a[i] (or A[i]).
 *
 * MEMORY FLOW:
 *   Host (CPU) arrays  ──cudaMemcpy──▶  Device (GPU) arrays
 *                                              │
 *                                        kernel launch
 *                                              │
 *   Host result        ◀──cudaMemcpy──  Device result
 *
 * For the dot product we use Thrust (see below) because a manual reduction
 * kernel is complex and Thrust does it in one line — showcasing its utility.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>            /* Thrust: wraps raw CUDA pointers          */
#include <thrust/inner_product.h>         /* Thrust: dot product (inner_product)       */
#include <thrust/transform_reduce.h>     /* Thrust: transform + reduce in one pass    */
#include <thrust/functional.h>            /* Thrust: plus<>, multiplies<>              */
#include "../include/math_ops.h"

/* ─── Helper: compute how many blocks we need ────────────────────────────── *
 *  For N elements and BLOCK_SIZE threads per block:
 *  blocks = ceil(N / BLOCK_SIZE).
 */
#define BLOCK_SIZE 256

static inline int grid_size(size_t n) {
    return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

/* ═════════════════════════════════════════════════════════════════════════════
 *  KERNELS — __global__ functions that run on the GPU
 * ═════════════════════════════════════════════════════════════════════════════ */

/* Vector addition: c[i] = a[i] + b[i] */
__global__ void kernel_vec_add(const double *a, const double *b, double *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;  /* global thread index */
    if (i < n) c[i] = a[i] + b[i];                     /* bounds check        */
}

/* Vector scaling: c[i] = alpha * a[i] */
__global__ void kernel_vec_scale(const double *a, double alpha, double *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = alpha * a[i];
}

/* Matrix addition: C[i] = A[i] + B[i]  (flat storage, same as vector add) */
__global__ void kernel_mat_add(const double *A, const double *B, double *C, size_t total) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) C[i] = A[i] + B[i];
}

/* Matrix scaling: C[i] = alpha * A[i] */
__global__ void kernel_mat_scale(const double *A, double alpha, double *C, size_t total) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) C[i] = alpha * A[i];
}

/* ═════════════════════════════════════════════════════════════════════════════
 *  HOST WRAPPERS — allocate GPU memory, launch kernels, copy results back
 * ═════════════════════════════════════════════════════════════════════════════ */

/* ── Vector addition ──────────────────────────────────────────────────────── */
void vec_add_cuda(const double *a, const double *b, double *c, size_t n) {
    double *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(double);

    /* Allocate device memory */
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    /* Copy input data:  host → device */
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    /* Launch kernel */
    kernel_vec_add<<<grid_size(n), BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    /* Copy result back:  device → host */
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    /* Free device memory */
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

/* ── Vector scaling ───────────────────────────────────────────────────────── */
void vec_scale_cuda(const double *a, double alpha, double *c, size_t n) {
    double *d_a, *d_c;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    kernel_vec_scale<<<grid_size(n), BLOCK_SIZE>>>(d_a, alpha, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_c);
}

/* ── Dot product — using Thrust ───────────────────────────────────────────── *
 *  Thrust eliminates the need to write a manual parallel reduction kernel.
 *  thrust::device_ptr wraps a raw CUDA pointer so Thrust algorithms can use it.
 *  inner_product computes Σ a[i]*b[i] in parallel.
 */
double vec_dot_cuda(const double *a, const double *b, size_t n) {
    double *d_a, *d_b;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    /* Wrap raw pointers for Thrust */
    thrust::device_ptr<double> t_a(d_a);
    thrust::device_ptr<double> t_b(d_b);

    /* Thrust inner_product: init=0.0, combines with +, pairs with * */
    double result = thrust::inner_product(t_a, t_a + n, t_b, 0.0);

    cudaFree(d_a); cudaFree(d_b);
    return result;
}

/* ── Vector norm — using Thrust transform_reduce ──────────────────────────── *
 *  We square each element (thrust::square), then sum (thrust::plus),
 *  and finally take sqrt on the CPU.  This demonstrates transform_reduce:
 *    reduce( transform(a[i]) )  in a single GPU pass.
 */

/* Functor that squares a value (needed by transform_reduce) */
struct square_op {
    __host__ __device__ double operator()(double x) const { return x * x; }
};

double vec_norm_cuda(const double *a, size_t n) {
    double *d_a;
    size_t bytes = n * sizeof(double);

    cudaMalloc(&d_a, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    thrust::device_ptr<double> t_a(d_a);

    /* Sum of squares on GPU, then sqrt on CPU */
    double sum_sq = thrust::transform_reduce(t_a, t_a + n,
                                              square_op(),        /* transform   */
                                              0.0,                /* initial val */
                                              thrust::plus<double>());  /* reduce */
    cudaFree(d_a);
    return sqrt(sum_sq);
}

/* ── Matrix addition ──────────────────────────────────────────────────────── */
void mat_add_cuda(const double *A, const double *B, double *C,
                  size_t rows, size_t cols) {
    size_t total = rows * cols;
    double *d_A, *d_B, *d_C;
    size_t bytes = total * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    kernel_mat_add<<<grid_size(total), BLOCK_SIZE>>>(d_A, d_B, d_C, total);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

/* ── Matrix scaling ───────────────────────────────────────────────────────── */
void mat_scale_cuda(const double *A, double alpha, double *C,
                    size_t rows, size_t cols) {
    size_t total = rows * cols;
    double *d_A, *d_C;
    size_t bytes = total * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

    kernel_mat_scale<<<grid_size(total), BLOCK_SIZE>>>(d_A, alpha, d_C, total);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_C);
}

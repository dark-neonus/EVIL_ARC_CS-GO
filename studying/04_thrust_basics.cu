/*
 * 04_thrust_basics.cu — Introduction to the Thrust library
 *
 * WHAT YOU LEARN:
 *   - Thrust = STL-like library for CUDA (ships with CUDA Toolkit)
 *   - thrust::device_vector: GPU-resident std::vector analogue
 *   - thrust::host_vector: CPU-resident counterpart
 *   - Algorithms: sort, reduce, transform, inner_product
 *   - Automatic memory management (no manual cudaMalloc/cudaFree)
 *
 * WHY THRUST:
 *   Writing reduction kernels (sum, min, max, dot product) by hand is
 *   error-prone and requires understanding warp-level primitives.
 *   Thrust does it optimally in one line.
 *
 * COMPILE:  nvcc 04_thrust_basics.cu -o thrust_basics && ./thrust_basics
 */

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

int main() {
    const int N = 10;

    /* ── 1. Create vectors ────────────────────────────────────────────── */
    /* host_vector lives on CPU */
    thrust::host_vector<float> h_vec(N);
    for (int i = 0; i < N; i++) h_vec[i] = (float)(N - i);

    printf("Original:  ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_vec[i]);
    printf("\n");

    /* device_vector lives on GPU — copying from host_vector transfers data */
    thrust::device_vector<float> d_vec = h_vec;

    /* ── 2. Sort on GPU ───────────────────────────────────────────────── */
    thrust::sort(d_vec.begin(), d_vec.end());

    /* Copy back to host for printing */
    h_vec = d_vec;
    printf("Sorted:    ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_vec[i]);
    printf("\n");

    /* ── 3. Reduce (sum) ──────────────────────────────────────────────── */
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());
    printf("Sum:       %.0f\n", sum);

    /* ── 4. Transform: square each element ────────────────────────────── */
    thrust::device_vector<float> d_squared(N);
    thrust::transform(d_vec.begin(), d_vec.end(),
                      d_squared.begin(),
                      thrust::square<float>());     /* built-in unary op */

    thrust::host_vector<float> h_sq = d_squared;
    printf("Squared:   ");
    for (int i = 0; i < N; i++) printf("%.0f ", h_sq[i]);
    printf("\n");

    /* ── 5. Inner product (dot product) ───────────────────────────────── */
    float dot = thrust::inner_product(d_vec.begin(), d_vec.end(),
                                      d_vec.begin(), 0.0f);
    printf("Dot(v,v):  %.0f\n", dot);

    /* ── 6. Sequence: fill with 0, 1, 2, ... ─────────────────────────── */
    thrust::device_vector<int> d_seq(N);
    thrust::sequence(d_seq.begin(), d_seq.end());    /* 0, 1, 2, ..., N-1 */

    thrust::host_vector<int> h_seq = d_seq;
    printf("Sequence:  ");
    for (int i = 0; i < N; i++) printf("%d ", h_seq[i]);
    printf("\n");

    /* Memory is freed automatically when vectors go out of scope */
    return 0;
}

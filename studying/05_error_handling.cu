/*
 * 05_error_handling.cu — Proper CUDA error checking
 *
 * WHAT YOU LEARN:
 *   - Every CUDA API call returns cudaError_t — ALWAYS check it
 *   - CUDA_CHECK macro: wraps any CUDA call, prints file/line on error
 *   - cudaGetLastError(): checks for kernel launch failures
 *   - Common errors: out of memory, invalid config, arch mismatch
 *
 * WHY IT MATTERS:
 *   CUDA errors are silent by default.  Without checking, your program
 *   may produce garbage output with no warning.
 *
 * COMPILE:  nvcc 05_error_handling.cu -o error_check && ./error_check
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* ── Macro: check CUDA return code, print error and exit on failure ──── */
#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void simple_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= 2.0f;
}

int main() {
    const int N = 100;
    float *d_data;

    /* Every CUDA call is wrapped — errors are caught immediately */
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    simple_kernel<<<1, 128>>>(d_data, N);

    /* cudaGetLastError: check if the kernel launch itself failed
     * (bad grid/block dims, invalid function, etc.) */
    CUDA_CHECK(cudaGetLastError());

    /* cudaDeviceSynchronize: also returns errors from kernel execution */
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_data));

    printf("All CUDA calls succeeded.\n");

    /* ── Intentional error: allocating too much memory ────────────────── */
    float *d_huge;
    cudaError_t err = cudaMalloc(&d_huge, (size_t)1024 * 1024 * 1024 * 100);
    if (err != cudaSuccess) {
        printf("Expected error: %s\n", cudaGetErrorString(err));
        /* Reset the error state so subsequent calls work */
        cudaGetLastError();
    }

    return 0;
}

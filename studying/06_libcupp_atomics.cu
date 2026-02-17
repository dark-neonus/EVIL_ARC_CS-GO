/*
 * 06_libcupp_atomics.cu — Using libcu++ (CUDA C++ Standard Library) atomics
 *
 * WHAT YOU LEARN:
 *   - libcu++ provides C++ standard library features that work on GPU
 *   - cuda::atomic<T>: GPU-compatible atomic type (like std::atomic)
 *   - Memory ordering: cuda::memory_order_relaxed (fastest for counters)
 *   - Scopes: cuda::thread_scope_device (visible to all GPU threads)
 *
 * WHY libcu++:
 *   - Standard C++ interface familiar from std::
 *   - Type-safe alternative to raw atomicAdd/atomicCAS
 *   - Supports advanced memory orderings for lock-free algorithms
 *
 * PRACTICAL USE IN OUR PROJECT:
 *   When accumulating integral sums across threads, atomics ensure
 *   correctness without explicit locks.
 *
 * COMPILE:  nvcc -std=c++17 06_libcupp_atomics.cu -o libcupp_demo && ./libcupp_demo
 *  (Requires CUDA 11+ for libcu++)
 */

#include <stdio.h>
#include <cuda/atomic>

/*
 * cuda::atomic<int, cuda::thread_scope_device>
 *   - int: the underlying type
 *   - thread_scope_device: all threads on the GPU can see updates
 *
 * Alternatives:
 *   thread_scope_block:  only threads in the same block see updates (faster)
 *   thread_scope_system: even CPU threads see updates (for unified memory)
 */

/* Kernel: each thread increments a shared counter */
__global__ void count_threads(cuda::atomic<int, cuda::thread_scope_device> *counter,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /* fetch_add: atomically adds 1 and returns old value */
        counter->fetch_add(1, cuda::memory_order_relaxed);
    }
}

/* Kernel: atomic accumulation of doubles (relevant to integral computation) */
__global__ void atomic_sum(cuda::atomic<double, cuda::thread_scope_device> *result,
                            const double *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        /* Atomically add data[i] to the running sum */
        result->fetch_add(data[i], cuda::memory_order_relaxed);
    }
}

int main() {
    const int N = 10000;

    /* ── Example 1: atomic counter ────────────────────────────────────── */
    cuda::atomic<int, cuda::thread_scope_device> *d_counter;
    cudaMalloc(&d_counter, sizeof(*d_counter));
    cudaMemset(d_counter, 0, sizeof(*d_counter));

    count_threads<<<(N + 255) / 256, 256>>>(d_counter, N);
    cudaDeviceSynchronize();

    int h_count;
    cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Atomic counter: %d (expected %d) — %s\n",
           h_count, N, h_count == N ? "PASS" : "FAIL");

    /* ── Example 2: atomic sum of doubles ─────────────────────────────── */
    double *h_data = (double *)malloc(N * sizeof(double));
    double expected = 0.0;
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0;  /* each element = 1.0, so sum = N */
        expected += h_data[i];
    }

    double *d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data, N * sizeof(double), cudaMemcpyHostToDevice);

    cuda::atomic<double, cuda::thread_scope_device> *d_sum;
    cudaMalloc(&d_sum, sizeof(*d_sum));
    cudaMemset(d_sum, 0, sizeof(*d_sum));

    atomic_sum<<<(N + 255) / 256, 256>>>(d_sum, d_data, N);
    cudaDeviceSynchronize();

    double h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Atomic sum: %.1f (expected %.1f) — %s\n",
           h_sum, expected, (h_sum == expected) ? "PASS" : "FAIL");

    cudaFree(d_counter);
    cudaFree(d_data);
    cudaFree(d_sum);
    free(h_data);
    return 0;
}

/*
 * 02_threads_and_blocks.cu — Understanding CUDA thread hierarchy
 *
 * WHAT YOU LEARN:
 *   - Thread hierarchy: Grid → Blocks → Threads
 *   - blockDim, blockIdx, threadIdx — built-in variables
 *   - Global thread ID = blockIdx.x * blockDim.x + threadIdx.x
 *   - Why we need bounds checking (grid may be larger than data)
 *
 * KEY IDEA:
 *   GPU launches thousands of threads.  Each thread knows its own ID
 *   via the built-in variables and processes one element.
 *
 *   ┌────────────── Grid ──────────────┐
 *   │  Block 0        Block 1     ...  │
 *   │ ┌──────────┐  ┌──────────┐       │
 *   │ │ t0 t1 t2 │  │ t0 t1 t2 │       │
 *   │ │ t3 t4 t5 │  │ t3 t4 t5 │       │
 *   │ └──────────┘  └──────────┘       │
 *   └─────────────────────────────────┘
 *
 * COMPILE:  nvcc 02_threads_and_blocks.cu -o threads_blocks && ./threads_blocks
 */

#include <stdio.h>

#define N 10

/* Each thread doubles one element */
__global__ void double_array(int *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Bounds check: we may launch more threads than elements */
    if (i < n) {
        a[i] = a[i] * 2;
    }
}

int main() {
    int h_a[N];  /* host (CPU) array */

    /* Fill with 0..9 */
    for (int i = 0; i < N; i++) h_a[i] = i;

    /* Allocate device (GPU) array */
    int *d_a;
    cudaMalloc(&d_a, N * sizeof(int));

    /* Copy data: CPU → GPU */
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    /* Launch with 4 threads/block → need ceil(10/4) = 3 blocks */
    int threads_per_block = 4;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    printf("Launching %d blocks × %d threads = %d total threads (for %d elements)\n",
           num_blocks, threads_per_block, num_blocks * threads_per_block, N);

    double_array<<<num_blocks, threads_per_block>>>(d_a, N);
    cudaDeviceSynchronize();

    /* Copy result back: GPU → CPU */
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: ");
    for (int i = 0; i < N; i++) printf("%d ", h_a[i]);
    printf("\n");

    cudaFree(d_a);
    return 0;
}

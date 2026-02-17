/*
 * 03_memory_model.cu — CUDA memory types: global, shared, registers
 *
 * WHAT YOU LEARN:
 *   - Global memory: large but slow (~500 cycles latency), visible to all threads
 *   - Shared memory: small (~48KB/block), fast (~1 cycle), shared within a block
 *   - Registers: fastest, private to each thread, limited quantity
 *   - __shared__ keyword for declaring shared memory
 *   - __syncthreads(): barrier — all threads in a block wait here
 *
 * EXAMPLE:
 *   We copy data into shared memory, reverse it within each block, write back.
 *   This demonstrates shared memory usage pattern: load → sync → compute → sync → store.
 *
 * COMPILE:  nvcc 03_memory_model.cu -o mem_model && ./mem_model
 */

#include <stdio.h>

#define BLOCK_SIZE 8

/* Reverse elements within each block using shared memory */
__global__ void block_reverse(int *d_out, const int *d_in, int n) {
    /*
     * __shared__: allocated once per block in fast on-chip memory.
     * All threads in the same block see the same shared memory array.
     */
    __shared__ int s_data[BLOCK_SIZE];

    int tid = threadIdx.x;                                  /* thread index within block */
    int gid = blockIdx.x * blockDim.x + threadIdx.x;       /* global index              */

    /* Step 1: each thread loads one element from global → shared */
    if (gid < n)
        s_data[tid] = d_in[gid];

    /*
     * __syncthreads(): barrier synchronization.
     * ALL threads in this block must reach this point before any can proceed.
     * Ensures shared memory is fully loaded before we read from it.
     */
    __syncthreads();

    /* Step 2: read from shared in reverse order, write to global */
    int reversed_tid = BLOCK_SIZE - 1 - tid;
    int reversed_gid = blockIdx.x * blockDim.x + reversed_tid;

    if (gid < n && reversed_gid < n)
        d_out[gid] = s_data[reversed_tid];
}

int main() {
    const int N = 16;
    int h_in[16], h_out[16];

    for (int i = 0; i < N; i++) h_in[i] = i;

    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block_reverse<<<num_blocks, BLOCK_SIZE>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Input:  "); for (int i = 0; i < N; i++) printf("%2d ", h_in[i]);
    printf("\nOutput: "); for (int i = 0; i < N; i++) printf("%2d ", h_out[i]);
    printf("\n(elements reversed within each block of %d)\n", BLOCK_SIZE);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

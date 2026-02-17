/*
 * 01_hello_gpu.cu — The simplest possible CUDA program
 *
 * WHAT YOU LEARN:
 *   - __global__ keyword: marks a function as a GPU kernel
 *   - <<<1, 1>>>: launch config — 1 block, 1 thread (minimal)
 *   - printf works inside CUDA kernels (since compute capability 2.0)
 *   - cudaDeviceSynchronize(): waits for GPU to finish before CPU continues
 *
 * COMPILE & RUN:
 *   nvcc 01_hello_gpu.cu -o hello_gpu && ./hello_gpu
 */

#include <stdio.h>

/* This function runs on the GPU */
__global__ void hello_kernel() {
    printf("Hello from GPU!  threadIdx=(%d,%d,%d)  blockIdx=(%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x,  blockIdx.y,  blockIdx.z);
}

int main() {
    printf("Hello from CPU!\n");

    /* Launch kernel: 1 block of 1 thread */
    hello_kernel<<<1, 1>>>();

    /* GPU calls are asynchronous — wait for kernel to finish */
    cudaDeviceSynchronize();

    printf("Back on CPU. Done.\n");
    return 0;
}

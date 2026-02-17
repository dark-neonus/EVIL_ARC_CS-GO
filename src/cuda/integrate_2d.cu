/*
 * integrate_2d.cu — CUDA implementation of 2D numerical integration
 *
 * PURPOSE:
 *   Compute ∫∫_D f(x,y) dA  over a 2D domain D (circle or triangle)
 *   using the GPU.  We discretise the bounding box into an nx×ny grid,
 *   assign one CUDA thread per grid cell, and let each thread:
 *     1) check whether its cell centre lies inside D,
 *     2) if yes — evaluate f and contribute  f(xi,yj)·ΔA  to the sum.
 *
 *   The partial sums are reduced on the GPU with atomicAdd.
 *
 * HOW CUDA WORKS HERE (short guide):
 *   - A "kernel" is a function prefixed with __global__; it runs on the GPU.
 *   - We launch it with kernel<<<grid, block>>>(...).
 *       • block = how many threads per block  (e.g. 16×16 = 256 threads).
 *       • grid  = how many blocks we need to cover all nx×ny cells.
 *   - Each thread figures out its (ix, iy) from built-in variables:
 *       ix = blockIdx.x * blockDim.x + threadIdx.x
 *   - Device memory is allocated with cudaMalloc, copied with cudaMemcpy.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>   /* CUDA runtime API: cudaMalloc, cudaMemcpy, etc. */
#include "../include/integral_2d.h"

/*
 * Portable atomicAdd for double.
 * Built-in atomicAdd(double*,double) requires compute capability ≥ 6.0.
 * This CAS-loop fallback works on any architecture.
 * If the built-in is available, the compiler picks it via overload resolution.
 */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double *addr, double val) {
    unsigned long long int *addr_ull = (unsigned long long int *)addr;
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ double atomicAddDouble(double *addr, double val) {
    return atomicAdd(addr, val);
}
#endif

/* ── Helper: the function we integrate ────────────────────────────────────
 *  __device__ means this function lives on the GPU and can only be called
 *  from other GPU code (kernels or other __device__ functions).
 *
 *  For demonstration we integrate f(x,y) = x² + y²  (known analytical result
 *  over a unit circle = π/2, handy for validation).
 */
__device__ double func_2d(double x, double y) {
    return x * x + y * y;
}

/* ── Helper: is point (x,y) inside the domain D? ─────────────────────────── */
__device__ int point_in_domain(double x, double y, Shape shape,
                               double cx, double cy, double R) {
    if (shape == SHAPE_CIRCLE) {
        /* Inside circle centred at (cx,cy) with radius R */
        double dx = x - cx;
        double dy = y - cy;
        return (dx * dx + dy * dy) <= (R * R);
    } else { /* SHAPE_TRIANGLE: equilateral triangle centred at (cx,cy), side R */
        /* Simple bounding: triangle with vertices (cx, cy+R), (cx-R, cy-R), (cx+R, cy-R) */
        double x0 = cx,     y0 = cy + R;         /* top vertex     */
        double x1 = cx - R, y1 = cy - R;         /* bottom-left    */
        double x2 = cx + R, y2 = cy - R;         /* bottom-right   */

        /* Barycentric sign test — classic "is point in triangle" */
        double d1 = (x - x1) * (y0 - y1) - (x0 - x1) * (y - y1);
        double d2 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2);
        double d3 = (x - x0) * (y2 - y0) - (x2 - x0) * (y - y0);
        int has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        int has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(has_neg && has_pos);             /* inside if all same sign */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  KERNEL: each thread handles one grid cell (ix, iy)
 *
 *  WHY SHARED MEMORY REDUCTION instead of one atomicAdd per thread:
 *
 *  Naive approach (one atomicAdd per thread):
 *    All ~N² threads hammer a single memory address → they serialize.
 *    Parallelism is completely lost.  GPU behaves worse than CPU.
 *
 *  Better approach (shared memory reduction):
 *    1. Each thread stores its value in shared memory (fast, per-block).
 *    2. Threads within the block cooperate to sum those values in log₂ steps.
 *    3. Only thread 0 of each block does ONE atomicAdd to the global result.
 *
 *  Result: atomic contention drops from N² → number_of_blocks.
 *  For a 1024² grid with 16×16 blocks: 1,048,576 → 4,096 atomic ops.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define BLOCK_X 16
#define BLOCK_Y 16

__global__ void kernel_integrate_2d(
    double *d_result,
    Shape   shape,
    double  cx, double cy, double R,
    double  x_min, double y_min,
    double  dx, double dy,
    int     nx, int     ny)
{
    /* ── Step 1: thread indices ───────────────────────────────────────── */
    int ix  = blockIdx.x * blockDim.x + threadIdx.x;
    int iy  = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;  /* linear id within block */

    /* ── Step 2: each thread computes its own contribution ───────────── */
    double val = 0.0;
    if (ix < nx && iy < ny) {
        double x = x_min + (ix + 0.5) * dx;
        double y = y_min + (iy + 0.5) * dy;
        if (point_in_domain(x, y, shape, cx, cy, R)) {
            val = func_2d(x, y) * dx * dy;
        }
    }

    /* ── Step 3: load into shared memory ─────────────────────────────── *
     *  One array per block, in fast on-chip memory.
     */
    __shared__ double s_sum[BLOCK_X * BLOCK_Y];
    s_sum[tid] = val;
    __syncthreads();  /* wait until all threads have written */

    /* ── Step 4: parallel reduction within the block ─────────────────── *
     *  Each iteration halves the number of active threads.
     *  Thread 0 ends up with the sum of all BLOCK_X*BLOCK_Y values.
     *
     *  stride=128: threads 0-127 each add the value 128 slots ahead.
     *  stride=64:  threads 0-63  each add the value 64  slots ahead.
     *  ...
     *  stride=1:   thread  0     adds s_sum[1].
     */
    for (int stride = (BLOCK_X * BLOCK_Y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    /* ── Step 5: one atomic per block (not per thread) ───────────────── */
    if (tid == 0) {
        atomicAddDouble(d_result, s_sum[0]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  HOST FUNCTION: sets up memory, launches kernel, retrieves result
 * ═══════════════════════════════════════════════════════════════════════════ */
double integrate_2d_cuda(IntegralParams2D p) {

    /* ── 1. Compute grid spacing ──────────────────────────────────────── */
    double dx = (p.x_max - p.x_min) / p.nx;
    double dy = (p.y_max - p.y_min) / p.ny;

    /* ── 2. Allocate device (GPU) memory for the result ───────────────── */
    double *d_result;
    cudaMalloc((void **)&d_result, sizeof(double));    /* one double on GPU  */
    cudaMemset(d_result, 0, sizeof(double));           /* initialise to zero */

    /* ── 3. Choose block and grid dimensions ──────────────────────────── *
     *  blockDim = (BLOCK_X, BLOCK_Y) = (16, 16) → 256 threads per block.
     *  gridDim  = enough blocks to cover all nx × ny cells.
     */
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((p.nx + block.x - 1) / block.x,
              (p.ny + block.y - 1) / block.y);

    /* ── 4. Launch the kernel ─────────────────────────────────────────── *
     *  <<<grid, block>>> is CUDA syntax: grid = how many blocks,
     *  block = how many threads per block.
     */
    kernel_integrate_2d<<<grid, block>>>(
        d_result,
        p.shape,
        p.center_x, p.center_y, p.radius,
        p.x_min, p.y_min,
        dx, dy,
        p.nx, p.ny
    );

    /* ── 5. Wait for kernel to finish ─────────────────────────────────── *
     *  Kernel launches are asynchronous — the CPU continues immediately.
     *  cudaDeviceSynchronize() blocks until all GPU work is done.
     */
    cudaDeviceSynchronize();

    /* ── 6. Copy result back to the CPU (device → host) ───────────────── */
    double result = 0.0;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    /* ── 7. Free GPU memory ───────────────────────────────────────────── */
    cudaFree(d_result);

    return result;
}

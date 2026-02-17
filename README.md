# EVIL_ARC_CS-GO

**Evaluation of Volumetric Integrals in Liquids with Attraction-Repulsion Colloids — CUDA Simulation with Graphical Overview**

Scientific semester project for the "Architecture of Computer Systems" course.

## What this project does

Numerically solves a system of nonlinear integral equations (mean-field DFT) describing the spatial density distribution of SALR particles confined between two walls — accelerated with NVIDIA CUDA.

## Project structure

```
├── configs/          # Simulation parameters (.cfg files)
├── include/          # Shared C/CUDA headers
├── src/
│   ├── core/         # DFT solver engine (future)
│   ├── cuda/         # GPU kernel implementations
│   ├── cpu/          # CPU reference implementations
│   ├── math/         # Vector/matrix utilities
│   └── utils/        # Config parser, I/O helpers
├── tests/            # Validation programs (CPU vs GPU)
├── scripts/          # Python plotting / post-processing
├── output/           # Runtime data (git-ignored)
├── studying/         # Standalone CUDA learning examples
├── docs/BUILD.md     # Build instructions
├── CMakeLists.txt    # CMake build system
├── Makefile          # Alternative Make build
└── configs/
    ├── default.cfg   # 2D test case
    └── test_3d.cfg   # 3D production run
```

## Build & run

Requires: **CMake ≥ 3.18**, **CUDA Toolkit** (nvcc), a C/C++ compiler, and a supported NVIDIA GPU.

> Full instructions: [docs/BUILD.md](docs/BUILD.md)

```bash
# Build everything
mkdir build && cd build
cmake ..
make

# Run tests
./test_integral_2d circle 1024
./test_math_ops 10000

# Run a studying example
./01_hello_gpu
```

## Studying examples (not part of the project)

| File | Topic |
|------|-------|
| `01_hello_gpu.cu` | Minimal kernel, launch syntax |
| `02_threads_and_blocks.cu` | Thread hierarchy, global ID |
| `03_memory_model.cu` | Shared memory, `__syncthreads` |
| `04_thrust_basics.cu` | Thrust vectors, sort, reduce |
| `05_error_handling.cu` | CUDA error checking macro |
| `06_libcupp_atomics.cu` | libcu++ atomics (`cuda::atomic`) |

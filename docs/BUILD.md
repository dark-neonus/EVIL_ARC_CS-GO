# How to build

## Requirements

- **CMake** ≥ 3.18
- **CUDA Toolkit** (nvcc)
- **C/C++ compiler** (gcc / g++)
- NVIDIA GPU with drivers installed

## Build steps

```bash
# 1. Create a build directory and go into it
mkdir build && cd build

# 2. Generate build files with CMake
cmake ..

# 3. Compile everything
make
```

That's it. Binaries appear inside `build/`.

## Run

```bash
# from the build/ directory:

# 2D integral test  (circle domain, 1024×1024 grid)
./test_integral_2d circle 1024

# math operations test  (vector size 10000)
./test_math_ops 10000

# any studying example
./01_hello_gpu
./04_thrust_basics
```

## Rebuild after changes

```bash
cd build && make
```

CMake only re-compiles files that changed.

## Clean rebuild

```bash
rm -rf build && mkdir build && cd build && cmake .. && make
```

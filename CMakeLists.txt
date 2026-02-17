# ═══════════════════════════════════════════════════════════════════════════════
#  SALR Particle Simulation — CMake build
# ═══════════════════════════════════════════════════════════════════════════════
cmake_minimum_required(VERSION 3.18)   # 3.18+ has native CUDA language support
project(EVIL_ARC_CS-GO LANGUAGES C CXX CUDA)

# ── Standard & optimisation ───────────────────────────────────────────────────
set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# GPU architecture targets — compile for multiple GPUs so the binary is portable.
# Override at configure time:  cmake .. -DCMAKE_CUDA_ARCHITECTURES=61
#
#   60 = Pascal  (GTX 1060/1070/1080, P100)
#   61 = Pascal  (GTX 1050/1070 Ti/1080 Ti)
#   70 = Volta   (V100, Titan V)
#   75 = Turing  (RTX 2060–2080, T4)
#   80 = Ampere  (A100, RTX 3090)
#   86 = Ampere  (RTX 3060–3080)
#   89 = Ada     (RTX 4060–4090)
set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89" CACHE STRING "CUDA architectures")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# ── Include headers ───────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ═══════════════════════════════════════════════════════════════════════════════
#  PROJECT TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# ── test_integral_2d ──────────────────────────────────────────────────────────
add_executable(test_integral_2d
    tests/test_integral_2d.cu
    src/cuda/integrate_2d.cu
    src/cpu/integrate_2d_cpu.c
)
target_link_libraries(test_integral_2d m)

# ── test_math_ops ─────────────────────────────────────────────────────────────
add_executable(test_math_ops
    tests/test_math_ops.c
    src/cuda/math_ops.cu
    src/cpu/math_ops_cpu.c
)
target_link_libraries(test_math_ops m)

# ═══════════════════════════════════════════════════════════════════════════════
#  STUDYING TARGETS (standalone learning examples, not part of main project)
# ═══════════════════════════════════════════════════════════════════════════════
file(GLOB STUDY_SOURCES studying/*.cu)

foreach(src ${STUDY_SOURCES})
    # extract filename without extension → becomes target name
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
endforeach()

# libcu++ atomics require sm_60+; override the arch list for this target
if(TARGET 06_libcupp_atomics)
    set_target_properties(06_libcupp_atomics PROPERTIES
        CUDA_ARCHITECTURES "60;61;70;75;80;86;89"
    )
endif()

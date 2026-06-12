cmake_minimum_required(VERSION 3.22)
project(cuda_uarch
    VERSION 0.1.0
    LANGUAGES CUDA CXX
    DESCRIPTION "GPU microarchitecture reverse-engineering via microbenchmarks"
)

# ─── C++ Standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── CUDA Architecture ──────────────────────────────────────────────────────
# Detects the local GPU compute capability automatically, fallback to 8.0
include(${CMAKE_SOURCE_DIR}/cmake/DetectCUDAArch.cmake)
detect_cuda_arch()

# ─── Build Options ───────────────────────────────────────────────────────────
option(BUILD_VERBOSE "Enable verbose compiler output" OFF)

# ─── Compiler / NVCC Flags ──────────────────────────────────────────────────
include(${CMAKE_SOURCE_DIR}/cmake/CompilerOptions.cmake)
set_cuda_flags()

# ─── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(benchmarks)

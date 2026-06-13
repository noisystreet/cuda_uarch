# AGENTS.md — AI Agent Guide

## Project Overview

GPU microarchitecture reverse-engineering via CUDA microbenchmarks.
Target: NVIDIA Ada Lovelace (sm_89) / RTX 4060 Laptop GPU.

## Build & Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# Run all benchmarks via Makefile
make run-latency       # Instruction latency
make run-throughput    # Instruction throughput / ILP
make run-mem-lat       # Global memory latency
make run-mem-bw        # Global memory bandwidth
make run-cache-size    # Cache hierarchy (size, line, associativity)
make run-shmem-bank    # Shared memory bank conflict
make run-scheduler     # Warp scheduler (divergence, concurrency, fairness)
make run-peak          # Peak compute (CUDA Core + Tensor Core)
make run-sfu           # Special function unit throughput
make run-denorm        # Tensor Core denormal behavior
make run-precision     # Numerical precision analysis (TC vs CUDA Core)
```

## Code Conventions

- **Language:** C++20, CUDA
- **Formatting:** `.clang-format` (LLVM-based, 4-space indent, 100 cols, Allman braces)
- **Run `clang-format -i`** on all `.cu` / `.h` files before committing
- **License:** MIT — add `// SPDX-License-Identifier: MIT` to new files
- **Benchmark output:** print lines prefixed with `RESULT,<label>,<median>,<mean>,<min>,<max>,<stddev>,<count>` for automated parsing by `plot_results.py`
- **Methodology:** single variable per experiment, median over repeated runs, prevent dead-code elimination (use output), warm-up before timing

## Project Structure

```
cuda_uarch/
├── benchmarks/
│   ├── common/          # config.h, timer.h, utils.h (shared infra)
│   ├── instruction/     # latency_probe, throughput_probe
│   ├── memory/          # global_mem_latency, global_mem_bw, cache_size_probe, shared_mem_bank
│   ├── scheduler/       # warp_scheduler_probe
│   └── advanced/        # peak_compute_probe, sfu_probe, denorm_probe
├── cmake/               # CMake modules
├── docs/                # PLAN.md, ANALYSIS.md
├── tools/               # check_env.sh, plot_results.py, install-precommit.sh
├── data/results/        # experimental CSV output
└── reports/figures/     # generated plots
```

## Naming

- Files: `snake_case.cu` / `snake_case.h`
- Types: `PascalCase` (e.g. `DeviceBuffer`, `GpuTimer`)
- Functions: `snake_case` (e.g. `get_device_info`, `compute_stats`)
- Macros: `UPPER_SNAKE_CASE` (e.g. `CUDA_CHECK`)
- Namespaces: `uarch` (lowercase)

## Agent Constraints

- **DO NOT** modify `docs/ANALYSIS.md` unless new benchmark data has been collected and validated
- **DO NOT** modify experimental results (RESULT CSV lines) in source code comments
- **ALWAYS** build and smoke-test a new benchmark before committing
- **ALWAYS** use `clang-format` before committing CUDA/C++ files
- **ALWAYS** use `#pragma unroll 1` on dependency-chain loops to prevent compiler optimisations from breaking measurements
- **ALWAYS** include a warm-up kernel launch before timed runs
- **DO NOT** merge experimental inference (e.g. "this implies N SM count") with established specification — note when a value is measured vs inferred
- **KEEP** analysis document language consistent with user's locale (currently Chinese)

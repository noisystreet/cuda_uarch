# AGENTS.md — AI Agent Guide

## Project Overview

GPU microarchitecture reverse-engineering via CUDA microbenchmarks.
Target: NVIDIA Ada Lovelace (sm_89) / RTX 4060 Laptop GPU.

## Build & Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# Run benchmarks
build/benchmarks/instruction/latency_probe   [chain_len] [repeats]
build/benchmarks/instruction/throughput_probe [unroll]    [repeats]
build/benchmarks/memory/global_mem_latency   [chain_len] [warps_per_sm] [repeats]
build/benchmarks/memory/global_mem_bw        [min_pow2]  [max_pow2]     [repeats]
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
│   ├── memory/          # global_mem_latency, global_mem_bw, cache_size_probe
│   ├── scheduler/       # (planned)
│   └── advanced/        # (planned)
├── cmake/               # CMake modules
├── docs/                # PLAN.md and other docs
├── tools/               # check_env.sh, plot_results.py, install-precommit.sh
└── data/results/        # experimental CSV output
```

## Naming

- Files: `snake_case.cu` / `snake_case.h`
- Types: `PascalCase` (e.g. `DeviceBuffer`, `GpuTimer`)
- Functions: `snake_case` (e.g. `get_device_info`, `compute_stats`)
- Macros: `UPPER_SNAKE_CASE` (e.g. `CUDA_CHECK`)
- Namespaces: `uarch` (lowercase)

# cuda_uarch — GPU Microarchitecture Reverse Engineering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-green)](https://developer.nvidia.com/cuda-toolkit)

Targeted microbenchmarks for NVIDIA GPU reverse engineering.
Systematically probes the CUDA software stack — Runtime → Driver → PTX → SASS —
to infer low-level microarchitecture details.

**Tested on:** RTX 4060 Laptop GPU (Ada Lovelace, sm_89)

> **Disclaimer:** This is a research / educational project. All findings are
> empirical estimates obtained through controlled experiments, not official
> specifications.

---

## Quick Start

```bash
# Prerequisites: CUDA Toolkit 12+ (nvcc, nvdisasm, cuobjdump), cmake 3.22+

# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)

# Or use the convenience Makefile
make build

# Run all benchmarks
make run-latency
make run-throughput
make run-mem-lat
make run-mem-bw
make run-cache-size
```

### Environment Check

```bash
bash tools/check_env.sh
```

---

## Benchmarks

| Benchmark | File | What It Measures |
|-----------|------|-----------------|
| **latency_probe** | `benchmarks/instruction/latency_probe.cu` | Instruction latency via dependency chains (IADD, FADD, FMUL, FFMA) |
| **throughput_probe** | `benchmarks/instruction/throughput_probe.cu` | Instruction throughput vs. ILP (FFMA with varying unroll factors) |
| **global_mem_latency** | `benchmarks/memory/global_mem_latency.cu` | Global memory load latency via pointer chasing |
| **global_mem_bw** | `benchmarks/memory/global_mem_bw.cu` | Read/write/copy bandwidth vs. array size |
| **cache_size_probe** | `benchmarks/memory/cache_size_probe.cu` | Cache size, cache line, and associativity probing |

### Planned

- Shared memory bank conflict analysis
- Warp scheduler / divergence probes
- Occupancy analysis
- Tensor Core benchmarks
- Warp shuffle and special function unit probes

---

## Results (RTX 4060 / Ada Lovelace)

### Instruction Latency

| Instruction | Measured Latency | Est. Cycles (@ 1.5 GHz) |
|-------------|-----------------|------------------------|
| FADD        | ~13.0 ns        | ~19.5 cycles |
| FMUL        | ~13.0 ns        | ~19.5 cycles |
| FFMA        | ~12.9 ns        | ~19.3 cycles |

> IADD latency was too low to measure reliably (optimized by compiler).
> FP operations show consistent ~19 cycle latency, matching Ada Lovelace
> expected pipeline depth.

### Cache Hierarchy

```
L1 Data Cache size  :  128 KB   (latency jump @ 128 KB working set)
L1 hit latency      :  ~41 ns
L2 Cache size       :   32 MB   (matches nvidia-smi reported value)
L2 hit latency      : ~133 ns
DRAM latency        : ~324 ns   (64 MB working set)
Cache line size     :  128 B    (latency jump @ 128 B stride)
```

### Global Memory Bandwidth

| Pattern | Bandwidth |
|---------|-----------|
| Read (float4)  | TBD |
| Write          | TBD |
| Copy           | TBD |

> Full bandwidth vs. size sweep data will be populated after running
> `global_mem_bw` with representative array sizes.

### Visualization

```bash
# Run a benchmark and capture results
make run-cache-size 2>&1 | tee data/results/cache_size.csv

# Generate plots
python3 tools/plot_results.py data/results/cache_size.csv --output-dir reports/figures/
```

---

## Project Structure

```
cuda_uarch/
├── benchmarks/
│   ├── common/          # Shared infra: config.h, timer.h, utils.h
│   ├── instruction/     # Instruction-level probes
│   ├── memory/          # Memory hierarchy probes
│   ├── scheduler/       # (planned)
│   └── advanced/        # (planned: Tensor Core, warp shuffle, SFU)
├── cmake/               # CMake modules
├── docs/                # Documentation
├── tools/               # check_env.sh, plot_results.py, install-precommit.sh
├── data/results/        # Experiment outputs (CSV)
└── reports/figures/     # Generated plots
```

---

## Methodology

- **Single variable:** Each experiment changes exactly one parameter
- **Median over mean:** Repeated runs, median selected for robustness
- **Dead-code prevention:** Results always written to output buffer
- **Warm-up:** Device initialized before timing begins
- **SASS verification:** Key kernels disassembled with `nvdisasm`

---

## Code Conventions

| Rule | Standard |
|------|----------|
| Language | C++20, CUDA |
| Formatting | `.clang-format` (LLVM-based) |
| License | MIT — add `// SPDX-License-Identifier: MIT` |
| Naming | `snake_case` files/functions, `PascalCase` types |
| Output | `RESULT,<label>,<median>,<mean>,<min>,<max>,<stddev>,<count>` |

Pre-commit hook auto-formats staged files:

```bash
bash tools/install-precommit.sh
```

---

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Dissecting the NVIDIA Turing GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1903.07486)

---

## License

MIT — see [LICENSE](LICENSE).

# ─── cuda_uarch convenience Makefile ─────────────────────────────────────
# Wraps CMake commands for common development tasks.
#
# Targets:
#   all / build     — configure + build
#   configure       — CMake configure (Release, sm_89)
#   debug           — CMake configure (Debug, sm_89)
#   clean           — remove build directory
#   rebuild         — clean + build
#   run-latency     — build + run latency_probe
#   run-throughput  — build + run throughput_probe
#   run-mem-lat     — build + run global_mem_latency
#   run-mem-bw      — build + run global_mem_bw
#   run-cache-size  — build + run cache_size_probe
#   run-shmem-bank  — build + run shared_mem_bank
#   run-scheduler   — build + run warp_scheduler_probe
#   run-peak        — build + run peak_compute_probe
#   run-sfu         — build + run sfu_probe
#   run-denorm      — build + run denorm_probe
#   run-precision   — build + run precision_probe
#   format          — clang-format all source files
#   check-env       — verify tool chain
#   help            — print this message

BUILD_DIR  ?= build
ARCH       ?= 89
CMAKE      ?= cmake
JOBS       ?= $(shell nproc 2>/dev/null || echo 4)

CMAKE_FLAGS  = -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)
CMAKE_DEBUG  = -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)

.PHONY: all build configure debug clean rebuild \
        run-latency run-throughput run-mem-lat run-mem-bw run-cache-size \
        run-shmem-bank run-scheduler run-peak run-sfu run-denorm run-precision format check-env help

# ─── Default ────────────────────────────────────────────────────────────────
all: build

# ─── Build ──────────────────────────────────────────────────────────────────
build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

debug:
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_DEBUG)

# ─── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build

# ─── Run benchmarks ─────────────────────────────────────────────────────────
run-latency: build
	$(BUILD_DIR)/benchmarks/instruction/latency_probe

run-throughput: build
	$(BUILD_DIR)/benchmarks/instruction/throughput_probe

run-mem-lat: build
	$(BUILD_DIR)/benchmarks/memory/global_mem_latency

run-mem-bw: build
	$(BUILD_DIR)/benchmarks/memory/global_mem_bw

run-cache-size: build
	$(BUILD_DIR)/benchmarks/memory/cache_size_probe

run-shmem-bank: build
	$(BUILD_DIR)/benchmarks/memory/shared_mem_bank

run-scheduler: build
	$(BUILD_DIR)/benchmarks/scheduler/warp_scheduler_probe

run-peak: build
	$(BUILD_DIR)/benchmarks/advanced/peak_compute_probe

run-sfu: build
	$(BUILD_DIR)/benchmarks/advanced/sfu_probe

run-denorm: build
	$(BUILD_DIR)/benchmarks/advanced/denorm_probe

run-precision: build
	$(BUILD_DIR)/benchmarks/advanced/precision_probe

# ─── Format ─────────────────────────────────────────────────────────────────
format:
	clang-format -i --style=file \
		$$(find . \( -name '*.cu' -o -name '*.h' \) ! -path './$(BUILD_DIR)/*' ! -path './.git/*')

# ─── Environment ────────────────────────────────────────────────────────────
check-env:
	@bash tools/check_env.sh

# ─── Help ───────────────────────────────────────────────────────────────────
help:
	@echo "cuda_uarch Makefile — available targets:"
	@echo ""
	@echo "  all / build       Configure and build (Release)"
	@echo "  configure         CMake configure (Release, sm_89)"
	@echo "  debug             CMake configure (Debug, sm_89)"
	@echo "  clean             Remove build directory"
	@echo "  rebuild           clean + build"
	@echo "  run-latency       Build + run latency_probe"
	@echo "  run-throughput    Build + run throughput_probe"
	@echo "  run-mem-lat       Build + run global_mem_latency"
	@echo "  run-mem-bw        Build + run global_mem_bw"
	@echo "  run-cache-size    Build + run cache_size_probe"
	@echo "  run-shmem-bank    Build + run shared_mem_bank"
	@echo "  run-scheduler     Build + run warp_scheduler_probe"
	@echo "  run-peak          Build + run peak_compute_probe"
	@echo "  run-sfu           Build + run sfu_probe"
	@echo "  run-denorm        Build + run denorm_probe"
	@echo "  run-precision     Build + run precision_probe"
	@echo "  format            clang-format all source files"
	@echo "  check-env         Verify tool chain"
	@echo "  help              Print this message"
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_DIR=$(BUILD_DIR)  ARCH=$(ARCH)  JOBS=$(JOBS)"

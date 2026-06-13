# ─── Common library (static) ──────────────────────────────────────────────────
add_library(uarch_common STATIC
    common/utils.cu
)

target_include_directories(uarch_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# ─── Per-benchmark executables ──────────────────────────────────────────────

add_subdirectory(instruction)
add_subdirectory(memory)
add_subdirectory(scheduler)
add_subdirectory(advanced)

# ─── Convenience alias ──────────────────────────────────────────────────────
# `make benchmarks` builds all
add_custom_target(benchmarks ALL
    DEPENDS
        latency_probe
        throughput_probe
        global_mem_latency
        global_mem_bw
        cache_size_probe
        shared_mem_bank
        warp_scheduler_probe
        peak_compute_probe
        sfu_probe
        denorm_probe
        precision_probe
        register_probe
    COMMENT "Building all microbenchmarks"
)

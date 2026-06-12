# ─── Global memory latency (pointer chasing) ──────────────────────────────
add_executable(global_mem_latency
    global_mem_latency.cu
)
target_link_libraries(global_mem_latency PRIVATE uarch_common)

# ─── Global memory bandwidth ──────────────────────────────────────────────
add_executable(global_mem_bw
    global_mem_bw.cu
)
target_link_libraries(global_mem_bw PRIVATE uarch_common)

# ─── Cache hierarchy probe (size, line, associativity) ────────────────────
add_executable(cache_size_probe
    cache_size_probe.cu
)
target_link_libraries(cache_size_probe PRIVATE uarch_common)

# ─── Shared memory bank conflict probe ────────────────────────────────────
add_executable(shared_mem_bank
    shared_mem_bank.cu
)
target_link_libraries(shared_mem_bank PRIVATE uarch_common)

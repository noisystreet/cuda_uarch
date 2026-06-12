# ─── Instruction latency probe (dependency chain) ─────────────────────────
add_executable(latency_probe
    latency_probe.cu
)
target_link_libraries(latency_probe PRIVATE uarch_common)

# ─── Instruction throughput probe (independent instructions) ─────────────
add_executable(throughput_probe
    throughput_probe.cu
)
target_link_libraries(throughput_probe PRIVATE uarch_common)

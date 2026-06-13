# ─── Warp scheduler probe (divergence, concurrency, fairness) ────────────
add_executable(warp_scheduler_probe
    warp_scheduler_probe.cu
)
target_link_libraries(warp_scheduler_probe PRIVATE uarch_common)

# ─── Warp shuffle / sync instruction probe ─────────────────────────────
add_executable(warp_shuffle_probe
    warp_shuffle_probe.cu
)
target_link_libraries(warp_shuffle_probe PRIVATE uarch_common)

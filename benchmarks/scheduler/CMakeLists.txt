# ─── Warp scheduler probe (divergence, concurrency, fairness) ────────────
add_executable(warp_scheduler_probe
    warp_scheduler_probe.cu
)
target_link_libraries(warp_scheduler_probe PRIVATE uarch_common)

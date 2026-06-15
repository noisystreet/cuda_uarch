# ─── Peak compute probe (CUDA Core + Tensor Core) ─────────────────────────
add_executable(peak_compute_probe
    peak_compute_probe.cu
)
target_link_libraries(peak_compute_probe PRIVATE uarch_common)

# sm_89 is needed for Tensor Core WMMA (sm_70+ is sufficient, but sm_89 adds TF32)
# CUDA architecture is already set globally

# ─── SFU (Special Function Unit) probe ─────────────────────────────────────
add_executable(sfu_probe
    sfu_probe.cu
)
target_link_libraries(sfu_probe PRIVATE uarch_common)

# ─── Denormal behavior probe (Tensor Core vs CUDA Core) ──────────────────
add_executable(denorm_probe
    denorm_probe.cu
)
target_link_libraries(denorm_probe PRIVATE uarch_common)

# ─── Numerical precision probe (TC vs CUDA Core) ──────────────────────────
add_executable(precision_probe
    precision_probe.cu
)
target_link_libraries(precision_probe PRIVATE uarch_common)

# ─── Register file probe (occupancy, pressure, renaming) ──────────────────
add_executable(register_probe
    register_probe.cu
)
target_link_libraries(register_probe PRIVATE uarch_common)

# ─── Occupancy & Block scheduling probe ───────────────────────────────────
add_executable(occupancy_probe
    occupancy_probe.cu
)
target_link_libraries(occupancy_probe PRIVATE uarch_common)

# ─── Tensor Core vs CUDA Core switching overhead probe ──────────────────
add_executable(switch_probe
    switch_probe.cu
)
target_link_libraries(switch_probe PRIVATE uarch_common)

# ─── Atomic operations probe (FP32 + INT32) ─────────────────────────────
add_executable(atomic_probe
    atomic_probe.cu
)
target_link_libraries(atomic_probe PRIVATE uarch_common)

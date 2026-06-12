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

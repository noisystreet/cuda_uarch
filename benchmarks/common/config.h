// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace uarch
{

// ─── Convenience GPU error check macro ─────────────────────────────────────
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

    // ─── Compile-time kernel configuration helpers ────────────────────────────

    /// Round-up division (constexpr)
    constexpr int div_up(int a, int b)
    {
        return (a + b - 1) / b;
    }

    /// Launch configuration packed struct
    struct LaunchConfig
    {
        dim3 grid_dim;
        dim3 block_dim;
        int shared_mem_bytes = 0;
        cudaStream_t stream = nullptr;
    };

    // ─── Device properties cache ───────────────────────────────────────────────
    struct DeviceInfo
    {
        int sm_count;
        int warp_size;
        int max_threads_per_sm;
        int max_regs_per_block;
        size_t shared_mem_per_block;
        size_t shared_mem_per_sm;
        size_t l2_cache_size;
        int major;
        int minor;

        [[nodiscard]] int max_warps_per_sm() const noexcept
        {
            return max_threads_per_sm / warp_size;
        }

        [[nodiscard]] std::string_view arch_name() const noexcept
        {
            if (major == 9)
                return "Blackwell";
            if (major == 8 && minor == 9)
                return "Ada Lovelace";
            if (major == 8 && minor == 6)
                return "Ampere (GA10x)";
            if (major == 8 && minor == 0)
                return "Ampere (GA100)";
            if (major == 7 && minor == 5)
                return "Turing";
            if (major == 7 && minor == 0)
                return "Volta";
            return "Unknown";
        }
    };

    /// Populate DeviceInfo from the current CUDA device.
    inline DeviceInfo get_device_info()
    {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        return DeviceInfo{
            .sm_count = prop.multiProcessorCount,
            .warp_size = prop.warpSize,
            .max_threads_per_sm = prop.maxThreadsPerMultiProcessor,
            .max_regs_per_block = prop.regsPerBlock,
            .shared_mem_per_block = prop.sharedMemPerBlock,
            .shared_mem_per_sm = prop.sharedMemPerMultiprocessor,
            .l2_cache_size = static_cast<size_t>(prop.l2CacheSize),
            .major = prop.major,
            .minor = prop.minor,
        };
    }

    /// Print device info to stdout
    inline void print_device_info(const DeviceInfo &info)
    {
        printf("╔══════════════════════════════════════════════╗\n");
        printf("║        Device Information                    ║\n");
        printf("╠══════════════════════════════════════════════╣\n");
        printf("║ Architecture  : %-34s ║\n", info.arch_name().data());
        printf("║ Compute Cap.  : %d.%d                            ║\n", info.major, info.minor);
        printf("║ SM count      : %-34d ║\n", info.sm_count);
        printf("║ Warp size     : %-34d ║\n", info.warp_size);
        printf("║ Max threads/SM: %-34d ║\n", info.max_threads_per_sm);
        printf("║ Max regs/block: %-34d ║\n", info.max_regs_per_block);
        printf("║ Shmem/block   : %-9zu bytes (%.1f KB)         ║\n", info.shared_mem_per_block,
               static_cast<double>(info.shared_mem_per_block) / 1024.0);
        printf("║ Shmem/SM      : %-9zu bytes (%.1f KB)         ║\n", info.shared_mem_per_sm,
               static_cast<double>(info.shared_mem_per_sm) / 1024.0);
        printf("║ L2 cache      : %-9zu bytes (%.1f MB)         ║\n", info.l2_cache_size,
               static_cast<double>(info.l2_cache_size) / 1024.0 / 1024.0);
        printf("╚══════════════════════════════════════════════╝\n\n");
    }

} // namespace uarch

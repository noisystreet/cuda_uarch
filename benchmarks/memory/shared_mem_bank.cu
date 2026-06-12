// SPDX-License-Identifier: MIT
/**
 * shared_mem_bank.cu — 共享内存 Bank Conflict 分析
 *
 * 实验：
 *   1. Bank Conflict 扫描  — 不同 stride 下的共享内存访问延迟
 *   2. 广播机制验证        — 所有线程读同一地址 vs 不同地址
 *   3. 多 warp 并发        — 更多 warp 下的 bank conflict 影响
 *
 * 背景：NVIDIA 共享内存分为 32 个 bank，每个 bank 4 字节宽。
 *   - stride=1:  连续访问，无 conflict（32 线程各访问不同 bank）
 *   - stride=32: 所有线程访问同一 bank → 32-way conflict（最坏情况）
 *   - stride=33: 错开映射，近似无 conflict
 *
 * Ada Lovelace (sm_89) 共享内存参数：
 *   - 32 banks × 4 B = 128 B/warp 无 conflict 带宽
 *   - 每 SM 100 KB 共享内存（max）
 *
 * Usage:
 *   ./shared_mem_bank [repeats]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// NVIDIA warp size is always 32 on all current GPUs
static constexpr int kWarpSize = 32;

// ════════════════════════════════════════════════════════════════════════════
//  Bank Conflict 扫描
// ════════════════════════════════════════════════════════════════════════════

/// 共享内存读写 kernel — 使用不同 stride 访问
///   - shmem_per_warp: 每个 warp 分配的共享内存段大小（float 个数）
///   - stride: 线程间地址步长（float 个数）
__global__ void shmem_bank_kernel(const float *__restrict__ input, float *__restrict__ output,
                                  int shmem_per_warp, int stride, int iterations)
{
    // 每个 block 分配共享内存
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    int bid = blockIdx.x;

    // 每个 warp 的数据在共享内存中的基址
    int warp_base = warp_id * shmem_per_warp;

    // 加载数据到共享内存（从全局内存）
    if (tid < blockDim.x)
    {
        smem[tid] = input[tid + bid * blockDim.x];
    }
    __syncthreads();

    float val = 0.0f;

    // 每迭代使用不同偏移，防止编译器 hoist 共享内存加载
    int base_offset = lane * stride;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        int idx = warp_base + (base_offset + i) % shmem_per_warp;
        val += smem[idx];
    }

    // 输出
    if (tid == 0)
        output[bid] = val;
}

// ════════════════════════════════════════════════════════════════════════════
//  广播测试 — 所有线程读同一地址
// ════════════════════════════════════════════════════════════════════════════

__global__ void shmem_broadcast_kernel(const float *__restrict__ input, float *__restrict__ output,
                                       int iterations)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int lane = tid % warpSize;

    if (tid < blockDim.x)
    {
        smem[tid] = input[tid + blockIdx.x * blockDim.x];
    }
    __syncthreads();

    float val = 0.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        // 所有 lane 读不同偏移，防止 hoist
        val += smem[i % 64];
    }

    if (tid == 0)
        output[blockIdx.x] = val;
}

// ════════════════════════════════════════════════════════════════════════════
//  多 warp 并发测试 — 增加 warp 数观察 conflict 累积效应
// ════════════════════════════════════════════════════════════════════════════

__global__ void shmem_warps_kernel(const float *__restrict__ input, float *__restrict__ output,
                                   int stride, int iterations, int warps_per_block)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // 只激活前 warps_per_block 个 warp
    if (warp_id >= warps_per_block)
        return;

    // 每个 warp 独享一段共享内存 (64 floats = 256 B)
    const int shmem_per_warp = 64;
    int warp_base = warp_id * shmem_per_warp;

    if (tid < blockDim.x)
    {
        smem[tid] = input[tid + blockIdx.x * blockDim.x];
    }
    __syncthreads();

    float val = 0.0f;
    int base_offset = lane * stride;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        int idx = warp_base + (base_offset + i) % shmem_per_warp;
        val += smem[idx];
    }

    if (tid == 0)
        output[blockIdx.x] = val;
}

// ════════════════════════════════════════════════════════════════════════════
//  测试驱动
// ════════════════════════════════════════════════════════════════════════════

void run_bank_conflict_scan(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 1. Bank Conflict 扫描 ═══\n");
    printf("Stride   Bank Pattern         Time (μs)  BW (GB/s)  Conflict\n");
    printf("------   -------------------   ---------  ---------  ---------\n");

    // 配置：每个 block 256 线程，1 block/SM，每个 warp 32 个 float
    const int block_size = 256;    // 8 warps/block
    const int shmem_per_warp = 64; // 256 B per warp（略大于 32，避免边界问题）
    const int shared_mem_per_block = shmem_per_warp * (block_size / kWarpSize) * sizeof(float);
    const int blocks = info.sm_count; // 1 block per SM 避免干扰
    const int num_items = blocks * block_size;
    const int iterations = 2000;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(blocks);
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    // 扫描 stride: 1, 2, 3, ..., 33, 64, 128, 256
    std::vector<int> strides;
    for (int s = 1; s <= 33; ++s)
        strides.push_back(s);
    strides.push_back(64);
    strides.push_back(128);
    strides.push_back(256);

    for (int stride : strides)
    {
        // Warm-up
        shmem_bank_kernel<<<blocks, block_size, shared_mem_per_block>>>(d_in.get(), d_out.get(),
                                                                        shmem_per_warp, stride, 4);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            shmem_bank_kernel<<<blocks, block_size, shared_mem_per_block>>>(
                d_in.get(), d_out.get(), shmem_per_warp, stride, iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }

        auto stats = compute_stats(samples);

        // 计算带宽：每个 warp 每迭代读取 shmem_per_warp 个 float
        // 总读取量 = blocks × (block_size/kWarpSize) × shmem_per_warp × iterations × 4 bytes
        double total_bytes = static_cast<double>(blocks) * (block_size / kWarpSize) *
                             shmem_per_warp * iterations * sizeof(float);
        double bw_gb_s = (total_bytes / (stats.median_us * 1e-6)) / 1e9;

        // 解析 bank conflict 程度
        const char *conflict_str;
        if (stride % 32 == 0)
            conflict_str = "32-way (最坏)";
        else if (stride % 16 == 0)
            conflict_str = "16-way";
        else if (stride % 8 == 0)
            conflict_str = "8-way";
        else if (stride % 4 == 0)
            conflict_str = "4-way";
        else if (stride % 2 == 0)
            conflict_str = "2-way";
        else
            conflict_str = "无 conflict";

        // bank pattern 说明
        char pattern[32];
        if (stride == 1)
            snprintf(pattern, sizeof(pattern), "lane[%d] → bank[%d]", stride, stride);
        else
            snprintf(pattern, sizeof(pattern), "stride=%d", stride);

        char label[128];
        snprintf(label, sizeof(label), "shmem_bank stride=%d smem_pw=%d", stride, shmem_per_warp);
        print_csv(label, samples);

        printf("  %-6d  %-20s  %9.1f  %9.1f  %s\n", stride, pattern, stats.median_us, bw_gb_s,
               conflict_str);
    }
}

void run_broadcast_test(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 2. 广播机制测试 ═══\n");
    printf("所有线程读同一地址 vs 不同地址\n");

    const int block_size = 256;
    const int shared_mem_per_block = block_size * sizeof(float);
    const int blocks = info.sm_count;
    const int num_items = blocks * block_size;
    const int iterations = 2000;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(blocks);
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    // 广播测试
    {
        shmem_broadcast_kernel<<<blocks, block_size, shared_mem_per_block>>>(d_in.get(),
                                                                             d_out.get(), 4);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            shmem_broadcast_kernel<<<blocks, block_size, shared_mem_per_block>>>(
                d_in.get(), d_out.get(), iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);
        char label[128];
        snprintf(label, sizeof(label), "shmem_broadcast iter=%d", iterations);
        print_csv(label, samples);

        double total_bytes = static_cast<double>(blocks) * (block_size / kWarpSize) * 64 *
                             iterations * sizeof(float);
        double bw_gb_s = (total_bytes / (stats.median_us * 1e-6)) / 1e9;
        printf("  广播（同地址）: %.1f μs  (BW: %.1f GB/s)\n", stats.median_us, bw_gb_s);
    }

    // 无 conflict 对比（stride=1）
    {
        shmem_bank_kernel<<<blocks, block_size, shared_mem_per_block>>>(d_in.get(), d_out.get(), 64,
                                                                        1, 4);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            shmem_bank_kernel<<<blocks, block_size, shared_mem_per_block>>>(d_in.get(), d_out.get(),
                                                                            64, 1, iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);
        double total_bytes = static_cast<double>(blocks) * (block_size / kWarpSize) * 64 *
                             iterations * sizeof(float);
        double bw_gb_s = (total_bytes / (stats.median_us * 1e-6)) / 1e9;
        printf("  无 conflict（stride=1）: %.1f μs  (BW: %.1f GB/s)\n", stats.median_us, bw_gb_s);
    }
}

void run_multi_warp_test(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 3. 多 Warp 并发 + Bank Conflict ═══\n");
    printf("Warp/Block  Stride=1 (μs)  Stride=32 (μs)  冲突/无冲突比\n");
    printf("----------  --------------  ---------------  -------------\n");

    const int iterations = 2000;
    const int blocks = info.sm_count;

    int warp_counts[] = {1, 2, 4, 8};

    for (int warps_per_block : warp_counts)
    {
        int block_size = warps_per_block * kWarpSize;
        // 每个 warp 64 float, 每个 block block_size/kWarpSize 个 warp
        int smem_per_warp = 64;
        int shared_mem = smem_per_warp * warps_per_block * sizeof(float);

        DeviceBuffer<float> d_in(blocks * block_size);
        DeviceBuffer<float> d_out(blocks);
        std::vector<float> h_in(blocks * block_size, 1.0f);
        d_in.upload(h_in.data());

        // stride=1 (no conflict)
        double time_nc = 0;
        {
            shmem_warps_kernel<<<blocks, block_size, shared_mem>>>(d_in.get(), d_out.get(), 1, 4,
                                                                   warps_per_block);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> samples;
            samples.reserve(repeats);
            for (int r = 0; r < repeats; ++r)
            {
                GpuTimer timer;
                timer.start();
                shmem_warps_kernel<<<blocks, block_size, shared_mem>>>(d_in.get(), d_out.get(), 1,
                                                                       iterations, warps_per_block);
                timer.stop();
                samples.push_back(timer.elapsed_us());
            }
            time_nc = compute_stats(samples).median_us;
        }

        // stride=32 (worst conflict)
        double time_wc = 0;
        {
            shmem_warps_kernel<<<blocks, block_size, shared_mem>>>(d_in.get(), d_out.get(), 32, 4,
                                                                   warps_per_block);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> samples;
            samples.reserve(repeats);
            for (int r = 0; r < repeats; ++r)
            {
                GpuTimer timer;
                timer.start();
                shmem_warps_kernel<<<blocks, block_size, shared_mem>>>(d_in.get(), d_out.get(), 32,
                                                                       iterations, warps_per_block);
                timer.stop();
                samples.push_back(timer.elapsed_us());
            }
            time_wc = compute_stats(samples).median_us;
        }

        printf("  %-8d     %12.1f     %13.1f     %.1f×\n", warps_per_block, time_nc, time_wc,
               time_wc / time_nc);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[])
{
    int repeats = 8;
    if (argc > 1)
        repeats = std::atoi(argv[1]);
    if (repeats <= 0)
        repeats = 8;

    auto info = get_device_info();
    print_device_info(info);
    warm_up_device();

    printf("共享内存参数:\n");
    printf("  Bank 数量: 32 (NVIDIA 标准)\n");
    printf("  Bank 宽度: 4 bytes\n");
    printf("  每 SM 共享内存: %zu bytes (%.1f KB)\n\n", info.shared_mem_per_sm,
           static_cast<double>(info.shared_mem_per_sm) / 1024.0);

    run_bank_conflict_scan(info, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    run_broadcast_test(info, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    run_multi_warp_test(info, repeats);

    printf("\nDone.\n");
    return 0;
}

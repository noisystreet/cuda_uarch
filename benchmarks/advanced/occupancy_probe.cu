// SPDX-License-Identifier: MIT
/**
 * occupancy_probe.cu — SM 占用率 (Occupancy) 与 Block 调度分析
 *
 * 实验：
 *   1. Block Size Sweep — 固定总线程数，改变 block 大小，观察吞吐变化
 *   2. 共享内存约束 — 固定 block 大小，增加每 block 共享内存，限制占用率
 *   3. 寄存器约束 — 通过 __launch_bounds__ 限制每 SM block 数
 *   4. 多 block 并发 — 少量大 block vs 大量小 block 的完成时间对比
 *
 * 背景：
 *   - Ada Lovelace (sm_89): 每 SM 最多 1536 线程 = 48 warps
 *   - 每 SM 最多 32 个 block（maxBlocksPerSM）
 *   - 每 SM 65536 寄存器、100 KB 共享内存
 *   - 实际占用率受限于三者中最严格的一个
 *
 * Usage:
 *   ./occupancy_probe [repeats]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ════════════════════════════════════════════════════════════════════════════
//  Kernel: 通用 FP32 FFMA 负载（可调节 ILP）
// ════════════════════════════════════════════════════════════════════════════

__global__ void workload_kernel(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 8 路 ILP
    float r0 = 1.0000001f, r1 = r0, r2 = r0, r3 = r0;
    float r4 = r0, r5 = r0, r6 = r0, r7 = r0;
    float a = 0.9999999f, b = 1.0000001f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = fmaf(r0, a, b);
        r1 = fmaf(r1, a, b);
        r2 = fmaf(r2, a, b);
        r3 = fmaf(r3, a, b);
        r4 = fmaf(r4, a, b);
        r5 = fmaf(r5, a, b);
        r6 = fmaf(r6, a, b);
        r7 = fmaf(r7, a, b);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  Kernel: 多 block 并发时间戳
// ════════════════════════════════════════════════════════════════════════════

__global__ void timestamp_kernel(uint64_t *__restrict__ start_times,
                                 uint64_t *__restrict__ end_times, int busy_loops)
{
    // 仅第一个 warp 的线程记录时间戳
    if (threadIdx.x == 0)
    {
        start_times[blockIdx.x] = clock64();
    }

    // 忙等（消耗 GPU 周期）
    volatile float sink = 0.0f;
    for (volatile int i = 0; i < busy_loops; ++i)
    {
        sink = fmaf(sink, 1.0000001f, 0.9999999f);
    }

    if (threadIdx.x == 0)
    {
        end_times[blockIdx.x] = clock64();
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Kernel: 共享内存 + 计算混合负载
// ════════════════════════════════════════════════════════════════════════════

__global__ void shared_mem_workload(float *__restrict__ output, int shmem_floats, int iterations)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // 初始化共享内存
    if (tid < shmem_floats)
    {
        smem[tid] = 1.0f;
    }
    __syncthreads();

    // 计算负载：读取共享内存并做 FMA
    float sum = 0.0f;
    float a = 1.0000001f, b = 0.9999999f;

    // 无共享内存时使用寄存器值
    float local_val = 1.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        if (shmem_floats > 0)
        {
            sum += smem[tid % shmem_floats];
        }
        else
        {
            sum += local_val;
        }
        sum = fmaf(sum, a, b);
    }

    if (tid == 0)
        output[blockIdx.x] = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 1: Block Size Sweep
// ════════════════════════════════════════════════════════════════════════════

void run_block_size_sweep(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 1. Block Size Sweep ═══\n");
    printf("  固定总线程 = %d (%d SM × max 1024 threads/SM)\n", info.sm_count * 1024,
           info.sm_count);
    printf("  调整 block 大小，观察吞吐变化\n\n");
    printf("  BlockSize  Blocks  Warps/SM     Time     Throughput   Occupancy\n");
    printf("  --------- ------  --------   -------   -----------   ---------\n");

    const int total_threads = info.sm_count * 1024; // 接近满占用
    const int iterations = 20000;
    DeviceBuffer<float> d_out(1);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    for (int i = 0; i < num_sizes; ++i)
    {
        int bs = block_sizes[i];
        int blocks = (total_threads + bs - 1) / bs; // 向上取整

        // 预热
        workload_kernel<<<1, bs>>>(d_out.get(), 4);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            workload_kernel<<<blocks, bs>>>(d_out.get(), iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);

        double total_ops =
            static_cast<double>(blocks) * bs * iterations * 8 * 2; // 8 FMA/iter × 2 ops
        double gops = total_ops / (stats.median_us * 1e-6) / 1e9;

        int warps_per_block = (bs + 31) / 32;
        int warps_per_sm = blocks * warps_per_block / info.sm_count;
        if (warps_per_sm > 48)
            warps_per_sm = 48; // 硬件上限

        char label[128];
        snprintf(label, sizeof(label), "blocksize bs=%d blocks=%d iter=%d", bs, blocks, iterations);
        print_csv(label, samples);

        printf("  %-7d    %-4d     %-3d       %8.1f  %10.0f   %d/%d\n", bs, blocks, warps_per_sm,
               stats.median_us, gops, warps_per_sm, 48);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 2: 共享内存约束
// ════════════════════════════════════════════════════════════════════════════

void run_shmem_occupancy(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 2. 共享内存约束 ═══\n");
    printf("  固定 block=256 threads, 增加每 block 共享内存\n");
    printf("  每 SM 最大共享内存: %zu bytes (%.1f KB)\n\n", info.shared_mem_per_sm,
           static_cast<double>(info.shared_mem_per_sm) / 1024.0);
    printf("  Shmem/Block    MaxBlk/SM    Time    Throughput\n");
    printf("  -----------   ----------  -------  -----------\n");

    const int threads = 256;
    const int blocks_target = info.sm_count * 4;
    const int iterations = 10000;
    DeviceBuffer<float> d_out(blocks_target);

    // 共享内存大小从 0 到超过 100 KB/SM（饱和点）
    int shmem_sizes_bytes[] = {0,     1024,  4096,  8192,  16384, 24576,
                               32768, 49152, 65536, 81920, 102400};
    int num_sizes = sizeof(shmem_sizes_bytes) / sizeof(shmem_sizes_bytes[0]);

    for (int i = 0; i < num_sizes; ++i)
    {
        int shmem_bytes = shmem_sizes_bytes[i];
        int shmem_floats = shmem_bytes / sizeof(float);

        // 预热
        shared_mem_workload<<<1, threads, shmem_bytes>>>(d_out.get(), shmem_floats, 4);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            shared_mem_workload<<<blocks_target, threads, shmem_bytes>>>(d_out.get(), shmem_floats,
                                                                         iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);

        // 估算 SM 可容纳的最大 block 数（受共享内存限制）
        int max_blk_by_shmem =
            (shmem_bytes > 0) ? info.shared_mem_per_sm / shmem_bytes : 100; // 无限制
        if (max_blk_by_shmem > 32)
            max_blk_by_shmem = 32; // 硬件上限

        double total_ops =
            static_cast<double>(blocks_target) * threads * iterations * 1 * 2; // 1 FMA/iter × 2 ops
        double gops = total_ops / (stats.median_us * 1e-6) / 1e9;

        char label[128];
        snprintf(label, sizeof(label), "shmem_occupancy shmem=%d iter=%d", shmem_bytes, iterations);
        print_csv(label, samples);

        printf("  %-7d B     %-5d        %8.1f  %8.0f\n", shmem_bytes, max_blk_by_shmem,
               stats.median_us, gops);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 3: Block 分发时间戳
// ════════════════════════════════════════════════════════════════════════════

void run_block_distribution(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 3. Block 分发/执行时间 ═══\n");
    printf("  不同 block/thread 配置下的总执行时间\n");
    printf("  每线程 busy_loops=500\n\n");
    printf("  Config                          Time (μs)\n");
    printf("  ------------------------------  ---------\n");

    struct
    {
        const char *label;
        int blocks;
        int threads;
    } configs[] = {
        {"24 blocks × 32 threads", info.sm_count, 32},
        {"24 blocks × 256 threads", info.sm_count, 256},
        {"48 blocks × 256 threads", info.sm_count * 2, 256},
        {"96 blocks × 256 threads", info.sm_count * 4, 256},
        {"192 blocks × 128 threads", info.sm_count * 8, 128},
        {"24 blocks × 1024 threads", info.sm_count, 1024},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    DeviceBuffer<uint64_t> d_start(4096);
    DeviceBuffer<uint64_t> d_end(4096);

    for (int i = 0; i < num_configs; ++i)
    {
        auto &c = configs[i];

        // 预热
        timestamp_kernel<<<1, 32>>>(d_start.get(), d_end.get(), 10);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> raw_times;
        raw_times.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            timestamp_kernel<<<c.blocks, c.threads>>>(d_start.get(), d_end.get(), 500);
            timer.stop();
            raw_times.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(raw_times);

        printf("  %-30s  %8.1f μs\n", c.label, stats.median_us);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 4: 多 block 并发
// ════════════════════════════════════════════════════════════════════════════

void run_concurrent_blocks(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 4. Block 并发度探测 ═══\n");
    printf("  发射 192 blocks × 128 threads, 不同 busy_loops\n");
    printf("  用总执行时间推断 block 是否并发启动\n\n");
    printf("  busy_loops    Total Time (μs)  Est. Per-Block\n");
    printf("  ----------   ----------------  --------------\n");

    const int blocks = info.sm_count * 8; // 192 for 24 SM
    const int threads = 128;
    DeviceBuffer<uint64_t> d_start(blocks);
    DeviceBuffer<uint64_t> d_end(blocks);

    int busy_loop_values[] = {100, 500, 2000, 10000};

    for (int busy : busy_loop_values)
    {
        timestamp_kernel<<<1, 32>>>(d_start.get(), d_end.get(), 10);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            timestamp_kernel<<<blocks, threads>>>(d_start.get(), d_end.get(), busy);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);

        // 粗略估算: 如果所有 block 并发，每 block 耗时 ≈ total / min_warps
        // 如果串行，每 block 耗时 ≈ total / blocks
        double per_block_serial_ns = stats.median_us * 1000.0 / blocks;
        double per_block_parallel_ns = (stats.median_us * 1000.0) / (blocks / (info.sm_count * 4));

        printf("  %-10d  %12.1f μs    %.2f ns (serial)\n", busy, stats.median_us,
               per_block_serial_ns);
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

    printf("==================================================\n");
    printf("  Occupancy & Block 调度分析\n");
    printf("  GPU: %s (sm_%d.%d)\n", info.arch_name().data(), info.major, info.minor);
    printf("  硬件限制: maxWarps/SM=%d  maxBlocks/SM=%d\n", 48, 32);
    printf("  Shmem/SM=%zu  Regs/SM=65536\n", info.shared_mem_per_sm);
    printf("==================================================\n\n");

    run_block_size_sweep(info, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    run_shmem_occupancy(info, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    run_block_distribution(info, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    run_concurrent_blocks(info, repeats);

    printf("\nDone.\n");
    return 0;
}

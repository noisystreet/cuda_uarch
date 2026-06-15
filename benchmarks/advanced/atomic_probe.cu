// SPDX-License-Identifier: MIT
//
// atomic_probe.cu — 原子操作吞吐/延迟探测
//
// 测量：
//   1. FP32 atomicAdd 吞吐（低争用 vs 高争用）
//   2. INT32 atomicAdd 吞吐（低争用 vs 高争用）
//   3. 争用程度扫描（threads 数从 1 warp → 多 warp → 多 block）
//   4. 与非原子操作写入带宽对比

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// ─── Kernel 1: FP32 atomicAdd — 集中争用 (所有线程写同一地址) ─────────────
__global__ void atomic_fp32_hotspot(float *__restrict__ counter, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = (float)((tid * 12347) % 9973) * 0.001f + 1.0f;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        atomicAdd(counter, val);
    }
}

// ─── Kernel 2: INT32 atomicAdd — 集中争用 ──────────────────────────────────
__global__ void atomic_int32_hotspot(int *__restrict__ counter, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = (tid * 12347) % 9973 + 1;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        atomicAdd(counter, val);
    }
}

// ─── Kernel 3: FP32 atomicAdd — 分散 (每个线程写不同地址) ─────────────────
__global__ void atomic_fp32_scattered(float *__restrict__ data, int stride, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = (tid * stride) % (gridDim.x * blockDim.x);
    float val = (float)((tid * 12347) % 9973) * 0.001f + 1.0f;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        atomicAdd(&data[idx], val);
    }
}

// ─── Kernel 4: INT32 atomicAdd — 分散 ─────────────────────────────────────
__global__ void atomic_int32_scattered(int *__restrict__ data, int stride, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = (tid * stride) % (gridDim.x * blockDim.x);
    int val = (tid * 12347) % 9973 + 1;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        atomicAdd(&data[idx], val);
    }
}

// ─── Kernel 5: 非原子 FP32 写入（用于对比） ──────────────────────────────────
__global__ void plain_fp32_write(float *__restrict__ data, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = (float)((tid * 12347) % 9973) * 0.001f + 1.0f;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        data[tid] = val;
        val += 0.001f;
    }
}

// ─── Kernel 6: 非原子 INT32 写入（用于对比） ──────────────────────────────────
__global__ void plain_int32_write(int *__restrict__ data, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = (tid * 12347) % 9973 + 1;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        data[tid] = val;
        val += 1;
    }
}

// ─── Host stats helper ──────────────────────────────────────────────────────
struct BenchStats
{
    double median_us;
    double mean_us;
    double min_us;
    double max_us;
    double stddev_us;
    int count;
};

static BenchStats run_benchmark(std::function<void()> launch_kernel, int samples = 8)
{
    std::vector<double> times;
    times.reserve(samples);

    for (int s = 0; s < samples; ++s)
    {
        uarch::GpuTimer timer;
        timer.start();
        launch_kernel();
        timer.stop();
        times.push_back(timer.elapsed_us());
    }

    std::sort(times.begin(), times.end());
    int n = (int)times.size();
    double median = (n % 2 == 0) ? (times[n / 2 - 1] + times[n / 2]) * 0.5 : times[n / 2];

    double sum = 0.0;
    for (auto t : times)
        sum += t;
    double mean = sum / n;

    double sq_sum = 0.0;
    for (auto t : times)
        sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / n);

    double min_val = times.front();
    double max_val = times.back();

    return {median, mean, min_val, max_val, stddev, n};
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main()
{
    auto info = uarch::get_device_info();
    uarch::print_device_info(info);
    printf("=== 原子操作吞吐/延迟探测 ===\n\n");

    const int repeat = 10000;

    // ─── 1. Hotspot contention sweep ─────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 1. 集中争用 (Hotspot) — 所有线程写同一地址                    │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-12s %12s %12s %12s %12s\n", "Threads", "FP32(us)", "FP32(ops/s)", "INT32(us)",
           "INT32(ops/s)");

    // Vary block sizes: 32, 64, 128, 256, 512, 1024
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    // Always 1 block for hotspot (all threads fight over 1 address)
    const int grid = 1;

    for (int bs : block_sizes)
    {
        // FP32 hot
        uarch::DeviceBuffer<float> d_fp32(1);
        CUDA_CHECK(cudaMemset(d_fp32.get(), 0, sizeof(float)));
        auto fp32_stats = run_benchmark(
            [&]()
            {
                atomic_fp32_hotspot<<<grid, bs>>>(d_fp32.get(), repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        // INT32 hot
        uarch::DeviceBuffer<int> d_int32(1);
        CUDA_CHECK(cudaMemset(d_int32.get(), 0, sizeof(int)));
        auto int32_stats = run_benchmark(
            [&]()
            {
                atomic_int32_hotspot<<<grid, bs>>>(d_int32.get(), repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        long long fp32_ops = (long long)grid * bs * repeat;
        long long int32_ops = fp32_ops;
        double fp32_rate = fp32_ops / (fp32_stats.median_us * 1e-6);
        double int32_rate = int32_ops / (int32_stats.median_us * 1e-6);

        printf("%-12d %12.2f %12.2e %12.2f %12.2e\n", bs, fp32_stats.median_us, fp32_rate,
               int32_stats.median_us, int32_rate);
    }
    printf("\n");

    // ─── 2. Scattered atomic with stride 1 ──────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 2. 分散写入 (Scattered) — 每个线程写独立地址 (stride=1)      │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-12s %12s %12s %12s %12s\n", "Threads", "FP32(us)", "FP32(ops/s)", "INT32(us)",
           "INT32(ops/s)");

    for (int bs : block_sizes)
    {
        int total_threads = grid * bs;
        uarch::DeviceBuffer<float> d_fp32(total_threads);
        uarch::DeviceBuffer<int> d_int32(total_threads);

        auto fp32_stats = run_benchmark(
            [&]()
            {
                atomic_fp32_scattered<<<grid, bs>>>(d_fp32.get(), 1, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        auto int32_stats = run_benchmark(
            [&]()
            {
                atomic_int32_scattered<<<grid, bs>>>(d_int32.get(), 1, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        long long ops = (long long)grid * bs * repeat;
        double fp32_rate = ops / (fp32_stats.median_us * 1e-6);
        double int32_rate = ops / (int32_stats.median_us * 1e-6);

        printf("%-12d %12.2f %12.2e %12.2f %12.2e\n", total_threads, fp32_stats.median_us,
               fp32_rate, int32_stats.median_us, int32_rate);
    }
    printf("\n");

    // ─── 3. Non-atomic write baseline ────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 3. 非原子写入对比 (Baseline)                                 │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-12s %12s %12s %12s %12s\n", "Threads", "PlainFP32", "PlainINT32", "FP32/Plain",
           "INT32/Plain");

    for (int bs : block_sizes)
    {
        int total_threads = grid * bs;
        uarch::DeviceBuffer<float> d_fp32(total_threads);
        uarch::DeviceBuffer<int> d_int32(total_threads);

        auto plain_fp32 = run_benchmark(
            [&]()
            {
                plain_fp32_write<<<grid, bs>>>(d_fp32.get(), repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        auto plain_int32 = run_benchmark(
            [&]()
            {
                plain_int32_write<<<grid, bs>>>(d_int32.get(), repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        // Re-run hotspot FP32 for this config
        uarch::DeviceBuffer<float> d_hot_fp32(1);
        auto fp32_hot = run_benchmark(
            [&]()
            {
                atomic_fp32_hotspot<<<grid, bs>>>(d_hot_fp32.get(), repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        uarch::DeviceBuffer<int> d_hot_int32(1);
        auto int32_hot = run_benchmark(
            [&]()
            {
                atomic_int32_hotspot<<<grid, bs>>>(d_hot_int32.get(), repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        double fp32_ratio = fp32_hot.median_us / plain_fp32.median_us;
        double int32_ratio = int32_hot.median_us / plain_int32.median_us;

        printf("%-12d %12.2f %12.2f %12.2fx %12.2fx\n", total_threads, plain_fp32.median_us,
               plain_int32.median_us, fp32_ratio, int32_ratio);
    }
    printf("\n");

    // ─── 4. Stride sweep on scattered atomic ────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 4. Stride 扫描 — 分散原子操作 vs 访存模式                   │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-10s %12s %12s\n", "Stride", "FP32(us)", "INT32(us)");

    const int bs_fixed = 256;
    std::vector<int> stride_vals = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    for (int s : stride_vals)
    {
        int total = grid * bs_fixed;
        uarch::DeviceBuffer<float> d_fp32(total);
        uarch::DeviceBuffer<int> d_int32(total);

        auto fp32_st = run_benchmark(
            [&]()
            {
                atomic_fp32_scattered<<<grid, bs_fixed>>>(d_fp32.get(), s, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        auto int32_st = run_benchmark(
            [&]()
            {
                atomic_int32_scattered<<<grid, bs_fixed>>>(d_int32.get(), s, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });

        printf("%-10d %12.2f %12.2f\n", s, fp32_st.median_us, int32_st.median_us);
    }
    printf("\n");

    return 0;
}
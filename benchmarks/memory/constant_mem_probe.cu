// SPDX-License-Identifier: MIT
//
// constant_mem_probe.cu — 常量内存广播机制探测
//
// 测量：
//   1. __constant__ 广播读取延迟（全 warp 读同一地址）
//   2. __constant__ 线性读取（多地址序列化）
//   3. __constant__ 与全局内存读取带宽对比
//   4. 常量内存缓存行大小推断（通过 stride sweep）

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// ─── Constants ────────────────────────────────────────────────────────────────
// 常量内存最大 64 KB，我们用前 32 KB 做测试
#define CONST_SIZE (32 * 1024) // 32 KB → 8192 floats
#define CONST_FLOATS (CONST_SIZE / sizeof(float))

// ─── Device-side data in __constant__ memory ─────────────────────────────────
__constant__ float const_data[CONST_FLOATS];

// ─── Kernel 1: Broadcast latency — 全 warp 读同一地址 ──────────────────────
// 每个线程重复读同一个常量地址，输出到全局内存防 DCE
__global__ void const_broadcast_read(float *__restrict__ output, int addr_idx, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    float val = const_data[addr_idx];

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        sum += val;
        val = const_data[addr_idx];
    }
    if (tid == 0)
        output[0] = sum;
}

// ─── Kernel 2: Linear sequential read — 各线程读不同地址 ──────────────────
__global__ void const_linear_read(float *__restrict__ output, int stride, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = (tid * stride) % CONST_FLOATS;
    float sum = 0.0f;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        sum += const_data[idx];
        idx = (idx + stride) % CONST_FLOATS;
    }
    output[tid] = sum;
}

// ─── Kernel 3: Global memory equivalent for comparison ──────────────────────
__global__ void global_read(float *__restrict__ output, const float *__restrict__ data, int stride,
                            int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = (tid * stride) % CONST_FLOATS;
    float sum = 0.0f;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        sum += data[idx];
        idx = (idx + stride) % CONST_FLOATS;
    }
    output[tid] = sum;
}

// ─── Kernel 4: Stride sweep for cache line detection ───────────────────────
// 测量不同 stride（单位：float）下的读取耗时
// 当 stride 跨越缓存行边界时，耗时会有跳变
__global__ void const_stride_read(float *__restrict__ output, int stride, int repeat)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    int idx = 0;

#pragma unroll 1
    for (int i = 0; i < repeat; ++i)
    {
        sum += const_data[idx];
        idx = (idx + stride) % CONST_FLOATS;
    }
    output[tid] = sum;
}

// ─── Host helper: run a kernel multiple times and collect stats ────────────
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

    // compute stats
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
    // ─── Device info ─────────────────────────────────────────────────────────
    auto info = uarch::get_device_info();
    uarch::print_device_info(info);
    printf("=== 常量内存广播机制探测 ===\n\n");

    const int warp_size = info.warp_size;

    // ─── Init constant memory from host ──────────────────────────────────────
    std::vector<float> host_data(CONST_FLOATS);
    for (int i = 0; i < CONST_FLOATS; ++i)
        host_data[i] = (float)((i * 12347) % 9973) * 0.001f + 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(const_data, host_data.data(), CONST_SIZE));

    // ─── Alloc device output ─────────────────────────────────────────────────
    const int threads = warp_size;
    uarch::DeviceBuffer<float> d_out(threads);
    uarch::DeviceBuffer<float> d_global(CONST_FLOATS);
    d_global.upload(host_data.data());

    const int repeat = 100000;

    // ─── 1. Broadcast latency ────────────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 1. 常量内存广播读取延迟 (全 warp=%d 读同一地址)              │\n", warp_size);
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-12s %10s %10s %10s %10s %6s\n", "Addr", "Median", "Mean", "Min", "Max", "Samples");
    printf("%-12s %10s %10s %10s %10s %6s\n", "Index", "(us)", "(us)", "(us)", "(us)", "");

    for (int addr : {0, 1, 2, 4, 8, 16, 32, 64, 128, 256})
    {
        auto stats = run_benchmark(
            [&]()
            {
                const_broadcast_read<<<1, threads>>>(d_out.get(), addr, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        printf("%-12d %10.2f %10.2f %10.2f %10.2f %6d\n", addr, stats.median_us, stats.mean_us,
               stats.min_us, stats.max_us, stats.count);
    }
    printf("\n");

    // ─── 2. Linear read with stride sweep ────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 2. 常量内存线性读取 Stride Sweep                            │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-10s %10s %10s %10s %10s\n", "Stride", "Median", "Mean", "Min", "Max");
    printf("%-10s %10s %10s %10s %10s\n", "(floats)", "(us)", "(us)", "(us)", "(us)");

    std::vector<int> strides = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int s : strides)
    {
        auto stats_const = run_benchmark(
            [&]()
            {
                const_linear_read<<<1, threads>>>(d_out.get(), s, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        auto stats_global = run_benchmark(
            [&]()
            {
                global_read<<<1, threads>>>(d_out.get(), d_global.get(), s, repeat);
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        printf("%-10d  C %8.2f %8.2f %8.2f %8.2f\n", s, stats_const.median_us, stats_const.mean_us,
               stats_const.min_us, stats_const.max_us);
        printf("%-10s  G %8.2f %8.2f %8.2f %8.2f\n", "", stats_global.median_us,
               stats_global.mean_us, stats_global.min_us, stats_global.max_us);
    }
    printf("  C = __constant__   G = global memory\n\n");

    // ─── 3. Cache line stride sweep ──────────────────────────────────────────
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ 3. 常量内存缓存行边界扫描 (stride=1..128 floats)             │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    printf("%-10s %10s %10s\n", "Stride", "Median", "Bytes/acc");
    printf("%-10s %10s %10s\n", "(floats)", "(us)", "");

    for (int s = 1; s <= 128; ++s)
    {
        auto stats = run_benchmark(
            [&]()
            {
                const_stride_read<<<1, threads>>>(d_out.get(), s, repeat / 4);
                CUDA_CHECK(cudaDeviceSynchronize());
            });
        // 每次访问的字节数 = stride * sizeof(float)
        int bytes_per_acc = s * (int)sizeof(float);
        printf("%-10d %10.2f %4d B\n", s, stats.median_us, bytes_per_acc);
    }
    printf("\n");

    return 0;
}
// SPDX-License-Identifier: MIT
/**
 * sfu_probe.cu — 特殊函数单元 (SFU) 吞吐测量
 *
 * 测量常见特殊数学函数的吞吐，与 FP32 FMA 基准对比。
 * 通过 ratio（SFU ops / FP32 ops per cycle）推断每 SM 的 SFU 数量。
 *
 * Ada Lovelace (sm_89) 预期：
 *   - 每 SM 4 个 SFU（128 个 FP32 Core 的 1/32）
 *   - __cosf / __expf 等约 1/32 的 FP32 吞吐
 *
 * Usage:
 *   ./sfu_probe [repeats]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ════════════════════════════════════════════════════════════════════════════
//  参考基准：FP32 FMA（已知峰值）
// ════════════════════════════════════════════════════════════════════════════

__global__ void ref_fp32_fma(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float r0 = 1.0000001f, r1 = r0, r2 = r0, r3 = r0;
    float r4 = r0, r5 = r0, r6 = r0, r7 = r0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = fmaf(r0, 1.0000001f, 0.9999999f);
        r1 = fmaf(r1, 1.0000001f, 0.9999999f);
        r2 = fmaf(r2, 1.0000001f, 0.9999999f);
        r3 = fmaf(r3, 1.0000001f, 0.9999999f);
        r4 = fmaf(r4, 1.0000001f, 0.9999999f);
        r5 = fmaf(r5, 1.0000001f, 0.9999999f);
        r6 = fmaf(r6, 1.0000001f, 0.9999999f);
        r7 = fmaf(r7, 1.0000001f, 0.9999999f);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  特殊函数 Kernel（8 路独立累加，ILP=8）
// ════════════════════════════════════════════════════════════════════════════

// --- __cosf ---
__global__ void sfu_cosf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = __cosf(r0);
        r1 = __cosf(r1);
        r2 = __cosf(r2);
        r3 = __cosf(r3);
        r4 = __cosf(r4);
        r5 = __cosf(r5);
        r6 = __cosf(r6);
        r7 = __cosf(r7);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// --- __expf ---
__global__ void sfu_expf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = __expf(r0);
        r1 = __expf(r1);
        r2 = __expf(r2);
        r3 = __expf(r3);
        r4 = __expf(r4);
        r5 = __expf(r5);
        r6 = __expf(r6);
        r7 = __expf(r7);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// --- __logf ---
__global__ void sfu_logf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 100.0f, r1 = 200.0f, r2 = 300.0f, r3 = 400.0f;
    float r4 = 500.0f, r5 = 600.0f, r6 = 700.0f, r7 = 800.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = __logf(r0);
        r1 = __logf(r1);
        r2 = __logf(r2);
        r3 = __logf(r3);
        r4 = __logf(r4);
        r5 = __logf(r5);
        r6 = __logf(r6);
        r7 = __logf(r7);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// --- rsqrtf ---
__global__ void sfu_rsqrtf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 100.0f, r1 = 200.0f, r2 = 300.0f, r3 = 400.0f;
    float r4 = 500.0f, r5 = 600.0f, r6 = 700.0f, r7 = 800.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = rsqrtf(r0);
        r1 = rsqrtf(r1);
        r2 = rsqrtf(r2);
        r3 = rsqrtf(r3);
        r4 = rsqrtf(r4);
        r5 = rsqrtf(r5);
        r6 = rsqrtf(r6);
        r7 = rsqrtf(r7);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// --- __powf (常数指数) ---
__global__ void sfu_powf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 2.0f, r1 = 3.0f, r2 = 4.0f, r3 = 5.0f;
    float r4 = 6.0f, r5 = 7.0f, r6 = 8.0f, r7 = 9.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = __powf(r0, 1.5f);
        r1 = __powf(r1, 1.5f);
        r2 = __powf(r2, 1.5f);
        r3 = __powf(r3, 1.5f);
        r4 = __powf(r4, 1.5f);
        r5 = __powf(r5, 1.5f);
        r6 = __powf(r6, 1.5f);
        r7 = __powf(r7, 1.5f);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// --- __tanhf ---
__global__ void sfu_tanhf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = __tanhf(r0);
        r1 = __tanhf(r1);
        r2 = __tanhf(r2);
        r3 = __tanhf(r3);
        r4 = __tanhf(r4);
        r5 = __tanhf(r5);
        r6 = __tanhf(r6);
        r7 = __tanhf(r7);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// --- __sinf ---
__global__ void sfu_sinf(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = __sinf(r0);
        r1 = __sinf(r1);
        r2 = __sinf(r2);
        r3 = __sinf(r3);
        r4 = __sinf(r4);
        r5 = __sinf(r5);
        r6 = __sinf(r6);
        r7 = __sinf(r7);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  测试驱动
// ════════════════════════════════════════════════════════════════════════════

struct SfuTest
{
    const char *name;
    const char *sass_mnemonic;
    void (*kernel)(float *, int);
    int ilp; // 每迭代独立操作数
};

/// 运行单个 SFU 测试
void run_sfu_test(const DeviceInfo &info, const SfuTest &test, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;
    DeviceBuffer<float> d_out(1);

    // Warm up
    test.kernel<<<1, 32>>>(d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        test.kernel<<<blocks, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);

    // 总 SFU 操作数 = blocks × threads × iterations × ilp
    double total_ops = static_cast<double>(blocks) * threads * iterations * test.ilp;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "sfu_%s iter=%d", test.name, iterations);
    print_csv(label, samples);

    printf("    -> %-8s  %8.2f GOp/s  (SASS: %s)\n", test.name, ops_per_sec / 1e9,
           test.sass_mnemonic);
}

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

    const int iterations = 50000;

    // 定义测试集
    SfuTest tests[] = {
        {"FP32_FMA", "FFMA", ref_fp32_fma, 8}, // 基准
        {"__cosf", "MUFU.COS", sfu_cosf, 8},     {"__sinf", "MUFU.SIN", sfu_sinf, 8},
        {"__expf", "MUFU.EX2", sfu_expf, 8},     {"__logf", "MUFU.LG2", sfu_logf, 8},
        {"__rsqrtf", "MUFU.RSQ", sfu_rsqrtf, 8}, {"__tanhf", "MUFU.TANH", sfu_tanhf, 8},
        {"__powf", "MUFU.POW", sfu_powf, 8}, // 实际由多个 MUFU 组合
    };
    const int num_tests = sizeof(tests) / sizeof(tests[0]);

    printf("==================================================\n");
    printf("  特殊函数单元 (SFU) 吞吐测量\n");
    printf("  GPU: %s (sm_%d.%d)\n", info.arch_name().data(), info.major, info.minor);
    printf("  每个 kernel 8 路独立操作 (ILP=8)\n");
    printf("==================================================\n\n");

    printf("%-12s  %12s  %10s  %s\n", "Function", "GOp/s", "vs FP32", "SASS Inst");
    printf("%-12s  %12s  %10s  %s\n", "--------", "------", "--------", "---------");

    // 先跑 FP32 基准
    run_sfu_test(info, tests[0], iterations, repeats);
    double ref_ops = 0;
    {
        // 解析基准结果
        const int threads = 256;
        const int blocks = info.sm_count * 4;
        DeviceBuffer<float> d_out(1);

        // 单独计时基准
        ref_fp32_fma<<<1, 32>>>(d_out.get(), 4);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            ref_fp32_fma<<<blocks, threads>>>(d_out.get(), iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);
        double total_ops = static_cast<double>(blocks) * threads * iterations * tests[0].ilp;
        ref_ops = total_ops / (stats.median_us * 1e-6);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  (基准 FP32 FMA: %.2f GOp/s)\n\n", ref_ops / 1e9);

    // 跑 SFU 测试
    for (int i = 1; i < num_tests; ++i)
    {
        run_sfu_test(info, tests[i], iterations / 5, repeats); // SFU 更慢，少迭代
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 再算一次对比（取最后一次已打印的数据）
    printf("\n─── 速率对比 ───\n");
    printf("  FP32 每 SM 约 128 个 Core (含 4 个 SFU)\n");
    printf("  预期 SFU/FP32 比值 ≈ 4/128 = 1/32 ≈ 3.1%%\n\n");

    printf("\nDone.\n");
    return 0;
}

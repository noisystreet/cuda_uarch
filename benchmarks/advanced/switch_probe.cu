// SPDX-License-Identifier: MIT
/**
 * switch_probe.cu — Tensor Core vs CUDA Core 切换开销探测
 *
 * 目的：
 *   检测在同一个 warp 内交替使用 Tensor Core (WMMA mma_sync) 和
 *   CUDA Core (FP32 FMA) 是否存在额外的切换开销。
 *
 * 背景：
 *   - Tensor Core 和 CUDA Core 在 Ada Lovelace 架构中共享 SM 但可能
 *     映射到不同的功能单元流水线。
 *   - 如果切换需要清空流水线或引入额外调度约束，则会产生可测量的开销。
 *
 * 方法：
 *   1. 纯 WMMA 基线：连续执行 N 次 FP16 mma_sync
 *   2. 纯 FMA 基线：连续执行 N 次 FP32 fmaf (ILP=16)
 *   3. 切换实验：WMMA + K×FMA 交替，K = 1,2,4,8,16,32,64
 *   4. 固定 WMMA 调用次数不变，对比纯方案与混合方案的时间差异
 *
 * 用法：
 *   ./switch_probe
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;
using namespace nvcuda::wmma;

// ════════════════════════════════════════════════════════════════════════════
//  常量
// ════════════════════════════════════════════════════════════════════════════

// 每次实验 WMMA 调用次数（固定）
constexpr int TC_CALLS = 64;

// 每个线程每轮 FMA 并行度 (ILP)
constexpr int FMA_ILP = 16;

// 切换间隔 (FMA ops per switch): 1, 2, 4, 8, 16, 32, 64
constexpr int SWITCH_INTERVALS[] = {1, 2, 4, 8, 16, 32, 64};
constexpr int NUM_INTERVALS = 7;

// 每次实验重复次数
constexpr int REPEATS = 8;

// 预热次数
constexpr int WARMUP = 2;

// ════════════════════════════════════════════════════════════════════════════
//  Kernel 1: 纯 Tensor Core WMMA 基线
// ════════════════════════════════════════════════════════════════════════════

__global__ void pure_tc_kernel(float *__restrict__ output, int tc_calls)
{
    // WMMA fragment 声明
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fragment<accumulator, 16, 16, 16, float> d_frag;

    fill_fragment(a_frag, __float2half(1.0001f));
    fill_fragment(b_frag, __float2half(0.9999f));
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

    // 依赖初始值，防止编译期常量折叠
    if (output[0] != 0.0f)
    {
        fill_fragment(c_frag, output[0]);
    }

#pragma unroll 1
    for (int i = 0; i < tc_calls; ++i)
    {
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    // 写回结果防止 DCE
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = d_frag.x[0];
        output[1] = d_frag.x[1];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Kernel 2: 纯 CUDA Core FMA 基线 (ILP = 16)
// ════════════════════════════════════════════════════════════════════════════

__global__ void pure_fma_kernel(float *__restrict__ output, int fma_iters)
{
    float a = 1.0000001f;
    float b = 0.9999999f;

    float r0 = 1.0f, r1 = 1.0f, r2 = 1.0f, r3 = 1.0f;
    float r4 = 1.0f, r5 = 1.0f, r6 = 1.0f, r7 = 1.0f;
    float r8 = 1.0f, r9 = 1.0f, r10 = 1.0f, r11 = 1.0f;
    float r12 = 1.0f, r13 = 1.0f, r14 = 1.0f, r15 = 1.0f;

#pragma unroll 1
    for (int i = 0; i < fma_iters; ++i)
    {
        r0 = fmaf(r0, a, b);
        r1 = fmaf(r1, a, b);
        r2 = fmaf(r2, a, b);
        r3 = fmaf(r3, a, b);
        r4 = fmaf(r4, a, b);
        r5 = fmaf(r5, a, b);
        r6 = fmaf(r6, a, b);
        r7 = fmaf(r7, a, b);
        r8 = fmaf(r8, a, b);
        r9 = fmaf(r9, a, b);
        r10 = fmaf(r10, a, b);
        r11 = fmaf(r11, a, b);
        r12 = fmaf(r12, a, b);
        r13 = fmaf(r13, a, b);
        r14 = fmaf(r14, a, b);
        r15 = fmaf(r15, a, b);
    }

    float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  Kernel 3: 混合 — WMMA 后跟 FMA
// ════════════════════════════════════════════════════════════════════════════

/// switch_interval: 每次 WMMA 调用后执行多少轮 FMA (每轮 FMA_ILP 条 FMA)
__global__ void switch_kernel(float *__restrict__ output, int tc_calls, int switch_interval)
{
    // ─── Tensor Core 状态 ───
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fragment<accumulator, 16, 16, 16, float> d_frag;

    fill_fragment(a_frag, __float2half(1.0001f));
    fill_fragment(b_frag, __float2half(0.9999f));
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

    // ─── CUDA Core 状态 (每个线程独立) ───
    float a = 1.0000001f;
    float b = 0.9999999f;
    float r0 = 1.0f, r1 = 1.0f, r2 = 1.0f, r3 = 1.0f;
    float r4 = 1.0f, r5 = 1.0f, r6 = 1.0f, r7 = 1.0f;
    float r8 = 1.0f, r9 = 1.0f, r10 = 1.0f, r11 = 1.0f;
    float r12 = 1.0f, r13 = 1.0f, r14 = 1.0f, r15 = 1.0f;

    // 依赖初始值
    if (output[0] != 0.0f)
    {
        fill_fragment(c_frag, output[0]);
        r0 = output[0];
    }

#pragma unroll 1
    for (int i = 0; i < tc_calls; ++i)
    {
        // ── Tensor Core WMMA ──
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;

        // ── CUDA Core FMA (switch_interval 轮 × FMA_ILP 条/轮) ──
#pragma unroll 1
        for (int j = 0; j < switch_interval; ++j)
        {
            r0 = fmaf(r0, a, b);
            r1 = fmaf(r1, a, b);
            r2 = fmaf(r2, a, b);
            r3 = fmaf(r3, a, b);
            r4 = fmaf(r4, a, b);
            r5 = fmaf(r5, a, b);
            r6 = fmaf(r6, a, b);
            r7 = fmaf(r7, a, b);
            r8 = fmaf(r8, a, b);
            r9 = fmaf(r9, a, b);
            r10 = fmaf(r10, a, b);
            r11 = fmaf(r11, a, b);
            r12 = fmaf(r12, a, b);
            r13 = fmaf(r13, a, b);
            r14 = fmaf(r14, a, b);
            r15 = fmaf(r15, a, b);
        }
    }

    // 写回防止 DCE
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = d_frag.x[0];
        float sum =
            r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
        output[1] = sum;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Kernel 4: 混合 — FMA 先于 WMMA (反向顺序)
// ════════════════════════════════════════════════════════════════════════════

__global__ void switch_rev_kernel(float *__restrict__ output, int tc_calls, int switch_interval)
{
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fragment<accumulator, 16, 16, 16, float> d_frag;

    fill_fragment(a_frag, __float2half(1.0001f));
    fill_fragment(b_frag, __float2half(0.9999f));
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

    float a = 1.0000001f;
    float b = 0.9999999f;
    float r0 = 1.0f, r1 = 1.0f, r2 = 1.0f, r3 = 1.0f;
    float r4 = 1.0f, r5 = 1.0f, r6 = 1.0f, r7 = 1.0f;
    float r8 = 1.0f, r9 = 1.0f, r10 = 1.0f, r11 = 1.0f;
    float r12 = 1.0f, r13 = 1.0f, r14 = 1.0f, r15 = 1.0f;

    if (output[0] != 0.0f)
    {
        fill_fragment(c_frag, output[0]);
        r0 = output[0];
    }

#pragma unroll 1
    for (int i = 0; i < tc_calls; ++i)
    {
        // ── CUDA Core FMA 先 ──
#pragma unroll 1
        for (int j = 0; j < switch_interval; ++j)
        {
            r0 = fmaf(r0, a, b);
            r1 = fmaf(r1, a, b);
            r2 = fmaf(r2, a, b);
            r3 = fmaf(r3, a, b);
            r4 = fmaf(r4, a, b);
            r5 = fmaf(r5, a, b);
            r6 = fmaf(r6, a, b);
            r7 = fmaf(r7, a, b);
            r8 = fmaf(r8, a, b);
            r9 = fmaf(r9, a, b);
            r10 = fmaf(r10, a, b);
            r11 = fmaf(r11, a, b);
            r12 = fmaf(r12, a, b);
            r13 = fmaf(r13, a, b);
            r14 = fmaf(r14, a, b);
            r15 = fmaf(r15, a, b);
        }

        // ── Tensor Core WMMA ──
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = d_frag.x[0];
        float sum =
            r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
        output[1] = sum;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  运行单个实验
// ════════════════════════════════════════════════════════════════════════════

/// 运行 kernel 并返回平均执行时间 (微秒)
static double run_kernel(const char *label, dim3 grid, dim3 block, int repeats, auto kernel,
                         auto &&...args)
{
    std::vector<double> samples;
    samples.reserve(repeats);

    DeviceBuffer<float> d_out(4);

    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        kernel<<<grid, block>>>(d_out.get(), args...);
        timer.stop();
        double us = timer.elapsed_us();
        samples.push_back(us);
    }

    // 取中位数
    auto stats = compute_stats(samples);
    print_csv(label, samples);
    return stats.median_us;
}

// ════════════════════════════════════════════════════════════════════════════
//  主函数
// ════════════════════════════════════════════════════════════════════════════

int main()
{
    // ── 打印设备信息 ──
    auto info = get_device_info();
    print_device_info(info);

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  Tensor Core vs CUDA Core 切换开销探测\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  配置: TC_CALLS=%d, 每轮 FMA_ILP=%d, REPEATS=%d\n", TC_CALLS, FMA_ILP, REPEATS);
    printf("  WMMA 精度: FP16 m16n16k16 (256 FMA/WMMA)\n");
    printf("  CUDA Core: FP32 FMA (ILP=16)\n\n");

    // ── 预热 ──
    printf("  预热 %d 轮...\n", WARMUP);
    DeviceBuffer<float> warmup_buf(4);
    for (int w = 0; w < WARMUP; ++w)
    {
        pure_tc_kernel<<<1, 32>>>(warmup_buf.get(), 4);
        CUDA_CHECK(cudaDeviceSynchronize());
        pure_fma_kernel<<<1, 32>>>(warmup_buf.get(), 64);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ──────────────────────────────────────────────────────────────
    //  实验 1: 纯 Tensor Core 基线
    // ──────────────────────────────────────────────────────────────
    printf("\n─── 实验 1: 纯 Tensor Core WMMA 基线 ───\n");
    printf("  WMMA 调用次数: %d\n", TC_CALLS);
    double t_tc = run_kernel("pure_tc", 1, 32, REPEATS, pure_tc_kernel, TC_CALLS);
    printf("  >>> 纯 TC 时间: %.2f us (%.2f us / WMMA call)\n\n", t_tc, t_tc / TC_CALLS);

    // ──────────────────────────────────────────────────────────────
    //  实验 2: 纯 CUDA Core FMA 基线 (对应各切换间隔的 FMA 总量)
    // ──────────────────────────────────────────────────────────────
    printf("─── 实验 2: 纯 CUDA Core FMA 基线 ───\n");
    printf("  FMA 总量 = TC_CALLS × switch_interval × FMA_ILP\n\n");

    double t_fma_pure[NUM_INTERVALS];
    for (int k = 0; k < NUM_INTERVALS; ++k)
    {
        int interval = SWITCH_INTERVALS[k];
        int fma_iters = TC_CALLS * interval;
        char label[64];
        snprintf(label, sizeof(label), "pure_fma_interval_%d", interval);

        t_fma_pure[k] = run_kernel(label, 1, 32, REPEATS, pure_fma_kernel, fma_iters);
        printf("  interval=%3d:  %d FMA iters → %.2f us (%.2f us / FMA_ILP_round)\n", interval,
               fma_iters, t_fma_pure[k], t_fma_pure[k] / fma_iters);
    }
    printf("\n");

    // ──────────────────────────────────────────────────────────────
    //  实验 3: 混合 — WMMA + FMA 交替
    // ──────────────────────────────────────────────────────────────
    printf("─── 实验 3: WMMA + FMA 交替 ───\n");
    printf("  每次 WMMA 后执行 interval 轮 FMA (每轮 ILP=%d)\n", FMA_ILP);
    printf("  若存在切换开销, 混合时间 > 纯 TC 时间 + 纯 FMA 时间\n\n");

    double t_switch[NUM_INTERVALS];
    for (int k = 0; k < NUM_INTERVALS; ++k)
    {
        int interval = SWITCH_INTERVALS[k];
        char label[64];
        snprintf(label, sizeof(label), "switch_interval_%d", interval);

        t_switch[k] = run_kernel(label, 1, 32, REPEATS, switch_kernel, TC_CALLS, interval);

        // 期望时间 = 纯 TC 时间 + 对应纯 FMA 时间
        double expected = t_tc + t_fma_pure[k];
        double overhead = t_switch[k] - expected;
        double overhead_pct = (overhead / expected) * 100.0;
        double overhead_per_switch = overhead / (TC_CALLS * 2); // 每轮 2 次切换: TC→FMA + FMA→TC

        printf("  interval=%3d:  %.2f us  (期望 %.2f us)  开销 %+.2f us (%+.2f%%)  "
               "每切换 %+.3f us\n",
               interval, t_switch[k], expected, overhead, overhead_pct, overhead_per_switch);
    }

    // ──────────────────────────────────────────────────────────────
    //  汇总
    // ──────────────────────────────────────────────────────────────
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  汇总\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  %12s  %10s  %10s  %10s  %12s\n", "interval", "t_mix(us)", "t_tc(us)", "t_fma(us)",
           "overhead(%)");
    for (int k = 0; k < NUM_INTERVALS; ++k)
    {
        int interval = SWITCH_INTERVALS[k];
        double expected = t_tc + t_fma_pure[k];
        double overhead_pct = ((t_switch[k] - expected) / expected) * 100.0;
        printf("  %12d  %10.2f  %10.2f  %10.2f  %+.2f\n", interval, t_switch[k], t_tc,
               t_fma_pure[k], overhead_pct);
    }

    // ──────────────────────────────────────────────────────────────
    //  实验 4 (额外): 反向切换 — 先 FMA 后 WMMA
    // ──────────────────────────────────────────────────────────────
    printf("\n─── 实验 4: 反向顺序 (先 FMA 后 WMMA) ───\n");

    for (int k = 0; k < NUM_INTERVALS; ++k)
    {
        int interval = SWITCH_INTERVALS[k];
        char label[64];
        snprintf(label, sizeof(label), "switch_rev_interval_%d", interval);

        double t = run_kernel(label, 1, 32, REPEATS, switch_rev_kernel, TC_CALLS, interval);
        double expected = t_tc + t_fma_pure[k];
        double overhead = t - expected;
        double overhead_pct = (overhead / expected) * 100.0;

        printf("  interval=%3d:  %.2f us  (期望 %.2f us)  开销 %+.2f us (%+.2f%%)\n", interval, t,
               expected, overhead, overhead_pct);
    }

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  分析完成\n");
    printf("══════════════════════════════════════════════════════════════\n");

    return 0;
}
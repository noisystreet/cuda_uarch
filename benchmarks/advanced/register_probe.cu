// SPDX-License-Identifier: MIT
/**
 * register_probe.cu — 寄存器文件分析
 *
 * 实验：
 *   1. Occupancy Sweep — 通过 __launch_bounds__(minBlocksPerSM) 间接约束
 *      寄存器使用量，测量不同寄存器预算下的吞吐
 *   2. 寄存器压力 — 固定线程数，增加每个线程的活跃寄存器数（通过更多
 *      独立累加器），观察吞吐拐点
 *   3. 依赖链长度 — 固定 ILP=1 但改变依赖链长度，检测寄存器重命名能力
 *
 * 背景：
 *   - Ada Lovelace 每 SM 65536 个 32 位寄存器
 *   - 每线程最大 255 个寄存器
 *   - 寄存器文件被所有 warp 共享，寄存器不足时 → 降低占用率
 *
 * Usage:
 *   ./register_probe [repeats]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ════════════════════════════════════════════════════════════════════════════
//  实验 1: Occupancy Sweep — 通过 __launch_bounds__ 约束寄存器
// ════════════════════════════════════════════════════════════════════════════

/// 高寄存器使用量的 FFMA kernel — 使用 N 个独立累加器
/// 通过 __launch_bounds__ 的 minBlocksPerSM 参数间接约束寄存器分配
template <int NRegs>
__launch_bounds__(256, 1) __global__
    void reg_pressure_kernel(const float *__restrict__ input, float *__restrict__ output,
                             int iterations)
{
    // NRegs 个独立寄存器 → 编译器必须分配至少 NRegs 个寄存器
    float r[NRegs];

    // 初始化
    for (int j = 0; j < NRegs; ++j)
        r[j] = input[(threadIdx.x + j) % 1024];

    float a = 1.0000001f;
    float b = 0.9999999f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
// 每路独立 FFMA
#pragma unroll
        for (int j = 0; j < NRegs; ++j)
        {
            r[j] = fmaf(r[j], a, b);
        }
    }

    // 求和输出
    float sum = 0.0f;
    for (int j = 0; j < NRegs; ++j)
        sum += r[j];
    if (threadIdx.x == 0)
        *output = sum;
}

/// 与上面相同但带有 __launch_bounds__ 的 minBlocksPerSM 约束
template <int NRegs, int MinBlocks>
__launch_bounds__(256, MinBlocks) __global__
    void reg_occupancy_kernel(const float *__restrict__ input, float *__restrict__ output,
                              int iterations)
{
    float r[NRegs];
    for (int j = 0; j < NRegs; ++j)
        r[j] = input[(threadIdx.x + j) % 1024];

    float a = 1.0000001f;
    float b = 0.9999999f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for (int j = 0; j < NRegs; ++j)
        {
            r[j] = fmaf(r[j], a, b);
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < NRegs; ++j)
        sum += r[j];
    if (threadIdx.x == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 2: 寄存器压力扫描（固定线程配置，增加每线程寄存器数）
// ════════════════════════════════════════════════════════════════════════════

/// 运行时配置寄存器压力 — 模板化 NRegs
template <int NRegs> void run_reg_pressure_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4; // 尝试填满 SM
    const int num_items = threads * blocks;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(1);
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    // 预热
    reg_pressure_kernel<NRegs><<<1, 256>>>(d_in.get(), d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        reg_pressure_kernel<NRegs><<<blocks, threads>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);

    double total_ops =
        static_cast<double>(blocks) * threads * iterations * NRegs * 2; // ×2 for FMA = 2 ops
    double gops = total_ops / (stats.median_us * 1e-6) / 1e9;

    char label[128];
    snprintf(label, sizeof(label), "register_pressure NRegs=%d iter=%d", NRegs, iterations);
    print_csv(label, samples);

    printf("  NRegs=%-3d  %8.1f μs  %8.0f GOp/s  ", NRegs, stats.median_us, gops);

    // 估算每线程寄存器数
    // 编译器实际分配的寄存器数可能比 NRegs 多（地址、循环变量等）
    printf("(~%d regs/thread)\n", NRegs + 4);
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 3: Occupancy Sweep — 固定 NRegs，改变 minBlocksPerSM
// ════════════════════════════════════════════════════════════════════════════

template <int NRegs, int MinBlocks>
void run_occupancy_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;
    const int num_items = threads * blocks;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(1);
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    reg_occupancy_kernel<NRegs, MinBlocks><<<1, 256>>>(d_in.get(), d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        reg_occupancy_kernel<NRegs, MinBlocks>
            <<<blocks, threads>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(blocks) * threads * iterations * NRegs * 2;
    double gops = total_ops / (stats.median_us * 1e-6) / 1e9;

    char label[128];
    snprintf(label, sizeof(label), "occupancy NRegs=%d MinBlk=%d iter=%d", NRegs, MinBlocks,
             iterations);
    print_csv(label, samples);

    printf("  NRegs=%-3d  MinBlocks=%-3d  %8.1f μs  %8.0f GOp/s\n", NRegs, MinBlocks,
           stats.median_us, gops);
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 4: 依赖链长度 — 检测寄存器重命名
// ════════════════════════════════════════════════════════════════════════════

/// 单一依赖链 kernel，ILP=1
/// chain_len 是每循环的依赖长度（通过展开实现）
template <int ChainLen>
__launch_bounds__(256, 1) __global__
    void dep_chain_kernel(float *__restrict__ output, int iterations)
{
    float r = 1.0000001f;
    float a = 0.9999999f;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
// 创建 ChainLen 长的依赖链：
// r = f(r) → r = f(r) → ... → r = f(r)
#pragma unroll
        for (int j = 0; j < ChainLen; ++j)
        {
            r = fmaf(r, a, 0.9999999f);
        }
    }

    if (threadIdx.x == 0)
        *output = r;
}

template <int ChainLen> void run_dep_chain_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;

    DeviceBuffer<float> d_out(1);

    dep_chain_kernel<ChainLen><<<1, 32>>>(d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        dep_chain_kernel<ChainLen><<<blocks, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);

    double total_ops = static_cast<double>(blocks) * threads * iterations * ChainLen;
    double gops = total_ops / (stats.median_us * 1e-6) / 1e9;

    // 每操作的延迟
    double ns_per_op = stats.median_us * 1000.0 / (iterations * ChainLen);

    char label[128];
    snprintf(label, sizeof(label), "dep_chain ChainLen=%d iter=%d", ChainLen, iterations);
    print_csv(label, samples);

    printf("  ChainLen=%-4d  %8.1f μs  %8.0f GOp/s  %6.2f ns/op\n", ChainLen, stats.median_us, gops,
           ns_per_op);
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
    printf("  寄存器文件分析\n");
    printf("  GPU: %s (sm_%d.%d)\n", info.arch_name().data(), info.major, info.minor);
    // Ada Lovelace: 65536 32-bit registers per SM, max 255 per thread
    printf("  寄存器文件 / SM: 65536 (32-bit regs, max 255/thread)\n");
    printf("==================================================\n\n");

    // ─── 实验 1: 寄存器压力扫描 ─────────────────────────────────────
    printf("─── 1. 寄存器压力扫描 ───\n");
    printf("  固定 block=96 (24×4), thread=256, 增加每线程寄存器数\n");
    printf("  NRegs     Time       Throughput    Estimated\n");
    printf("  -----    -------    -----------    ----------\n");

    const int iterations1 = 20000;

    // 从 2 到 128 个寄存器，2 倍步进
    run_reg_pressure_test<2>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<4>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<8>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<16>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<32>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<48>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<64>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<80>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<96>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_reg_pressure_test<128>(info, iterations1, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ─── 实验 2: Occupancy Sweep ─────────────────────────────────────
    printf("\n─── 2. Occupancy Sweep (__launch_bounds__) ───\n");
    printf("  固定 NRegs=32，改变 minBlocksPerSM 约束\n");
    printf("  编译器被迫减少寄存器 → 更多并发 block → 吞吐变化\n\n");
    printf("  NRegs    MinBlk     Time       Throughput\n");
    printf("  -----    ------    -------    -----------\n");

    const int iterations2 = 20000;

    run_occupancy_test<32, 1>(info, iterations2, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_occupancy_test<32, 2>(info, iterations2, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_occupancy_test<32, 4>(info, iterations2, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_occupancy_test<32, 8>(info, iterations2, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_occupancy_test<32, 16>(info, iterations2, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ─── 实验 3: 依赖链长度 ──────────────────────────────────────────
    printf("\n─── 3. 依赖链长度（寄存器重命名检测） ───\n");
    printf("  固定 ILP=1，增加串行依赖长度\n");
    printf("  有重命名: 链长度不应线性影响延迟\n");
    printf("  无重命名: 长链延迟线性增加\n\n");
    printf("  ChainLen    Time       Throughput    ns/op\n");
    printf("  --------   -------    -----------   ------\n");

    const int iterations3 = 10000;

    run_dep_chain_test<1>(info, iterations3, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_dep_chain_test<2>(info, iterations3 / 2, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_dep_chain_test<4>(info, iterations3 / 4, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_dep_chain_test<8>(info, iterations3 / 8, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_dep_chain_test<16>(info, iterations3 / 16, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_dep_chain_test<32>(info, iterations3 / 32, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n─── 分析提示 ───\n");
    printf("  实验 1: 如果吞吐在 NRegs=X 处骤降，说明 X 个寄存器\n");
    printf("          触发了占用率降低（寄存器文件满）\n");
    printf("  实验 2: __launch_bounds__ 不改变 SASS 代码，只影响 occupancy\n");
    printf("  实验 3: 如果 ns/op 随链长增加而上升，说明依赖链无法\n");
    printf("          被寄存器重命名完全消除\n\n");

    printf("Done.\n");
    return 0;
}

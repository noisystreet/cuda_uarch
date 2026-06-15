// SPDX-License-Identifier: MIT
// associativity_probe.cu — 缓存关联度 (Associativity) 精确测量
//
// 方法：自包含指针追逐（Chase-in-Data）
//
// 原理：
//   对于容量 C、缓存行 L 的 N 路组相联缓存，stride = C 保证所有
//   data[i] 映射到同一缓存集合。将指针追逐链直接嵌入 data 数组：
//   data[perm[i] × stride] = perm[(i+1) % N]，每条循环迭代只执行
//   一个从 data[idx × stride] 的加载——结果既作为下一轮 idx，又
//   累加到 val。
//
//   当活跃同集合地址数 M ≤ 关联度时，所有加载命中 L1；
//   当 M > 关联度时，LRU 淘汰触发 L2 缺失，每加载延迟跃升。
//   跃变点即为关联度。
//
// 参考:
//   L1 = 128 KB, L2 = 32 MB, line = 128 B (由 cache_size_probe 测得)
//   L1 stride = 128 KB, L2 stride = 32 MB

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

// ════════════════════════════════════════════════════════════════════════════
//  Kernel: 自包含指针追逐
// ════════════════════════════════════════════════════════════════════════════
//  data[idx × stride] 保存"下一个 idx"，同时也可用作伪随机值累加。
//  单线程，严格依赖链——每条指令必须等前一条的加载完成。
__global__ void chase_in_data(const int *__restrict__ data, int *__restrict__ output, int num_addrs,
                              int stride_elems, int chain_len)
{
    // 起始 idx 必须是有效范围内的值，确保 ptr→指向 data 中的一个元素
    // 该元素的值 → 下一个 ptr → ...
    int idx = 0;
    int val = 0;

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        idx = data[idx * stride_elems];
        val ^= idx; // 每个 idx 不同，确保结果变化
    }
    *output = val;
}

// ════════════════════════════════════════════════════════════════════════════
//  辅助函数
// ════════════════════════════════════════════════════════════════════════════

/// 构建自包含指针追逐数据
/// data[perm[i] × stride_elems] = perm[(i+1) % N]
/// perm 是 0..N-1 的随机排列
static void build_chase_data(std::vector<int> &data, int num_addrs, int stride_elems, unsigned seed)
{
    std::vector<int> perm(num_addrs);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::fill(data.begin(), data.end(), 0);
    for (int i = 0; i < num_addrs; ++i)
        data[perm[i] * stride_elems] = perm[(i + 1) % num_addrs];
}

/// 测量单次运行
struct Result
{
    double median_us;
    double per_access_ns;
};

static Result measure(std::function<void()> launch, int samples, int chain_len)
{
    std::vector<double> us;
    us.reserve(samples);
    for (int s = 0; s < samples; ++s)
    {
        uarch::GpuTimer timer;
        timer.start();
        launch();
        timer.stop();
        us.push_back(timer.elapsed_us());
    }

    std::sort(us.begin(), us.end());
    int n = (int)us.size();
    double med = (n % 2 == 0) ? (us[n / 2 - 1] + us[n / 2]) * 0.5 : us[n / 2];

    return {med, med * 1000.0 / chain_len};
}

/// 通用关联度探测
/// @param label       缓存层级名称（"L1" / "L2"）
/// @param cache_size  缓存容量（字节）
/// @param max_n       最高测试地址数
/// @param chain_len   指针追逐链长度
/// @param samples     采样次数
static void probe_assoc(const char *label, size_t cache_size, int max_n, int chain_len, int samples)
{
    const int stride_elems = (int)(cache_size / sizeof(int));
    uarch::DeviceBuffer<int> d_out(1);

    printf("\n═══ %s 关联度探测 (Chase-in-Data) ═══\n", label);
    printf("%s = %zu bytes, stride = %zu bytes (%d ints)\n", label, cache_size, cache_size,
           stride_elems);
    printf("单线程自包含指针追逐, %d 次/run, %d 次采样\n", chain_len, samples);
    printf("------------------------------------------------------------\n");
    printf("  Addrs │ Total(us) │ PerAcc(ns) │  Δ(ns)  │  注释\n");
    printf("------------------------------------------------------------\n");

    double prev_ns = 0.0;

    for (int n = 1; n <= max_n; ++n)
    {
        size_t data_elems = (size_t)(n - 1) * stride_elems + 1;

        // 检查显存是否够
        if (data_elems > (1ULL << 30)) // > 1G 元素时检查
        {
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            size_t needed = data_elems * sizeof(int);
            if (needed >= free_mem)
            {
                printf("  %5d  │  %10s │  %10s │  %7s │ SKIP: 显存不足\n", n, "-", "-", "-");
                continue;
            }
        }

        std::vector<int> h_data(data_elems, 0);
        build_chase_data(h_data, n, stride_elems, 42);

        uarch::DeviceBuffer<int> d_data(data_elems);
        d_data.upload(h_data.data());

        // Warm-up
        chase_in_data<<<1, 1>>>(d_data.get(), d_out.get(), n, stride_elems, 100);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto res = measure(
            [&]()
            {
                chase_in_data<<<1, 1>>>(d_data.get(), d_out.get(), n, stride_elems, chain_len);
                CUDA_CHECK(cudaDeviceSynchronize());
            },
            samples, chain_len);

        double delta = (n == 1) ? 0.0 : res.per_access_ns - prev_ns;
        const char *note = (delta > 10.0) ? " *** 跃变 ***" : "";
        printf("  %5d  │  %10.2f │  %10.2f │  %7.1f │%s\n", n, res.median_us, res.per_access_ns,
               delta, note);
        fflush(stdout);
        prev_ns = res.per_access_ns;
    }
    printf("------------------------------------------------------------\n");
    printf("关联度 = PerAcc △ > 10ns 首次出现前的 Addrs 值\n");
}

// ════════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════════
int main()
{
    auto info = uarch::get_device_info();
    uarch::print_device_info(info);
    uarch::warm_up_device();

    printf("=== 缓存关联度精确测量 ===\n");
    printf("基于 cache_size_probe 测得: L1=128KB, L2=32MB, line=128B\n\n");

    const int chain_len = 50000; // 更多迭代 → 更低噪声
    const int samples = 12;

    // L1: 128 KB, N = 1..64（步进 1）
    probe_assoc("L1", 128 * 1024, 64, chain_len, samples);

    // L2: 32 MB, N = 1..24（步进 1, 需要约 700 MB）
    // 如显存不足, max_n 自动跳过（见 probe_assoc 中内存检查）
    probe_assoc("L2", 32 * 1024 * 1024, 24, chain_len, samples);

    printf("\nDone.\n");
    return 0;
}
// SPDX-License-Identifier: MIT
/**
 * warp_scheduler_probe.cu — Warp 调度策略分析
 *
 * 三个实验：
 *   1. Warp Divergence     — 分支分歧对吞吐的影响
 *   2. Warp Concurrency    — 活跃 warp 数与延迟隐藏的关系
 *   3. Inter-Warp Latency  — 多 warp 对内存延迟的隐藏效果
 *
 * Usage:
 *   ./warp_scheduler_probe [repeats]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ════════════════════════════════════════════════════════════════════════════
//  1. Warp Divergence Probe
// ════════════════════════════════════════════════════════════════════════════

/// FFMA kernel with controlled divergence.
///   - divergent_threads: number of threads (out of warpSize) that take
///     the 'then' path; the rest take the 'else' path.
///   - When divergent_threads = 0 or 32 → uniform (no divergence)
///   - When divergent_threads = 16 → half divergence
template <int DivergentThreads>
__global__ void divergence_ffma(const float *__restrict__ input, float *__restrict__ output,
                                int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    float a = 1.0000001f;
    float b = 0.9999999f;
    float r = input[tid % 1024];

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        if constexpr (DivergentThreads == 0)
        {
            // All threads skip — uniform branch
            r = fmaf(r, a, b);
        }
        else if constexpr (DivergentThreads == 32)
        {
            // All threads enter — uniform branch
            r = fmaf(r, a, b);
        }
        else
        {
            // Divergent: first N threads take one path, rest take another
            if (lane < DivergentThreads)
            {
                r = fmaf(r, a, b); // path A
            }
            else
            {
                r = fmaf(r * 0.5f, a, b); // path B (different op count)
            }
        }
    }

    output[tid] = r;
}

/// Run divergence test for a specific divergence pattern.
template <int N> void run_divergence_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads_per_block = 256;
    const int blocks = info.sm_count * 2; // 2 blocks/SM
    const int num_items = threads_per_block * blocks;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(num_items);
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    // Warm up
    divergence_ffma<N><<<blocks, threads_per_block>>>(d_in.get(), d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        divergence_ffma<N><<<blocks, threads_per_block>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * num_items;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    double active_ratio = (N > 0 && N < 32) ? static_cast<double>(32) / 32.0 : 1.0;
    // For divergent warps, warp serializes the two paths.
    // Expected throughput = uniform_throughput * active_ratio
    // where active_ratio accounts for serialized execution.

    char label[128];
    snprintf(label, sizeof(label), "divergence N=%d iter=%d threads=%d", N, iterations, num_items);
    print_csv(label, samples);
    printf("    -> Throughput: %.1f GOp/s  (divergent threads: %d/32)\n", ops_per_sec / 1e9, N);
}

void probe_divergence(int repeats)
{
    printf("\n═══ 1. Warp Divergence Probe ═══\n");
    printf("Measures FFMA throughput with controlled branch divergence.\n");
    printf("N=0 or N=32 = uniform (no divergence), N=16 = 50%% divergent.\n\n");
    printf("%-8s  %16s  %s\n", "N/32", "Throughput (GOp/s)", "Interpretation");
    printf("%-8s  %16s  %s\n", "----", "------------------", "-------------");

    const int iterations = 10000;

    // Test a range of divergence patterns
    constexpr int kDivPatterns[] = {0, 8, 16, 24, 32};

    for (int n : kDivPatterns)
    {
        switch (n)
        {
        case 0:
            run_divergence_test<0>(get_device_info(), iterations, repeats);
            break;
        case 8:
            run_divergence_test<8>(get_device_info(), iterations, repeats);
            break;
        case 16:
            run_divergence_test<16>(get_device_info(), iterations, repeats);
            break;
        case 24:
            run_divergence_test<24>(get_device_info(), iterations, repeats);
            break;
        case 32:
            run_divergence_test<32>(get_device_info(), iterations, repeats);
            break;
        default:
            break;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Print interpretation
        const char *interp;
        if (n == 0 || n == 32)
            interp = "uniform (full speed)";
        else if (n == 16)
            interp = "max divergence (2× serialized)";
        else
            interp = "partial divergence";
        printf("    -> %s\n\n", interp);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  2. Warp Concurrency Probe  (occupancy / latency hiding)
// ════════════════════════════════════════════════════════════════════════════

// Forward declaration
static std::vector<int> build_chain(int size, unsigned seed = 42);

/// Memory-latency-bound kernel (pointer chase with limited ILP).
/// Lower ILP means more sensitive to warp count for latency hiding.
__global__ void latency_hiding_kernel(const int *__restrict__ next, int *__restrict__ output,
                                      int chain_len, int start_idx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = start_idx + tid;
    int val = 0;

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        idx = next[idx];
        val ^= idx;
    }
    if (tid == 0)
        *output = val;
}

/// Measure throughput while sweeping warps per SM.
void probe_concurrency(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 2. Warp Concurrency Probe (Latency Hiding) ═══\n");
    printf("Pointer chase with varying warps per SM.\n");
    printf("More warps → more latency hiding → higher effective throughput.\n\n");
    printf("%-16s  %16s  %s\n", "Warps/SM", "Latency (ns)", "Interpretation");
    printf("%-16s  %16s  %s\n", "--------", "------------", "-------------");

    const int chain_len = 10000;

    // Build random index chain
    const int chain_size = 1 << 20; // 1M elements (4 MB)
    auto chain = build_chain(chain_size);
    DeviceBuffer<int> d_next(chain_size);
    d_next.upload(chain.data());
    DeviceBuffer<int> d_output(1);

    int start_idx = 0;

    // Sweep warps per SM from 1 to near max (48 for Ada)
    const int warp_sizes[] = {1, 2, 4, 8, 16, 32, 48};

    for (int warps_per_sm : warp_sizes)
    {
        int threads = warps_per_sm * info.warp_size * info.sm_count;
        if (threads > info.max_threads_per_sm * info.sm_count)
            continue;

        int block_size = std::min(threads, 256);
        int blocks = div_up(threads, block_size);

        // Warm up
        latency_hiding_kernel<<<blocks, block_size>>>(d_next.get(), d_output.get(), 4, start_idx);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            latency_hiding_kernel<<<blocks, block_size>>>(d_next.get(), d_output.get(), chain_len,
                                                          start_idx);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }

        auto stats = compute_stats(samples);
        double ns_per_load = (stats.median_us * 1000.0) / chain_len;
        double effective_bw = 0.0;

        char label[128];
        snprintf(label, sizeof(label), "concurrency warps/SM=%d blocks=%d threads=%d", warps_per_sm,
                 blocks, threads);
        print_csv(label, samples);

        const char *interp;
        if (ns_per_load > 300)
            interp = "no hiding (≈ DRAM)";
        else if (ns_per_load > 150)
            interp = "partial hiding";
        else
            interp = "well hidden";

        printf("    -> %.1f ns/load  (%s)\n\n", ns_per_load, interp);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  3. Warp Scheduling Fairness Probe
// ════════════════════════════════════════════════════════════════════════════

/// Two different kernels run concurrently in different blocks.
/// Kernel A: short-running (16 iterations)
/// Kernel B: long-running (1024 iterations)
/// Measure completion order to infer scheduling fairness.

__global__ void kernel_short(volatile int *flag, int id)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            __threadfence();
        }
        *flag = id;
    }
}

__global__ void kernel_long(volatile int *flag, int id)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0)
    {
        for (int i = 0; i < 1024; ++i)
        {
            __threadfence();
        }
        *flag = id;
    }
}

void probe_fairness(const DeviceInfo &info, int /*repeats*/)
{
    printf("\n═══ 3. Warp Scheduling Fairness (Qualitative) ═══\n");
    printf("Launches short and long kernels concurrently to observe\n");
    printf("whether the scheduler prioritises short tasks.\n\n");

    DeviceBuffer<int> d_flag1(1);
    DeviceBuffer<int> d_flag2(1);

    // Launch short and long kernels concurrently via different streams
    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    // Start timing
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));
    CUDA_CHECK(cudaEventRecord(start_ev, s1));

    kernel_short<<<1, 32, 0, s1>>>(d_flag1.get(), 1);
    kernel_long<<<1, 32, 0, s2>>>(d_flag2.get(), 2);

    // Wait for short kernel
    CUDA_CHECK(cudaEventRecord(stop_ev, s1));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));
    float short_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&short_ms, start_ev, stop_ev));

    // Wait for long kernel
    CUDA_CHECK(cudaStreamSynchronize(s2));

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    int h_flag1 = 0, h_flag2 = 0;
    d_flag1.download(&h_flag1);
    d_flag2.download(&h_flag2);

    printf("  Short kernel (16x fence) completed at: %.0f μs  (flag=%d)\n", short_ms * 1000.0,
           h_flag1);
    printf("  Long kernel (1024x fence) completed at: same kernel launch\n");
    printf("  Short kernel time: %.2f ms\n", short_ms);
    printf("  => Without priority: short finishes long before long.\n");
    printf("  => If scheduler used priority inversion or fairness:\n");
    printf("     the short task would NOT complete first.\n\n");
}

// ════════════════════════════════════════════════════════════════════════════
//  Helper: build random permutation chain
// ════════════════════════════════════════════════════════════════════════════

static std::vector<int> build_chain(int size, unsigned seed)
{
    std::vector<int> perm(size);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
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

    probe_divergence(repeats);
    probe_concurrency(info, repeats);
    probe_fairness(info, repeats);

    printf("\nDone.\n");
    return 0;
}

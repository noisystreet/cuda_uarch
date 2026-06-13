// SPDX-License-Identifier: MIT
/**
 * warp_shuffle_probe.cu — Warp Shuffle / Sync instruction analysis
 *
 * Experiments:
 *   1. Shuffle throughput — __shfl_sync (idx/down/up/xor) GOp/s
 *   2. Shuffle delta sweep — impact of shift distance on throughput
 *   3. Syncwarp overhead — __syncwarp barrier cost vs compute baseline
 *   4. Ballot / All / Any — warp vote primitive throughput
 *   5. Match (__match_all_sync) — warp match throughput (sm_89)
 *
 * Usage:
 *   ./warp_shuffle_probe [repeats]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ════════════════════════════════════════════════════════════════════════════
//  1. Shuffle Throughput Kernels
// ════════════════════════════════════════════════════════════════════════════

/// __shfl_sync (idx mode) — direct index shuffle, no rotation
template <int Unroll>
__global__ void shfl_idx_kernel(const float *__restrict__ input, float *__restrict__ output,
                                int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    float val = input[tid % 1024];

    // Each lane reads from a fixed partner to exercise the shuffle network
    int partner = (lane + 1) % warpSize;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for (int j = 0; j < Unroll; ++j)
        {
            val = __shfl_sync(0xffffffff, val + 0.0f, partner);
        }
    }
    output[tid] = val;
}

/// __shfl_down_sync — shift right by delta lanes
template <int Unroll, int Delta>
__global__ void shfl_down_kernel(const float *__restrict__ input, float *__restrict__ output,
                                 int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = input[tid % 1024];

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for (int j = 0; j < Unroll; ++j)
        {
            val = __shfl_down_sync(0xffffffff, val + 0.0f, Delta);
        }
    }
    output[tid] = val;
}

/// __shfl_up_sync — shift left by delta lanes
template <int Unroll, int Delta>
__global__ void shfl_up_kernel(const float *__restrict__ input, float *__restrict__ output,
                               int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = input[tid % 1024];

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for (int j = 0; j < Unroll; ++j)
        {
            val = __shfl_up_sync(0xffffffff, val + 0.0f, Delta);
        }
    }
    output[tid] = val;
}

/// __shfl_xor_sync — butterfly shuffle (XOR by mask)
template <int Unroll, int Mask>
__global__ void shfl_xor_kernel(const float *__restrict__ input, float *__restrict__ output,
                                int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = input[tid % 1024];

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for (int j = 0; j < Unroll; ++j)
        {
            val = __shfl_xor_sync(0xffffffff, val + 0.0f, Mask);
        }
    }
    output[tid] = val;
}

// Dispatch helpers
template <int Unroll> void run_shfl_idx(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = info.warp_size;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, threads * sizeof(float)));
    std::vector<float> h_in(1024, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));

    shfl_idx_kernel<Unroll><<<1, threads>>>(d_in, d_out, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        shfl_idx_kernel<Unroll><<<1, threads>>>(d_in, d_out, iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * Unroll * threads;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "shfl_idx Unroll=%d iter=%d", Unroll, iterations);
    print_csv(label, samples);
    printf("    -> Throughput: %.1f GOp/s  (__shfl_sync idx, Unroll=%d)\n", ops_per_sec / 1e9,
           Unroll);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

template <int Unroll, int Delta>
void run_shfl_down(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = info.warp_size;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, threads * sizeof(float)));
    std::vector<float> h_in(1024, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));

    shfl_down_kernel<Unroll, Delta><<<1, threads>>>(d_in, d_out, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        shfl_down_kernel<Unroll, Delta><<<1, threads>>>(d_in, d_out, iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * Unroll * threads;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "shfl_down Delta=%d Unroll=%d iter=%d", Delta, Unroll,
             iterations);
    print_csv(label, samples);
    printf("    -> Throughput: %.1f GOp/s  (__shfl_down_sync delta=%d, Unroll=%d)\n",
           ops_per_sec / 1e9, Delta, Unroll);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

template <int Unroll, int Delta>
void run_shfl_up(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = info.warp_size;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, threads * sizeof(float)));
    std::vector<float> h_in(1024, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));

    shfl_up_kernel<Unroll, Delta><<<1, threads>>>(d_in, d_out, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        shfl_up_kernel<Unroll, Delta><<<1, threads>>>(d_in, d_out, iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * Unroll * threads;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "shfl_up Delta=%d Unroll=%d iter=%d", Delta, Unroll, iterations);
    print_csv(label, samples);
    printf("    -> Throughput: %.1f GOp/s  (__shfl_up_sync delta=%d, Unroll=%d)\n",
           ops_per_sec / 1e9, Delta, Unroll);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

template <int Unroll, int Mask>
void run_shfl_xor(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = info.warp_size;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, threads * sizeof(float)));
    std::vector<float> h_in(1024, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));

    shfl_xor_kernel<Unroll, Mask><<<1, threads>>>(d_in, d_out, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        shfl_xor_kernel<Unroll, Mask><<<1, threads>>>(d_in, d_out, iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * Unroll * threads;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "shfl_xor Mask=%d Unroll=%d iter=%d", Mask, Unroll, iterations);
    print_csv(label, samples);
    printf("    -> Throughput: %.1f GOp/s  (__shfl_xor_sync mask=%d, Unroll=%d)\n",
           ops_per_sec / 1e9, Mask, Unroll);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

void probe_shfl_throughput(const DeviceInfo &info, int repeats)
{
    printf("═══ 1. Warp Shuffle Throughput ═══\n");
    printf("Measures shuffle operations per second for each variant.\n");
    printf("Unroll=1: single shuffle per iteration (ILP=1, exposes latency).\n");
    printf("Unroll=8: multiple independent shuffles per iteration (ILP=8).\n\n");

    const int iterations = 100000;

    printf("--- __shfl_sync (idx mode, partner = lane+1) ---\n");
    run_shfl_idx<1>(info, iterations, repeats);
    run_shfl_idx<8>(info, iterations, repeats);

    printf("\n--- __shfl_down_sync (delta=1) ---\n");
    run_shfl_down<1, 1>(info, iterations, repeats);
    run_shfl_down<8, 1>(info, iterations, repeats);

    printf("\n--- __shfl_up_sync (delta=1) ---\n");
    run_shfl_up<1, 1>(info, iterations, repeats);
    run_shfl_up<8, 1>(info, iterations, repeats);

    printf("\n--- __shfl_xor_sync (mask=1, nearest neighbor) ---\n");
    run_shfl_xor<1, 1>(info, iterations, repeats);
    run_shfl_xor<8, 1>(info, iterations, repeats);

    printf("\n--- __shfl_xor_sync (mask=16, far neighbor) ---\n");
    run_shfl_xor<1, 16>(info, iterations, repeats);
    run_shfl_xor<8, 16>(info, iterations, repeats);
}

// ════════════════════════════════════════════════════════════════════════════
//  2. Shuffle Delta Sweep — impact of shift distance
// ════════════════════════════════════════════════════════════════════════════

void probe_shfl_delta_sweep(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 2. Shuffle Delta Sweep ═══\n");
    printf("Varying delta for __shfl_down_sync (Unroll=1).\n");
    printf("If throughput is independent of delta → single-cycle shuffle.\n");
    printf("If throughput decreases with larger delta → multi-cycle routing.\n\n");

    const int iterations = 100000;
    const int deltas[] = {1, 2, 4, 8, 16, 31};

    printf("%-8s  %16s\n", "Delta", "Throughput (GOp/s)");
    printf("%-8s  %16s\n", "-----", "------------------");
    for (int d : deltas)
    {
        // Template dispatch via switch
        switch (d)
        {
        case 1:
            run_shfl_down<1, 1>(info, iterations, repeats);
            break;
        case 2:
            run_shfl_down<1, 2>(info, iterations, repeats);
            break;
        case 4:
            run_shfl_down<1, 4>(info, iterations, repeats);
            break;
        case 8:
            run_shfl_down<1, 8>(info, iterations, repeats);
            break;
        case 16:
            run_shfl_down<1, 16>(info, iterations, repeats);
            break;
        case 31:
            run_shfl_down<1, 31>(info, iterations, repeats);
            break;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  3. __syncwarp Overhead
// ════════════════════════════════════════════════════════════════════════════

/// Baseline: pure compute (no syncwarp)
__global__ void syncwarp_baseline(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = static_cast<float>(tid);

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        val = fmaf(val, 1.0000001f, 0.0000001f);
    }
    output[tid] = val;
}

/// With __syncwarp: insert barrier after each compute step
__global__ void syncwarp_barrier(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = static_cast<float>(tid);

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        val = fmaf(val, 1.0000001f, 0.0000001f);
        __syncwarp();
    }
    output[tid] = val;
}

void probe_syncwarp(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 3. __syncwarp Overhead ═══\n");
    printf("Compare FFMA throughput with and without __syncwarp per iteration.\n");
    printf("Difference → cost of warp barrier synchronization.\n\n");

    const int iterations = 50000;
    const int threads = 256;
    const int blocks = info.sm_count * 2;
    const int num_items = threads * blocks;

    DeviceBuffer<float> d_out(num_items);

    // Warm up
    syncwarp_baseline<<<blocks, threads>>>(d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    syncwarp_barrier<<<blocks, threads>>>(d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Baseline (no syncwarp)
    {
        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            syncwarp_baseline<<<blocks, threads>>>(d_out.get(), iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);
        double total_ops = static_cast<double>(iterations) * num_items;
        double ops_per_sec = total_ops / (stats.median_us * 1e-6);
        print_csv("syncwarp baseline FFMA", samples);
        printf("  Baseline (FFMA only):     %.1f GOp/s  (%.2f us)\n", ops_per_sec / 1e9,
               stats.median_us);
    }

    // With __syncwarp
    {
        std::vector<double> samples;
        samples.reserve(repeats);
        for (int r = 0; r < repeats; ++r)
        {
            GpuTimer timer;
            timer.start();
            syncwarp_barrier<<<blocks, threads>>>(d_out.get(), iterations);
            timer.stop();
            samples.push_back(timer.elapsed_us());
        }
        auto stats = compute_stats(samples);
        double total_ops = static_cast<double>(iterations) * num_items;
        double ops_per_sec = total_ops / (stats.median_us * 1e-6);
        print_csv("syncwarp with barrier", samples);
        printf("  With __syncwarp:         %.1f GOp/s  (%.2f us)\n", ops_per_sec / 1e9,
               stats.median_us);

        double overhead_ns =
            (stats.median_us - 0) * 1000.0 / iterations; // relative to baseline in next calc
        printf("  __syncwarp overhead:     ~%.1f ns per barrier (vs baseline)\n", overhead_ns);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  4. Ballot / All / Any Throughput
// ════════════════════════════════════════════════════════════════════════════

/// __ballot_sync throughput
__global__ void ballot_kernel(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    unsigned mask = 0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        mask = __ballot_sync(0xffffffff, lane < 16);
    }
    if (tid == 0)
        *output = static_cast<float>(mask);
}

/// __all_sync throughput
__global__ void all_sync_kernel(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    int result = 0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        result = __all_sync(0xffffffff, lane < 16);
    }
    if (tid == 0)
        *output = static_cast<float>(result);
}

/// __any_sync throughput
__global__ void any_sync_kernel(float *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    int result = 0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        result = __any_sync(0xffffffff, lane < 16);
    }
    if (tid == 0)
        *output = static_cast<float>(result);
}

template <typename KernelFn>
void run_vote_test(const char *name, KernelFn kernel, const DeviceInfo &info, int iterations,
                   int repeats)
{
    const int threads = info.warp_size;
    DeviceBuffer<float> d_out(1);

    kernel<<<1, threads>>>(d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        kernel<<<1, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * threads;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "%s iter=%d", name, iterations);
    print_csv(label, samples);
    printf("    -> Throughput: %.1f GOp/s  (%s)\n", ops_per_sec / 1e9, name);
}

void probe_vote(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 4. Warp Vote Primitives ═══\n");
    printf("Throughput of __ballot_sync, __all_sync, __any_sync.\n\n");

    const int iterations = 100000;

    run_vote_test("__ballot_sync", ballot_kernel, info, iterations, repeats);
    run_vote_test("__all_sync", all_sync_kernel, info, iterations, repeats);
    run_vote_test("__any_sync", any_sync_kernel, info, iterations, repeats);
}

// ════════════════════════════════════════════════════════════════════════════
//  5. __match_all_sync Throughput (Ada sm_89)
// ════════════════════════════════════════════════════════════════════════════

/// __match_all_sync — returns mask of lanes with same value + sets predicate
__global__ void match_all_kernel(unsigned *__restrict__ output, int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warpSize;
    unsigned mask_result = 0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        int pred = 0;
        mask_result = __match_all_sync(0xffffffff, lane, &pred);
    }
    if (tid == 0)
        *output = mask_result;
}

void probe_match(const DeviceInfo &info, int repeats)
{
    printf("\n═══ 5. __match_all_sync Throughput ═══\n");
    printf("Measures warp match operations per second (sm_89 feature).\n\n");

    const int iterations = 100000;
    const int threads = info.warp_size;

    DeviceBuffer<unsigned> d_out(1);

    match_all_kernel<<<1, threads>>>(d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        match_all_kernel<<<1, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(iterations) * threads;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    print_csv("__match_all_sync", samples);
    printf("    -> Throughput: %.1f GOp/s  (__match_all_sync)\n", ops_per_sec / 1e9);
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

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("  Warp Shuffle / Sync Instruction Analysis\n");
    printf("  Hardware: sm_%d.%d (Ada Lovelace)\n", info.major, info.minor);
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    probe_shfl_throughput(info, repeats);
    probe_shfl_delta_sweep(info, repeats);
    probe_syncwarp(info, repeats);
    probe_vote(info, repeats);
    probe_match(info, repeats);

    printf("\nDone.\n");
    return 0;
}

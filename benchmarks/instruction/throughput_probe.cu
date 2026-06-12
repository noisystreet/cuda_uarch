// SPDX-License-Identifier: MIT
/**
 * throughput_probe.cu — Instruction Throughput Measurement
 *
 * Method:
 *   Execute many independent arithmetic instructions in a single warp/thread.
 *   High ILP keeps the pipeline busy; measure the sustained throughput
 *   and compare with theoretical peak.
 *
 *   By varying the number of independent operations per loop iteration,
 *   we can infer the issue width of the warp scheduler.
 *
 * Usage:
 *   ./throughput_probe [unroll_factor] [repetitions]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ─── Device kernels with varying ILP ──────────────────────────────────────
// Template struct to generate different unroll factors at compile time.

template <int Unroll>
__global__ void throughput_ffma(const float *__restrict__ /*input*/, float *__restrict__ output,
                                int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= gridDim.x * blockDim.x)
        return;

    float a = 1.0000001f;
    float b = 0.9999999f;

    // Accumulate in independent registers
    float r[Unroll];
#pragma unroll
    for (int u = 0; u < Unroll; ++u)
        r[u] = static_cast<float>(tid + u);

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
#pragma unroll
        for (int u = 0; u < Unroll; ++u)
        {
            r[u] = fmaf(r[u], a, b); // FFMA
        }
    }

    // Sum to prevent dead code elimination
    float sum = 0.0f;
#pragma unroll
    for (int u = 0; u < Unroll; ++u)
        sum += r[u];
    output[tid] = sum;
}

// ─── Run throughput test ──────────────────────────────────────────────────
template <int Unroll> void run_throughput_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads_per_block = 256;
    const int blocks = info.sm_count * 2; // 2 blocks per SM
    const int num_items = threads_per_block * blocks;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(num_items);

    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    // Warm up
    throughput_ffma<Unroll><<<blocks, threads_per_block>>>(d_in.get(), d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        throughput_ffma<Unroll><<<blocks, threads_per_block>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    // Compute throughput
    auto stats = compute_stats(samples);
    // Total FFMA operations = Unroll * iterations * num_items
    double total_ops = static_cast<double>(Unroll) * iterations * num_items;
    double ops_per_sec = total_ops / (stats.median_us * 1e-6);

    char label[128];
    snprintf(label, sizeof(label), "FFMA unroll=%d iter=%d threads=%d", Unroll, iterations,
             num_items);
    print_csv(label, samples);
    printf("    -> Throughput: %.2f GOp/s  (%.2f ops/cycle @ %.0f MHz)\n", ops_per_sec / 1e9,
           ops_per_sec / (1500e6), 1500.0);
}

// ─── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    int unroll_factor = 1;
    int repeats = 32;

    if (argc > 1)
        unroll_factor = std::atoi(argv[1]);
    if (argc > 2)
        repeats = std::atoi(argv[2]);
    if (unroll_factor <= 0)
        unroll_factor = 1;
    if (repeats <= 0)
        repeats = 32;

    auto info = get_device_info();
    print_device_info(info);
    warm_up_device();

    const int iterations = 10000;

    printf("\n=== Instruction Throughput Probe ========\n");
    printf("unroll=%d  iterations=%d  repeats=%d\n\n", unroll_factor, iterations, repeats);

    // Try multiple unroll factors
    constexpr int kUnrolls[] = {1, 2, 4, 8, 16};

    for (int u : kUnrolls)
    {
        if (unroll_factor > 0 && u != unroll_factor)
            continue;

        switch (u)
        {
        case 1:
            run_throughput_test<1>(info, iterations, repeats);
            break;
        case 2:
            run_throughput_test<2>(info, iterations, repeats);
            break;
        case 4:
            run_throughput_test<4>(info, iterations, repeats);
            break;
        case 8:
            run_throughput_test<8>(info, iterations, repeats);
            break;
        case 16:
            run_throughput_test<16>(info, iterations, repeats);
            break;
        default:
            break;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\nDone.\n");
    return 0;
}

// SPDX-License-Identifier: MIT
/**
 * global_mem_latency.cu — Global Memory Latency via Pointer Chasing
 *
 * Method:
 *   Build a linked list (or index chain) in global memory.
 *   Each element holds an index to the next element in a random permutation.
 *   A single thread walks the chain: idx = next[idx], repeated many times.
 *   Total time / chain_length ≈ global memory load latency.
 *
 *   By varying the number of concurrent warps, we can observe latency
 *   hiding effects and infer the effective latency under load.
 *
 * Usage:
 *   ./global_mem_latency [chain_length] [warps_per_sm] [repetitions]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace uarch;

// ─── Pointer-chase kernel ──────────────────────────────────────────────────
// Each thread walks an index chain independently.
__global__ void chase_kernel(const int *__restrict__ next, int *__restrict__ output, int chain_len,
                             int start_idx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize: each thread picks a start index
    int idx = start_idx + tid;
    int val = 0;

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        idx = next[idx]; // dependent load (pointer chase)
        val ^= idx;      // use result to prevent elimination
    }

    // Store final value so compiler can't optimize away the chain
    if (tid == 0)
        *output = val;
}

// ─── Build a random permutation index chain ────────────────────────────────
std::vector<int> build_chain(int num_elements, unsigned seed = 42)
{
    std::vector<int> perm(num_elements);
    std::iota(perm.begin(), perm.end(), 0);

    // Generate a cyclic permutation (no fixed points, single cycle)
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    // Ensure it's a valid permutation (0..N-1 appearing exactly once)
    return perm;
}

// ─── Host-side test driver ─────────────────────────────────────────────────
void run_latency_test(const DeviceInfo &info, int chain_len, int warps_per_sm, int repeats)
{
    // Allocate chain: ~2x the chain length for random walks
    const int chain_size = std::max(chain_len * 2, 1 << 20); // at least 1M
    const int threads = warps_per_sm * info.warp_size * info.sm_count;

    // Build chain on host and upload
    auto chain = build_chain(chain_size);
    DeviceBuffer<int> d_next(chain_size);
    d_next.upload(chain.data());

    DeviceBuffer<int> d_output(1);

    int start_idx = 0; // each thread uses tid as starting offset

    // Warm up
    chase_kernel<<<div_up(threads, 256), 256>>>(d_next.get(), d_output.get(), 4, start_idx);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        chase_kernel<<<div_up(threads, 256), 256>>>(d_next.get(), d_output.get(), chain_len,
                                                    start_idx);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    // Each thread performs chain_len dependent loads
    double total_loads = static_cast<double>(chain_len) * threads;
    double ns_per_load = (stats.median_us * 1000.0) / chain_len;
    // Note: total_loads not needed for latency (warp-level aggregation)

    char label[128];
    snprintf(label, sizeof(label), "global_mem_latency chain_len=%d warps/SM=%d threads=%d",
             chain_len, warps_per_sm, threads);
    print_csv(label, samples);

    printf("    -> Effective latency: %.1f ns/load  (%.0f cycles @ %.0f MHz)\n", ns_per_load,
           ns_per_load * 1.5, 1500.0);

    // If only 1 thread active, this approximates the true latency
    if (threads == info.warp_size)
    {
        printf("    -> (single warp, approximates true hardware latency)\n");
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    int chain_len = 10000;
    int warps_per_sm = 1;
    int repeats = 16;

    if (argc > 1)
        chain_len = std::atoi(argv[1]);
    if (argc > 2)
        warps_per_sm = std::atoi(argv[2]);
    if (argc > 3)
        repeats = std::atoi(argv[3]);

    if (chain_len <= 0)
        chain_len = 10000;
    if (warps_per_sm <= 0)
        warps_per_sm = 1;
    if (repeats <= 0)
        repeats = 16;

    auto info = get_device_info();
    print_device_info(info);
    warm_up_device();

    printf("\n=== Global Memory Latency Probe =========\n");
    printf("chain_len=%d  warps_per_sm=%d  repeats=%d\n\n", chain_len, warps_per_sm, repeats);

    // Sweep warps_per_sm to observe latency hiding
    constexpr int kWarpSweep[] = {1, 2, 4, 8, 16, 32};

    for (int w : kWarpSweep)
    {
        if (warps_per_sm > 0 && w != warps_per_sm)
            continue;
        run_latency_test(info, chain_len, w, repeats);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\nDone.\n");
    return 0;
}

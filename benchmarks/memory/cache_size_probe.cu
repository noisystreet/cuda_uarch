// SPDX-License-Identifier: MIT
/**
 * cache_size_probe.cu — L1/L2 Cache Hierarchy Analysis
 *
 * Three experiments:
 *   1. Cache size   — pointer-chase with growing working set; latency
 *                      jumps reveal cache capacities.
 *   2. Cache line   — fixed small working set, vary stride; constant
 *                      latency up to cache-line boundary.
 *   3. Associativity — access N addresses mapping to the same cache set;
 *                      latency jump reveals associativity.
 *
 * Usage:
 *   ./cache_size_probe [repeats]
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

// ════════════════════════════════════════════════════════════════════════════
//  1. Cache Size Probe  (pointer chasing)
// ════════════════════════════════════════════════════════════════════════════

/// Build a random permutation within [0, size) forming a single cycle.
static std::vector<int> build_chain(int size, unsigned seed)
{
    std::vector<int> perm(size);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);
    // Ensure perm[size-1] -> perm[0] to close the cycle
    // (already a permutation, so chasing will cycle through all elements)
    return perm;
}

/// Pointer-chase kernel.
__global__ void chase_kernel(const int *__restrict__ next, int *__restrict__ output, int chain_len,
                             int start_idx)
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

/// Measure average load latency (ns) for a given working set size.
double measure_size_latency(int working_set_bytes, int repeats)
{
    // Convert bytes → number of int elements
    int num_elems = working_set_bytes / sizeof(int);
    num_elems = std::max(num_elems, 1024); // at least 1K elements

    auto chain = build_chain(num_elems, 42);
    DeviceBuffer<int> d_next(num_elems);
    d_next.upload(chain.data());
    DeviceBuffer<int> d_output(1);

    int chain_len = std::min(num_elems, 100000);
    int start_idx = 0;

    // Single warp probe (no latency hiding)
    const int threads = 32;
    const int blocks = 1;

    // Warm-up
    chase_kernel<<<blocks, threads>>>(d_next.get(), d_output.get(), 4, start_idx);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        chase_kernel<<<blocks, threads>>>(d_next.get(), d_output.get(), chain_len, start_idx);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    // ns per access = (median_us * 1000) / chain_len
    return (stats.median_us * 1000.0) / chain_len;
}

void probe_cache_size(int repeats)
{
    printf("\n═══ 1. Cache Size Probe (pointer chasing) ═══\n");
    printf("%-20s  %12s  %12s\n", "Working Set", "Size (KB)", "Latency (ns)");
    printf("%-20s  %12s  %12s\n", "-----------", "---------", "-----------");

    // Sweep from 4 KB to 64 MB (powers of 2)
    for (int log2 = 12; log2 <= 26; ++log2)
    {
        int bytes = 1 << log2; // 4 KB → 64 MB
        double lat_ns = measure_size_latency(bytes, repeats);

        printf("%-20s  %12d  %12.1f\n", "", bytes / 1024, lat_ns);
        fflush(stdout);

        // Short-circuit if latency is obviously L2+ (>= 150 ns) at small
        // sizes — indicates something is wrong with the method, but still
        // useful data.
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  2. Cache Line Size Probe  (stride sweep)
// ════════════════════════════════════════════════════════════════════════════

/// Sequential access kernel with configurable stride.
__global__ void stride_read_kernel(const char *__restrict__ base, int *__restrict__ output,
                                   int num_accesses, int stride_bytes)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid * stride_bytes;
    int val = 0;

    for (int i = 0; i < num_accesses; ++i)
    {
        val += base[idx];
        idx += stride_bytes;
    }
    if (tid == 0)
        *output = val;
}

double measure_line_latency(int stride_bytes, int working_set_bytes, int repeats)
{
    int num_bytes = working_set_bytes;
    int num_accesses = num_bytes / std::max(stride_bytes, 1);
    num_accesses = std::max(num_accesses, 1024);

    DeviceBuffer<char> d_buf(num_bytes);
    DeviceBuffer<int> d_output(1);

    const int threads = 32;
    const int blocks = 1;

    // Warm-up
    stride_read_kernel<<<blocks, threads>>>(d_buf.get(), d_output.get(), 4, stride_bytes);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        stride_read_kernel<<<blocks, threads>>>(d_buf.get(), d_output.get(), num_accesses,
                                                stride_bytes);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    return (stats.median_us * 1000.0) / num_accesses;
}

void probe_cache_line(int repeats)
{
    // Small working set (4 KB) — should stay in L1
    const int working_set = 4096;

    printf("\n═══ 2. Cache Line Size Probe (stride sweep) ═══\n");
    printf("Working set = %d bytes (fits L1)\n", working_set);
    printf("%-12s  %12s\n", "Stride (B)", "Latency (ns)");
    printf("%-12s  %12s\n", "----------", "-----------");

    // Sweep stride: 4, 8, 16, 32, 64, 128, 256, 512, 1024
    int strides[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int s : strides)
    {
        double lat_ns = measure_line_latency(s, working_set, repeats);
        printf("%-12d  %12.1f\n", s, lat_ns);
        fflush(stdout);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  3. Cache Associativity Probe  (conflict miss)
// ════════════════════════════════════════════════════════════════════════════

/// Access N addresses that map to the same cache set (determined by stride).
__global__ void conflict_kernel(const int *__restrict__ data, int *__restrict__ output,
                                int num_addresses, int stride)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid;
    int val = 0;

    for (int i = 0; i < num_addresses; ++i)
    {
        val += data[idx * stride];
    }
    if (tid == 0)
        *output = val;
}

double measure_conflict_latency(int num_addresses, int stride, int cache_size, int repeats)
{
    // Allocate cache_size bytes so that stride maps to the same set
    int num_elems = cache_size / sizeof(int);
    num_elems = std::max(num_elems, 1024);

    DeviceBuffer<int> d_data(num_elems);
    DeviceBuffer<int> d_output(1);

    const int threads = 1; // single thread to isolate effect
    const int blocks = 1;

    // Warm-up
    conflict_kernel<<<blocks, threads>>>(d_data.get(), d_output.get(), 4, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        conflict_kernel<<<blocks, threads>>>(d_data.get(), d_output.get(), num_addresses, stride);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    return (stats.median_us * 1000.0) / num_addresses;
}

void probe_associativity(int repeats)
{
    // Target a specific cache level (e.g. L1 ~128 KB on Ada)
    // Use stride = cache_line_size * num_sets_for_our_range
    // On Ada L1: 128 KB, 128 B/line → 1024 sets, 32-way?
    // We'll sweep associativity from 1 to 64.

    const int cache_size = 128 * 1024; // assume L1 is ~128KB (Ada)
    const int line_size = 128;         // assume 128 B/line
    const int num_sets = cache_size / line_size;

    printf("\n═══ 3. Cache Associativity Probe ═══\n");
    printf("Assumed L1: %d KB, %d B/line → %d sets\n", cache_size / 1024, line_size, num_sets);
    printf("%-18s  %12s\n", "Num Addresses", "Latency (ns)");
    printf("%-18s  %12s\n", "-------------", "-----------");

    // Sweep number of addresses mapping to the same set
    // Stride = num_sets (so they all map to set 0)
    int stride = num_sets;

    for (int n = 1; n <= 64; n *= 2)
    {
        double lat_ns = measure_conflict_latency(n, stride, cache_size, repeats);
        printf("%-18d  %12.1f\n", n, lat_ns);
        fflush(stdout);
    }
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

    // Run the three probes
    probe_cache_size(repeats);
    probe_cache_line(repeats);
    probe_associativity(repeats);

    printf("\nDone.\n");
    return 0;
}

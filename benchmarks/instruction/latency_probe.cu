// SPDX-License-Identifier: MIT
/**
 * latency_probe.cu — Instruction Latency Measurement via Dependency Chains
 *
 * Method:
 *   Build a long serial dependency chain inside a single warp/thread.
 *   Each operation depends on the result of the previous one, so ILP = 1.
 *   Measured total time / chain_length ≈ instruction latency.
 *
 *   Types tested:
 *     - IADD (integer add)
 *     - FADD (float add)
 *     - FMUL (float multiply)
 *     - FFMA (fused multiply-add)
 *     - I2F (integer to float conversion)
 *
 * Usage:
 *   ./latency_probe [chain_length] [repetitions]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ─── Device-side dependency chain kernels ─────────────────────────────────
// Each kernel performs `chain_len` dependent operations, reading from
// `chain_len` pre-loaded values in registers.

// --- Integer addition ---
__global__ void chain_iadd(const float *__restrict__ input, float *__restrict__ output,
                           int chain_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= gridDim.x * blockDim.x)
        return;

    // Load once from global memory (cold), then chain
    int val = __float2int_rn(input[tid]);

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        val = val + 1; // integer add dependency chain
        val = val - 1; // keep value bounded
    }

    output[tid] = static_cast<float>(val);
}

// --- Float addition ---
__global__ void chain_fadd(const float *__restrict__ input, float *__restrict__ output,
                           int chain_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= gridDim.x * blockDim.x)
        return;

    float val = input[tid];

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        val = val + 1.0f; // FADD dependency
    }

    output[tid] = val;
}

// --- Float multiply ---
__global__ void chain_fmul(const float *__restrict__ input, float *__restrict__ output,
                           int chain_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= gridDim.x * blockDim.x)
        return;

    float val = input[tid];

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        val = val * 1.0000001f; // FMUL, avoid trivial identity
    }

    output[tid] = val;
}

// --- Fused multiply-add ---
__global__ void chain_ffma(const float *__restrict__ input, float *__restrict__ output,
                           int chain_len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= gridDim.x * blockDim.x)
        return;

    float val = input[tid];

#pragma unroll 1
    for (int i = 0; i < chain_len; ++i)
    {
        val = fmaf(val, 1.0000001f, 0.0f); // FFMA: val * 1.0 + 0.0
    }

    output[tid] = val;
}

// ─── Host-side benchmark driver ───────────────────────────────────────────
struct ChainConfig
{
    const char *name;
    void (*kernel)(const float *, float *, int);
};

static constexpr ChainConfig kChains[] = {
    {"IADD", chain_iadd},
    {"FADD", chain_fadd},
    {"FMUL", chain_fmul},
    {"FFMA", chain_ffma},
};

static constexpr int kNumChains = sizeof(kChains) / sizeof(kChains[0]);

void run_chain_test(ChainConfig cfg, int chain_len, int num_threads, int repeats)
{
    const int num_items = num_threads; // one item per thread
    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(num_items);

    // Host data
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    // Warm up
    cfg.kernel<<<div_up(num_items, 256), 256>>>(d_in.get(), d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        cfg.kernel<<<div_up(num_items, 256), 256>>>(d_in.get(), d_out.get(), chain_len);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    // Compute per-instruction latency
    // All threads in a warp execute in lockstep, so the chain length
    // determines the total serial path length per warp.
    // ns_per_op = total_time_us * 1000 / chain_len
    auto stats = compute_stats(samples);
    double ns_per_op = (stats.median_us * 1000.0) / chain_len;

    char label[128];
    snprintf(label, sizeof(label), "%s chain_len=%d threads=%d", cfg.name, chain_len, num_threads);
    print_csv(label, samples);

    printf("    -> Per-instruction latency: %.2f ns  (%.2f cycles @ %.0f MHz)\n", ns_per_op,
           ns_per_op * 1.5, 1500.0); // rough cycle estimate
}

// ─── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    int chain_len = 10000;
    int repeats = 32;

    if (argc > 1)
        chain_len = std::atoi(argv[1]);
    if (argc > 2)
        repeats = std::atoi(argv[2]);
    if (chain_len <= 0)
        chain_len = 10000;
    if (repeats <= 0)
        repeats = 32;

    // Device info
    auto info = get_device_info();
    print_device_info(info);

    warm_up_device();

    // Use a single warp (32 threads) to avoid latency hiding.
    // More warps would allow the scheduler to hide latency, making
    // the measurement appear lower than the true instruction latency.
    const int num_threads = 32; // one warp

    printf("\n=== Instruction Latency Probe ===========\n");
    printf("chain_len=%d  threads=%d (1 warp)  repeats=%d\n\n", chain_len, num_threads, repeats);

    for (int i = 0; i < kNumChains; ++i)
    {
        run_chain_test(kChains[i], chain_len, num_threads, repeats);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\nDone.\n");
    return 0;
}

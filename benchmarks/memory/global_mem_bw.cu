// SPDX-License-Identifier: MIT
/**
 * global_mem_bw.cu — Global Memory Bandwidth Measurement
 *
 * Method:
 *   Launch kernels that read/write contiguous arrays of various sizes.
 *   Measure achieved bandwidth for:
 *     - Read-only (float4 vectorized)
 *     - Write-only
 *     - Copy (read + write)
 *   Sweep array size to observe bandwidth saturation.
 *
 * Usage:
 *   ./global_mem_bw [min_pow2] [max_pow2] [repetitions]
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace uarch;

// ─── Read-only bandwidth kernel (float4 vectorised) ────────────────────────
__global__ void read_bw(const float *__restrict__ src, float *__restrict__ dummy, int num_elements)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < num_elements / 4; i += stride)
    {
        float4 v = reinterpret_cast<const float4 *>(src)[i];
        sum += v.x + v.y + v.z + v.w;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
        *dummy = sum;
}

// ─── Write-only bandwidth kernel ──────────────────────────────────────────
__global__ void write_bw(float *__restrict__ dst, int num_elements)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_elements; i += stride)
    {
        dst[i] = 1.0f;
    }
}

// ─── Copy bandwidth kernel (read + write) ─────────────────────────────────
__global__ void copy_bw(const float *__restrict__ src, float *__restrict__ dst, int num_elements)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_elements; i += stride)
    {
        dst[i] = src[i];
    }
}

// ─── Templated test runner ─────────────────────────────────────────────────
template <typename KernelFn>
void run_bw_test(const DeviceInfo &info, const char *label, int num_elements, int repeats,
                 KernelFn kernel, float bytes_per_element, auto &&...kernel_args)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;

    // Warm up
    kernel<<<blocks, threads>>>(kernel_args..., num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        kernel<<<blocks, threads>>>(kernel_args..., num_elements);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_bytes = static_cast<double>(num_elements) * bytes_per_element;
    double bw_gb_s = (total_bytes / (stats.median_us * 1e-6)) / 1e9;

    char line[128];
    snprintf(line, sizeof(line), "%s size=%d (%.1f MB)", label, num_elements, total_bytes / 1e6);
    print_csv(line, samples);
    printf("    -> Bandwidth: %.1f GB/s\n", bw_gb_s);
}

// ─── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    int min_pow2 = 20; // 2^20 = 1M floats = 4 MB
    int max_pow2 = 28; // 2^28 = 256M floats = 1 GB
    int repeats = 16;

    if (argc > 1)
        min_pow2 = std::atoi(argv[1]);
    if (argc > 2)
        max_pow2 = std::atoi(argv[2]);
    if (argc > 3)
        repeats = std::atoi(argv[3]);
    if (min_pow2 <= 0)
        min_pow2 = 20;
    if (max_pow2 <= 0)
        max_pow2 = 28;
    if (repeats <= 0)
        repeats = 16;

    auto info = get_device_info();
    print_device_info(info);
    warm_up_device();

    printf("\n=== Global Memory Bandwidth Probe =======\n");
    printf("size range: 2^%d ~ 2^%d floats  repeats=%d\n\n", min_pow2, max_pow2, repeats);

    // Buffers (reused across sweep to avoid realloc)
    const int max_elems = 1 << max_pow2;
    DeviceBuffer<float> d_src(max_elems);
    DeviceBuffer<float> d_dst(max_elems);
    DeviceBuffer<float> d_dummy(1);

    std::vector<float> h_src(max_elems, 1.0f);
    d_src.upload(h_src.data());

    for (int p = min_pow2; p <= max_pow2; ++p)
    {
        int num_elements = 1 << p;

        // Read bandwidth (float4 vectorized)
        run_bw_test(info, "read_bw_vec4", num_elements, repeats, read_bw, sizeof(float),
                    d_src.get(), d_dummy.get());

        // Write bandwidth
        run_bw_test(info, "write_bw", num_elements, repeats, write_bw, sizeof(float), d_dst.get());

        // Copy bandwidth (read + write)
        run_bw_test(info, "copy_bw", num_elements, repeats, copy_bw, 2 * sizeof(float), d_src.get(),
                    d_dst.get());

        CUDA_CHECK(cudaDeviceSynchronize());
        printf("\n");
    }

    printf("Done.\n");
    return 0;
}

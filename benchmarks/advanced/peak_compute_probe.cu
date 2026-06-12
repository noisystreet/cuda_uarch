// SPDX-License-Identifier: MIT
/**
 * peak_compute_probe.cu — 峰值算力测量
 *
 * 测试内容：
 *   1. CUDA Core FP32 FMA
 *   2. CUDA Core FP64 FMA  (Ada 上 rate 极低)
 *   3. CUDA Core INT32 ADD
 *   4. Tensor Core FP16 MMA (WMMA)
 *   5. Tensor Core INT8 MMA
 *   6. Tensor Core TF32 MMA (需 sm_80+)
 *
 * 理论峰值（RTX 4060 Ada Lovelace, sm_89, 1.5 GHz）：
 *   FP32:  24 SM × 128 CUDA Core × 2 FMA × 1.5 GHz = 9.216 TFLOPS
 *   FP16:  24 SM × 4 TC × 256 FMA × 2 × 1.5 GHz  = 73.728 TFLOPS
 *   INT8:  24 SM × 4 TC × 512 × 2 × 1.5 GHz       = 147.456 TOPS
 *   TF32:  24 SM × 4 TC × 128 × 2 × 1.5 GHz       = 36.864 TFLOPS
 *
 * Usage:
 *   ./peak_compute_probe [repeats]
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
//  CUDA Core — FP32 FMA 吞吐
// ════════════════════════════════════════════════════════════════════════════

__global__ void peak_fp32_fma(const float *__restrict__ input, float *__restrict__ output,
                              int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float a = 1.0000001f;
    float b = 0.9999999f;

    float r0 = input[tid % 1024];
    float r1 = r0, r2 = r0, r3 = r0;
    float r4 = r0, r5 = r0, r6 = r0, r7 = r0;
    float r8 = r0, r9 = r0, r10 = r0, r11 = r0;
    float r12 = r0, r13 = r0, r14 = r0, r15 = r0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
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
    if (tid % stride == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  CUDA Core — FP64 FMA 吞吐
// ════════════════════════════════════════════════════════════════════════════

__global__ void peak_fp64_fma(const double *__restrict__ input, double *__restrict__ output,
                              int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    double a = 1.0000001;
    double b = 0.9999999;

    double r0 = input[tid % 1024];
    double r1 = r0, r2 = r0, r3 = r0;
    double r4 = r0, r5 = r0, r6 = r0, r7 = r0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 = fma(r0, a, b);
        r1 = fma(r1, a, b);
        r2 = fma(r2, a, b);
        r3 = fma(r3, a, b);
        r4 = fma(r4, a, b);
        r5 = fma(r5, a, b);
        r6 = fma(r6, a, b);
        r7 = fma(r7, a, b);
    }

    double sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    if (tid % stride == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  CUDA Core — INT32 ADD 吞吐
// ════════════════════════════════════════════════════════════════════════════

__global__ void peak_int32_add(const int *__restrict__ input, int *__restrict__ output,
                               int iterations)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    int r0 = input[tid % 1024];
    int r1 = r0, r2 = r0, r3 = r0;
    int r4 = r0, r5 = r0, r6 = r0, r7 = r0;
    int r8 = r0, r9 = r0, r10 = r0, r11 = r0;
    int r12 = r0, r13 = r0, r14 = r0, r15 = r0;

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        r0 += 1;
        r1 += 1;
        r2 += 1;
        r3 += 1;
        r4 += 1;
        r5 += 1;
        r6 += 1;
        r7 += 1;
        r8 += 1;
        r9 += 1;
        r10 += 1;
        r11 += 1;
        r12 += 1;
        r13 += 1;
        r14 += 1;
        r15 += 1;
    }

    int sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    if (tid % stride == 0)
        *output = sum;
}

// ════════════════════════════════════════════════════════════════════════════
//  Tensor Core — FP16 MMA
// ════════════════════════════════════════════════════════════════════════════

__global__ void peak_tc_fp16(float *__restrict__ output, int iterations)
{
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fragment<accumulator, 16, 16, 16, float> d_frag;

    fill_fragment(a_frag, __half(1.0));
    fill_fragment(b_frag, __half(1.0));
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Write fragment element directly (avoid store_matrix_sync to local mem)
        float *frag_ptr = reinterpret_cast<float *>(&d_frag);
        *output = frag_ptr[0];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Tensor Core — INT8 MMA
// ════════════════════════════════════════════════════════════════════════════

__global__ void peak_tc_int8(int *__restrict__ output, int iterations)
{
    fragment<matrix_a, 16, 16, 16, signed char, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, signed char, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int> c_frag;
    fragment<accumulator, 16, 16, 16, int> d_frag;

    fill_fragment(a_frag, (signed char)1);
    fill_fragment(b_frag, (signed char)1);
    fill_fragment(c_frag, 0);
    fill_fragment(d_frag, 0);

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        int *frag_ptr = reinterpret_cast<int *>(&d_frag);
        *output = frag_ptr[0];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Tensor Core — TF32 MMA
// ════════════════════════════════════════════════════════════════════════════

__global__ void peak_tc_tf32(float *__restrict__ output, int iterations)
{
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;
    fragment<accumulator, 16, 16, 8, float> d_frag;

    fill_fragment(a_frag, 1.0f);
    fill_fragment(b_frag, 1.0f);
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float *frag_ptr = reinterpret_cast<float *>(&d_frag);
        *output = frag_ptr[0];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  CUDA Core 测试驱动
// ════════════════════════════════════════════════════════════════════════════

void run_fp32_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;
    const int num_items = threads * blocks;

    DeviceBuffer<float> d_in(num_items);
    DeviceBuffer<float> d_out(1);
    std::vector<float> h_in(num_items, 1.0f);
    d_in.upload(h_in.data());

    peak_fp32_fma<<<1, 32>>>(d_in.get(), d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        peak_fp32_fma<<<blocks, threads>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_fmas = static_cast<double>(blocks) * threads * iterations * 16;
    double total_ops = total_fmas * 2;
    double tflops = total_ops / (stats.median_us * 1e-6) / 1e12;

    char label[128];
    snprintf(label, sizeof(label), "cuda_core_fp32_fma iter=%d", iterations);
    print_csv(label, samples);

    double theoretical = 24.0 * 128 * 2 * 1.5 / 1000.0; // 9.216 TFLOPS
    double util_pct = (tflops / theoretical) * 100.0;
    printf("    -> FP32: %.2f TFLOPS/s  (理论峰值 %.2f TFLOPS, 利用率 %.1f%%)\n", tflops,
           theoretical, util_pct);
}

void run_fp64_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;
    const int num_items = threads * blocks;

    DeviceBuffer<double> d_in(num_items);
    DeviceBuffer<double> d_out(1);
    std::vector<double> h_in(num_items, 1.0);
    d_in.upload(h_in.data());

    peak_fp64_fma<<<1, 32>>>(d_in.get(), d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        peak_fp64_fma<<<blocks, threads>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_fmas = static_cast<double>(blocks) * threads * iterations * 8;
    double total_ops = total_fmas * 2;
    double tflops = total_ops / (stats.median_us * 1e-6) / 1e12;

    char label[128];
    snprintf(label, sizeof(label), "cuda_core_fp64_fma iter=%d", iterations);
    print_csv(label, samples);

    double theoretical = 24.0 * 128 * 2 * 1.5 / 32.0 / 1000.0; // ~0.288 TFLOPS
    double util_pct = (tflops / theoretical) * 100.0;
    printf("    -> FP64: %.4f TFLOPS/s  (理论峰值 %.3f TFLOPS, 利用率 %.1f%%)\n", tflops,
           theoretical, util_pct);
}

void run_int32_test(const DeviceInfo &info, int iterations, int repeats)
{
    const int threads = 256;
    const int blocks = info.sm_count * 4;
    const int num_items = threads * blocks;

    DeviceBuffer<int> d_in(num_items);
    DeviceBuffer<int> d_out(1);
    std::vector<int> h_in(num_items, 1);
    d_in.upload(h_in.data());

    peak_int32_add<<<1, 32>>>(d_in.get(), d_out.get(), 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        peak_int32_add<<<blocks, threads>>>(d_in.get(), d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_ops = static_cast<double>(blocks) * threads * iterations * 16;
    double tops = total_ops / (stats.median_us * 1e-6) / 1e12;

    char label[128];
    snprintf(label, sizeof(label), "cuda_core_int32_add iter=%d", iterations);
    print_csv(label, samples);

    double theoretical = 24.0 * 128 * 1.5 / 1000.0; // INT32: 1 op/cycle/core = ~4.6 TOPS
    double util_pct = (tops / theoretical) * 100.0;
    printf("    -> INT32: %.2f TOPS/s  (理论峰值 %.2f TOPS, 利用率 %.1f%%)\n", tops, theoretical,
           util_pct);
}

// ════════════════════════════════════════════════════════════════════════════
//  Tensor Core 测试驱动
// ════════════════════════════════════════════════════════════════════════════

void run_tc_fp16_test(const DeviceInfo &info, int iterations, int repeats)
{
    // FP16 TC: 每 MMA = 256 FMA (512 FLOPs), 每 SM 4 TC
    // 理论: 24 × 4 × 256 × 2 × 1.5 GHz = 73.728 TFLOPS
    double theoretical = 24.0 * 4 * 256 * 2 * 1.5 / 1000.0;

    // TC test with 1 warp per block
    const int threads = 32;
    const int blocks = info.sm_count;
    DeviceBuffer<float> d_out(1);

    peak_tc_fp16<<<1, 32>>>(d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        peak_tc_fp16<<<blocks, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_threads = static_cast<double>(blocks) * threads;
    double total_ops = total_threads * iterations * 512;
    double tflops = total_ops / (stats.median_us * 1e-6) / 1e12;
    double util_pct = (tflops / theoretical) * 100.0;

    char label[128];
    snprintf(label, sizeof(label), "tensor_core_fp16_mma iter=%d", iterations);
    print_csv(label, samples);
    printf("    -> FP16 TC: %.2f TFLOPS/s  (理论峰值 %.2f TFLOPS, 利用率 %.1f%%)\n", tflops,
           theoretical, util_pct);
}

void run_tc_int8_test(const DeviceInfo &info, int iterations, int repeats)
{
    // INT8 TC: 每 MMA = 512 INT8 ops, 每 SM 4 TC
    // 理论: 24 × 4 × 512 × 2 × 1.5 GHz = 147.456 TOPS
    double theoretical = 24.0 * 4 * 512 * 2 * 1.5 / 1000.0;

    const int threads = 64;
    const int blocks = info.sm_count;
    DeviceBuffer<int> d_out(1);

    peak_tc_int8<<<1, 32>>>(d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        peak_tc_int8<<<blocks, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_threads = static_cast<double>(blocks) * threads;
    double total_ops = total_threads * iterations * 512 * 2;
    double tops = total_ops / (stats.median_us * 1e-6) / 1e12;
    double util_pct = (tops / theoretical) * 100.0;

    char label[128];
    snprintf(label, sizeof(label), "tensor_core_int8_mma iter=%d", iterations);
    print_csv(label, samples);
    printf("    -> INT8 TC: %.2f TOPS/s  (理论峰值 %.2f TOPS, 利用率 %.1f%%)\n", tops, theoretical,
           util_pct);
}

void run_tc_tf32_test(const DeviceInfo &info, int iterations, int repeats)
{
    // TF32 TC: 每 MMA = 128 FMA (256 FLOPs), 每 SM 4 TC
    // 理论: 24 × 4 × 128 × 2 × 1.5 GHz = 36.864 TFLOPS
    double theoretical = 24.0 * 4 * 128 * 2 * 1.5 / 1000.0;

    const int threads = 64;
    const int blocks = info.sm_count;
    DeviceBuffer<float> d_out(1);

    peak_tc_tf32<<<1, 32>>>(d_out.get(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r)
    {
        GpuTimer timer;
        timer.start();
        peak_tc_tf32<<<blocks, threads>>>(d_out.get(), iterations);
        timer.stop();
        samples.push_back(timer.elapsed_us());
    }

    auto stats = compute_stats(samples);
    double total_threads = static_cast<double>(blocks) * threads;
    double total_ops = total_threads * iterations * 128 * 2;
    double tflops = total_ops / (stats.median_us * 1e-6) / 1e12;
    double util_pct = (tflops / theoretical) * 100.0;

    char label[128];
    snprintf(label, sizeof(label), "tensor_core_tf32_mma iter=%d", iterations);
    print_csv(label, samples);
    printf("    -> TF32 TC: %.2f TFLOPS/s  (理论峰值 %.2f TFLOPS, 利用率 %.1f%%)\n", tflops,
           theoretical, util_pct);
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

    const int iterations = 100000;

    printf("==================================================\n");
    printf("  峰值算力测试\n");
    printf("  GPU: %s (sm_%d.%d)\n", info.arch_name().data(), info.major, info.minor);
    printf("==================================================\n\n");

    printf("─── CUDA Core ───\n");
    run_fp32_test(info, iterations, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_fp64_test(info, iterations / 10, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_int32_test(info, iterations, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n─── Tensor Core ───\n");
    run_tc_fp16_test(info, 100, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_tc_int8_test(info, iterations / 1000, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_tc_tf32_test(info, iterations / 1000, repeats);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nDone.\n");
    return 0;
}

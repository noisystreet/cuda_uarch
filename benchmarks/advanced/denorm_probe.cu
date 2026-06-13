// SPDX-License-Identifier: MIT
/**
 * denorm_probe.cu — Tensor Core 非规格化数 (Denormal/Subnormal) 行为探测
 *
 * 目的：检测 Tensor Core 在处理非规格化浮点数时是否将其刷新为零 (FTZ)，
 *       以及 CUDA Core 在相同条件下的行为作为对照。
 *
 * 背景：
 *   - 非规格化数（denormal/subnormal）是指指数位全 0、尾数非 0 的数，
 *     用于表示接近零的极小值。
 *   - NVIDIA GPU 默认启用 FTZ (Flush-To-Zero)，可通过 --ftz=false 禁用。
 *   - Tensor Core 的 FTZ 行为可能与 CUDA Core 不同，且不受编译器选项控制。
 *
 * 方法：
 *   1. 构造已知的非规格化 FP16 值
 *   2. 通过 WMMA 执行 FP16 矩阵乘法（使用非规格化输入）
 *   3. 检查输出是否保留非规格化结果
 *   4. 与 CUDA Core FP16 运算对比
 *   5. 同样测试 TF32 精度
 *
 * Usage:
 *   ./denorm_probe
 */

#include "config.h"
#include "timer.h"
#include "utils.h"

#include <mma.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace uarch;
using namespace nvcuda::wmma;

// ════════════════════════════════════════════════════════════════════════════
//  辅助函数 — 构造/检测非规格化数
// ════════════════════════════════════════════════════════════════════════════

/// FP16 位操作联合体
union fp16_bits
{
    unsigned short bits;
    __half value;

    static fp16_bits from_bits(unsigned short b)
    {
        fp16_bits u;
        u.bits = b;
        return u;
    }
};

/// 检查 FP16 是否为非规格化数（指数位全 0 且尾数非 0）
static bool is_denorm_fp16(__half v)
{
    fp16_bits u;
    u.value = v;
    unsigned short exp = (u.bits >> 10) & 0x1F;
    unsigned short mant = u.bits & 0x03FF;
    return (exp == 0 && mant != 0);
}

/// 检查 FP16 是否为零（正零或负零）
static bool is_zero_fp16(__half v)
{
    fp16_bits u;
    u.value = v;
    return (u.bits == 0 || u.bits == 0x8000);
}

/// 检查 FP16 是否为规格化正数
static bool is_normal_pos_fp16(__half v)
{
    fp16_bits u;
    u.value = v;
    unsigned short exp = (u.bits >> 10) & 0x1F;
    return (exp > 0 && exp < 31 && !(u.bits & 0x8000));
}

/// FP32 位操作联合体
union fp32_bits
{
    unsigned int bits;
    float value;

    static fp32_bits from_bits(unsigned int b)
    {
        fp32_bits u;
        u.bits = b;
        return u;
    }
};

static bool is_denorm_fp32(float v)
{
    fp32_bits u;
    u.value = v;
    unsigned int exp = (u.bits >> 23) & 0xFF;
    unsigned int mant = u.bits & 0x7FFFFF;
    return (exp == 0 && mant != 0);
}

// ─── 打印 FP16 位表示 ─────────────────────────────────────────────────────
static void print_fp16_bits(const char *label, __half v)
{
    fp16_bits u;
    u.value = v;
    unsigned short sign = (u.bits >> 15) & 1;
    unsigned short exp = (u.bits >> 10) & 0x1F;
    unsigned short mant = u.bits & 0x03FF;
    printf("  %-40s bits=0x%04X  sign=%d  exp=%d (bias=15)  mant=0x%03X  ", label, u.bits, sign,
           exp, mant);
    printf("value=%g", __half2float(v));
    if (exp == 0 && mant == 0)
        printf(" [zero]");
    else if (exp == 0)
        printf(" [denorm]");
    else if (exp == 31)
        printf(" [inf/nan]");
    else
        printf(" [normal]");
    printf("\n");
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 1: CUDA Core FP16 乘法 — 对照基准
// ════════════════════════════════════════════════════════════════════════════

__global__ void cuda_core_fp16_mul(__half *__restrict__ out, __half a, __half b)
{
    out[0] = a * b;
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 2: Tensor Core FP16 WMMA — 非规格化输入
// ════════════════════════════════════════════════════════════════════════════

/// 用 WMMA 计算 D = A × B + C，其中 A/B 可能是非规格化数
/// 输出 d_frag[0]（第一个累加器元素）到 output
__global__ void tc_fp16_mma_denorm(__half *__restrict__ output, __half a_val, __half b_val,
                                   int repeat)
{
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fragment<accumulator, 16, 16, 16, float> d_frag;

    // 用给定值填充整个 tile
    fill_fragment(a_frag, a_val);
    fill_fragment(b_frag, b_val);
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

    for (int i = 0; i < repeat; ++i)
    {
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    // 将结果写回 — 直接访问 fragment 元素避免 store_matrix_sync 本地内存问题
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = __float2half(d_frag.x[0]);
        output[1] = __float2half(d_frag.x[1]);
        output[2] = __float2half(d_frag.x[2]);
        output[3] = __float2half(d_frag.x[3]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  实验 3: Tensor Core TF32 WMMA — 非规格化输入
// ════════════════════════════════════════════════════════════════════════════

__global__ void tc_tf32_mma_denorm(float *__restrict__ output, float a_val, float b_val, int repeat)
{
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
    fragment<accumulator, 16, 16, 8, float> c_frag;
    fragment<accumulator, 16, 16, 8, float> d_frag;

    fill_fragment(a_frag, a_val);
    fill_fragment(b_frag, b_val);
    fill_fragment(c_frag, 0.0f);
    fill_fragment(d_frag, 0.0f);

    for (int i = 0; i < repeat; ++i)
    {
        mma_sync(d_frag, a_frag, b_frag, c_frag);
        c_frag = d_frag;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = d_frag.x[0];
        output[1] = d_frag.x[1];
        output[2] = d_frag.x[2];
        output[3] = d_frag.x[3];
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  辅助：构造测试值
// ════════════════════════════════════════════════════════════════════════════

/// 构造一个 FP16 非规格化数
/// 最小的正非规格化数: bits=0x0001 = 2^-24 ≈ 5.96e-8
/// 最大的非规格化数: bits=0x03FF = (1 - 2^-10) * 2^-14 ≈ 6.1e-5
static __half make_denorm_fp16(unsigned short mantissa)
{
    // 指数全 0，尾数 = mantissa（10 bit）
    return fp16_bits::from_bits(mantissa & 0x03FF).value;
}

/// 构造 FP16 规格化最小正数: bits=0x0400 = 2^-14 ≈ 6.1e-5
static __half make_min_normal_fp16()
{
    return fp16_bits::from_bits(0x0400).value; // exp=1, mant=0
}

/// 构造 FP32 非规格化数（指数全 0，尾数非 0）
static float make_denorm_fp32(unsigned int mantissa)
{
    return fp32_bits::from_bits(mantissa & 0x7FFFFF).value;
}

// ════════════════════════════════════════════════════════════════════════════
//  测试执行
// ════════════════════════════════════════════════════════════════════════════

void probe_cuda_core_fp16()
{
    printf("\n═══ 1. CUDA Core FP16 乘法 — 对照基准 ═══\n");
    printf("  FTZ 编译选项: --ftz=true (我们使用 --use_fast_math)\n\n");

    struct TestCase
    {
        const char *label;
        __half a;
        __half b;
    };

    __half normal_val = make_min_normal_fp16();     // 最小规格化正数
    __half denorm_small = make_denorm_fp16(0x0001); // 最小正非规格化数
    __half denorm_large = make_denorm_fp16(0x03FF); // 最大非规格化数

    TestCase tests[] = {
        {"normal × normal", normal_val, normal_val},
        {"normal × denorm(small)", normal_val, denorm_small},
        {"denorm(small) × denorm(small)", denorm_small, denorm_small},
        {"denorm(large) × denorm(large)", denorm_large, denorm_large},
        {"normal × denorm(large)", normal_val, denorm_large},
        {"denorm(small) × zero", denorm_small, __half(0.0)},
        {"zero × zero", __half(0.0), __half(0.0)},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    DeviceBuffer<__half> d_out(4);

    for (int i = 0; i < num_tests; ++i)
    {
        auto &t = tests[i];

        cuda_core_fp16_mul<<<1, 32>>>(d_out.get(), t.a, t.b);
        CUDA_CHECK(cudaDeviceSynchronize());

        __half h_out[4];
        d_out.download(h_out);

        printf("  Test %d: %s\n", i + 1, t.label);
        print_fp16_bits("  A input", t.a);
        print_fp16_bits("  B input", t.b);
        printf("  A * B = ");
        print_fp16_bits("", h_out[0]);

        if (is_denorm_fp16(h_out[0]))
            printf("  >>> 结果保留了非规格化数 ✓\n");
        else if (is_zero_fp16(h_out[0]))
            printf("  >>> 结果被刷新为零 (FTZ) ✗\n");
        else if (is_normal_pos_fp16(h_out[0]))
            printf("  >>> 结果是规格化数\n");
        printf("\n");
    }
}

void probe_tensor_core_fp16()
{
    printf("\n═══ 2. Tensor Core FP16 WMMA — 非规格化输入 ═══\n");
    printf("  使用 mma_sync (m16n16k16) 测试 FP16 Tensor Core\n");
    printf("  填充整个 tile 用同一个值，检查第一个累加器输出\n\n");

    struct TcTestCase
    {
        const char *label;
        __half tile_val; // 用于填充 A 和 B 矩阵
    };

    __half normal_val = make_min_normal_fp16();
    __half denorm_small = make_denorm_fp16(0x0001);
    __half denorm_med = make_denorm_fp16(0x0200);
    __half denorm_large = make_denorm_fp16(0x03FF);

    TcTestCase tests[] = {
        {"tile all = min normal", normal_val},
        {"tile all = denorm(0x0001, min)", denorm_small},
        {"tile all = denorm(0x0200, mid)", denorm_med},
        {"tile all = denorm(0x03FF, max)", denorm_large},
        {"tile all = zero", __half(0.0)},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    DeviceBuffer<__half> d_out(4);

    for (int i = 0; i < num_tests; ++i)
    {
        auto &t = tests[i];

        // 预热
        tc_fp16_mma_denorm<<<1, 32>>>(d_out.get(), t.tile_val, t.tile_val, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 正式运行（重复 8 次 MMA 累积效应）
        tc_fp16_mma_denorm<<<1, 32>>>(d_out.get(), t.tile_val, t.tile_val, 8);
        CUDA_CHECK(cudaDeviceSynchronize());

        __half h_out[4];
        d_out.download(h_out);

        printf("  Test %d: %s\n", i + 1, t.label);
        print_fp16_bits("  Tile fill value (A&B)", t.tile_val);
        print_fp16_bits("  MMA result[0]", h_out[0]);
        print_fp16_bits("  MMA result[1]", h_out[1]);
        print_fp16_bits("  MMA result[2]", h_out[2]);
        print_fp16_bits("  MMA result[3]", h_out[3]);

        // 分析行为
        bool any_denorm = is_denorm_fp16(h_out[0]) || is_denorm_fp16(h_out[1]) ||
                          is_denorm_fp16(h_out[2]) || is_denorm_fp16(h_out[3]);
        bool all_zero = is_zero_fp16(h_out[0]) && is_zero_fp16(h_out[1]) &&
                        is_zero_fp16(h_out[2]) && is_zero_fp16(h_out[3]);

        if (any_denorm)
            printf("  >>> Tensor Core 保留了非规格化数 ✓\n");
        else if (all_zero && !is_zero_fp16(t.tile_val))
            printf("  >>> Tensor Core 将非规格化输入刷新为零 (FTZ) ✗\n");
        else if (!is_zero_fp16(t.tile_val))
            printf("  >>> 结果全为零，但输入非零 — FTZ 发生 ✗\n");
        else
            printf("  >>> 全零输入 → 全零输出 ✓\n");

        printf("  ---- 预期: denorm × denorm = denorm（若非规格化数受支持）\n\n");
    }
}

void probe_tensor_core_tf32()
{
    printf("\n═══ 3. Tensor Core TF32 WMMA — 非规格化输入 ═══\n");
    printf("  TF32 精度: 8 位指数（同 FP32），10 位尾数（同 FP16）\n");
    printf("  非规格化 TF32 定义为指数全 0\n\n");

    // TF32 的 denormal 区段与 FP32 相同（8 位指数）
    float denorm_val = make_denorm_fp32(0x000001); // 最小正 denorm FP32
    float small_norm = 1.17549e-38f;               // 最小正规格化 FP32
    float one_val = 1.0f;

    struct
    {
        const char *label;
        float val;
    } tests[] = {
        {"tile all = min FP32 denorm (0x000001)", denorm_val},
        {"tile all = min normal FP32", small_norm},
        {"tile all = 1.0 (normal baseline)", one_val},
        {"tile all = 0.0", 0.0f},
    };
    int num_tests = sizeof(tests) / sizeof(tests[0]);

    DeviceBuffer<float> d_out(4);

    for (int i = 0; i < num_tests; ++i)
    {
        auto &t = tests[i];

        tc_tf32_mma_denorm<<<1, 32>>>(d_out.get(), t.val, t.val, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_out[4];
        d_out.download(h_out);

        printf("  Test %d: %s\n", i + 1, t.label);
        fp32_bits u_in;
        u_in.value = t.val;
        printf("  Input value = %g (bits=0x%08X)\n", t.val, u_in.bits);

        fp32_bits u;
        for (int j = 0; j < 4; ++j)
        {
            u.value = h_out[j];
            printf("  MMA result[%d] = %g  (bits=0x%08X)%s\n", j, h_out[j], u.bits,
                   is_denorm_fp32(h_out[j]) ? " [denorm]"
                                            : (h_out[j] == 0.0f ? " [zero]" : " [normal]"));
        }

        bool any_denorm = false;
        bool all_zero = true;
        for (int j = 0; j < 4; ++j)
        {
            if (is_denorm_fp32(h_out[j]))
                any_denorm = true;
            if (h_out[j] != 0.0f)
                all_zero = false;
        }

        if (any_denorm)
            printf("  >>> Tensor Core (TF32) 保留了非规格化数 ✓\n");
        else if (all_zero && t.val != 0.0f)
            printf("  >>> Tensor Core (TF32) 将非零输入刷新为零 ✗\n");
        else
            printf("  >>> 全零输出（符合预期）\n");
        printf("\n");
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  额外：编译器 FTZ 设置验证
// ════════════════════════════════════════════════════════════════════════════

__global__ void ftz_test_kernel(float *c, const float *a, const float *b)
{
    *c = fmaf(*a, *b, 0.0f);
}

void probe_ftz_setting()
{
    printf("\n═══ 附录: 编译器 FTZ 设置验证 ═══\n");

#ifdef __CUDA_FLUSH_DENORMALS_TO_ZERO__
    printf("  __CUDA_FLUSH_DENORMALS_TO_ZERO__ 已定义\n");
    printf("  编译器启用全局 FTZ（来自 --ftz=true / --use_fast_math）\n");
#else
    printf("  __CUDA_FLUSH_DENORMALS_TO_ZERO__ 未定义\n");
    printf("  编译器未启用全局 FTZ\n");
#endif

    // 通过 CUDA Core FP32 运算验证
    float denorm_a = make_denorm_fp32(0x000001);
    float denorm_b = make_denorm_fp32(0x000002);

    float *d_a;
    cudaMalloc(&d_a, sizeof(float));
    float *d_b;
    cudaMalloc(&d_b, sizeof(float));
    float *d_c;
    cudaMalloc(&d_c, sizeof(float));

    cudaMemcpy(d_a, &denorm_a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &denorm_b, sizeof(float), cudaMemcpyHostToDevice);

    // 执行 FP32 乘加（使用默认 FMA）
    ftz_test_kernel<<<1, 1>>>(d_c, d_a, d_b);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_c;
    cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    fp32_bits u_a, u_b;
    u_a.value = denorm_a;
    u_b.value = denorm_b;
    printf("  Denorm input:  %g (0x%08X) × %g (0x%08X)\n", denorm_a, u_a.bits, denorm_b, u_b.bits);
    printf("  FMA result:    %g", h_c);
    if (h_c == 0.0f)
        printf(" → FTZ 生效（刷新为零）\n");
    else if (is_denorm_fp32(h_c))
        printf(" → 非规格化数保留\n");
    else
        printf(" → 规格化数\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ════════════════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════════════════

int main(int /*argc*/, char * /*argv*/[])
{
    auto info = get_device_info();
    print_device_info(info);
    warm_up_device();

    printf("==================================================\n");
    printf("  Tensor Core / CUDA Core 非规格化数行为探测\n");
    printf("  硬件: %s (sm_%d.%d)\n", info.arch_name().data(), info.major, info.minor);
    printf("==================================================\n");

    printf("\n背景：非规格化数 (denormal/subnormal) 是指数位全 0、\n");
    printf("尾数非 0 的浮点数，用于填补接近零的精度损失。\n");
    printf("NVIDIA GPU 默认启用 FTZ (Flush-To-Zero)。\n");
    printf("本实验编译使用 --use_fast_math (含 --ftz=true)。\n");

    probe_ftz_setting();

    probe_cuda_core_fp16();

    probe_tensor_core_fp16();

    probe_tensor_core_tf32();

    printf("\n═══ 结论 ═══\n");
    printf("  如果 Tensor Core 结果中出现非规格化数:\n");
    printf("    → Tensor Core 保留非规格化数，与 CUDA Core 一致\n");
    printf("  如果非零输入产生全零结果:\n");
    printf("    → Tensor Core 存在额外的 FTZ 行为\n");
    printf("  注意: --use_fast_math 全局启用 FTZ，\n");
    printf("  如需精确测量需编译时不带 --ftz=true\n\n");

    printf("Done.\n");
    return 0;
}

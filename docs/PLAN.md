# GPU 软件栈逆向分析规划

## 一、项目背景与目标

通过构造 microbenchmark 对 NVIDIA GPU 的软件栈（CUDA Runtime → Driver → PTX → SASS）进行逆向分析，
推断 GPU 微架构的底层细节，包括但不限于：指令发射宽度、流水线结构、缓存层次与延迟、调度器策略、内存子系统等。

---

## 二、整体路线图

```
Phase 1: 环境搭建与工具链           ✅ 完成
    ↓
Phase 2: 基准测试框架               ✅ 完成
    ↓
Phase 3: 指令级微架构探测           ✅ 完成
    ↓
Phase 4: 内存子系统分析              ✅ 完成
    ↓
Phase 5: 调度器与并行模型            🔄 部分完成
    ↓
Phase 6: 高级专题                    🔄 部分完成
    ↓
Phase 7: 综合分析报告               ✅ 完成 (docs/ANALYSIS.md)
```

---

## 三、各阶段详细规划与完成状态

### Phase 1: 环境搭建与工具链 (✅ 完成)

| 任务 | 说明 | 状态 |
|------|------|:----:|
| 硬件信息采集 | `nvidia-smi -a` 获取 GPU 型号、显存、驱动版本等 | ✅ |
| CUDA 工具链 | 确认 `nvcc`、`cuobjdump`、`nvdisasm` 版本 | ✅ |
| 性能分析工具 | 配置 Nsight Compute (`ncu`)、Nsight Systems (`nsys`) | ✅ |
| 二进制工具 | `nvdisasm` 反汇编 SASS, `cuobjdump` 提取 cubin | ✅ |
| Python 分析栈 | pandas / numpy / matplotlib 用于数据后处理 | ✅ |

**产出:** `tools/check_env.sh` + `tools/plot_results.py` + `cmake/DetectCUDAArch.cmake`

---

### Phase 2: 基准测试框架 (✅ 完成)

```
benchmarks/
├── common/          # config.h, timer.h, utils.h (公共库)
├── instruction/     # 指令级探针 (2 probes)
├── memory/          # 内存层次探针 (4 probes)
├── scheduler/       # 调度器探针 (1 probe)
└── advanced/        # 高级专题 (4 probes)
```

关键设计要点：
- 使用 `cudaEvent` 进行 GPU 端高精度计时 ✅
- 控制变量法：每次只改变一个参数 ✅
- 统计有效性：每次实验重复多次，取中位数 ✅
- 防 DCE：所有 kernel 写回结果防止死代码消除 ✅
- 预热：计时前发射一次短运行 kernel ✅
- 结果输出：`RESULT,<label>,<median>,<mean>,<min>,<max>,<stddev>,<count>` ✅

**产出:** CMake 构建系统 + `AGENTS.md` 规则约束

---

### Phase 3: 指令级微架构探测 (✅ 完成)

#### 3.1 指令发射宽度 (已覆盖)
- 通过 ILP sweep 推断 warp scheduler 发射宽度 ✅
- **发现:** Ada Lovelace 每 SM 128 个 FP32 Core，ILP=16 时达 98% 理论峰值

#### 3.2 指令延迟 (✅ 完成)
- 依赖链法测量 4 种指令延迟
- ✅ FADD: 12.88 ns (~19.3 cycles)
- ✅ FMUL: 12.89 ns (~19.3 cycles)
- ✅ FFMA: 12.89 ns (~19.3 cycles)
- ✅ IADD: < 0.1 ns (编译器优化)

#### 3.3 指令吞吐 (✅ 完成)
- FFMA ILP sweep (unroll=1/2/4/8/16) ✅
- 峰值吞吐: 2998 ops/cycle (ILP=16) = 98% 理论峰值 ✅

#### 3.4 SASS 反汇编分析 (🔲 待完成)
- 对关键 kernel 使用 `nvdisasm` 反汇编
- 分析 SASS 指令编码格式
- 推断 functional unit 布局

---

### Phase 4: 内存子系统分析 (✅ 完成)

#### 4.1 全局内存带宽与延迟 (✅ 完成)
- **延迟:** Pointer chasing 测得 162.9 ns (1 warp/SM) ✅
- **带宽:** 连续访问读取 218 GB/s (85% 理论峰值) ✅
- **写带宽:** 212 GB/s ✅
- **拷贝带宽:** 204 GB/s ✅

#### 4.2 共享内存 (✅ 完成)
- **Bank conflict stride 扫描:** stride 1~33 + 64/128/256 ✅
- **广播机制:** 同地址读取 1852 GB/s (2.2× 加速) ✅
- **多 warp 并发:** 1/2/4/8 warp 下 conflict 影响 ✅
- **发现:** Ada bank conflict 影响仅 ~1.7%，远小于理论值

#### 4.3 L1/L2 缓存层次 (✅ 完成)
- **L1 大小:** 128 KB (延迟跳变点) ✅
- **L1 延迟:** ~41 ns (~62 cycles) ✅
- **L2 大小:** 32 MB (与 nvidia-smi 一致) ✅
- **L2 延迟:** ~133 ns (~200 cycles) ✅
- **Cache line:** 128 B ✅
- **L1 关联度:** 🔲 方法需改进

#### 4.4 常量内存与纹理内存 (🔲 待完成)
- 常量内存广播机制探测
- 纹理内存缓存行为分析

---

### Phase 5: 调度器与并行模型 (🔄 部分完成)

#### 5.1 Warp 调度策略 (✅ 完成)
- **分歧 (divergence):** 分歧导致 ~14% 吞吐损失 ✅
- **公平性:** 轮转调度 (round-robin)，无优先级 ✅
- **并发 warp 上限:** 48 (硬件限制) ✅

#### 5.2 Block 调度 (🔲 待完成)
- **SM 占用率:** 调整 block 大小和寄存器/共享内存使用量
- **block 分配策略:** 观察 block 在 SM 间的分布

#### 5.3 寄存器文件 (🔲 待完成)
- **寄存器数量:** 通过 `__launch_bounds__` 限制寄存器使用量
- **寄存器重命名:** 通过寄存器依赖链探测重命名能力

---

### Phase 6: 高级专题 (🔄 部分完成)

#### 6.1 Tensor Core 分析 (🔄 部分完成)
- ✅ FP16/INT8/TF32 峰值吞吐 (WMMA)
- ✅ 非规格化数行为 (denorm FTZ) 阈值发现
- ✅ 数值精度分析 (FP32 内部累加器)
- 🔲 Tile size 对比 (16x16x16 vs 8x8x4 等)
- 🔲 Tensor Core vs CUDA Core 切换开销

#### 6.2 混合精度与特殊指令 (🔲 待完成)
- __shfl / __shfl_down / __shfl_xor 等 warp shuffle 指令延迟
- __syncwarp / __ballot_sync 等同步指令开销

#### 6.3 功耗与频率 (🔲 待完成)
- 不同负载下的 GPU 频率变化（nvidia-smi dmon）
- 功耗-性能权衡分析

#### 其他高级专题 (已完成)
- ✅ 特殊函数单元 (SFU) 吞吐: cosf/sinf/expf/logf/rsqrtf/tanhf/powf
- ✅ 共享内存 Bank Conflict 分析
- ✅ 非规格化数行为探测

---

### Phase 7: 综合分析报告 (✅ 完成)

- **产出:** `docs/ANALYSIS.md` — 13 章完整分析报告 ✅
- 覆盖：指令延迟/吞吐、缓存层次、内存带宽/延迟、Warp 调度、峰值算力、
  SFU、Bank Conflict、非规格化数行为、数值精度 ✅
- 与文献交叉验证（Ada Lovelace 架构规格） ✅

---

## 四、项目结构

```
cuda_uarch/
├── CMakeLists.txt             # 根 CMake (C++20, CUDA)
├── Makefile                   # 便捷构建/运行命令
├── .clang-format              # 代码格式配置
├── .gitignore
├── AGENTS.md                  # AI Agent 指南（含约束规则）
├── PLAN.md                    # 本规划文件
├── cmake/
│   ├── CompilerOptions.cmake  # NVCC / CXX 编译选项
│   └── DetectCUDAArch.cmake   # 自动探测本地 GPU compute capability
├── tools/
│   ├── check_env.sh           # 环境检查
│   └── plot_results.py        # CSV 结果可视化
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── common/
│   │   ├── config.h           # DeviceInfo, CUDA_CHECK, LaunchConfig
│   │   ├── timer.h            # GpuTimer (cudaEvent), HostTimer, SampleStats
│   │   ├── utils.h            # DeviceBuffer RAII, fill_random, warm_up
│   │   └── utils.cu           # (编译占位)
│   ├── instruction/
│   │   ├── CMakeLists.txt
│   │   ├── latency_probe.cu       # ✅ 指令延迟
│   │   └── throughput_probe.cu    # ✅ 指令吞吐与 ILP
│   ├── memory/
│   │   ├── CMakeLists.txt
│   │   ├── global_mem_latency.cu  # ✅ 全局内存延迟
│   │   ├── global_mem_bw.cu       # ✅ 全局内存带宽
│   │   ├── cache_size_probe.cu    # ✅ 缓存大小/行大小/关联度
│   │   └── shared_mem_bank.cu     # ✅ 共享内存 Bank Conflict
│   ├── scheduler/
│   │   ├── CMakeLists.txt
│   │   └── warp_scheduler_probe.cu # ✅ Warp 调度（分歧/并发/公平性）
│   └── advanced/
│       ├── CMakeLists.txt
│       ├── peak_compute_probe.cu   # ✅ CUDA Core + Tensor Core 峰值
│       ├── sfu_probe.cu            # ✅ 特殊函数单元 (SFU)
│       ├── denorm_probe.cu         # ✅ 非规格化数行为
│       └── precision_probe.cu      # ✅ 数值精度分析
├── build/                     # CMake 构建目录
├── docs/
│   ├── PLAN.md                # 本规划文件
│   └── ANALYSIS.md            # 13 章完整分析报告
├── data/
│   └── results/               # 实验结果 CSV
└── reports/
    └── figures/               # 图表输出
```

---

## 五、完成状态汇总

| 优先级 | 模块 | 说明 | 状态 |
|:-----:|------|------|:----:|
| P0 | Phase 1 + 2 | 环境搭建 + 基准框架 | ✅ |
| P0 | 指令延迟 + 吞吐 | FADD/FMUL/FFMA/IADD | ✅ |
| P0 | 全局内存延迟与带宽 | Pointer chasing + 连续访问 | ✅ |
| P1 | 缓存层次 | L1=128KB, L2=32MB, line=128B | ✅ |
| P1 | 共享内存 Bank Conflict | stride 扫描 + 广播 | ✅ |
| P1 | Warp 调度 | 分歧/并发/公平性 | ✅ |
| P1 | **常量内存广播机制** | 常量内存读取延迟验证 | 🔲 |
| P2 | 峰值算力 | CUDA Core + Tensor Core | ✅ |
| P2 | 特殊函数单元 (SFU) | 7 种 MUFU 指令 | ✅ |
| P2 | Tensor Core 数值精度 | FP16/TF32 精度对比 | ✅ |
| P2 | 非规格化数行为 | Denorm FTZ 阈值 | ✅ |
| P2 | **寄存器文件** | `__launch_bounds__` + 寄存器依赖链 | 🔲 |
| P2 | **Block 调度 / Occupancy** | 占用率 + block 分配策略 | 🔲 |
| P2 | **Warp Shuffle / 同步指令** | `__shfl_sync` / `__syncwarp` 等 | 🔲 |
| P2 | **SASS 反汇编** | `nvdisasm` 分析指令编码 | 🔲 |
| P2 | **原子操作** | `atomicAdd` 吞吐和延迟 | 🔲 |
| P3 | **功耗-频率曲线** | nvidia-smi dmon 监测 | 🔲 |
| P3 | **纹理/常量内存** | 纹理缓存行为 | 🔲 |
| P3 | **PCIe 带宽** | Host↔Device 传输 | 🔲 |
| P3 | **CUDA Stream 并发** | 多 stream 并发执行 | 🔲 |
| P3 | **UVM 统一内存** | page fault 和数据迁移 | 🔲 |

**图例:** ✅ = 已完成  🔄 = 部分完成  🔲 = 待完成

---

## 六、方法论原则

1. **控制变量法:** 每次实验只改变一个参数，其余保持恒定
2. **统计有效性:** 每次实验重复多次，取中位数而非平均值（抗 outlier）
3. **交叉验证:** 使用多种独立方法验证同一结论
4. **文献对照:** 与公开的 GPU 架构资料（NVIDIA 官方文档、论文）交叉比对
5. **渐进式深入:** 从宏观指标逐步细化到微观细节

---

## 七、参考资源

- NVIDIA CUDA Programming Guide
- NVIDIA Profiler/Nsight Compute 文档
- PTX ISA 文档
- 相关论文（如 Dissecting the NVIDIA Turing GPU Architecture via Microbenchmarking）
- Hot Chips 公开资料

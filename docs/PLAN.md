# GPU 软件栈逆向分析规划

## 一、项目背景与目标

通过构造 microbenchmark 对 NVIDIA GPU 的软件栈（CUDA Runtime → Driver → PTX → SASS）进行逆向分析，
推断 GPU 微架构的底层细节，包括但不限于：指令发射宽度、流水线结构、缓存层次与延迟、调度器策略、内存子系统等。

---

## 二、整体路线图

```
Phase 1: 环境搭建与工具链
    ↓
Phase 2: 基准测试框架
    ↓
Phase 3: 指令级微架构探测
    ↓
Phase 4: 内存子系统分析
    ↓
Phase 5: 调度器与并行模型
    ↓
Phase 6: 高级专题 (Tensor Core, 指令吞吐, 功耗)
    ↓
Phase 7: 综合分析报告
```

---

## 三、各阶段详细规划

### Phase 1: 环境搭建与工具链

| 任务 | 说明 |
|------|------|
| 硬件信息采集 | `nvidia-smi -a` 获取 GPU 型号、显存、驱动版本等 |
| CUDA 工具链 | 确认 `nvcc`、`cuobjdump`、`nvdisasm` 版本 |
| 性能分析工具 | 配置 Nsight Compute (`ncu`)、Nsight Systems (`nsys`) |
| 二进制工具 | `nvdisasm` 反汇编 SASS, `cuobjdump` 提取 cubin |
| Python 分析栈 | pandas / numpy / matplotlib 用于数据后处理 |

**产出:** 环境检查脚本 `tools/check_env.sh` + 工具链可用性确认。

---

### Phase 2: 基准测试框架

设计一个可复用的 microbenchmark 框架：

```
benchmarks/
├── common/          # 公共头文件、计时宏、预热逻辑
├── instruction/     # 指令级探针
├── memory/          # 内存层次探针
├── scheduler/       # 调度器探针
└── advanced/        # 高级专题
```

关键设计要点：
- 使用 `clock64()` 或 `nvtx` 进行高精度计时
- 控制变量法：每次只改变一个参数
- 自动遍历：通过模板元编程生成不同配置的 kernel
- 结果输出为结构化格式（CSV/JSON），方便后处理

---

### Phase 3: 指令级微架构探测

通过精心构造的 kernel 推断指令流水线细节。

#### 3.1 指令发射宽度
- 构造不同 ILP (Instruction-Level Parallelism) 级别的 kernel
- 测量同数量指令在不同 ILP 下的吞吐
- 推断每个 warp scheduler 每周期发射的指令数

#### 3.2 指令延迟
- 采用依赖链法（pointer chasing）：`a = b[b[b[...]]]`
- 对每种指令类型（FMA、IADD、LD/ST 等）构建长依赖链
- 根据总耗时 / 依赖深度 计算指令延迟

#### 3.3 指令吞吐
- 在 warp 内展开大量独立指令，消除延迟隐藏影响
- 测量不同指令类型（FP32/FP64/INT32/HADD2 等）的峰值吞吐
- 对比理论峰值与实测值

#### 3.4 SASS 反汇编分析
- 对关键 kernel 使用 `nvdisasm` 反汇编
- 分析 SASS 指令编码格式
- 推断 functional unit 布局

---

### Phase 4: 内存子系统分析

#### 4.1 全局内存带宽与延迟
- **延迟:** 通过 pointer chasing 测量全局内存延迟
- **带宽:** 构造连续/随机访问模式，测量实际带宽
- **并发:** 调整同时访问的 warp 数量，观察带宽饱和曲线

#### 4.2 共享内存
- **bank conflict:** 构造不同 stride 的访问模式，测量性能退化
- **容量与延迟:** 通过多次读写测量共享内存实际延迟

#### 4.3 L1/L2 缓存层次
- **缓存大小:** 逐步增大工作集，观察延迟跳变点 → 推断 cache size
- **行大小 (cache line):** 改变访问步长，探测 cache line 长度
- **关联度:** 通过冲突 misses 推断缓存关联度
- **逐出策略:** 分析不同访问模式下的命中率模式

#### 4.4 常量内存与纹理内存
- 常量内存广播机制探测
- 纹理内存缓存行为分析

---

### Phase 5: 调度器与并行模型

#### 5.1 Warp 调度策略
- **调度粒度:** 构造 divergent warp，观察性能损失
- **调度优先级:** 探测是否存在优先级反转或 fair scheduling
- **并发 warp 上限:** 逐步增加 block 内 warp 数，观察资源瓶颈

#### 5.2 Block 调度
- **SM 占用率:** 调整 block 大小和寄存器/共享内存使用量
- **block 分配策略:** 观察 block 在 SM 间的分布

#### 5.3 寄存器文件
- **寄存器数量:** 通过 `__launch_bounds__` 限制寄存器使用量
- **寄存器重命名:** 通过寄存器依赖链探测重命名能力

---

### Phase 6: 高级专题

#### 6.1 Tensor Core 分析
- 不同精度的 Tensor Core 指令吞吐
- Tile size (16x16x16 / 8x8x4 等) 对性能的影响
- Tensor Core vs CUDA Core 的切换开销

#### 6.2 混合精度与特殊指令
- __shfl / __shfl_down / __shfl_xor 等 warp shuffle 指令延迟
- __syncwarp / __ballot_sync 等同步指令开销
- __fma_rn / __cosf / __expf 等特殊函数吞吐

#### 6.3 功耗与频率
- 不同负载下的 GPU 频率变化（nvidia-smi dmon）
- 功耗-性能权衡分析

---

### Phase 7: 综合分析

- 将各阶段结果汇总，构建完整的 GPU 微架构模型
- 与已知文献（如 GPU 论文、Hot Chips 资料）交叉验证
- 形成可发布的逆向分析报告

---

## 四、项目结构

```
cuda_uarch/
├── CMakeLists.txt             # 根 CMake (C++20, CUDA)
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
│   │   ├── latency_probe.cu    # 指令延迟（依赖链法）
│   │   └── throughput_probe.cu # 指令吞吐（独立指令展开）
│   ├── memory/
│   │   ├── CMakeLists.txt
│   │   ├── global_mem_latency.cu  # 全局内存延迟（pointer chasing）
│   │   └── global_mem_bw.cu       # 全局内存带宽（读/写/copy）
│   ├── scheduler/             # (规划中)
│   │   └── CMakeLists.txt
│   └── advanced/              # (规划中)
│       └── CMakeLists.txt
├── build/                     # CMake 构建目录
├── data/
│   └── results/               # 实验结果 CSV
└── reports/
    └── figures/               # 图表输出
```

---

## 五、执行优先级

| 优先级 | 模块 | 理由 |
|--------|------|------|
| P0 | Phase 1 + 2 | 基础框架，必须最先完成 |
| P0 | 指令延迟 + 吞吐 (Phase 3.2, 3.3) | 最基础的分析，其他实验依赖此方法论 |
| P0 | 全局内存延迟与带宽 (Phase 4.1) | 理解内存子系统的基础 |
| P1 | 缓存层次 (Phase 4.3) | 需要先完成内存延迟方法论 |
| P1 | 共享内存 bank conflict (Phase 4.2) | 独立但重要 |
| P1 | Warp 调度 (Phase 5.1) | 理解并行执行模型 |
| P2 | 寄存器文件 (Phase 5.3) | 较深入的微架构分析 |
| P2 | Block 调度 (Phase 5.2) | 需要完整框架 |
| P2 | Tensor Core (Phase 6.1) | 高级专题，依赖前面的方法论 |
| P3 | 功耗/频率 (Phase 6.3) | 辅助分析，可最后进行 |

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

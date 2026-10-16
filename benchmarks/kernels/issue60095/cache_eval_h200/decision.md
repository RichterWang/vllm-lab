# #60095 split autotune 缓存：评估与处理建议

日期：2026-10-16  
评估对象：`vllm-60095-pr`，HEAD `d6f0b7a4166b325b4dda6512e2e06dcb581923ba` 及其未提交实验改动。

## 结论

建议从候选实现中移除启动时 split autotune 和持久缓存，保留确定性的静态 split 选择。H200 数据显示，缓存能修正少数 batch/词表组合的 split，并改善这些点的 kernel 设备时间；但在整个 K=6 测试集上设备时间收益有限，同步 Python wall time没有总体收益，冷启动校准成本又较高。现有证据不足以把启动 autotune 缓存作为通用性能优化提交。

这不是对静态 split 规则整体收益的结论。此次 H200 A/B 比较的是**同一候选源码中的静态 fallback 与缓存选择**，没有比较当前静态候选与 #60097 或 main。因此，移除缓存后，当前 split 算法是否优于目标 baseline，仍需单独复测。

## H200 实测依据

实验在 NVIDIA H200（132 SM）上运行，PyTorch 2.13.0+cu130、CUDA runtime 13.0、Triton 3.7.1。覆盖 V=131072、151936、262144；K=6 共 33 个形状，并对大 batch 补测 K=1、32。57 个形状的数值正确性检查均通过。

- K=6 全集：缓存相对静态 fallback 的设备时间几何平均 speedup 为 **1.022×**；同步 Python wall time几何平均 speedup 为 **0.993×**。
- 缓存只在 **7/33** 个 K=6 形状改变选择：V=131072/151936、B=47–128，将 64 splits 改为 32 splits。对应单点设备时间改善约 **8.5%–13.4%**；其余多数点基本不变。
- V=262144 没有 split 选择变化，整体近乎持平。设备时间最差点为 V=262144、B=8，speedup **0.993×**，差异很小。
- 冷校准为 **13.8–18.4 秒**，其中包含候选 kernel 的冷 JIT 编译。它不能表述为纯校准成本；预热 JIT 后单独复测的纯校准时间未完成。
- 持久缓存命中初始化中位数约 **9.45 ms**，观测最大值约 **43.14 ms**。
- 对少数获益形状，按每次节省约 0.7–2.1 微秒估算，冷成本需要数百万次同形状调用才可能摊平。该估算假设单一形状重复，不能代替实际混合流量分析。
- K=1/6/32 的已测大 batch 中，复用 K=6 选择通常距各自实测最佳 split 不到约 4%；没有证据要求把 K 加入缓存 key。

因此，数据更支持“静态启发式在少数 H200 形状上切分偏多，缓存以现场测量修正这些点”，而不是“启动 autotune 对一般 H200 工作负载都有净收益”。

## 本地源码处理

已在 `/home/minstrel/vllm_lab/worktrees/vllm-60095-pr` 的 `vllm-60095-pr` 分支移除缓存专属代码：

- 删除新建的 `logprob_autotune.py`。
- 删除 `gpu_worker.py` 中由 `VLLM_TOKEN_LOGPROB_AUTOTUNE` 控制的启动校准入口。
- 删除 logprob 模块中的进程内校准表、profile 注入函数及缓存查询；默认路径现在直接使用静态 split 规则。
- 删除只验证缓存 profile/K 复用的两个测试；保留 tile kernel、数值正确性和静态 split 选择测试。

其余已有的候选 kernel、静态 split 算法及相关正确性测试仍在工作树中。此次没有提交、推送，也没有运行测试或新的 GPU 测量。缓存评估原始结果仍留在 `results/kernel/issue60095/cache_eval_h200_20261016T043426Z/`；其校准脚本是历史复现实验材料，依赖缓存模块，不适用于当前已移除缓存的源码状态。

## 下午额度恢复后的 H200 复测

下一轮应确认**无缓存的当前候选**相对 PR baseline 的收益，而不是再次做缓存/静态 A/B。建议：

1. 先记录并核实远端 worktree 的分支、HEAD、源码 diff、CUDA/PyTorch/Triton 与 GPU 状态；确认当前源码不再含缓存入口。
2. 固定当前静态候选与干净的 #60097 PR baseline，使用同一 H200、同一编译环境、同一输入形状和计时方法，交替运行至少三轮。报告各自实际 split、kernel 设备时间、同步 Python wall time和正确性。
3. 覆盖本轮的 V/B/K 形状，并保留此前能体现差异的 V=131072/151936、B=47–128；同时检查小 batch、B=192/256 和 V=262144 的回退风险。
4. 如果还要报告“相对 main”的提升，应把 main 单独列为第二个基线。候选与 main 的差异可能同时包含 #60097 kernel 改动，不能把全部增益归因于本地 split 规则。
5. 对主要结论分别给设备端和 wall time；只有同方向、可重复且幅度超过计时波动的收益才作为性能结论。H200 缓存实验中 wall time没有总体改善，因此不能只凭设备时间宣布端到端改善。

## 资料

- 完整实验记录：[issue60095/end.md](../end.md)
- H200 原始分析、CSV/JSON/JSONL、环境和源码快照：`/home/minstrel/vllm_lab/results/kernel/issue60095/cache_eval_h200_20261016T043426Z/`
- 本报告与实验脚本目录：`/home/minstrel/vllm_lab/benchmarks/kernels/issue60095/cache_eval_h200/`

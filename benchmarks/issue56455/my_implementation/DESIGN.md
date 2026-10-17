# 独立 scheduler 方案设计稿（暂停版）

## 目标

在部署方给定的静态服务边界内，动态决定每个 scheduler step 如何组合 decode 与 prefill 工作。目标是在 TTFT、TPOT/TTOT 及 decode ITL 的 P50/P99 不劣于 baseline 的约束下，提高可持续完成请求吞吐。吞吐应通过相同负载下的端到端测试评估，不能以单步排入更多请求或 token 代替。

静态 `max_num_seqs` 是服务配置和实验约束，不由算法在线调整。调度算法可以根据当前负载调整每步准入数和 prefill cap，但不能假设能绕过静态序列上限。KV 是否可分配继续由现有 KV allocator 决定；性能预测和显存容量检查分开处理。

## 双层决策

### 外层：负载控制器

读取当前运行中 decode 的请求数和上下文长度、已有 prefill 的剩余工作量、等待队列的估计 prefill 工作量及最老等待时间，并结合 KV 可用容量、token/input budget、候选执行路径和最近的端到端延迟反馈，决定：

- 本步允许多少等待请求进入；
- 本步给 prefill 的 token cap；
- 是否先推进已驻留的 prefill，或暂缓引入新的 prefill；
- 本步允许的保守执行时间预算。

队列预计工作量用于衡量积压压力；等待时间用于防止长请求被持续跳过。不能只把各请求的预测执行时间简单相加，因为批处理会改变成本，而且这种规则可能长期饿死长请求。

### 内层：token 选择器

在外层给定的 cap 和准入范围内，结合 decode 请求的 token 需求、各 prefill 的上下文位置与剩余 chunk，选择本步执行请求及其 token 数。已有 decode 需要有明确的服务保障；新旧 prefill 共同受 token/input budget、静态序列上限和实际 KV 分配结果约束。未执行完的长 prefill 继续分块推进。

对于有 (p_i) 个新 token、已有 (l_i) 个上下文 token 的 prefill，可以用

\[
H=\sum_i\left(p_i l_i+\frac{p_i(p_i+1)}{2}\right)
\]

作为 attention 工作量的近似特征。它不能单独预测 step latency；预测还需考虑 decode 上下文读取、矩阵运算、batch 形状、padding、attention backend 和 CUDA Graph 执行模式。

## 延迟预测与回退

先用目标硬件、模型和执行配置的少量标定构建 LUT。LUT 按 decode 工作量、prefill 工作量及实际执行路径区分候选批次形状，并提供延迟分位数和安全余量。运行时使用正确关联到该 batch 的 GPU 执行时间缓慢校正预测；不能把 scheduler CPU 间隔直接当作 GPU step latency。配置或形状不匹配、预测数据缺失或校正不可信时，回退 baseline 调度。

CUDA Graph mode 是成本模型输入之一。需要实测并记录每步实际采用的 `FULL`、`PIECEWISE` 或 `NONE`，不能仅依据配置推断。batch token 数和 padding 后形状可能决定是否有匹配的已捕获图。

## 公平性与边界

- decode 已满且静态序列位也已满时，算法不能接纳新 prefill；它只能通过安排已有请求、等待名额释放来推进。增加静态上限属于部署配置变更，不是本算法职责。
- 新请求持续到达时，使用等待时长或等价的老化机制避免长时间饥饿；不能永远只选短 prefill。
- 长 prefill 必须能分块；持续有新 prefill 时，已有 chunk 也要获得推进机会。
- decode 不应被无限期连续延后；若策略允许短暂延期，必须定义最大连续延期并验证 ITL P99。
- KV 空间不足时遵守现有分配器和 scheduler 的失败/抢占语义；性能估算不能代替 KV block 检查。
- SLO 预算不足时，优先满足已定义的 decode 服务约束并回退到 baseline，不以超出单步 token/input budget 的方式追求吞吐。

## 验收方法

先用诊断运行验证每步准入、decode 服务、prefill chunk 推进、KV 门槛、等待原因和实际 graph mode。性能对照关闭 trace，在相同模型、硬件、到达表和静态配置下比较 baseline 与候选。最终报告共同的 TTFT、TPOT/TTOT、逐 token ITL 分位数及持续到达负载下的完成吞吐；SSE event gap 只能作为代理指标，不能代替逐 token ITL。

## 当前状态

上述内容是目标设计，不代表当前代码已经实现。现有实验实现主要验证固定 prefill cap/profile；双层动态准入、上下文工作量模型、graph-aware 延迟表与在线校正均未完成。项目于 2026-10-17 暂停，原因和已得结论见仓库根目录的 `benchmarks/issue56455/closeout_20261017.md`。

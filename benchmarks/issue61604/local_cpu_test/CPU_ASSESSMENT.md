# Issue #61604 CPU 阶段评估

本报告的最新有效结论见 [本轮测试报告](/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/REPORT_2026-10-18.md)。

旧版 adverse 结果因 probe 计数包含额外 oracle 查询，已按用户要求删除；不能再引用旧 probe 计数。旧候选首查 memo 的计时也不代表当前修改。修正后的本轮性能数据、正确性结果、动态 replay、ABA 和 memo 内存检查均按新报告记录。

patch 已加 4 个完整命中 block 的 memo 门限。17 项定向 memo 测试、完整 prefix-cache 文件 189 项均通过；组合 replay 的调度、block、refcount 和释放顺序与真实 base 一致。门限后的稳定等待完整 `schedule()` 中位数下降 59.2%；固定亲和 churn replay 中，短命中不建 memo但比 base 慢约 4.4%，长前缀 memo-on 比 base 慢约 6.3%（各三轮，毫秒级）。当前保留门限但不冻结候选；在 H100/H200 验证目标服务负载后再决定是否保留优化。

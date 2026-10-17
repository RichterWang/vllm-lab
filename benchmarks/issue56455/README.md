# vLLM PR #56455 实验归档

本项目于 2026-10-17 暂停。此目录只保留设计、结论和状态索引；源码实验保存在内层 worktree，原始运行数据保存在本机被 Git 忽略的 `results/issue56455/`。

- [阶段收束、对外讨论草稿与恢复条件](closeout_20261017.md)
- [独立双层调度方案设计稿](my_implementation/DESIGN.md)
- 内层代码检查点：worktree `/home/minstrel/vllm_lab/worktrees/vllm-56455-mypr`，branch `vllm-56455-trail`，base `4b839c3378c6a40fa613328a9b97069ed83b181b`

## 本地实验数据索引

所有运行数据当前受 `.gitignore` 排除，不会随仓库提交或推送。常用目录按实验分类保存：

- `results/issue56455/main_h100_20261015/`：H100 baseline 与原 PR 对照；另有打包副本 `main_h100_20261015.tar.gz`。
- `results/issue56455/fixed_h100_20261015/`：H100 固定到达时序实验。
- `results/issue56455/local_comparison/`、`local_gpu_smoke/`：3070 Ti 本地实现对照。
- `results/issue56455/calibration/`、`shared_cap_profiles/`：prefill latency 标定和实验 profile。
- `results/issue56455/tail_completion/`：384、448 和有界尾段对照。
- `results/issue56455/fixed_arrival_performance/`、`fixed_arrival_generalization/`：固定到达时序与相邻负载对照。
- `results/issue56455/diagnostics/`：scheduler trace、队列 gate、资源释放及配对分析。
- `results/issue56455/correctness/`：正确性验证输出。

除非恢复项目需要重新分析，先保留原目录和文件名，避免破坏报告中已有的路径引用。仓库内的结论摘要以 `closeout_20261017.md` 为准。

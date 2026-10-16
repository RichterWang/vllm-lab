# H200：当前 split 缓存与同源码静态策略对照

本实验只比较一个 `vllm-60095-pr` worktree 内的两种选择状态。静态组在
计时前移除该 V/dtype 的进程内选择表，缓存组恢复同一源码的已校准表；
两组都调用 `compute_token_logprobs(..., num_vocab_splits=None)`。不引用
`#60097` 或 main 的 kernel 结果作为直接基线，也不修改 vLLM 源码。

运行前必须确认 `vllm-60095-pr` 目录确实有 `logprob_autotune.py`、
`_AUTOTUNED_SPLITS` 和 K=6 校准实现。脚本会记录工作树 HEAD、branch、
dirty status、完整 diff 和关键源码快照；实现不匹配时会停止。

```bash
RUN=/workspace/vllm_lab/results/kernel/issue60095/cache_eval_h200_$(date -u +%Y%m%dT%H%M%SZ)
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  /workspace/vllm_lab/src/vllm/.venv/bin/python \
  /workspace/vllm_lab/benchmarks/kernels/issue60095/cache_eval_h200/bench_same_source.py \
  --vllm-root /workspace/vllm_lab/worktrees/vllm-60095-pr \
  --binary-root /workspace/vllm_lab/src/vllm \
  --output-dir "$RUN" |& tee "$RUN.log"

/workspace/vllm_lab/src/vllm/.venv/bin/python \
  /workspace/vllm_lab/benchmarks/kernels/issue60095/cache_eval_h200/analyze.py "$RUN"
```

脚本拒绝覆盖已有结果目录。默认 V=131072/151936/262144、bf16，全部
指定 B 测 K=6，大 B=96/128/192/256 另测 K=1/32；每个形状扫描
1/2/4/8/16/32/64 split。每种策略和候选值进行三轮交替计时，并对
FP64 oracle 做正确性校验。`initialization.json` 单独保存空选择缓存的
校准开销和后续命中开销。`raw.jsonl`、`results.csv`/JSON、环境与源码
快照保留全部原始数据；`analyze.py` 生成表格和两张图。

校准成本在当前 Triton 编译缓存状态下测量，报告中必须说明该状态；
它不等于整个 vLLM 进程或模型的启动时间。CUDA Graph 设备时间和同步
Python wall 时间分别报告。只有确认远端 H200 当前工作树和空闲 GPU 后
才能执行本命令。

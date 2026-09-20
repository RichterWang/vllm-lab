# vLLM 架构阅读路线

本文对应源码提交
`fbe8a157fbdd4b2f813ae6395397232de97d5c5d`。路径均相对于
`/home/minstrel/vllm_lab/src/vllm`。

## 本实验固定的边界

- 模型：`Qwen/Qwen3-0.6B`，revision
  `c1899de289a04d12100db370d81485cdf75e47ca`。
- GPU：RTX 3070 Ti Laptop，8 GiB，SM86。
- 数据类型：FP16；上下文上限：4096；最大并发序列：8。
- 首轮使用 eager execution，暂不把 CUDA Graph 引入变量。
- 使用 vLLM 原生 sampler。FlashInfer sampler 的首次执行还会触发 CUDA
  JIT，因此留到独立的编译工具链实验。
- HTTP 服务只绑定 `127.0.0.1:8000`。

## 离线调用链

```text
experiments/01_offline_chat.py
  -> vllm/entrypoints/llm.py: LLM
  -> Renderer / chat template / tokenization
  -> vllm/engine/arg_utils.py: EngineArgs
  -> vllm/config/vllm.py: VllmConfig
  -> vllm/v1/engine/llm_engine.py: LLMEngine
  -> InputProcessor
  -> vllm/v1/engine/core_client.py: SyncMPClient
  -> vllm/v1/engine/core.py: EngineCore
  -> vllm/v1/core/sched/scheduler.py: Scheduler
  -> UniProcExecutor -> GPUWorker
  -> vllm/v1/worker/gpu/model_runner.py: GPUModelRunner (V2)
  -> vllm/model_executor/models/qwen3.py: Qwen3ForCausalLM
  -> vllm/v1/worker/gpu/sample/sampler.py: Sampler
  -> Scheduler.update_from_output()
  -> OutputProcessor -> RequestOutput
```

`LLM` 是最适合第一次跟踪的入口：它把用户参数变成 `EngineArgs`，再创建
`LLMEngine`。`LLM.chat()` 仍会应用 chat template 和 tokenize；当前默认的
V1 multiprocessing 配置通过 `SyncMPClient` 连接 EngineCore 子进程。它没有
HTTP/FastAPI 层，因而仍是观察模型加载、KV cache、调度和采样的最短路径。

## 在线调用链

```text
vllm serve
  -> vllm/entrypoints/cli/serve.py: ServeSubcommand
  -> vllm/entrypoints/launchers/api_server/entry.py
  -> POST /v1/chat/completions
  -> vllm/entrypoints/openai/chat_completion/api_router.py
  -> vllm/entrypoints/openai/chat_completion/serving.py: OpenAIServingChat
  -> vllm/v1/engine/async_llm.py: AsyncLLM
  -> vllm/v1/engine/core_client.py: AsyncMPClient
  -> EngineCore 子进程
  -> Scheduler -> UniProcExecutor -> GPUWorker
  -> GPUModelRunner (V2) -> Qwen3ForCausalLM -> Sampler
  -> OpenAI ChatCompletionResponse
```

在线路径在同一个推理核心外增加了参数校验、模型名路由、chat template、异步
请求生命周期、流式/非流式转换和指标。旧的
`vllm/entrypoints/openai/api_server.py` 在当前提交只是兼容导出并已标记弃用，
阅读时应从 `entrypoints/launchers/` 开始。

## 一次 engine step 发生什么

核心循环可从 `vllm/v1/engine/core.py` 的 `EngineCore.step()` 阅读：

1. `Scheduler.schedule()` 在 token budget 下选择运行中、新到达或恢复的请求，
   同时给它们分配 KV cache block。
2. 单卡基线的 `UniProcExecutor` 将 `SchedulerOutput` 交给 `GPUWorker`；本机
   日志确认 worker 选择了 V2 `GPUModelRunner`。
3. model runner 准备 token、position 和 KV 元数据，调用 Qwen3 forward，
   并采样下一批 token。
4. sampler 并不直接产生公开的 `RequestOutput`。它的输出先经过
   `Scheduler.update_from_output()` 更新请求状态和释放资源，再以
   `EngineCoreOutputs` 送到 `OutputProcessor` 或在线 serving 层。
5. 下一轮不是固定区分“prefill batch”和“decode batch”；当前 scheduler 以每个
   请求的已计算 token 与目标 token 差额统一决定本轮工作，因此可以覆盖
   chunked prefill、prefix cache 和 speculative decoding。

这正是 continuous batching 的核心观察点：新请求可以在旧请求仍处于 decode
时进入后续 step，而不必等待一个静态 batch 整体结束。

## 模型如何接入

Hugging Face `config.json` 声明 `architectures=["Qwen3ForCausalLM"]`。
`vllm/model_executor/models/registry.py` 将这个名字映射到
`vllm/model_executor/models/qwen3.py`。因此研究一个新模型时，建议依次查看：

1. HF 配置中的 `architectures`；
2. vLLM model registry 的映射；
3. 对应模型类的 forward、权重加载与 tensor-parallel 规则；
4. 相关 `tests/models/` 测试如何与 Transformers 参考实现对齐。

## 已观察到的资源数据

| 项目 | 结果 |
|---|---:|
| 权重在 GPU 上的日志值 | 1.12 GiB |
| 离线模型加载后显存 | 6454 MiB / 8192 MiB |
| 在线服务运行时显存 | 6590 MiB / 8192 MiB |
| 在线 KV cache 容量 | 46,704 tokens |
| 4096 token 理论最大并发 | 11.40x |
| 服务退出后显存 | 9 MiB |

`gpu-memory-utilization=0.80` 控制的是引擎可规划的 GPU 内存预算，不等同于
权重大小。除权重外，KV cache、PyTorch/CUDA 上下文和工作区都会占用显存。

## 建议的下一组源码实验

1. 在 `Scheduler.schedule()` 周围仅增加本地调试日志，观察 4 个请求每轮的
   scheduled token 数；不要把调试日志提交到 PR。
2. 对比单请求与并发请求的 `/metrics`，记录 running、waiting、KV cache
   usage 与 token counters。
3. 去掉 `--enforce-eager`，单独测量 CUDA Graph 的启动时间和稳态延迟。
4. 配好隔离的 CUDA JIT 工具链后再启用 FlashInfer sampler，避免把 Python
   架构学习和本地编译问题混在一起。
5. 真正修改源码时，从最新 `upstream/main` 新建一条功能分支，补最小测试，
   执行目标测试和 pre-commit，再向 `vllm-project/vllm` 提交 PR。

现有输出只证明推理与服务链路可用；4 路请求成功也不证明相对吞吐提升。
0.6B 小模型对架构问题的自然语言回答可能不精确，不能替代源码阅读、参考
答案验证或正式 benchmark。

## 推荐源码阅读顺序

1. 入口与配置：`vllm/entrypoints/llm.py`、`vllm/engine/arg_utils.py`、
   `vllm/config/vllm.py`。
2. 前后处理：`vllm/renderers/`、`vllm/v1/engine/input_processor.py`、
   `vllm/v1/engine/output_processor.py`、`vllm/outputs.py`。
3. 核心与进程边界：`vllm/v1/engine/llm_engine.py`、`core_client.py`、
   `core.py`。
4. 调度与 KV：`vllm/v1/core/sched/scheduler.py`、
   `vllm/v1/core/kv_cache_manager.py`。
5. 单卡执行：`vllm/v1/executor/uniproc_executor.py`、
   `vllm/v1/worker/gpu_worker.py`、`vllm/v1/worker/gpu/model_runner.py`。
6. 模型与采样：`vllm/model_executor/models/registry.py`、`qwen3.py`、
   `vllm/v1/worker/gpu/sample/sampler.py`。
7. 最后增加 HTTP：`vllm/entrypoints/cli/serve.py`、
   `vllm/entrypoints/launchers/`、`vllm/entrypoints/openai/`、
   `vllm/v1/engine/async_llm.py`。

Model Runner 存在 V1/V2 两条实现；本机日志中的 `Using V2 Model Runner` 决定
了本实验应读 `vllm/v1/worker/gpu/model_runner.py`。旧教程常引用的
`vllm/v1/worker/gpu_model_runner.py` 是另一条 legacy/fallback 路径。

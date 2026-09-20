# vLLM Lab 阶段完成报告

完成时间：2026-09-20（Asia/Singapore）
阶段结论：**当前远端机器上的小模型运行、OpenAI API、服务端并发观察和干净
关停验收全部通过。**

## 1. 验收边界

本阶段验证以下内容：

1. 固定 Hugging Face 模型可以完整下载并由 editable vLLM 加载；
2. 离线 `LLM.chat()` 可以在 GPU 上生成非空输出；
3. OpenAI 兼容健康、版本、模型列表和聊天接口可用；
4. 4 个客户端请求期间，vLLM 指标确实观察到 4 个同时运行的 engine request；
5. 服务能被脚本收回，端口、进程、队列、KV cache 和显存恢复为空闲状态；
6. 实验配置、脚本、结果、环境记录和架构阅读路线已经与上游 fork 分离。

本阶段不把小模型生成文本当作知识正确性测试，也不把并发重叠当作吞吐提升
证明。语义质量需要参考答案/数据集，性能结论需要顺序基线、预热、重复次数和
统计分析；二者不在本阶段的 `passed` 定义中。

## 2. 未改动的系统边界

- NVIDIA 驱动仍为 `580.65.06`。
- IsaacLab 使用的 `infra_dev` Conda 环境未修改。
- 没有安装或替换系统级 `/usr/local/cuda`。
- 上游 fork 的 `AGENTS.md` 未修改；一次性实施记录保存在本实验仓库。

## 3. 硬件与 Python 环境

| 项目 | 接受状态 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU，8192 MiB，SM86 |
| Python | 3.12.13 |
| uv | 0.12.10 |
| 项目环境 | `/home/minstrel/vllm_lab/src/vllm/.venv`，约 8.7 GiB |
| PyTorch | `2.13.0+cu132`；PyTorch CUDA runtime 13.2 |
| Triton | 3.7.1 |
| vLLM runtime 版本 | `0.29.1rc1.dev405+gfbe8a157f` |
| 安装元数据版本 | `0.29.1rc1.dev405+gfbe8a157f.precompiled` |

运行时版本不带 `.precompiled`，editable distribution 元数据带该后缀；它们来自
同一源码提交，是源码版本与安装变体两种表示，不是 commit 漂移。

## 4. vLLM fork 状态

- 路径：`/home/minstrel/vllm_lab/src/vllm`。
- fork remote：`git@github.com:RichterWang/vllm.git`。
- upstream remote：`https://github.com/vllm-project/vllm.git`。
- 当前学习分支：`learning/cu130-fbe8a157`。
- 当前提交：`fbe8a157fbdd4b2f813ae6395397232de97d5c5d`。
- 工作树：干净。

该 SHA 与官方 cu130 预编译扩展匹配。fork 的 `main` 已同步并推送到
`41c4a3ed4e45792739bd8a5bbcd01269b06eae2c`。学习分支只用于固定环境；正式
PR 应从届时最新的 `upstream/main` 新建独立功能分支。

## 5. 固定模型

- Hugging Face ID：`Qwen/Qwen3-0.6B`。
- revision：`c1899de289a04d12100db370d81485cdf75e47ca`。
- 本地解析后的 snapshot：约 1.5 GiB。
- `model.safetensors`：1,503,300,328 bytes。
- 架构声明：`Qwen3ForCausalLM`。
- `.incomplete` 文件：0。
- 该模型公开可读，本次匿名访问成功，不要求 Hugging Face token。

## 6. 已落地的项目结构

```text
/home/minstrel/vllm_lab
├── configs/qwen3-0.6b.env
├── docs/environment_bootstrap.md
├── docs/architecture_walkthrough.md
├── environment/installed-packages.txt
├── experiments/01_offline_chat.py
├── scripts/download_model.sh
├── scripts/run_offline_chat.sh
├── scripts/serve_qwen3_0_6b.sh
├── scripts/smoke_openai_api.py
├── scripts/concurrent_batch_test.py
├── scripts/verify_acceptance.py
├── scripts/run_acceptance.sh
├── results/*.json
└── src/vllm/                    # 独立 Git 仓库
```

根目录是实验仓库，`src/vllm` 是独立上游 fork。`.gitignore` 排除嵌套仓库、
虚拟环境、缓存和大日志，但保留小型 JSON 验收证据。VS Code 以实验根目录为
workspace，解释器指向 `src/vllm/.venv/bin/python`，pytest 工作目录指向
`src/vllm`。

实验根仓库已建立本地 `main` 基线提交，所有上述脚本、配置、文档和 JSON
证据均已跟踪；该实验仓库目前没有配置 remote，也没有执行远端推送。

## 7. 脚本可靠性修正

最终验收前又完成了一轮只读审查并修复以下问题：

- 每次运行先移除旧结果，失败时写 `failed`，成功结果用临时文件原子替换，
  避免早期异常后遗留旧 `passed`。
- 结果记录 UTC 时间、完整 vLLM SHA、模型 revision 和关键运行参数。
- API 客户端与服务端共同使用 `LAB_HOST`/`LAB_PORT`，不再维护两套地址真值。
- 离线与在线都固定 `generation_config=vllm`。
- GPU 快照显式选择 `LAB_GPU_INDEX=0`，不再把多 GPU 的多行输出错误压成一行。
- API smoke test 可等待服务就绪；一键脚本在退出路径中收回自己启动的服务。
- 并发测试不检查模型对架构问题的自然语言答案，而是用长请求持续采样
  `/metrics`，把 engine 同时运行请求数作为服务端重叠证据。

## 8. 最终验收证据

### 离线推理

- 状态：`passed`。
- 模型加载：13.844 秒。
- 生成：0.438 秒。
- prompt/completion：31/54 tokens。
- 加载后显存：6454 MiB / 8192 MiB。
- V1 Engine、V2 Model Runner、FlashAttention 2 和原生 sampler 均由日志确认。

### OpenAI API

- `/health`：HTTP 200。
- `/version`：HTTP 200，版本与固定源码一致。
- `/v1/models`：HTTP 200，发布名为 `qwen3-0.6b`。
- `/v1/chat/completions`：HTTP 200，返回非空内容。
- 单次 completion：0.323 秒，25 prompt tokens、38 completion tokens。

### 服务端并发观察

- 4/4 请求均为 HTTP 200，每个请求固定生成 128 tokens。
- 并发阶段墙钟时间：1.165 秒。
- `/metrics` 采样数：79。
- `peak_running_requests`：4。
- `peak_waiting_requests`：0。
- `peak_kv_cache_usage`：0.013708（约 1.37%）。
- 结束时 running=0、waiting=0、KV cache usage=0。
- 服务累计 counters：121 prompt tokens、550 generation tokens。

`peak_running_requests=4` 证明服务端确实同时持有这 4 个请求；本阶段没有将其
外推为相对顺序执行的吞吐增益。

### 关停与资源释放

- `127.0.0.1:8000`：已关闭。
- vLLM/EngineCore 残留进程：0。
- GPU 显存：9 MiB used、8079 MiB free。
- GPU utilization 最终复核：0%。
- `results/acceptance.json`：`passed`，6 项布尔检查全部为 true。

## 9. 结果与复现入口

最终结果：

- `results/offline_chat.json`
- `results/openai_api.json`
- `results/concurrent_batch.json`
- `results/acceptance.json`

当前机器上一键重复：

```bash
cd /home/minstrel/vllm_lab
./scripts/run_acceptance.sh
```

从零创建 Miniconda/uv 环境、检出匹配 commit 并安装官方 cu130 wheel 的命令
记录在 `docs/environment_bootstrap.md`。官方 wheel 索引当前可用，且命令使用
完整 40 位 SHA，不再使用会被忽略的 `nightly` 字符串。

`environment/installed-packages.txt` 保存当前 `pip freeze` 的 407 条诊断
记录。它反映实际接受环境，但不是跨平台 hash lock；源码仓库的 x86_64 固定
测试依赖应优先从
`requirements/test/cuda.txt` 安装。当前环境早期曾从未锁定的
`requirements/test/cuda.in` 安装，因此报告明确保留这一区别。

## 10. 已知限制

1. 自动 FlashInfer sampler 首次 warm-up 会尝试 CUDA JIT，而当前 shell 没有
   系统级 `/usr/local/cuda`。本基线通过官方开关
   `VLLM_USE_FLASHINFER_SAMPLER=0` 使用原生 sampler；这不是核心推理故障。
2. 上游固定的 `decord==0.6.0` 能 import，但其旧 CPython 3.6 wheel tag 会被
   `uv pip check` 报告为平台不匹配。
3. 当前固定源码的整个 `tests/test_version.py` 为 6 passed、7 failed，原因是
   开发版版本元组/major-version 测试假设与该 snapshot 不一致；单独的
   `test_version_is_defined` 与 16 个 logger tests 已通过。
4. 本次生成内容只做非空检查。0.6B 模型可能对架构概念回答不准确，不能替代
   源码事实或正式 eval。

## 11. 架构理解结果与下一阶段

离线和在线入口最终汇合到相同的 V1 EngineCore：

```text
LLM.chat / OpenAI serving
  -> Renderer + InputProcessor
  -> SyncMPClient / AsyncMPClient
  -> EngineCore
  -> Scheduler
  -> UniProcExecutor -> GPUWorker
  -> V2 GPUModelRunner -> Qwen3ForCausalLM -> Sampler
  -> Scheduler.update_from_output
  -> OutputProcessor / OpenAI response
```

详细源码路径、V1/V2 runner 区分和阅读顺序已写入
`docs/architecture_walkthrough.md`。

下一阶段建议只聚焦 scheduler：记录每个 engine step 的 scheduled token 和
KV block 变化，建立顺序/并发的正式 benchmark，并把临时观测代码与可提交的
功能修改分离。准备实际 PR 时，再检查重复 issue/PR、同步最新 upstream、建立
功能分支、补最小测试、运行目标测试与 pre-commit，并由提交者逐行理解改动。

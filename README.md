# vLLM Lab

这个仓库保存可重复运行的 vLLM 学习实验；真正的 vLLM fork 位于
`src/vllm/`，并保持为一个独立 Git 仓库。这样实验脚本、模型配置和运行结果
不会混入未来要提交给上游的 PR。

## 目录

- `configs/`：固定的模型 revision 与低显存运行参数。
- `docs/`：环境说明、源码调用链和后续阅读路线。
- `experiments/`：直接调用 editable vLLM 的离线实验。
- `scripts/`：下载、离线推理、启动服务和接口验收入口。
- `results/`：可纳入版本管理的精简验收结果；大体积日志被忽略。
- `src/vllm/`：独立的 `vllm-project/vllm` fork。

## 快速重复实验

当前环境已经安装在 `src/vllm/.venv`。所有脚本都直接解析这个解释器，因此在
实验仓库根目录执行即可：

```bash
cd /home/minstrel/vllm_lab
./scripts/run_acceptance.sh
```

这个入口会自动完成模型校验、离线推理、服务启动、接口/并发 smoke test、
服务关闭和资源释放检查。也可以分步执行：

```bash
cd /home/minstrel/vllm_lab
./scripts/download_model.sh
./scripts/run_offline_chat.sh
```

启动仅监听本机回环地址的 OpenAI 兼容服务：

```bash
./scripts/serve_qwen3_0_6b.sh
```

在另一个终端完成单请求和 4 路并发 smoke test：

```bash
cd /home/minstrel/vllm_lab
source configs/qwen3-0.6b.env
src/vllm/.venv/bin/python scripts/smoke_openai_api.py
src/vllm/.venv/bin/python scripts/concurrent_batch_test.py
```

需要交互开发时可激活环境：

```bash
conda activate vllm-dev
cd /home/minstrel/vllm_lab/src/vllm
source .venv/bin/activate
```

VS Code 应打开 `/home/minstrel/vllm_lab` 作为工作区；项目设置会选择
`src/vllm/.venv/bin/python`，测试工作目录则指向 vLLM 源码仓库。

完整验收结论见 `AGENT.md`，从零重建环境见
`docs/environment_bootstrap.md`，源码阅读路线见
`docs/architecture_walkthrough.md`。

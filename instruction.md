# vLLM Lab 操作说明

## 进入已经安装好的开发环境

```bash
conda activate vllm-dev
cd /home/minstrel/vllm_lab/src/vllm
source .venv/bin/activate
```

当前 editable 源码分支为 `learning/cu130-fbe8a157`，对应官方 cu130
预编译 wheel 的提交：

```text
fbe8a157fbdd4b2f813ae6395397232de97d5c5d
```

不要使用 `VLLM_PRECOMPILED_WHEEL_COMMIT=nightly`。当前提交的安装代码只接受
40 位 commit SHA，字符串 `nightly` 会被忽略。

## 下载固定模型快照

```bash
cd /home/minstrel/vllm_lab
./scripts/download_model.sh
```

## 一键完成全部 smoke acceptance

```bash
cd /home/minstrel/vllm_lab
./scripts/run_acceptance.sh
```

脚本会等待服务就绪，并在测试结束后负责关停服务、检查端口和显存。正常情况
无需手工维护后台 PID。

## 离线推理

```bash
./scripts/run_offline_chat.sh
```

## 启动本地 OpenAI 服务

```bash
./scripts/serve_qwen3_0_6b.sh
```

服务只监听 `127.0.0.1:8000`。在另一个终端运行：

```bash
source /home/minstrel/vllm_lab/configs/qwen3-0.6b.env
/home/minstrel/vllm_lab/src/vllm/.venv/bin/python \
  /home/minstrel/vllm_lab/scripts/smoke_openai_api.py

/home/minstrel/vllm_lab/src/vllm/.venv/bin/python \
  /home/minstrel/vllm_lab/scripts/concurrent_batch_test.py
```

结束服务使用 `Ctrl-C`。正常退出后，`127.0.0.1:8000` 不再监听，模型占用的
显存也会释放。

## 结果位置

- `results/offline_chat.json`：离线 `LLM.chat()` smoke test。
- `results/openai_api.json`：健康检查、模型列表和聊天接口 smoke test。
- `results/concurrent_batch.json`：4 路并发请求 smoke test。
- `results/acceptance.json`：关停后汇总验收。
- `AGENT.md`：本阶段完整实施和验收报告。

这些测试验证的是下载、加载、调度、生成和接口链路。Qwen3-0.6B 的回答内容
不作为知识准确性基准；并发请求成功也不单独证明吞吐提升。需要评测正确性或
性能时，应分别引入参考答案或可控 benchmark。

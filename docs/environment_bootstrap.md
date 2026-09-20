# 从零重建 vLLM Lab 环境

本文记录当前机器上的环境边界和重建步骤。目标是隔离 vLLM 开发，不改动
NVIDIA 580.65.06 驱动，也不使用或修改 IsaacLab 的 `infra_dev`。

## 1. 仓库分层

实验仓库和上游 fork 是两个独立 Git 仓库：

```text
/home/minstrel/vllm_lab/          # 实验、配置、结果和笔记
└── src/vllm/                     # RichterWang/vllm fork
```

重建嵌套 fork：

```bash
mkdir -p /home/minstrel/vllm_lab/src
git clone git@github.com:RichterWang/vllm.git \
  /home/minstrel/vllm_lab/src/vllm
cd /home/minstrel/vllm_lab/src/vllm
git remote add upstream https://github.com/vllm-project/vllm.git
git fetch upstream
git switch --create learning/cu130-fbe8a157 \
  fbe8a157fbdd4b2f813ae6395397232de97d5c5d
```

若 `upstream` 已存在，不重复添加。这个固定提交用于重复本实验；真正开发 PR
时应从届时最新的 `upstream/main` 新建另一条功能分支。

## 2. 创建独立 Python 环境

```bash
/home/minstrel/miniconda3/bin/conda create -n vllm-dev \
  -c conda-forge python=3.12 uv=0.12.10 -y

cd /home/minstrel/vllm_lab/src/vllm
/home/minstrel/miniconda3/envs/vllm-dev/bin/uv venv \
  --python /home/minstrel/miniconda3/envs/vllm-dev/bin/python \
  --seed
source .venv/bin/activate
```

不要在 `(base)` 中直接安装 vLLM，也不要借用 `infra_dev` 的 `nvcc`。预编译
安装不要求系统级 CUDA toolkit；NVIDIA 驱动只负责提供兼容的运行时接口。

## 3. 安装与源码提交匹配的预编译扩展

必须传完整的 40 位提交 SHA；字符串 `nightly` 不是有效 commit 值。当前 wheel
索引已发布该提交的 cu130 x86_64 artifact：

```text
https://wheels.vllm.ai/fbe8a157fbdd4b2f813ae6395397232de97d5c5d/cu130/vllm/
```

执行：

```bash
cd /home/minstrel/vllm_lab/src/vllm
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=fbe8a157fbdd4b2f813ae6395397232de97d5c5d \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
/home/minstrel/miniconda3/envs/vllm-dev/bin/uv pip install -U -e . \
  --torch-backend=auto
```

如果未来该 commit/variant 不再可取，应显式提供已归档 wheel 的本地路径：

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/absolute/path/to/matching-vllm.whl \
/home/minstrel/miniconda3/envs/vllm-dev/bin/uv pip install -U -e . \
  --torch-backend=auto
```

不能拿别的源码提交配这个 wheel；Python 代码与 C++/CUDA 扩展必须来自同一
commit。

## 4. 安装贡献工具和测试依赖

```bash
cd /home/minstrel/vllm_lab/src/vllm
source .venv/bin/activate

/home/minstrel/miniconda3/envs/vllm-dev/bin/uv pip install \
  -r requirements/lint.txt
/home/minstrel/miniconda3/envs/vllm-dev/bin/uv pip install \
  -r requirements/test/cuda.txt --torch-backend=auto

.venv/bin/pre-commit install
.venv/bin/pre-commit install --hook-type commit-msg
```

`requirements/test/cuda.txt` 是 x86_64 的固定版本集合；直接安装未锁定的
`requirements/test/cuda.in` 会得到随时间变化的依赖。本机当前接受环境早期
使用了 `.in`，因此实际包版本单独保存在
`environment/installed-packages.txt`，该文件是诊断快照，不是跨平台 lock。

## 5. 自检

```bash
cd /home/minstrel/vllm_lab/src/vllm
source .venv/bin/activate

.venv/bin/python -c \
  'import torch, vllm; print(vllm.__version__); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())'
.venv/bin/vllm --version
git status --short --branch
```

然后在实验仓库运行完整 smoke acceptance：

```bash
cd /home/minstrel/vllm_lab
./scripts/run_acceptance.sh
```

这会下载/校验固定模型、运行离线推理、临时启动本地服务、执行单请求和服务端
并发观察、关闭服务，并验证端口与显存已经恢复。

## 6. CUDA 编译与 JIT 的边界

当前环境验证的是预编译 vLLM 核心。FlashInfer sampler 的首次运行可能需要
可发现的 `nvcc` 做 JIT；修改 vLLM C++/CUDA kernel 也需要完整源码编译工具链。
这两项应建立在 vLLM 自己的环境中，并按上游 incremental build 文档配置，
不能为了它们更换现有驱动，也不能改动 `infra_dev`。

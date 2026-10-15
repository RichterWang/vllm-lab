# Issue #60095：PR #60097 split 选择优化实验日志

> 2026-10-15 开始。此文件是本轮工作的持续记录；结论以实际保存的测量文件为准。
> 本轮所有分支、代码和结果仅保存在本地，不推送 GitHub。

## 目标与边界

目标是改进 PR #60097 的 `compute_token_logprobs` 自动词表切分选择。
PR 作者已经实现一次读取词表和 split kernel。本轮只把**自动选择策略**当作候选改动，
不把 PR 作者的 kernel 优化算作本项目的原创成果。

PR 讨论在 H200 上指出 `B=128,V=151936,K=6,bf16` 时，auto 选 2 splits，
约 47.5 µs；强制 16/32 splits 约 18 µs。这里的 H200 数字来自
[PR #60097 讨论](https://github.com/vllm-project/vllm/pull/60097#issuecomment-6074007271)，
本地 H200 独立复现只比较过 base 与 PR auto，并没有扫描强制 splits。

本轮先用 RTX 3070 Ti 做规则发现和正确性检查。3070 Ti 的性能结论只适用于本机测试范围；
要声称 H200 改善，仍需之后在 H200 上测候选规则。

## 工作分支与环境

| 仓库 | 本地分支 | 起点 |
|---|---|---|
| 外层 `vllm-lab` | `codex/issue60095-split-selection` | `3becc8a5acb9e6f9fa70fa629ece642b1b1ea63d` |
| vLLM PR worktree | `codex/issue60095-split-selection` | PR #60097 `0ddd6017793c943a2091a5800f0cbac1daaa3e02` |

本机：NVIDIA GeForce RTX 3070 Ti Laptop GPU，SM 8.6，8 GiB；驱动 580.65.06。
现有 vLLM 环境：PyTorch 2.13.0+cu132、CUDA runtime 13.2、Triton 3.7.1。
本轮在用户开启的 full access 环境执行，`nvidia-smi` 与 PyTorch 均能看到 GPU；
之前受限的工具沙箱不暴露 `/dev/nvidia*`，不代表机器 CUDA 环境损坏。

## 实施方案

1. **建立原始 PR 的基线。** 固定 `V=151936,K=6,bf16`，扫描 B=1–128 的代表点，
   加测当前自动规则的切换边界；每个点测 `auto,1,2,4,8,16,32`。记录自动选择的
   实际 split、正确性、CUDA Graph 设备时间、主机同步时间和环境元数据。
2. **分析选择损失。** 对同形状的强制 split，比较 auto 与最快选择；用至少三轮测量
   的中位数和逐轮范围识别稳定差距。把最优点作为设计依据，不把单次最快值当成结论。
3. **形成候选策略。** 优先考虑少量离线生成的、按硬件能力和形状分段的规则；
   推理调用时只执行常量时间选择，不现场 autotune。未测硬件保留原 PR 规则，
   保持 `VLLM_BATCH_INVARIANT` 强制 fused 路径。限制 split 值集合，避免新增大量
   Triton 编译变体。
4. **验证与迭代。** 在独立于规则拟合的 batch/vocab 点上测候选与原 PR；运行
   float64 oracle 正确性检查和相关现有测试。主要判据是目标形状稳定改善，
   保护集不出现显著退化；报告逐点数值，而不是只报总体平均。
5. **后续课题。** 只有选择策略稳定后，才以独立实验测空 Triton cache 的首次编译
   和编译变体数。本轮不以模型服务 TTFT 或并发数据支撑 kernel 结论。

## 实验记录

### 2026-10-15 00:59：启动

- 核对了两个仓库均无未提交改动，建立上表两个新分支；没有推送。
- 核对 PR 实现：原自动规则的目标程序数是 `2 × SM`，每片至少 8192 logits，
  split 数向下取 2 的幂，最多 64；开启 `VLLM_BATCH_INVARIANT` 时返回 1。
- 核对本地与 H200 历史结果。H200 独立复现的三轮 PR auto 中位数为：
  B=1 8.02 µs、B=16 8.67 µs、B=32 13.98 µs、B=64 25.27 µs、
  B=128 46.99 µs。它们没有强制 split 对照，不能用于拟合最佳 split。
- GPU 在正常执行环境可见；尚未启动本轮新测量。

### 2026-10-15 01:01：第一次 3070 Ti split 扫描

- 运行了 `bench_token_logprobs.py`，固定 `V=151936,K=6,bf16`；B 包含
  `1,2,3,4,5,6,7,8,10,11,12,13,14,16,20,23,24,25,31,32,40,46,47,48,63,64,96,128`。
  每个 B 测 `auto,1,2,4,8,16,32,64`，合计 224 个配置；224/224 均通过
  float64 oracle 正确性检查。原始结果：
  `results/kernel/issue60095/split-selection/sweep1/{result.json,result.csv}`。
- benchmark 新增 `effective_splits` 和 `gpu_sm_count` 字段，不改计时方法。
- 单轮设备时间提示明显选择损失：B=1 auto 16 splits 为 10.92 µs，
  强制 64 为 6.01 µs；B=24 auto 2 为 66.69 µs，强制 16 为 21.04 µs；
  B=47 auto 1 为 131.11 µs，强制 8 为 37.39 µs；B=128 auto 1 为
  150.58 µs，强制 32 为 95.20 µs。单轮结果只用于定位，最终策略依赖重复测量。
- 原规则在本机的切换点为 B=6（16→8）、B=12（8→4）、B=24（4→2）、
  B=47（2→1）。因此已把这些边界两侧纳入扫描。
- 已启动第二、第三轮，并交叉改变 split 测试顺序；分析脚本
  `split_selection/analyze_split_sweeps.py` 将输出逐形状中位数、范围和最佳强制 split 对照。

### 2026-10-15 01:05：三轮基线与扩展词表完成

- `V=151936` 的 28 个 B × 8 种 split × 3 轮，以及
  `V∈{32768,65536,100000,131072}` 的 28 个形状 × 8 种 split × 3 轮，
  合计 1344 行；分析脚本检查全部 `status=ok,correct=true`，没有失败行。
  原始数据在 `results/kernel/issue60095/split-selection/{sweep1,sweep2,sweep3,vocab1,vocab2,vocab3}/`。
- 三轮中位数汇总在 `results/kernel/issue60095/split-selection/{baseline_analysis,vocab_analysis}/`。
  对 `V=151936`：B=1 原 auto 10.92 µs、最佳强制 64 为 6.02 µs；
  B=24 为 65.77 对 21.04 µs；B=47 为 126.97 对 37.02 µs；
  B=128 为 148.97 对 94.24 µs。强制最优只作离线 oracle，不在推理请求中扫描。
- 新增 `split_selection/fit_split_rule.py`，用实测强制 split 中位数评估候选的
  `target_per_sm/min_chunk/max_splits` 常量组合。56 个已测形状上，
  `target_per_sm=16,min_chunk=1024,max_splits=64` 的离线预测为几何平均
  2.14×、最差 1.42×（相对原 auto），且无 >5% 退化。这**只是按已有数据重放**；
  需要真正修改 auto 后在留出形状上复测。
- 已启动小词表 `V∈{3001,8192,16384}` 扫描，检验更积极切分是否引入短词表退化。

### 2026-10-15 01:20：确定候选规则，保留独立留出集

- 小词表探索：`V=3001` 的 fused 路径约 3.9–4.6 µs，切分没有稳定明显收益；
  `V=8192/16384` 的强制 8/16 splits 在多数 B 显著优于原 auto。
  因此候选仅在 `V>=8192` 使用较积极的规则；小词表沿用原规则。
- 离线搜索 `target_per_sm∈{2,4,8,16,32,64,128}`、
  `min_chunk∈{1024,2048,4096,8192,16384}`、`max_splits∈{16,32,64}`
  及最小词表门槛。选择较保守的 `(64,1024,64,min_vocab=8192)`：
  在训练形状和单轮小词表形状上，用实测强制 split 时间重放，几何平均预测
  1.97×、最差约 1.00×，无 >5% 退化。它不是新的 GPU 实测结论。
- 已在 vLLM 实验分支加入设备/形状查表，仅对 `(compute capability 8.6, 46 SM)`
  且 `V>=8192` 启用上述参数；其他设备和更小词表保持 PR 原选择，
  `VLLM_BATCH_INVARIANT` 仍强制 1。新增一个针对选择行为的测试。
- 候选 smoke：B=1,V=151936 的 auto 选 64，约 5.95 µs；
  B=128,V=151936 选 16，约 96.80 µs；B=1,V=3001 保持 1。
  6/6 smoke 配置通过 float64 oracle。
- 尝试运行现有 `tests/v1/sample/test_logprobs.py` 中的选择测试，但 pytest 在收集阶段
  因 PR worktree 未安装 `_vllm_fa2_C/_vllm_fa3_C` 扩展而停止；当前共享编译扩展
  安装在 `src/vllm`。后续用隔离的选择函数检查和 benchmark oracle 验证，
  并在结论中保留这个测试限制。
- **预先固定留出验证形状**：`V∈{49152,98304,200000}`，
  `B∈{1,9,17,33,49,65,127,192,256}`。此数据不参与上面的离线拟合。
  另外对关键形状集 `V∈{8192,32768,65536,131072,151936}`、
  `B∈{1,8,16,24,32,47,64,96,128}` 做三轮交叉 A/B。
- 为公平 A/B，新建了只读参考 worktree
  `worktrees/vllm-60097-pr-reference`，仍固定在 `0ddd601779...`；
  实验分支位于 `worktrees/vllm-60097-pr`。比较脚本检查两边源码 SHA256、
  硬件/运行时和形状集合一致。

### 2026-10-15 01:27：关键形状三轮交叉 A/B 完成

- 参考组是干净的 PR #60097 源码，候选组是本地未提交的选择规则；两组均由同一
  benchmark、同一编译扩展和同一 GPU 运行。源码 SHA256 不同，避免仅用相同 Git
  commit 误认两组相同。三轮轮流改变先后顺序，逐形状取三轮 CUDA Graph 时间中位数。
- 45 个形状（5 个词表长度 × 9 个 batch size）均为 `status=ok,correct=true`。
  相对于 PR auto，候选规则逐形状加速比的几何平均为 **2.042×**，最小为
  **1.172×**，没有超过 5% 的退化。
- `V=151936,K=6,bf16` 的代表数据（原 PR → 候选，单位 µs）：

  | B | 原 PR auto split/时间 | 候选 split/时间 | 加速 |
  |---:|---:|---:|---:|
  | 1 | 16 / 8.69 | 64 / 5.91 | 1.47× |
  | 24 | 2 / 66.12 | 64 / 22.17 | 2.98× |
  | 47 | 1 / 126.10 | 32 / 37.08 | 3.40× |
  | 128 | 1 / 135.60 | 16 / 94.93 | 1.43× |

- 完整原始结果和逐形状报告：
  `results/kernel/issue60095/split-selection/ab_key_20261014T172317_445802925/`。
  这组结果证明本机关键形状上的改进；还需对预先声明的留出形状验证。

### 2026-10-15 01:30：独立留出形状验证与主机开销发现

- 留出集为 `V∈{49152,98304,200000}`、`B∈{1,9,17,33,49,65,127,192,256}`，
  共 27 个形状，三轮交叉 A/B。全部通过 float64 oracle；CUDA Graph 设备时间
  几何平均加速 **1.918×**，最小加速 **1.014×**，无超过 5% 的设备时间退化。
  完整数据：`results/kernel/issue60095/split-selection/ab_holdout_20261014T172515_568557916/`。
- 进一步检查 benchmark 的 `warm_wall_ms`（包含 Python 调度、kernel launch 和
  GPU 同步）：关键集 45 个形状中 24 个、留出集 27 个形状中 9 个的候选中位数
  比原 PR 慢超过 5%。特别是 `V=8192` 时原规则走单 kernel，候选改为两阶段；
  B=1 原约 17.7 µs，候选约 38.8 µs，虽然 CUDA Graph 设备时间为 7.81→3.72 µs。
  `V=151936,B=24` 的主机同步时间则由 70.6 降至 38.8 µs。
- **解释与决策**：CUDA Graph 数字证明 GPU kernel 工作加快，却不能据此声称每个
  请求调用更快。需要修正规则以保护短任务，并测选择函数本身开销；最终报告同时
  给出设备时间和主机同步时间，避免只挑有利指标。

### 2026-10-15 01:34：第二版候选与新验证集

- 第一版在 8192 词表的同步调用时间约慢 2.2 倍，不能直接作为完整性能改进提交。
  第二版只在本机 `(SM 8.6,46 SM)`、`V>=49152` 且 `B×V>=2,300,000` 时用
  `(target_per_sm=64,min_chunk=1024)`；其他形状沿用 PR 原规则。
  门槛来自第一版 A/B 的同步调用时间转折点，属于在前一留出集上继续调参。
- 因前一留出集已参与修正，**不能再将它称为第二版规则的独立验证**。预先登记
  第二版新的未见形状：`V∈{55296,114688,180224}`、
  `B∈{7,15,19,27,39,55,73,111,173,224}`；三轮交叉 A/B，分别检查
  GPU 设备时间和同步调用时间。关键集与旧留出集只是回归对照。
- 单独计时选择函数，第一版原 PR 约 3.03 µs、候选约 6.88 µs/次（本机
  10000 次 Python 调用粗测）。第二版先判断形状门槛，绝大多数不适用形状
  避免额外能力查询；主机调用结果仍需复测。

### 2026-10-15 01:37：第二版关键集回归结果

- 关键集 45 形状三轮交叉 A/B，所有测量与 float64 oracle 一致。CUDA Graph
  设备时间几何平均 **1.393×**，同步调用时间几何平均 **1.242×**。
  源数据：`results/kernel/issue60095/split-selection/ab_key_20261014T172827_030403158/`。
- 真正改变 split 的 17 个形状：设备时间几何平均 **2.385×**、最小 **1.417×**；
  同步调用时间几何平均 **1.829×**、最小 **1.086×**。
- 其余 28 个形状沿用原 split；设备时间几何平均约 **1.004×**，同步调用
  时间约 **0.982×**。同 split 组有两项设备时间和五项同步调用时间超过 5%
  的退化，最大约 6.6%；主要发生在 `V=8192`、`V=32768` 和很小 batch。
  这些形状没有切换 kernel 路径，差异可能来自 Python 判断开销、GPU 频率或
  仅 10 次 warm wall 调用的测量噪声。不能直接把它们解释为 kernel 退化。
- 第二版的选择阈值由前一轮数据得出；接下来用先登记的第三套新形状验证。

### 2026-10-15 01:40：第二版独立新形状验证

- 对预先登记的 `V∈{55296,114688,180224}`、十个 B，共 30 个形状做三轮
  交叉 A/B；180/180 行均为 `status=ok,correct=true`。全部形状的设备时间
  几何平均 **1.538×**、同步调用时间几何平均 **1.341×**。
  原始数据：`results/kernel/issue60095/split-selection/ab_guarded_holdout_20261014T172922_388927645/`。
- 改变 split 的 21 个形状：设备时间几何平均 **1.878×**、最小 **1.011×**；
  同步调用时间几何平均 **1.532×**、最小 **0.976×**（约 2.4% 慢，低于 5%）。
  仍没有足够证据声称每个受影响形状都改善：其中部分大 batch 的设备收益接近零。
- 保持原 split 的 9 个形状：设备时间几何平均 **0.966×**、同步调用时间
  **0.982×**。三项设备时间下降超过 5%，最差约 11%；同步调用只有一项
  下降超过 5%，最差约 6.4%。这些形状的 GPU kernel 配置完全相同，故针对
  异常点再做交叉复测，以区分频率变化、计时误差和主机选择开销。

### 2026-10-15 01:43：异常点复测、额外正确性与 baseline 对照

- 对 `V∈{114688,180224}`、`B∈{7,15,19}` 三轮交叉复测：先前设备时间
  看似下降 8%–11% 的未切换 split 点，此次均约为 **1.000×**，说明原异常
  主要是计时/频率波动。未切换 split 的同步调用时间仍有 1%–6% 的变慢，
  与额外 Python 判断开销及短样本计时误差相容；不能排除微小真实开销。
  数据：`results/kernel/issue60095/split-selection/ab_fallback_check_20261014T173026_351907194/`。
- 候选对 `float16` 的 `K=1,32`、`float32` 的 `K=1,32`、`bfloat16` 的
  `K=32` 各测九个形状，**45/45** 均通过 float64 oracle；此前主实验的
  `bfloat16,K=6` 也全部通过。额外检查仅验证正确性，较小的计时参数不用于
  性能结论。数据：`results/kernel/issue60095/split-selection/dtype_correctness_20261014T174019/`。
- 应用户要求，加入与 PR **之前**的 baseline auto（三轮、相同环境和关键形状）
  对照。baseline commit 为 `d09ff7763747bbe7cd153edd8017479804f47be7`；
  候选基于 PR commit `0ddd6017793c943a2091a5800f0cbac1daaa3e02` 加
  本地未提交规则。关键集 45 形状的设备时间几何平均 **2.910×**、同步调用
  时间 **1.858×**。最小设备加速 **1.059×**；同步调用有九项超过 5%
  退化，最差 **0.774×**，主要是短任务引入两阶段 kernel 的 PR 既有开销。
  数据：`results/kernel/issue60095/split-selection/ab_baseline_key_20261014T174137_843147812/`。
- **归因边界**：候选 vs baseline 包括 PR #60097 作者的 kernel 改动；
  本轮自行实现的收益只由候选 vs 干净 PR #60097 的对照支持。
- baseline auto 与候选在第二版独立新形状集 30 形状的三轮对照：设备时间
  几何平均 **3.589×**、最小 **1.828×**；同步调用时间几何平均 **2.664×**、
  最小 **1.638×**，无超过 5% 的退化。数据：
  `results/kernel/issue60095/split-selection/ab_baseline_guarded_holdout_20261014T174237_777733413/`。
- 独立导入候选源并覆盖形状门槛、SM8.6 选择、Hopper fallback 与
  `VLLM_BATCH_INVARIANT`，选择函数 **11/11** 断言通过。现有 pytest 模块仍因
  PR worktree 缺少匹配的本地编译扩展无法收集；这不等价于上游测试通过。
- 为调查小幅同步调用退化，给 benchmark 加入 `--wall-repetitions`，并在
  六个复测形状上改为每轮 **100 次**同步调用、三轮交叉。改变 split 的两项
  同步调用加速约 1.44×；同 split 的四项中最差仍约 0.942×（约 1.8 µs）。
  单独计时选择函数，同 split 形状原 PR 3.01 µs、候选 3.22 µs/次；
  当前证据无法把整段 1.8 µs 都归因于选择函数，可能含频率/计时差异。
  这是保留的性能风险，H200 与服务级实验需继续观察。数据：
  `results/kernel/issue60095/split-selection/ab_pr_fallback_check_20261014T174641_328283900/`。
- 已保存可从固定 PR commit 应用的本地补丁
  `benchmarks/kernels/issue60095/split_selection/selector_candidate_sm86.patch`，SHA256
  `7a337c59f7fef6c402e3bb1a97afe3ad633692eb05e9c44e443189e1377bce31`；
  在干净的 PR 参考 worktree 上执行 `git apply --check` 通过。它目前只针对
  SM8.6/46 SM；拷贝到 H200 不会使 H200 采用新策略。

## 本机最终结果与归因

| 对照 | 形状 | GPU 设备时间几何平均加速 | 同步调用时间几何平均加速 | 说明 |
|---|---:|---:|---:|---|
| 原 PR auto → 第二版候选，关键集 | 45 | 1.393× | 1.242× | 拟合/回归形状，不算独立验证 |
| 原 PR auto → 第二版候选，新形状集 | 30 | 1.538× | 1.341× | 第二版规则的独立形状验证 |
| PR 之前 baseline auto → 第二版候选，关键集 | 45 | 2.910× | 1.858× | 包含 PR 作者的 kernel 收益 |
| PR 之前 baseline auto → 第二版候选，新形状集 | 30 | 3.589× | 2.664× | 包含 PR 作者的 kernel 收益 |

所有表中数字是三轮交叉运行后，对每个形状取时间中位数，再对形状加速比取
几何平均；不是某个单请求的端到端加速。关键集与新形状集等权统计，未按线上
请求频率加权。当前最有说服力的原创结论是：在 RTX 3070 Ti 的新形状集上，
第二版选择规则相对干净的 PR #60097 auto 提升 GPU 设备时间 1.538×、
同步调用时间 1.341×。改变 split 的 21 个形状同步调用时间最差约 0.976×；
未改变 split 的形状仍出现 1%–6% 的同步调用波动，应在更长测量和服务路径中复核。

对主要目标 `V=151936,K=6,bfloat16`，三轮关键集里的代表形状如下。两列
对照来自分别执行的交叉 A/B，不能把不同实验轮次的时间相减当成新的配对实验：

| B | 原 PR auto → 候选 split | 原 PR → 候选设备 µs | 设备加速 | 原 PR → 候选同步 µs | 同步加速 | baseline → 候选设备 µs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 → 16 | 9.01 → 9.59 | 0.94× | 31.99 → 32.40 | 0.99× | 133.43 → 9.52 |
| 24 | 2 → 64 | 66.61 → 21.89 | 3.04× | 72.06 → 37.39 | 1.93× | 178.43 → 21.88 |
| 47 | 1 → 32 | 126.40 → 37.01 | 3.42× | 129.53 → 42.47 | 3.05× | 183.84 → 36.99 |
| 128 | 1 → 16 | 135.76 → 94.65 | 1.43× | 139.49 → 99.90 | 1.40× | 213.25 → 94.48 |

B=1 未改变 split，0.94× 的设备计时属于同路径间的波动，不能解释为
选择规则改变了 kernel；它也提醒我们不能把几何平均收益推广到每个请求。

**不能声称** H200 已经因这个补丁提速。当前补丁只匹配 SM8.6、46 SM，
在 H200 上会走 PR 原逻辑。也不能把候选对 baseline 的全部收益写成个人贡献；
那其中有 PR 作者的一次读取词表、在线规约和 split kernel 改进。

## H200 下一轮：实施条件与执行顺序

### 1. 先准备可传输的确定版本

今晚没有提交或推送任何内容，云端 `main` 不包含这里的脚本和补丁。要在 H200
继续，先把本地 `benchmarks/kernels/issue60095/` 整个目录传到
`/workspace/vllm_lab/benchmarks/kernels/issue60095/`，或在以后获准提交和推送后
克隆指定外层分支。不能拿远端 `main` 代替本轮未提交工作。补丁文件为
`split_selection/selector_candidate_sm86.patch`，但它在 H200 上只是保存本地实现的参考。
`results/` 当前被外层 `.gitignore` 忽略；如果要保留完整原始测量，必须单独
传输、归档，或明确挑选精简结果再纳入版本管理。

在云端保留三个独立源码版本：

| 名称 | 固定源码 | 用途 |
|---|---|---|
| `vllm-60097-base` | `d09ff7763747bbe7cd153edd8017479804f47be7` | PR 前 baseline |
| `vllm-60097-pr-reference` | `0ddd6017793c943a2091a5800f0cbac1daaa3e02` | 干净 PR auto / 强制 split 扫描 |
| `vllm-60097-pr` | 同一 PR commit + 本地实验补丁 | 候选策略 |

建议把 vLLM fork 克隆在 `src/vllm`，加上
`https://github.com/vllm-project/vllm.git` 为 upstream，然后按固定 SHA 建三个
worktree；不要切换当前学习用 checkout 的分支。可执行的骨架如下，前提是外层
实验目录已经传到 `/workspace/vllm_lab`：

```bash
cd /workspace/vllm_lab
LAB=/workspace/vllm_lab
mkdir -p "$LAB/src" "$LAB/worktrees"
git clone git@github.com:RichterWang/vllm.git "$LAB/src/vllm"
git -C "$LAB/src/vllm" remote add upstream https://github.com/vllm-project/vllm.git
git -C "$LAB/src/vllm" fetch upstream main refs/pull/60097/head
git -C "$LAB/src/vllm" cat-file -e d09ff7763747bbe7cd153edd8017479804f47be7^{commit}
git -C "$LAB/src/vllm" cat-file -e 0ddd6017793c943a2091a5800f0cbac1daaa3e02^{commit}
git -C "$LAB/src/vllm" worktree add --detach "$LAB/worktrees/vllm-60097-base" d09ff7763747bbe7cd153edd8017479804f47be7
git -C "$LAB/src/vllm" worktree add --detach "$LAB/worktrees/vllm-60097-pr-reference" 0ddd6017793c943a2091a5800f0cbac1daaa3e02
git -C "$LAB/src/vllm" worktree add -b codex/issue60095-h200-tuning "$LAB/worktrees/vllm-60097-pr" 0ddd6017793c943a2091a5800f0cbac1daaa3e02
git -C "$LAB/worktrees/vllm-60097-pr" apply --check "$LAB/benchmarks/kernels/issue60095/split_selection/selector_candidate_sm86.patch"
git -C "$LAB/worktrees/vllm-60097-pr" apply "$LAB/benchmarks/kernels/issue60095/split_selection/selector_candidate_sm86.patch"
```

路径需以云端实际 Git 工作目录核对；若 PR head 已变动，仍需确认上述固定 SHA
存在，不能静默替换。脚本的 `--binary-root` 默认指向 `src/vllm`：按该 vLLM
版本的 CUDA 安装说明创建 `src/vllm/.venv`，安装/构建匹配的本地扩展，并用
同一个解释器、同一个扩展根目录运行所有 microbenchmark。准备条件至少包括：

- H200 在容器内可见，`nvidia-smi` 和独立 Python 进程的
  `torch.cuda.is_available()` 均成功；打印 GPU 名称、SM 数、驱动、CUDA、
  PyTorch、Triton 与 `pip freeze`，保存到结果目录。
- CUDA 驱动和所装 PyTorch runtime 兼容；源码扩展可导入且与当前源码 ABI
  匹配。若 baseline 无法共用扩展，则分别构建并记录这一实验差异。
- 保证 GPU 空闲、功率/频率策略一致、热身和 Triton cache 状态一致；每轮
  交叉改变 reference/candidate 顺序。首次编译与稳定时间分别测。
- 有足够显存容纳最大 `B×V` logits 与 float64 oracle；若大形状 OOM，就明确
  剔除并记录，不偷偷减小形状后沿用原结论。

### 2. H200 先测 PR auto 与强制 split，再设计 H200 规则

当前 SM8.6 规则不会改变 H200。先在干净 PR 参考版本上固定
`K=6,bfloat16`，扫描 `B∈{1,8,16,24,32,47,64,96,128,192,256}`、
`V∈{32768,65536,98304,151936,200000}`，每形状测
`auto,1,2,4,8,16,32,64`，三轮更换 split 顺序。重点复核 PR 讨论的
`B=128,V=151936`：auto 2 splits 与强制 16/32 splits。保存每行的
`effective_splits`、正确性、CUDA Graph 时间、同步调用时间、源码 SHA。
参考命令（每轮用不同输出目录）：

```bash
PY=/workspace/vllm_lab/src/vllm/.venv/bin/python
LAB=/workspace/vllm_lab
"$PY" "$LAB/benchmarks/kernels/issue60095/bench_token_logprobs.py" \
  --vllm-root "$LAB/worktrees/vllm-60097-pr-reference" \
  --binary-root "$LAB/src/vllm" \
  --select 1,8,16,24,32,47,64,96,128,192,256 \
  --vocab-sizes 32768,65536,98304,151936,200000 \
  --splits auto,1,2,4,8,16,32,64 \
  --wall-repetitions 100 \
  --output-dir "$LAB/results/kernel/issue60095/h200/forced_round1"
```

只有实测后才在 H200 实验分支增加 SM9.0 对应的选择规则；用离线数据拟合，
请求到来时不现场 autotune。不能简单复用 3070 Ti 的阈值。把
`V∈{49152,114688,180224}`、`B∈{3,7,15,31,63,127,129,223}`
预留为 H200 新形状，不参与规则拟合。若发现其他风险点，可追加保护集，
但必须写明追加时间，不能伪称预留集。

### 3. H200 三臂验证、冷启动与服务路径

规则固定后用相同形状、相同数据、相同编译扩展做三轮交叉比较：
`baseline auto`、`PR #60097 auto`、`H200 candidate auto`。现有
`split_selection/run_selector_ab.sh` 可用 `key pr` 和 `key baseline` 进行 3070 Ti 既定形状的
双臂比较；H200 自定义形状可直接重复上面的 benchmark 命令，再交给
`split_selection/compare_auto_ab.py --reference <三轮目录> --candidate <三轮目录>
--output-dir <分析目录>`。将 `WALL_REPETITIONS=100` 用于同步调用计时，
并使用至少三轮而非单轮最小值。

报告所有形状的三轮时间、中位数、范围、最小加速、>5% 退化数、失败数。
对 `float16/float32/bfloat16`、`K=1/6/32` 做正确性抽样；在安装了匹配
扩展的候选环境运行 `tests/v1/sample/test_logprobs.py` 的相关测试。
另用不同的空 `TRITON_CACHE_DIR` 比较首次编译、首次调用和实际编译变体数，
避免把热 cache 的首调用当成冷启动。若要声称 TTFT/TPOT/吞吐量收益，再
启动同一模型并发送真正携带 `logprobs` 参数的请求，记录线上请求的 TTFT、
ITL、TPOT、吞吐量及错误率；普通随机文本 benchmark 不一定触发这个 kernel。

建议的提交门槛：H200 目标形状在设备时间和同步调用时间都稳定改善；预留
形状无明确的大幅退化；正确性、编译成本和服务路径没有不可接受的倒退。
任何门槛未达到都如实写在报告中，不用 3070 Ti 数据代替 H200 结论。

## H200 结果补齐后的报告格式

在本文新增“`H200 实测，日期，GPU/驱动/源码 SHA`”小节，至少填：

| 形状 B/V/K/dtype | baseline auto µs | PR auto split/µs | H200 候选 split/µs | 候选/PR 设备加速 | 候选/PR 同步调用加速 | 正确性 |
|---|---:|---:|---:|---:|---:|---|
| `128/151936/6/bf16` | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

表后写三轮逐形状原始文件路径、所用加速定义、几何平均和最差值、退化形状
清单、冷启动/变体数、如果做了服务测试则写真实的端到端指标。分开陈述：
`baseline → PR` 是作者已有贡献，`PR → H200 候选` 才是本项目的增量。
若 H200 候选无提升或过度依赖特定模型形状，也需要保留失败实验和原因。

## 向 vLLM 社区提交 PR 的准备与组织

1. 提交前重查 #60097 的实时状态、讨论和最近 commit。若它尚未合入，先在
   讨论中给出独立复现与 H200 新证据，确认维护者是否欢迎叠加选择规则；
   避免提交重复优化。若已合入，以合入后的 upstream `main` 重建基线，
   重新跑测试和性能数据，不能直接沿用旧 SHA 的比较。
2. 在个人 vLLM fork 的新分支上，仅放生产选择逻辑和相关单元测试；
   benchmark、原始数据与长篇实验报告留在外层 `vllm-lab`，PR 描述链接到
   可访问的精简报告。代码中解释设备选择范围与 fallback，避免把单机
   数字硬编码成未经验证的全局默认。
3. PR 描述依次写：问题与复现形状、原 PR auto 的损失、所提规则及为何没有
   运行时 autotune、H200 和 3070 Ti 环境、三轮 A/B 表格、所有退化、
   正确性与编译成本、可复制命令、适用边界。附上原作者 PR 与 issue 链接，
   明确原创部分只在选择策略。不要用 baseline→候选的总加速充当原创收益。
4. 通过相关上游测试、格式检查、`git diff --check`；让 CI 在 PR 页面运行。
   必要时补服务级请求，尤其是同步调用时间或 TTFT 被质疑时。
5. 今晚只保留本地分支和补丁；没有提交、推送、创建 PR。后续如需发布，
   先完成 H200 实验与回归检查，再决定是否值得发 PR。

## 剩余限制与文件索引

- 只在单台 RTX 3070 Ti 测了最终规则；没有最终候选的 H200、冷启动编译或
  真实服务请求数据。H200 早先的 base/PR 独立复现见
  `report_h200_60097/independent_h200_reproduction.md`，不能代替新候选验证。
- 现有 pytest 收集因编译扩展路径不匹配失败；独立选择断言和 kernel oracle
  通过，但不能宣称上游测试全绿。warm wall 每轮仅 10 次调用，微小差异
  需要增加重复数。CUDA Graph 时间没有 Python launch 开销。
- 当前规则只为本机硬件启用，是可复现原型，还不是可直接向上游推广的策略。
- 实验入口：`bench_token_logprobs.py`；交叉 runner：`split_selection/run_selector_ab.sh`；
  比较脚本：`split_selection/compare_auto_ab.py`；离线规则拟合：`split_selection/fit_split_rule.py`；
  本地补丁：`split_selection/selector_candidate_sm86.patch`。原始数据全部保存在忽略的
  `results/kernel/issue60095/split-selection/`，请勿在清理仓库时误删。

## 验收状态（2026-10-15）

- [x] 本地独立分支、干净 PR 参考 worktree 和 PR 前 baseline worktree。
- [x] 强制 split 扫描、规则拟合、二次规则修正、三轮交叉 A/B 与独立新形状验证。
- [x] 与 PR auto、与 PR 前 baseline auto 分别比较；补充同步调用时间与异常点复测。
- [x] 不同 dtype/K 的 oracle、选择函数隔离断言、脚本语法与 diff 空白检查。
- [x] 本文记录 H200 环境条件、实验矩阵、报告格式和上游 PR 准备要求。
- [ ] H200 上针对当前 PR 的强制 split 扫描及 H200 专属候选策略。
- [ ] 匹配扩展环境中的上游 pytest、冷启动编译和真实服务请求验证。
- [ ] 上游 PR：尚不满足提交证据门槛，也未提交、推送或创建。

## 2026-10-15 后续实验：初始化校准与缓存

用户要求继续在现有 `codex/issue60095-split-selection` 实验分支上验证：
每次 vLLM 初始化先检查对应 GPU/源码/软件环境的校准结果；命中则直接加载，
缺失才测不同 split，后续请求只使用已保存的选择表。需要分别量化首次校准
与缓存命中的启动成本，并对干净 PR #60097 auto 和 PR 前 baseline auto
做三轮 A/B。当前 3070 Ti 实验完成后仍须在 H200 上重新校准与复测，不能
复制 3070 Ti 的缓存结果。此节以下持续记录新增实现与实际数据。

### 本轮实现（本地未提交、未推送）

- 外层仓库和候选 vLLM worktree 都保持在 `codex/issue60095-split-selection`；PR 参考仍是 `0ddd6017793c943a2091a5800f0cbac1daaa3e02`，PR 前 baseline 是 `d09ff7763747bbe7cd153edd8017479804f47be7`。没有切换或重命名分支。
- 候选的 `vllm/v1/worker/gpu_worker.py` 在 `load_model()` 结束后、KV cache 可用内存探测前，检测 `VLLM_TOKEN_LOGPROB_AUTOTUNE=1`。开启后每次初始化调用校准入口；缓存命中直接加载表，未命中才测试。默认关闭以避免在尚未验证的设备上无条件增加启动开销。校准失败会记录异常并沿用现有规则。
- 新文件 `vllm/v1/worker/gpu/sample/logprob_autotune.py` 用 GPU UUID、型号/算力/SM/显存、PyTorch/CUDA/Triton 版本、kernel 与校准器源码哈希、V/K/dtype、B 与 split 网格生成键。使用文件锁防止并发进程重复测试，原子写 JSON；缓存格式不符或数据无效时重测。默认目录在 `VLLM_CACHE_ROOT/token_logprob_autotune`，实验可用 `VLLM_TOKEN_LOGPROB_AUTOTUNE_CACHE_DIR` 指定。
- 校准矩阵固定 B=`1,8,16,32,64,128`，split=`1,2,4,8,16,32,64`，默认 K=6。可以设置 `VLLM_TOKEN_LOGPROB_AUTOTUNE_KS=1,2,6,11` 在启动时预热多个 K；各 K 的选择表同时保存在进程内，每个 K 独立持久化。每个 split 先通过 FP64 oracle 正确性检查，再做三轮交错、同步 wall 计时；只有相对原 PR 选择至少快 5% 才改选。其余 B 取下一个已测 B 的选择，超过最大 B 则退回旧策略。缓存只用于完全匹配的 V/dtype/K；未覆盖组合走原有固定规则。`VLLM_BATCH_INVARIANT` 仍强制不分割。
- benchmark 入口新增 `--initialize-autotune` 和 `--autotune-cache-dir`；`split_selection/run_selector_ab.sh autotune_vocab pr|baseline` 对同一 V/K/dtype 的 9 个 B 做三轮交叉 A/B。旧 PR 选择函数签名不同，benchmark 已做兼容。新增选择表与 fallback 的单元测试，但完整 pytest 仍受之前记录的扩展加载问题限制。

### 3070 Ti 实测：校准与缓存成本

环境：RTX 3070 Ti Laptop GPU，46 SM、8 GiB，PyTorch `2.13.0+cu132`，Triton `3.7.1`；V=151936、bf16、K=6。以下时间只覆盖 `initialize_split_autotune()` 调用，不包含模型加载和整个 Python 进程启动。

| 情况 | 结果 | 耗时 |
|---|---|---:|
| 空选择缓存，已有 Triton 编译缓存 | 首次完成校准并写盘 | 0.460 s（多 K 扩展前的同一校准算法，一次） |
| 空选择缓存、空 Triton 编译缓存 | 首次完成校准并写盘 | 1.283 s（多 K 扩展前的同一校准算法，一次） |
| 有效选择缓存 | 不重测，直接加载 | 六次 A/B 中 0.280–0.301 ms 为常见值；另有一次 26.636 ms |
| 独立进程缓存命中断言 | 不重测；中间 B 与 K fallback 正确 | 14.31 ms（一次） |

缓存命中耗时出现进程调度/设备属性查询相关长尾；不能只声称固定 0.3 ms。空编译缓存样本只有一次，也不能当作跨环境的稳定启动成本。三次独立 K=6 校准的 B=16 分别选出 64、16、32 splits，显示接近最优的选择仍受测量噪声影响；它们相对原 PR 的 4 splits 都有明显余量，但上线前应增加重复轮次或设置近似最优时偏向较少 split 的规则。原始数据：`results/kernel/issue60095/autotune/final_cold/`、`final_empty_run/`，以及下述 A/B 目录中的 `result.json`。多 K 扩展后的当前源码，冷编译缓存的 K=1/2/6/11 连续校准成本见末节；当前源码的 K=6 单项校准在最终 PR A/B 中为 0.983 s（已有 Triton 编译缓存）。

### 3070 Ti 实测：三轮交叉 A/B

同一 benchmark、同一编译扩展根目录，V=151936、bf16、K=6，B=`1,8,16,24,32,47,64,96,128`。每个请求结果都通过 FP64 oracle；最终代码的 PR 对照第一轮候选进程在计时前重新校准（0.983 s），后两轮缓存命中（0.28–0.29 ms）。baseline 对照三轮候选全部命中（0.29–0.32 ms）。加速比是每个形状先取三轮时间中位数，再计算参考/候选；总体为各形状加速比几何平均。CUDA Graph 时间不含 Python 调用开销；同步 wall 包含 Python 调用与同步。

| 参考 | CUDA Graph 几何平均加速 | 最差形状 | 同步 wall 几何平均加速 | 最差形状 | 超过 5% 退化 |
|---|---:|---:|---:|---:|---:|
| PR #60097 auto | **2.118×** | 0.999× | **1.856×** | 1.000× | 0/9 |
| PR 前 baseline auto | **5.892×** | 2.256× | **4.276×** | 2.160× | 0/9 |

PR 对照原始目录：`results/kernel/issue60095/split-selection/ab_pr_autotune_vocab_20261015T032801_468622201/`；baseline 对照：`results/kernel/issue60095/split-selection/ab_baseline_autotune_vocab_20261015T032841_083201267/`。每个目录有六次 `result.json`/CSV 与 `analysis/comparison.csv`、`summary.json`。baseline→PR 的已有收益属于原作者；本项目的增量应主要看 PR→候选。以上只证明 3070 Ti 的 kernel 层结果，尚无 H200 或服务级 TTFT 结论。

### 新 K 首次出现时是否马上校准

目前**没有**在请求路径对新 K 同步校准。K 会改变收集 token 的宽度与 Triton 的 `TOPK_BLOCK_SIZE` 编译常量；任意 K 的缓存不能无证据复用。现实现仅对命中的 K 使用选择表，其他 K 走原有选择规则，保证请求不中断。

为了估计取舍，单独在 K=7、V=151936、空 Triton 编译缓存下测得：不校准、首次调用原规则的 wall 约 **654.83 ms**（含该路径首次编译）；先同步完成全部 split 扫描约 **1.275 s**，之后首次调用约 0.167 ms。两组是独立进程的一次性测量，仅说明量级；新增 K 的同步校准会把约秒级工作放在该请求前面，还可能与线上模型共享 GPU/显存，因此目前不值得默认启用。

可采用的下一步是：首次遇到新 K 时先按旧规则服务并记录该 K；由显式维护操作或确认 worker 空闲后的后台任务做一次校准，完成后原子安装到进程表并持久化。正式实现前须解决后台 CUDA 与推理流竞争、显存峰值、线程安全、重复任务去重，以及新 K 对应 Triton 变体的编译数量。更简单的第一步是在初始化参数中指定一小组预计使用的 K，启动时批量校准；代价是启动时间随 K 数近似增长。H200 需先测 K=1/6/7/16 等使用频率与真实 TTFT，再决定是否值得做运行期扩展。

### H200 复测实施条件与步骤（尚未执行）

1. 在 H200 上拉取**当前本地改动发布后的对应版本**；发布前先确认分支 SHA 和源码差异。本机当前未提交、未推送，不能把云端旧分支当作本轮代码。保留 clean PR `0ddd6017` 和 baseline `d09ff776` worktree，以及匹配的 vLLM 扩展环境。
2. 在独立缓存目录设置 `VLLM_TOKEN_LOGPROB_AUTOTUNE=1`；记录 H200 UUID/SM/显存、驱动、PyTorch/CUDA/Triton 版本和源码哈希。先用空选择缓存与空 Triton 缓存测首次校准秒数和新增编译变体/缓存体积；重启进程至少三次量缓存命中开销，验证 H200 会生成不同键、不复用 3070 Ti 的表。修改源码或 K/dtype/V 后验证缓存失效重测。
3. 先强制扫描 split 1/2/4/8/16/32/64，B 覆盖 1、8、16、24、32、47、64、96、128 及切换边界，至少 V=151936/K=6/bf16。然后分别运行 `split_selection/run_selector_ab.sh autotune_vocab pr` 和 `... baseline` 三轮交叉 A/B；保存逐形状正确性、选择、设备时间、同步 wall 和退化。若要声称跨 K 有收益，另测 K=1/7/16 与真实采样配置。
4. 检查服务 `load_model` 的实际日志、KV 可用内存及 TTFT。校准在模型加载后、内存 profiling 前发生，但临时分配/编译仍须通过完整服务启动验证。首次启动额外成本必须与缓存命中启动分开报告。只有确认服务收益和退化边界，才考虑向上游提交 PR。
5. H200 报告按“环境与版本→校准成本/缓存命中→PR 对照逐形状→baseline 对照逐形状→正确性和退化→服务级结果→适用边界”组织。原始 CSV/JSON 与运行命令一并保留；清楚区分作者的 baseline→PR 收益与本项目 PR→候选增量。若 H200 选择无增益，也保留失败结果并说明原因。

本轮最后检查：`compileall` 和两仓库 `git diff --check` 通过；完整上游 pytest 与 H200/真实服务尚未验证。所有更改仍只在本地工作区，未提交、未推送。

### K 的含义与启动时预热多个 K（补充）

这里的 K 是 `compute_token_logprobs(logits, token_ids)` 的 `token_ids.shape[1]`，即每个 logits 行要返回概率的 token 个数；不是模型层数、KV cache 长度或请求 batch。普通 top-k 路径会放入 1 个实际采样 token，再附加请求的 top-k token，因此 `logprobs=0/1/5/10` 分别形成 K=`1/2/6/11`（前提是实际走 logprobs 计算路径）。如果请求指定 `logprob_token_ids`，K=`1+max(top-k 数、自定义 token ID 数)`；prompt logprobs 与 speculative decode 也调用此函数。模型主要决定 V/dtype，K 主要由请求配置决定，不能靠“覆盖常见模型”推断常见 K。

新版本允许在**初始化时**设置 `VLLM_TOKEN_LOGPROB_AUTOTUNE_KS=1,2,6,11`；默认仍仅 K=6，以免没有相关请求的服务白付编译与扫描成本。每个 K 单独缓存并在同一 worker 进程叠加选择表。未列出的 K 仍走原规则，不在首个请求中同步校准。建议已有工作负载以 `logprobs=5` 为主就保持 K=6；确实常用 `logprobs=0/1/5/10` 才预热 `1,2,6,11`。若实际只用其中两种，应只填对应两个 K；不要为了覆盖所有整数 K 而全量预热。自定义 token ID 请求应先统计生产流量的实际 K 分布再决定。

在 3070 Ti、V=151936、bf16、空选择缓存和空 Triton 编译缓存下，按 K=`1,2,6,11` 顺序初始化：逐个成本 **1.299/0.581/0.586/0.583 s**，总计 **3.049 s**。同一配置下重启进程再次加载：第一个 K 15.47 ms，后续各 K 0.16–0.22 ms，总计 **16.11 ms**。原始记录：`results/kernel/issue60095/autotune/multi_k_cold.jsonl` 与 `multi_k_hit.jsonl`。另做 K=1/2/6/11、B=16/64/128 的 split 扫描：最佳或近最佳 split 多落在 8–64，K 变化对选择的影响小于 B 和词表；见 `results/kernel/issue60095/autotune/k_sweep_{1,2,6,11}/`。这说明不应把“多 K 必然带来相同比例收益”当作已验证结论，也提示将来可研究安全的 K 分桶，减少启动测试数量。

**H200 复测新增项：**先测默认 K=6 的冷启动与 PR/baseline A/B；再以真实请求分布选 1–3 个额外 K，报告每增加一个 K 的校准时间、编译缓存体积和选择差异。只有额外 K 带来显著性能差距，才建议将它们列入服务的启动 K 列表。上述本地 K 扫描不能代替 H200 数据。

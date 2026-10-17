# Issue #61604 工作记录与设计交接

## 2026-10-18 上卡实现、负载与最佳 shape 交付方案（当前执行依据）

用户已明确：H100/H200 不必复现作者消费级 GPU 的数字或相同容量配置。作者实验用于说明可比较的机制与指标；H 卡主实验按实际 step 时间、缓存容量与自然等待重试校准。本节优先于下面保留的旧阶段计划。“上卡”指实验部署，尚不意味着生产服务上线。

### 实现与实验快照

- 使用当前生命周期版本方案：仅失败准入且实际命中达到门限时建立 memo；逐块校验完整 key 和版本，失效处回退原查询；成功 reset 清全体 memo；请求成功准入/free 时清理，弱引用处理遗留请求。不引入树、依赖索引或新的准入策略。
- 启用范围保持单 full-attention group、无 EAGLE drop、DCP/PCP=1。门限 4 是实验起点，不宣称最佳；4/8/16 的选择必须使用独立于最终报告的调参阶段。同一候选不能按每个测试 shape 暗中换门限。
- 固定真实 base a2fb4827324bf023075799ef3d70d347834b8137 和完整 patch 快照，记录所有改动文件、脚本、模型 revision 与环境指纹。只记录 manager 文件不足以标识此次新增的 block 生命周期修改。远端启动时核查实际导入路径和文件哈希。
- 对照至少包含 base 和完整 patch。机制诊断可增加 patch 关闭 memo，识别生命周期字段维护及公共路径成本；旧全局失效版本只作可选消融。性能运行不保留逐 block 探针，不在 CPU/GPU timed path 中复制校验工作。
- 服务控制器负责独立重启、readiness、预热、负载执行、排空、日志与退出；负载生成器保存实际 token IDs、随机种子和请求到达时间；汇总器按运行轮次比较。先检查该基线 CLI，本文参数并非可直接执行的命令。

### 模型选择对结果的影响

模型不是中性的测试背景。它会影响 scheduler 与 GPU 执行的相对耗时、KV block 容量/压缩方式、是否自然产生长等待，也可能直接决定 memo 的启用条件。base 与 patch 的同模型配对结果可以回答“这个模型/负载上的变化”；跨模型绝对 tokens/s 不能直接比较，也不能把某一模型的 E2E 增益外推到所有模型。

- **首选机制验证和寻找高信号 shape：`Qwen/Qwen3-0.6B`。** 这是 issue 作者 GPU 对比所用的模型 ID。它轻、适合单 H100/H200，短 decode step 有机会让 scheduler CPU 时间更可见；这是放大信号的理由，不代表真实生产模型的收益。固定模型 revision 与 tokenizer。候选 H100/H200 上用基线校准可用 KV block 数，必要时固定较小的 block 配额造成自然等待；每次启动须验证 coordinator 为单一 full-attention `UnitaryKVCacheCoordinator`、spec 精确为 `FullAttentionSpec`，且 patch 的 `_memoize_prefix_hits` 为 True。模型名本身不足以证明路径启用。
- **代表性确认：再选实际目标模型，优先 8B/14B 级 dense full-attention 模型。** 同样确认 spec 和启用状态，分别报告结果。它可以检查最佳小模型 shape 是否只因极短 GPU step 显得突出，但不应与小模型合并成一个平均数。
- **DeepSeek-V4-Flash / DSV4-Flash 不适合当前候选的主测试。** 当前候选明确要求 `type(coordinator.kv_cache_spec) is FullAttentionSpec`；DSV4 的 `DeepseekV4Attention.get_kv_cache_spec()` 返回 `MLAAttentionSpec`，并且模型使用压缩/稀疏的 hybrid CSA/HCA 路径。因此 patch 中 `_memoize_prefix_hits` 会关闭；即使服务成功跑通，也没有测到这项优化。DSpark/MTP speculative 配置、超大权重和多卡通信还会额外改变调度形态。官方 vLLM recipe 将模型标为 284B 总参数/13B 激活，并列出 H200 验证及 FP8 变体至少约 200GB VRAM 的需求；单张 H100/H200 不适合作为这轮设置。若未来扩展支持 DSV4，需要独立设计其 cache spec/group 语义和正确性覆盖，再做该模型实验，不能通过放宽类型判断强行开启。
- 每个性能 run 必须记录模型仓库与 revision、量化/权重精度、chat template 与 token IDs、attention/KV backend、cache spec/group 结构、speculative decoding、world size，以及运行时 memo gate 是否开启。若 gate 未开启，该 run 只能是其他问题的基线，不能纳入 #61604 的 patch 收益。

模型依据：本 worktree 的候选门控在 `worktrees/issue-61604-pr/vllm/v1/core/kv_cache_manager.py`；DSV4 返回 MLA spec 的实现见 `worktrees/issue-61604-pr/vllm/models/deepseek_v4/attention.py`。官方 DeepSeek-V4-Flash 配方/硬件信息：<https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml>。Issue 作者的 Qwen3-0.6B 端到端结果：<https://github.com/vllm-project/vllm/issues/61604#issuecomment-6185593469>。

### H 卡最佳 shape 的构造与选择

必须交付“最佳已测 shape”及其独立确认结果，可以是适用条件明确的合成高收益场景。不能承诺必有正收益，也不能用最大的一次随机波动填充结果。若所有确认结果均无收益，仍报告最佳已测配置及区间，并据此停止或调整方案。

1. 先用一个支持范围内、上下文长度足够的小型 dense full-attention 模型，例如现有 Qwen3-0.6B 资产（以配置实际校验为准）。小模型和短 GPU step 有机会放大 scheduler 占比，这是实验假设。另选一个实际目标模型作确认，不能把小模型结果外推到全部模型。
2. 只用 base 诊断校准真实 KV 压力。候选共享前缀 8K/16K/32K tokens（不得超过模型原生可用上下文），唯一后缀初始 128、输出初始 256，并发初始 16/32/64。按顺序先调整容量到约 2–8 个实际运行请求，再扫描前缀和并发；不做全笛卡尔积。若没有持续重试，可在后缀 128/512、输出 128/512 中进一步校准。
3. KV 配额从实际共享 prefix 占用和活跃请求的唯一尾部需求估算，然后以诊断校准。共享块只占一份，不能按每请求完整 prompt 简单相乘。容量既须容纳可复用前缀和正常推进，也须真实触发等待；block size、总 block 数和实际字节数同时记录。另跑默认容量对照。H200 容量更大不等于更容易触发该问题。
4. 设置足够的 max_num_seqs 和 token budget，确认阻塞原因是 allocate_slots 空间不足，而不是序列上限或 token budget；保存连续失败重试分布及实际命中块数。不允许 monkeypatch 分配失败、插入 sleep 或人为拖慢 base。压力负载靠真实请求到达、生成和缓存占用形成。
5. 固定少量校准合格的 shape，再进行探索性 base/patch 配对；选择同步输出吞吐相对改善最大的可重复 shape，记录同时约束的完成率和延迟。选择空间、轮次数与全部候选结果公开。主门限先固定为 4；若要调门限，先完成调参并重新固定候选。
6. 锁定最佳 shape 后，换独立 seed/到达轨迹、重启服务，做至少 5 对独立确认轮次，AB/BA 交替；推荐主结果 10 对。选优样本不混入确认统计。200 请求仅是短试跑起点；正式运行预先固定足够请求量，使单轮稳态约 60 秒以上，并保证 p99 的样本量。不能因 patch 更快就给它更少请求。
7. 最佳同步 shape 必须原样测默认异步，并做至少一个邻近长度/并发配置，判断收益是否仅存在于极窄点。作者 16K/128/256、并发32配置可作参考点，无需强制复现其 1130-block 配额或 9%–12% 数值。

### 必测用例

| 用例 | 构造 | 必须证明的内容 |
| --- | --- | --- |
| 高收益目标 | 长共享前缀、自然 KV 压力、持续失败准入和重试 | 长命中确实反复查询，memo 在实际模型执行中复用 |
| 相同 shape 默认容量 | 恢复正常 KV 配额，其他条件固定 | 有多少正常服务流量能触发收益；无等待亦为有效结果 |
| 多前缀 churn | A 长前缀等待；不共享前缀的 B 请求持续到达、完成并自然驱逐 | B 驱逐存在且 A memo 被复用；同时记录 A 的 block 是否也被驱逐，不能假定 A 永久有效 |
| 不利场景 | 短前缀、零命中、首次查询立即准入；前部频繁失效 | 公共路径与生命周期维护是否造成吞吐或尾延迟退化 |
| 服务生命周期 | 有界突发、逐步到达、客户端取消、自然抢占、请求完成后排空 | 无死锁/长期无法准入、取消正确、活动引用和 memo 数量回落 |
| 冷/热缓存 | 相同请求集分别从空 prefix cache 和固定预热开始 | 冷轮体现首次查找/建立成本，热轮体现重复复用；加载/编译启动时间独立记录 |

高收益、默认容量和不利场景均需同步/异步。用固定并发负载测饱和吞吐，用固定的预生成开放到达轨迹测排队/延迟；两者分别报告。闭环客户端完成更快会产生不同到达时刻，不能声称两版本的到达轨迹完全相同。

### 指标与正确性口径

- 主性能指标：成功完成请求数、错误/超时/取消数、实际输出 token 数、输出 tokens/s、requests/s、总运行时间。使用实际输出数作分子，说明计时窗口是否包括排空；`吞吐提升 = patch/base - 1`，`耗时降低 = 1 - patch/base`，不能互换分母。
- 服务质量：TTFT、ITL、TPOT、端到端延迟的 p50/p95/p99；注明按请求还是按 token 采样，排队时间单列。不能通过丢请求或改变输出长度取得收益。
- 机制诊断：schedule/lookup 的 CPU 累计与分位耗时，block-map probe、失败准入数、每请求连续重试分布、实际命中 blocks、memo 建立/复用/失效位置/回退长度、缓存驱逐、preemption、running/waiting 数量和 free blocks。不要只报总 cache hit rate。
- 资源：scheduler 进程 CPU、主机/NUMA/绑核与其他负载、GPU step/空闲间隙（抽样 profiler）、GPU 利用率与显存；RSS、memo 条目/记录块数峰值及排空后残留。缓存仍占显存不等于泄漏，应检查活动引用与记录是否随轮次持续增长。
- 数值正确性：固定 tokenizer、prompt token IDs、采样参数和实际输出长度，记录输出 token IDs/终止原因；同 shape 做 base-base 重复以估计自身波动。对固定输入的小批量另与禁用 prefix cache 的结果作检查；差异应定位到具体 token/可获取的 logprob，并调查是否来自调度、数值非确定性或 stale KV。base-base 自身有差异不能自动豁免 patch 的差异。
- 放宽物理身份后，返回块必须是完整 key 的有效副本并覆盖连续前缀；引用/容量/空闲队列操作针对实际副本，无重复释放、使用中覆盖或排空后引用滞留。GPU 服务日志无法独立证明这些内部性质，因此沿用 CPU 定向检查并在诊断运行抽样断言，不能只靠输出成功率。

### 最终报告与决策

交付三组结果：H 卡最佳已测 shape 的独立确认、相同 shape 默认异步、默认容量/不利场景。每组给 base/patch 的绝对吞吐与延迟、配对变化、轮次范围及以轮次为单位的 95% 区间，保留全部有效轮次；尾延迟样本不足或区间较宽必须标为未定。

最佳 shape 完整列出 GPU/CPU、模型 revision、输入/共享/输出长度、block size/容量、并发/到达率、同步状态、门限、真实重试与 memo 复用比例。与作者只比较机制和负载条件，不能把跨硬件的绝对 tokens/s 当作实现优劣。作者已有消费级 GPU 同步收益与异步不明确的结果，见 issue 的两条原始评论：
https://github.com/vllm-project/vllm/issues/61604#issuecomment-6185593469
https://github.com/vllm-project/vllm/issues/61604#issuecomment-6186782334

最优条件下确认有收益且机制证据吻合，才有端到端的正面结论；若仅 CPU 降低，则如实定位为 scheduler CPU 优化。一般场景的退化容忍度须在正式运行前写入 manifest（建议起点：吞吐 2%、p99 延迟 5%，属于项目决策值，不是已获维护者认可的标准）；以区间判断是否排除超限退化，数据不足即未定。任何未解释的引用/内容正确性问题阻止候选冻结。GPU 测试通过后才决定是否冻结合入候选和提交 PR。

脚本路径 `benchmarks/issue61604/hopper/`，结果路径 `results/issue61604/hopper/<run-id>/`。本节只完成设计与评估，尚未部署 GPU、创建付费资源或执行上述测试。

## 2026-10-18 最新 CPU 验证状态

### 按物理 block 生命周期校验的选择性失效实验

在 PR worktree 实验性加入 `KVCacheBlock._hash_lifecycle_version`：设置/清除主 cache hash 时递增。等待请求 memo 保存逻辑 hash、物理 block 与版本；复用时逐 block 核对 hash/version，第一处不匹配就截断 memo 并从普通 cache map 查找恢复。成功 reset 仍通过 cache-map 对象替换清空所有 memo。门限保持 4，生产默认值未改变；memo 不增加 KV block 引用。实验结果与正确性边界见 `results/issue61604/local_cpu_test/LIFECYCLE_INVALIDATION_REPORT_2026-10-18.md`。

定向覆盖 ABA、hash move、无关 block eviction 与选中副本的分配/释放：4 passed。完整 `test_prefix_caching.py` 在门限 4 和临时门限 16 下各为 191 passed；门限 16 测试期间发现无关驱逐测试只构造四 block hit，已改为门限相对场景后重跑通过，最终恢复默认 4。Ruff 检查通过。Base 与 lifecycle patch 的组合 scheduler replay 都是 37 步，21 次失败分配、17 次 key 删除、17 次抢占、一次取消；调度 trace hash 完全相同，最终 block refcount 全为 0。ABA 单测允许重查后选中另一个仍有效的同 hash 副本，并确认实际选中的副本承担引用增加和释放。

同一脚本固定逻辑 CPU 2，五轮比较 100 次真实 `schedule()`：lifecycle 无 churn 中位 11.918 ms，B churn 12.254 ms；旧全局失效行为模拟的 B churn 为 20.143 ms。每个 B churn 场景都实际删除 73 个 B 映射。Lifecycle 策略让 A 保持 1 次完整查表/1 次 memo 建立/100 次失败准入及 8192-token 命中；全局清理模拟产生 36 次额外 A 查表与 memo 重建。相对无 churn，生命周期版 churn 多 0.336 ms（2.8%）；相对全局清理 churn 少 7.889 ms（耗时降低 39.2%；反向比较，全局策略耗时高 64.4%）。现行实现已移除上一版全局失效原型使用、但 lifecycle 机制不需要的 `removal_epoch` 删除计数器，避免在删除热路径上保留无用维护成本。详情与 raw JSON 在上面的报告和 `results/issue61604/local_cpu_test/lifecycle-invalidation-20261018/`。早期一组计时因探针重复扫描 memo 被识别为污染，已删除，不作为证据。

结论：这组实验支持继续推进小范围的 block 生命周期校验，暂时没有依据引入树或依赖索引。它不是服务吞吐证据：A 只有一个等待者、准入失败被强制、A 由 producer 固定持有、B 的分配释放不计入 schedule 时间，也没有 GPU E2E。候选不冻结；后续仍需 review reset、hash move/promotion、alias、物理 block 复用和不支持配置回退，并在 H100/H200 目标负载测吞吐与延迟。

### 长前缀 A 被无关前缀 B churn 全局失效

完成了此前缺少的“误伤” replay：A 有 8192-token/512-block 的长前缀等待，A 的 producer 始终持有这些物理 block；每步由互不相交的 B 前缀自然产生两块缓存并驱逐旧 B 映射。A 的失败准入通过 `reserved_blocks` 固定下来，以隔离重复查询成本。固定在逻辑 CPU 2，base/patch、无 churn/B churn 四种条件各 5 轮交错，每轮计时 100 次真实 `scheduler.schedule()`；B 的分配与释放发生在被计时的 schedule 调用之间。

Patch 无 churn 时 A 只做 1 次完整查表、99 次 memo fast path，累计 schedule 为 11.545 ms。B churn 时有 73 次 B 映射删除，造成 A memo 全局失效 36 次；A 完整查表与 memo 重建均从 1 次增至 37 次，累计 schedule 为 20.125 ms（比 patch 无 churn 高 8.580 ms / 74.3%）。Base 无 churn/B churn 中位数为 20.622/20.835 ms，变化很小；patch churn 的 20.125 ms 已接近 base churn。36 次额外 A 查表各命中 512 blocks 后遇到第一个 miss，折合 18,468 次额外 block-map probe。五轮计数完全一致，A 每步仍命中相同 8192 tokens，物理 block ID digest 不变。详见 `results/issue61604/local_cpu_test/GLOBAL_INVALIDATION_REPORT_2026-10-18.md` 和 `results/issue61604/local_cpu_test/global-invalidation-20261018/`。

这证明了受控高 churn 场景下，全局 epoch 会清掉仍有效的 A memo，重复查表/重建的成本远大于单次 invalidate 函数本身；在该合成场景中约消掉 92% 的 patch 调度时间优势。但它固定只有一个 waiter、强制失败准入并排除了 B 分配时间，不能代表服务吞吐。暂不直接引入 per-key 依赖索引；若 H100/H200 目标负载出现相近的无关 churn 与长等待共存，再评估最小的 selective invalidation，并复跑行为、ABA 与内存检查。

门限验证补充：门限临时设为 16 时，完整 `test_prefix_caching.py` 189 项全通过；memo 测试已改为读取当前门限并扩展 hit 长度。恢复生产门限 4 后，同一完整文件仍为 189 passed。门限选择的 GPU 端到端测试仍待 H100/H200。

### 本轮热路径与门限消融

按建议先把请求状态检查移到 weak memo 查询之前，并将单次 `.get(request)` 的结果传给 memo 校验函数，避免命中时重复查 weak map。`test_prefix_caching.py -k waiting_prefix_memo` 为 17 passed；Ruff 与 Python 编译检查通过。然后用进程内门限覆盖做 base、memo-off、4/8/16 block 消融，生产默认值仍是 4。

3 轮配对结果：短前缀两 block 命中时 base 2.536 ms、memo-off 2.571 ms、门限 4/8/16 分别 2.668/2.626/2.656 ms；均不建 memo。五 block churn 时 base 2.285 ms、memo-off 2.351 ms、门限 4/8/16 为 2.419/2.384/2.372 ms；4-block 门限建立一个五-block memo，8/16 不建立。8192-token 稳定等待（512 blocks）时，base/memo-off/4/8/16 中位数为 289.551/287.488/118.049/118.357/118.615 ms；memo-on 三种门限都将 probe 从 513000 降到 1512，调度、命中和等待队列轨迹一致，释放压力后均准入。数据和限制见 `results/issue61604/local_cpu_test/ABLATION_REPORT_2026-10-18.md`。

当前结论：门限 16 会跳过测得的五-block churn memo，同时保留 512-block 长等待收益；完整 prefix-cache 正确性文件在门限 16 与恢复后的默认门限 4 下都为 189 passed。性能消融每个短场景只有 3 轮，仍不足以靠 CPU 小幅差异决定最终门限；生产默认值保持 4，门限选型留到 H100/H200 局部服务负载验证。

### 前一轮基准（四-block 门限，消融前）

按用户确认，patch 现在只在失败准入且至少命中 4 个完整 block 时建立 memo；门限检查也前移到两个分配失败调用点，短前缀跳过 memo 建立函数。3-block/4-block 边界单测通过，17 项 memo 定向测试通过，完整 prefix-cache 文件 189 passed，Ruff 通过。详细数据与限制见 `results/issue61604/local_cpu_test/REPORT_2026-10-18.md`。

门限后的 20-request/4-prefix 组合 replay 为 37 步：base 和 patch 都有 21 次失败分配、17 次 cache mapping 删除、17 次抢占；37 个释放事件及逐步调度、物理 block、refcount 完全一致。Patch memo 峰值一条、记录 5 个 block，并进入 memo 查询 3 次。随后对短场景和 5-block churn 各做 5 组固定 CPU、低插桩配对 replay：短场景无 memo，patch 中位多 0.124 ms（+4.92%）；churn patch 多 0.110 ms（+4.83%）；两者 trace 相同，且各 5/5 配对 patch 较慢。两三毫秒 replay 的百分比需谨慎解释，但此信号不再能简单归为偶然噪声。分段 profile 显示短场景差值落在 prefix lookup 和 allocator 检查；churn 还支付 memo 建立/校验。详细分析见 `results/issue61604/local_cpu_test/OVERHEAD_BREAKDOWN_2026-10-18.md`。

候选仍不冻结：稳定等待 1000 步 schedule 中位数 269.70 → 110.13 ms（−59.2%），探查数 513000 → 1512；短请求即使不建立 memo，也有约 0.12 ms 的 replay 增量，5-block churn 也有约 0.11 ms 增量。Profile 样本只有两轮，只用于定位：patch 短场景 prefix lookup 约多 37 μs、allocator 约多 24 μs；churn 的 memo 建立约 28 μs、fast-path 校验约 16 μs、全局失效约 6.5 μs。下一步先评估请求状态前置、WeakKeyDictionary 重复查找消除、8/16-block 门限或多次失败后再建 memo；这些尚未实现。随后在 H100/H200 目标服务负载验证；不先改成 radix tree。当前没有 GPU E2E。

## 目标与当前阶段

问题：请求已有较长的 prefix-cache 命中，但因为 KV 空间不足无法准入；每次 scheduler 重试都会从头查询同一段 block hash，消耗调度线程 CPU 时间。

目标：保持命中语义、准入规则和 KV 分配行为正确，复用等待请求此前查询中的有效部分，减少 hash 表查询。

②重复检查、③CPU 稳定等待复现、④失效与生命周期设计、⑤最小范围实现已完成初版。memo 建立门限已加入 patch，CPU 正确性、动态 replay、短/长 churn 和稳定等待已复测；行为一致且稳定等待收益保留，但 churn 有小幅回退信号。候选暂不冻结，需在 H100/H200 目标服务负载验证。GPU 端到端验证尚未执行；本机 RTX 3070 Ti 不做 E2E。

## 工作区与代码归属

- 外层实验仓库：`/home/minstrel/vllm_lab`，分支 `my-61604-pr`。脚本、结果和交接文档放在 `benchmarks/issue61604/`。
- 基线：`/home/minstrel/vllm_lab/worktrees/issue-61604-base`，detached HEAD。
- 实现：`/home/minstrel/vllm_lab/worktrees/issue-61604-pr`，分支 `issue-61604-pr`。
- 两个内层 worktree 起点均为 `a2fb4827324bf023075799ef3d70d347834b8137`，来自 issue 中作者使用的 vLLM 基线。
- 外层保留实验仓库历史，不能把外层提交当成内层 vLLM 实现的起点。现有两个内层 worktree 注册在外层的 Git common directory 中；目录层次不意味着独立 Git 仓库。
- 初版实现与新增测试尚未提交。不要把工作区 HEAD 当作补丁版本标识；测量 JSON 同时记录 manager 文件 SHA256。
- 用户已明确：外层已有图表可以不提交。

主要代码入口（相对实现 worktree）：

| 文件/方法 | 作用 |
| --- | --- |
| `vllm/v1/core/kv_cache_manager.py` / `__init__` | 启用条件与 `_prefix_hits` |
| 同文件 / `get_computed_blocks` | 在原返回与事件处理逻辑之前选择查询路径 |
| 同文件 / `_find_memoized_prefix` | 校验旧前缀、查询后续 blocks、更新 memo |
| 同文件 / `allocate_slots`、`free`、`reset_prefix_cache` | 生命周期清理 |
| `vllm/v1/core/block_pool.py` | 主 hash、别名、驱逐、hash 移动、reset 的语义依据 |
| `vllm/v1/core/kv_cache_coordinator.py` | Unitary coordinator 的原始查询路径 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | FullAttentionManager 的原始命中遍历 |
| `tests/v1/core/test_prefix_caching.py` | 新增 `test_waiting_prefix_memo_*` 测试 |

## 独立实现与重复检查

issue：<https://github.com/vllm-project/vllm/issues/61604>。

本次检索未发现对应的公开实现或作者补丁。GitHub CLI 不可用，使用只读 REST API 查询 issue/comments、引用 61604 的 PR、prefix/memo、prefix/lookup（全部三页）以及作者 PR。快照保存在 `evidence/`。这是当时的搜索结果，不能证明不存在未公开代码；上游提交前必须重新检查。

相关但不同的工作：#59322 优化 offload key 更新中的重复遍历；#54366 修改等待队列驱动的驱逐选择，在单次分配事务内 memoize；#58837 修改准入策略。

当前代码根据 issue 描述独立编写，没有应用作者原始补丁。未发布 GitHub 评论或 PR。

## 最小范围与回退

只有以下条件同时满足才启用：

- prefix caching 开启，且请求通过原有 `prefix_cache_lookup_enabled()` 检查。
- coordinator 的精确类型为 `UnitaryKVCacheCoordinator`。
- spec 的精确类型为 `FullAttentionSpec`；暂不包含子类。
- `eagle_group_ids` 为空，即没有 EAGLE block drop。
- DCP、PCP 均为 1。
- 请求状态为 `WAITING` 或 `PREEMPTED`。

其余配置仍调用原 coordinator 查询。当前类型判断是保守限制；不要为了扩大覆盖率直接改成 `isinstance` 而不检查子类语义。混合 attention、细粒度命中和 context parallel 不在初版范围内。

## 数据结构与查询算法

`_prefix_hits` 是 `WeakKeyDictionary[Request, list[tuple[BlockHash, KVCacheBlock, int]]]`。

- key 是 Request 对象身份，不是 request_id，避免 ID 复用继承旧记录。
- value 记录从首块开始的连续、可校验命中前缀。每项保存请求 hash、block 对象引用和主 hash 生命周期版本。
- value 不保存 Request 引用。弱 key 允许无其他引用的请求被回收。
- 保存 Python block 引用不会增加 KV `ref_cnt`，不会 pin block，也不会改变 free queue 顺序。

已有 memo 的查询（首次查询走 coordinator）：

1. 计算上限 `min((request.num_tokens - 1) // block_size, len(request.block_hashes))`，保留最后一个 token 重算语义。
2. 从头检查旧记录：请求当前位置的 hash 必须仍等于保存值，block 主 hash 必须等于该 hash 与 group 0 的组合，且生命周期版本必须与记录一致。
3. 遇到长度上限或第一个不一致就停止，删除 memo 的后续部分。
4. 从校验结束的位置继续调用 `get_cached_block(hash, [0])`，直到 miss 或上限。
5. 只有主 hash 匹配的连续前缀才能写入 memo。别名命中仍可以用于当前返回，但不保存该块及其后面的块。
6. 返回 blocks 与命中 token 数，继续原来的事件和返回值包装逻辑。

即使旧前缀全部有效，也必须继续查询其尾部，不能漏掉等待期间新增的命中。算法仍然是 O(前缀块数)，收益来自常数开销下降，不是 O(1) 查询。

## 缓存有效性与生命周期

| 事件 | 当前处理 |
| --- | --- |
| KV 空间不足，分配返回 None | 没有记录时从本次命中建立 memo；已有有效记录则保留 |
| 请求 hash 改变或长度缩短 | 在不匹配处截断，再查询 |
| 缓存映射删除、block 驱逐、hash 移动、复用 | 受影响 block 的生命周期版本变化；该请求在首个失效处截断并回退原查表，无关 block 删除不清除 memo |
| 旧前缀仍有效且尾部新增缓存 | 从旧命中尾部继续查找 |
| 别名 hash 命中 | 保守地重新查询，不进入 memo |
| 分配成功 | 在正常返回或 delayed-cache 返回之前清除该请求 memo |
| `free(request)` | 清除该请求 memo，随后调用原释放逻辑 |
| prefix cache reset 成功 | 清空全部 memo |
| reset 失败 | 不清空；后续查询仍逐块校验 |
| 请求失去其他强引用 | WeakKeyDictionary 自动移除记录 |

有效性依赖主 hash 的建立和清除经过 set_block_hash/reset_hash 并更新生命周期版本，以及成功 reset 清空 memo/替换映射对象。不能只删除主 hash 映射却保留 block 的 hash/version。允许保留仍有效的等价副本，不要求其始终是普通查表的首选物理 block；分配和释放必须针对实际选中副本。当前生产实现已移除 removal_epoch。

同一 block 被复用后若再次获得相同完整 chained hash，沿用现有 prefix cache 的 hash 正确性假设。未来测试还应检查重复 hash 对应多个物理 block 时，选择差异是否影响 refcount、驱逐顺序与调度轨迹；不能只凭 token 命中数相同就宣称所有行为等价。

## CPU 复现与证据口径

脚本：`reproduce.py`，命令和原始结果索引见 `README.md`。

使用现有 `/home/minstrel/vllm_lab/src/vllm/.venv/bin/python`，通过 `PYTHONPATH` 切换源码。设置 `VLLM_TARGET_DEVICE=cpu`、`HF_HUB_OFFLINE=1`，读取本地 Qwen3-0.6B 配置，不执行模型 forward。

真实 scheduler/KV manager 场景：holder 缓存 8192 tokens，直接占用 pool 剩余容量，waiter 使用相同前缀并增加 128 tokens。连续调度 1000 步均无法准入；释放占用后断言可以准入。

- 计时与计数只涵盖 1000 个被阻塞步骤，不包含最后一次成功准入。
- probes 指 `get_cached_block` 调用数，不是全部字典操作。
- 插桩有开销，当前计时包含插桩成本。
- trace hash 只覆盖这些步骤的空 `num_scheduled_tokens`；不能据此宣称动态调度完全等价。
- 所有三轮按 base、pr 交替执行。数据位于 `base-1.json` 至 `pr-3.json`，汇总为 `summary.json`。

| 指标 | 基线 | 初版 |
| --- | ---: | ---: |
| 每轮 probes | 513000 | 1512 |
| 1000 步 schedule 总时间中位数 | 264.20 ms | 105.32 ms |
| 阻塞分配次数 | 1000 | 1000 |

该场景刻意保持缓存稳定，偏向 memo 最有利的情况。约 60.1% 的 CPU 时间下降和约 99.7% 的查询减少不能外推为服务吞吐提升，也不能直接与作者不同负载的数据比较。

### 本轮补充：动态 scheduler replay、失效边界与不利场景

脚本位于 `local_cpu_test/scheduler_replay.py`、`local_cpu_test/adverse_cpu.py`；原始结果归档在 `/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/scheduler_replay/` 与 `.../adverse/`。测试仅运行 CPU scheduler/KV manager，没有模型 forward 或 GPU E2E。

**真实推进 replay：** 模拟模型输出，让 12 个相同长前缀请求经历分批到达、准入、prefill/generation、EOS 完成；受限 5-block pool 自然产生 6 次抢占，并取消一个已等待请求。逐步对比 scheduler 输出、block id、request queue、抢占计数、结束状态及 block refcount。Base/patch 共 28 步逐项 trace 完全一致；12 个请求都结束，`req-1` 为取消请求，最终 pool refcount 为 `[0,0,0,0,0]`。SHA256：`bea12fd5f1efd99667acdd18abd589af8c1aab3f50b1232dcc33fb1336ac18e3`。这是确定性 synthetic model-output replay，不等价于真实模型执行。

**失效与生命周期新增测试：** `test_prefix_caching.py` 中 `test_waiting_prefix_memo_*` 定向 11 项通过，完整文件 183 passed。新增覆盖 BlockPool `move_block_hashes`、同一 hash 的多个物理 block（删除原首选后必须选择重复 block，且与原 coordinator oracle 一致）、失败 reset 不清 memo、弱 key 回收、相同 request_id 的新 Request 不继承记录。原有用例覆盖内容/主 hash失效、eviction、reset 成功、pool 重用不 pin、别名回退、请求前缀扩展、分配成功/free 清理与配置回退。真实 replay 也覆盖自然抢占与取消。日志：`memo-targeted-tests-final.log`、`pr-prefix-cache-tests-final.log`。

**不利性能对照：** 第一版 benchmark 错误地强制关闭 memo，并把额外 oracle 查询放入计时区间；其结果与结论已撤销并删除。修正后的 `adverse_cpu.py` 分别测 base、patch memo on、patch memo off；请求准备在计时前完成，oracle 与 probe 计数放在独立 pass。请求以初始 `WAITING` 状态进入首次查询。3 轮交错、每轮每场景 500 样本，覆盖全 hash miss、64/512/8192-token hit 和首块失效。lookup+立即准入中位数：memo on 对 base 分别 +1.1%、+8.8%、+11.5%、+19.0%、+31.8%；patch memo off 与 base 在中长前缀接近。完整数据和指纹见 `results/issue61604/local_cpu_test/adverse/summary.json`。

**历史实现（本轮修改前）：** 当时的代码在符合条件的 `WAITING`/`PREEMPTED` 请求第一次查询时就建立 memo；分配成功/free/reset 时清理。之前关于“首查走原路径、重试才建立 memo”的描述不符合代码，已撤回。修正后的 benchmark 会直接测量当前实现的首查成本与重试复用成本，结果出来前不据此判断是否需要改实现。

历史 CPU 结论：旧版完整前缀重试记录为零次 get_cached_block，首块 miss 记录为一次；首次查询计数混入 oracle，原比较撤销。旧版长前缀立即准入慢 19.0%。本轮新版已完成针对性和完整 prefix-cache 测试，以及 adverse A/B；本节其余行保留为历史版本记录，当前结论以本文件顶部和 REPORT_2026-10-18.md 为准。

## 已验证与下一步

完整 `test_prefix_caching.py`：旧 patch 183 passed，含新增参数展开后的用例；使用 `--noconftest` 绕过无关全局/GPU fixtures，不等同于上游完整 CI。Ruff 检查通过。

现有新增测试覆盖前缀增长、中间块失效、reset、pool 复用且不 pin、准入和 free 清理、别名回退、EAGLE 原路径结果对比、DCP/PCP 禁用，以及本轮补充的 hash move、重复 hash 物理块选择、弱引用和 request ID 复用。

后续优先顺序：

1. 门限后短/长 churn 每种只有 3 轮，CPU replay 显示相对 base 分别约 +4.4%/+6.3%；这些是微秒级 `schedule()` CPU 差异，不等价 serving 吞吐。记录在 `threshold-retest-20261018/`。
2. 当前保留 4-block 门限和 ABA、多物理 block、reset、request ID 复用与弱引用覆盖。若未来调整失效机制，需再次逐步对照真实 base 的调度、物理 block、refcount 和释放顺序。
3. 目标服务负载的 GPU E2E 是下一项决策证据：在 H100/H200 单卡上分别测同步和默认异步，检查吞吐、TTFT、ITL、尾延迟、输出和波动。
4. 根据端到端收益决定是否保留 memo 优化，再检查最新主线和重复 PR，进行人工逐行审查及上游要求的验证。

当前没有授权对外发评论，也没有创建 PR。若进入上游提交阶段，遵守内层 `AGENTS.md` 的重复检查、人工审查和 PR checklist 要求。

## 下一阶段：H100 / H200 单卡端到端验证计划

### 评估结论

当前 CPU 阶段可以收尾，下一步在一台 H100 或 H200 上验证端到端效果；仅在发现新问题或修改实现后补针对性 CPU 验证。首轮选择可用的单卡即可，不需要同时租两种卡或先跑多卡。第二种卡仅用于复核已成立的结论。

该改动发生在 scheduler CPU 路径，不是 GPU kernel 优化。更短的 GPU step 可能提高 CPU 开销占比，这是待验证的假设；默认异步调度仍可能隐藏收益。CPU 型号、分配核数、频率、NUMA 与负载和 GPU 型号同样需要记录。不能将不同主机的差异单独归因于 H100/H200。

H 系列机器如果 KV 容量充足，测试可能根本触发不了等待重试。必须分别测试受控 KV 压力与默认容量，明确前者是机制验证，后者用于判断实际适用性。

作者在 RTX 3080 Ti / RTX 4090 上报告同步吞吐约 +9.3% / +11.6%，默认异步约 +0.9% / +0.5%，并未建立异步收益。它们是作者实现的数据，不是我们的 H 系列预期值。
来源：<https://github.com/vllm-project/vllm/issues/61604#issuecomment-6185593469>、<https://github.com/vllm-project/vllm/issues/61604#issuecomment-6186782334>。

### 执行任务与先后顺序

- [ ] **G0：固定本轮实验快照（不等于冻结合入候选）。** 保存 base SHA、实现 patch、全部改动文件 SHA256、测试结果与实验脚本版本。远端不能只 checkout 同一个 HEAD 就称为 patch；当前改动尚未提交，必须显式同步并核验。
- [x] **G1：本地正确性与动态 replay 已完成当前阶段验证。** 最新 prefix-cache 文件为 191 passed，覆盖生命周期与副本引用；组合 replay 已通过。允许等价副本选择差异，不能只靠逐步物理 ID 完全一致判断正确性。代码变化或 GPU 暴露问题时做针对性复测。
- [ ] **G2：编写单卡测试脚本和 manifest。** 包括远端源码校验、启动/停止自己创建的服务、readiness、固定 warmup、负载生成、请求/服务器日志、失败退出和结果归档。先根据该基线的 CLI help 核实参数；本文参数是实验设计，不是已验证可直接执行的命令。
- [ ] **G3：同机环境 smoke run。** 选一张可用 H100 或 H200，单独运行 base/patch，确认各自导入路径、修改文件指纹、模型加载、请求成功，以及初版 fast path 确实满足启用条件。TP=PP=DCP=PCP=1，不启用 speculative decoding、KV connector/offload。
- [ ] **G4：仅用基线校准 KV 压力。** 校准下面的主场景，使真实模型运行期间反复出现“长前缀命中 + 空间不足”，而非仅 max_num_seqs 限制或 token budget 耗尽。记录并冻结 block 数、队列/运行数量、命中长度、重试次数和预抢占情况；不能根据 patch 速度挑参数。
- [ ] **G5：执行同步、异步和回归对照。** 先各做一组短运行确认流程，正式数据使用下面的矩阵和配对规则。
- [ ] **G6：解释结果并归档。** 分开报告 CPU 成本、同步吞吐、异步吞吐及尾延迟；若只有机制验证有收益，明确适用范围。满足前述步骤后才决定是否扩大到其他模型/另一种 H 卡及最新主线。

### 建议起始负载与矩阵

可选的作者参考场景（不要求复现，也不作为 H 卡主场景的硬约束）：Qwen3-0.6B，固定模型 revision、tokenizer 与 dtype；16384-token 共享前缀、128-token 后缀、256 输出 tokens，200 请求，并发 32，temperature=0、ignore_eos。明确禁用/固定 thinking 和 chat template 行为，保存实际输入 token 序列或其指纹，并验证长度，不能把文本长度当 token 长度。

保持 prefix caching 开启、block size=16（若所选后端不支持则明确记录差异），max model length 至少覆盖输入和输出。`num_gpu_blocks_override=1130` 仅作为作者参数的校准起点，不承诺在该版本/后端会得到相同运行请求数。保持 max_num_seqs 高于受容量限制的运行请求数，token budget 能覆盖所需 prefill；具体值经基线校准后固定。

| 场景 | 同步 | 默认异步 | 目的 |
| --- | --- | --- | --- |
| 长共享前缀 + 受控 KV 压力 | 必测 | 必测 | 主收益与重叠隐藏效应 |
| 相同负载 + 默认 KV 容量 | 必测 | 必测 | 实际容量下的触发频率与收益 |
| 短前缀或无共享前缀，低等待 | 必测 | 必测 | 首次查询、零命中和 memo 维护回归 |
| 多前缀轮换、持续到达造成驱逐 | 必测 | 必测 | 失效频繁时的成本与正确性 |

低等待对照不能仅删掉 KV 限制后假定成立，要用诊断数据确认。先跑最小矩阵，再根据结果扩大前缀长度和并发，不盲目扫描全部组合。

### A/B 控制与指标

- 同一主机、同一 GPU、同一环境和模型资产，base 与 patch 顺序运行，不同时占用 GPU。两者仅改变候选源码；记录 Python/PyTorch/CUDA/驱动、attention backend、CUDA Graph 配置、实际 scheduler 类型、全部服务和客户端参数。
- 每次独立重启服务，用相同 warmup 建立同样的前缀缓存状态。控制编译、图捕获、模型加载不进入稳态计时。固定到达序列、请求内容和 seed；独立记录客户端耗时，确认客户端不是吞吐瓶颈。
- 正式主场景至少 5 个配对轮次，预先固定 AB/BA 顺序交替，保留全部有效轮次；必要的重复轮数增加应同时作用于 base/patch，不能因为收益不理想而挑轮次。任何作废运行都记录原因与日志。
- 主指标：完成请求数、失败数、输出 token 数、输出 tokens/s、端到端耗时；辅助指标为 TTFT、ITL/TPOT 的 p50/p95/p99、每请求延迟、preemption 数、GPU 利用率与显存。
- 诊断运行额外记录 prefix 查询/重试次数、命中长度、memo 校验与失效、`schedule()` CPU 时间、KV 可用 block 数、队列长度。性能运行关闭高频 Python 探针；两个版本使用相同插桩设置。诊断与性能数据分开存放。
- 以“运行轮次”为统计单位报告配对变化、范围和置信区间；同一轮内的 200 个请求不是 200 个独立 A/B 实验。5 对只是起点，尾延迟与小幅收益若不稳定，应报告证据不足。
- 保存输出 token IDs/内容及终止原因，做 base-base 和 base-patch 比较。temperature=0 不保证所有执行条件下逐 token 相同；不一致需结合基线自身波动、批次和调度变化调查，不能自动解释为无害，也不能把完成率当 KV 正确性证明。

### 结果判断与交付

1. 先判断是否触发目标机制。几乎没有等待重查的运行不能用来证明该优化无效，但必须作为适用范围证据保留。
2. 同步收益需要配对改善方向稳定且超出运行波动，并由诊断中的 CPU 成本下降支持。当前不预设“必须提升 9%”或任何由作者数字推导的目标。
3. 默认异步无明显改善是允许的结论；任何稳定的吞吐或尾延迟退化都需要解释。置信区间跨零时报告未测出明确差异，不能声称两者严格等价或无回归。
4. 为正常负载定义实际可接受的退化幅度时，应在正式测试前写入 manifest，不能看完结果再决定阈值。尾延迟改善/退化需要足够请求与轮次支持。
5. 不同容量、调度方式和模型的结果分别报告，不混成一个总提升百分比。

脚本放在 `benchmarks/issue61604/hopper/`；结果统一放在 `results/issue61604/hopper/<run-id>/`，包含 manifest、源码 patch/指纹、测试命令与结果、每轮 JSON、请求结果、服务器日志、诊断 profile、汇总脚本和结论。记录本地/远端保存位置；较大的原始产物是否纳入 Git 单独按仓库惯例处理。

本节是待执行计划。尚未分配或启动 H 系列 GPU，也没有 H100/H200 实测结果；既有其他 issue 的 H 卡数据不能作为本任务证据。

### 本机 CPU 重测记录（2026-10-18）

用户要求先在现有 CPU 环境确认稳定等待收益。测试入口、命令及结果解释见 `local_cpu_test/README.md` 与 `local_cpu_test/run_cpu_test.sh`；脚本在外层目录，结果写入独立的 `results/issue61604/local_cpu_test/`，保留之前 `benchmarks/issue61604/` 下的三轮文件。

本机 Intel Core i7-12700H，20 个逻辑 CPU 可用。base 的 prefix-cache 全套为 172 passed；patch 为 180 passed（包含 memo 用例）。五轮按每轮 base 后 patch 配对：1000 个阻塞步骤中 schedule 中位数 265.07 → 104.68 ms（−60.5%），查表调用 513000 → 1512（−99.7%），lookup 中位数 240.97 → 81.78 ms（−66.1%）。五对 trace hash 相同，均有 1000 次分配失败，并在释放容量后准入。详见 `results/issue61604/local_cpu_test/summary.json` 和 `manifest.json`。

这只是稳定等待场景的复测，不是动态到达/缓存 churn 的 scheduler 负载测试。现有单测覆盖 eviction、reset、增长、别名回退和 block 不 pin；动态真实调度负载性能验证仍是开始 H 系列 A/B 前的待办。CPU 测量的插桩有开销，脚本按 base 后 patch 顺序，正式 GPU 配对要按 A/B、B/A 交替并继续固定服务状态。

### 动态到达与缓存驱逐 CPU scheduler replay（2026-10-18）

入口：`local_cpu_test/run_dynamic_cpu_test.sh`；结果：`/home/minstrel/vllm_lab/results/issue61604/local_cpu_test/dynamic/`。本机 Intel Core i7-12700H。三组配对、AB/BA 交替，每轮 500 scheduler steps；每 5 步到达一个新 waiting 请求，并驱逐一个缓存前缀 block，累计 100 到达、100 驱逐。等待头请求仍因 KV 容量受阻。命中从 8192 降至 6592 tokens。

三轮 base/patch 的调度 trace、逐步 hit 长度 trace、waiting queue 长度 trace 完全相同；每轮 500 次分配失败，释放压力后原头请求均准入。schedule 中位数 119.48 → 51.23 ms（−57.1%），lookup 中位数 105.48 → 38.35 ms（−63.6%），查表 231650 → 1012（−99.6%）。逐轮 JSON、日志、`summary.json` 与 `manifest.json` 保存上述路径。

限定：这是增长等待队列、定时驱逐的真实 scheduler replay；附加请求保持排队，没有执行 model forward 或被逐一完成。结果支持 memo 在该人为构造的 churn 场景仍保持行为一致且减少重复探查，但不代表通用 serving 吞吐。明天 H100/H200 端到端 A/B 仍应独立进行。用户明确不需要在本机 RTX 3070 Ti 做 GPU 端到端测试。

## 下一个阶段：
对，精细失效机制新增的维护开销，可能超过它挽回的收益，但目前还没测过，不能直接定论。 并且我前面的判断需要修正：失效函数只花约 6.5 μs，不代表全局失效的总代价只有这些。它还会让仍有效的 memo 丢失，导致后续重新查表、重建记录；这部分才可能是主要损失。
现在建议按这个顺序推进：
1. 先减少现有实现的常见路径成本。
   memo 表为空时直接跳过 membership 查询；有记录时只取一次 memo，避免入口 in、内部再 .get()；同时检查准入成功和释放时是否在对空表做无用清理。这些改动范围小，优先保持现有失效语义。
2. 用消融对照确定门限，而不是直接把 4 改成 16。
   比较真实 base、patch 关闭 memo、门限为 4/8/16 的 patch。沿用短前缀、五 block churn、稳定长前缀等待三类负载，判断：
   - 关闭 memo 后仍剩多少公共路径成本；
   - 跳过五 block memo 能消除多少 churn 回退；
   - 长前缀收益是否保留。
   当前五 block 命中只有几十个 token 量级，确实可能不足以摊薄建立成本。但提高门限也无法消除所有公共路径检查。
3. 已测全局失效造成的“误伤”。
   构造两个互不相关的前缀：A 上有长前缀等待请求，B 持续发生驱逐。观察 B 的删除让 A 重新查表、重建 memo 多少次，以及增加多少调度时间。这个场景能直接判断精细失效是否值得做；此前只有一条 memo 的短 replay 不足以回答。
4. 只有第 3 步显示明显损失，再设计按 key 的失效机制。
   届时比较版本号、反向依赖索引等方案，并验证重复物理 block、ABA 和记录回收。并查集暂时不适合作为实现选择。
此段为旧阶段计划，已被顶部生命周期实验取代：当前上卡候选使用 block 生命周期校验，无需再实现全局 epoch 或 selective invalidation 原型。上卡需确认真实等待重试及无关 churn，门限选型与端到端收益仍待测。

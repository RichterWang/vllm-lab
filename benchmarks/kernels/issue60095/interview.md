# Issue #60095 / PR #60097 背景

## 这个 PR 要解决什么问题？

PR #60097 优化的是 vLLM 计算生成 token 的 logprob（对数概率）时使用的 kernel。

- 计算logprob的原因：通常是因为调用者需要查看或利用模型对输出的置信度。比如评估模型、比较候选续写、计算文本的似然分数，或在 API 请求中返回 token 的 logprob。
但普通文本生成并不一定需要把 logprob 返回给用户。采样器可以直接根据 logits 选下一个 token；只有请求 logprobs 等功能时，vLLM 才需要额外计算并提供这些概率信息。PR #60097 优化的是这部分额外计算。

当模型生成下一个 token 时，会为词表中的每个 token 计算一个分数（logit）。计算某个生成 token 的 logprob，需要将它的 logit 与整张词表的 logits 一起归一化。词表较大、decode batch 较小时，如果每条请求都由单个 GPU program 处理，GPU 上可并行的 program 数量可能太少，部分 GPU 计算单元会空闲。

PR 的目标是在这种场景下提高并行度，并减少重复读取词表数据造成的开销。

## 词表在这里有什么作用？

词表（vocabulary）是模型可输出 token 的集合。模型针对每个 token 输出一个 logit。给定目标 token `t`，它的 logprob 可写为：

```text
logprob(t) = logit(t) - logsumexp(all vocabulary logits)
```

其中 `logsumexp` 对整张词表的 logits 做归一化。因此，即使只请求一个或几个 token 的 logprob，通常也需要扫描整张词表来计算归一化项。

当前 benchmark 使用词表大小 151,936 的合成 logits 来模拟这段计算，不加载模型。

### Logit 是什么？这里是否在做 Softmax？

Logit 是模型输出层为每个候选 token 产生的原始分数，还不是概率。模型先根据上下文计算隐藏状态，再由输出层把隐藏状态映射为词表大小的一组 logits。

要把 logits 转为概率，需要进行 Softmax 归一化。计算目标 token 的 logprob 时，可以不显式生成整张词表的概率向量，而直接计算：

```text
logprob(t) = logit(t) - logsumexp(all vocabulary logits)
```

其中 `logsumexp` 是 Softmax 的归一化项。PR 优化的是高效计算该归一化项及目标 token 的 logprob。

### 为什么计算 logprob 需要整张词表？

目标 token 的 logit 只是一个未归一化分数。要把它转换为概率，需要除以所有 token 的指数分数之和：

```text
P(t) = exp(logit(t)) / sum(exp(logit(v)) for v in vocabulary)
```

等价地，在 log 空间中：

```text
logprob(t) = logit(t) - logsumexp(all vocabulary logits)
```

因此，即使只返回实际生成 token 的 logprob，也需要整张词表的 logits 来计算归一化分母。

## 为什么切分词表？

原实现一次由一个 GPU program 处理一条请求的整张词表。decode batch 较小时，请求数少，每条请求对应的 program 也少；与此同时，每条请求要处理的词表很大。这可能导致并行度不足，GPU 计算单元利用不充分。

PR 可以把每条请求的词表范围切成多个块，让多个 GPU program 分别计算局部结果，再合并这些局部结果并求出目标 token 的 logprob。这样能增加并行度，但也会产生中间结果和合并开销。batch 较大时，未切分的单个 program 路径可能更合适。

## `auto` 是什么？

benchmark 中的 `auto` 使用 PR #60097 实现的自动切分策略。PR 根据 batch size、词表大小以及 GPU 的计算单元数量选择 split 数，尝试平衡并行收益与切分、合并开销。base 实现没有该参数，因此 base 测试只运行原始路径；PR 测试可以比较 `auto` 与手动指定的 split 数。

当前实验是在 RTX 3070 Ti 上进行的。它能帮助验证代码路径并观察该 GPU 上的趋势，但不能直接代表 H100/H200 上的性能，也不能据此确认数据中心 GPU 上启发式策略的最优 split 数。

### Benchmark 中的 batch size 指什么？

这个 kernel 的 batch size 是一次计算中处理的 logits 行数；每一行对应一条序列在当前 decode 步的词表分数。它通常接近该 decode 步实际进入此计算的序列数，但不一定等于 HTTP 请求数：一个请求可能要求生成多个序列（例如 `n > 1`），而调度器每一步实际处理的序列数也会变化。

因此，benchmark 中的 batch size 是 kernel 输入的行数，用来模拟不同大小的 decode batch；它不是 HTTP 并发请求数的直接等价物。

## 当前 benchmark 的边界

- 输入是合成 logits，不包含真实模型的前向计算。
- 测量聚焦 token-logprob kernel，不代表完整服务的端到端延迟。
- RTX 3070 Ti 上的结果只能说明该设备与当前输入配置下的表现。
- 若要验证 PR 讨论中针对 H100/H200 的性能现象，应在相应 GPU 上运行相同的 base/PR A/B 测试。

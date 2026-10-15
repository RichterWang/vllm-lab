# First RTX 3070 Ti run (2026-10-15)

Raw files: `results/issue56455/20261015T132446Z/` (local, Git ignored).
GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8,192 MiB. Model:
Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
Each arm used a fresh server, `max_num_seqs=8`,
`max_num_batched_tokens=512`, 32 requests and concurrency 16.

| Arm | Successful | Throughput (req/s) | P99 TTFT (ms) | P99 stream gap (ms) | Max stream gap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 32/32 | 21.72 | 418.79 | 6.99 | 10.31 |
| PR slots=0 | 32/32 | 21.88 | 414.66 | 6.47 | 9.06 |
| PR slots=2 | 32/32 | 21.55 | 436.66 | 6.45 | 340.80 |

The 340.8 ms maximum came from one gap in request 0. There was no comparable
gap above 50 ms in the other arms. This is a single-run observation, and the
aggregate P99 stream gap did not increase. The requests shared most of their
prefix, the client measured stream chunks rather than exact tokens, and the
GPU/model differ from the author's workload. No performance conclusion follows
from this smoke run.

The first two startup attempts failed before serving because the new worktrees
did not contain the installed CUDA extensions and generated FlashMLA interface.
`link_extensions.sh` now links both into each worktree. All three servers then
started and completed the request set. The worktrees are clean according to
`git status --short`.

Next experiment: put a unique prefix at the start of each request and vary
prompt length to force more real prefill work; use three alternating runs per
arm, sample waiting/running/KV metrics, and inspect per-request long gaps.
The later H200 run should use Qwen2.5-7B-Instruct and the same ShareGPT request
sequence for stock, PR disabled, and PR enabled.

# vLLM serving benchmark

This directory contains the performance phase that follows the functional
acceptance suite. It measures request-level latency, engine concurrency, KV
cache use, and GPU activity without modifying the upstream vLLM source tree.

## Measurement model

The client uses the streaming OpenAI-compatible `/v1/completions` endpoint.
For each measured request it records:

- TTFT: request start to the first non-empty streamed text chunk;
- E2E latency: request start to the end of the stream;
- TPOT: `(E2E - TTFT) / (completion_tokens - 1)`;
- server-reported prompt and completion token counts.

In parallel it samples `/metrics` for running/waiting requests, KV-cache use,
and cumulative token counters. `nvidia-smi` provides GPU memory, utilization,
and power time series. Warm-up requests are completed before all samplers and
the measured clock start.

TTFT is based on the first non-empty HTTP stream chunk, so it includes client,
HTTP, and server scheduling overhead. TPOT uses the final server usage count;
stream chunks are not assumed to contain exactly one token.

## Run

From the lab root, with the checked-in vLLM environment already installed:

```bash
./benchmarks/run_matrix.sh
```

The orchestrator refuses to reuse an existing service on the configured port.
For every mode it starts a fresh server, waits for health, runs all enabled
scenarios, and stops the process group before starting the next mode. This
keeps eager and compilation/CUDA Graph state isolated.

The default matrix enables one acceptance scenario for both modes. The two
additional concurrency scenarios are checked in with `enabled: false`; enable
them only after the first run succeeds. Expand input/output lengths in the
same way.

## Results

Each run is written below `results/benchmark/<UTC-time>_<vllm-sha>/`:

```text
server_<mode>.log
<mode>/manifest.json
<mode>/<scenario>/summary.json
<mode>/<scenario>/requests.jsonl
<mode>/<scenario>/engine_metrics.jsonl
<mode>/<scenario>/gpu_metrics.jsonl
summary.json
summary.csv
```

The resolved default compilation/CUDA Graph policy must be read from the
corresponding server log. Therefore the mode is deliberately named
`default_hybrid`, rather than claiming that every operation used CUDA Graph.

At least 20 valid request samples are required before p95 is treated as more
than exploratory. Results from the 0.6B model on one 8 GiB RTX 3070 Ti are a
local scheduler/serving baseline, not a production-scale performance claim.

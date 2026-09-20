# vLLM Development Environment Implementation Log

Started: 2026-09-19 (Asia/Singapore)
Completed: 2026-09-20 (Asia/Singapore)

## Scope

Set up an isolated vLLM contributor environment without changing the NVIDIA
580.65.06 driver or the existing `infra_dev` Conda environment used by
IsaacLab.

## Repository

- Fork remote: `git@github.com:RichterWang/vllm.git`
- Upstream remote: `https://github.com/vllm-project/vllm.git`
- `main` was fast-forwarded to upstream commit
  `41c4a3ed4e45792739bd8a5bbcd01269b06eae2c` and pushed to the fork.
- Learning environment branch:
  `learning/cu130-fbe8a157`
- The learning branch is pinned to
  `fbe8a157fbdd4b2f813ae6395397232de97d5c5d`, matching the available official
  cu130 nightly wheel.

## Python Environment

- Bootstrap Conda environment: `vllm-dev`
- Python: 3.12.13
- Package manager: uv 0.12.10
- Project environment: `src/vllm/.venv`
- Editable package:
  `vllm==0.29.1rc1.dev405+gfbe8a157f.precompiled`
- PyTorch: `2.13.0+cu132`
- Triton: `3.7.1`

The vLLM package was installed editable from the source tree. Its compiled
extensions came from the official wheel matching the source commit and the
cu130 variant. Runtime, lint, and `requirements/test/cuda.in` dependencies were
installed into `.venv`.

## Developer Tooling

- Installed vLLM lint dependencies.
- Installed the complete CUDA test dependency set.
- Installed the repository pre-commit and commit-msg hooks.
- Configured the `/home/minstrel/vllm_lab` VS Code workspace to use
  `src/vllm/.venv/bin/python` and run tests from `src/vllm`.

## Notes

- `infra_dev` was not modified.
- The NVIDIA driver was not modified.
- The repository's `AGENTS.md` was not used for this log because its editing
  policy explicitly excludes one-off implementation records and requires that
  file to remain a compact set of reusable contributor instructions.

## Validation

- PyTorch detected the NVIDIA GeForce RTX 3070 Ti Laptop GPU with compute
  capability 8.6 and 7.9 GiB usable memory.
- A CUDA tensor operation completed successfully.
- `vllm._C_stable_libtorch`, `vllm._moe_C_stable_libtorch`, and the bundled
  FlashAttention extension imported successfully.
- The `vllm` CLI and pre-commit executable started successfully.
- Lightweight Python tests passed: 16 passed in `tests/test_logger.py` plus
  `tests/test_version.py::test_version_is_defined`.
- The upstream-pinned legacy `decord==0.6.0` package imports successfully, but
  `uv pip check` reports its old CPython 3.6 wheel tag as a platform mismatch.
  It was retained because it is explicitly pinned by
  `requirements/test/cuda.in`.
- Running all of `tests/test_version.py` produced 7 failures and 6 passes due
  to version assumptions in this pinned development snapshot (the generated
  prerelease version tuple has six elements and several test cases use major
  version 1 while the implementation asserts major version 0). This is an
  upstream source/test snapshot issue rather than an environment or CUDA
  installation failure.

## Small-Model Runtime and API Phase

Completed: 2026-09-20 (Asia/Singapore)

- Created a separate lab repository layout for pinned configs, scripts,
  experiments, notes, environment metadata, and compact results. The nested
  vLLM fork remains an independent repository.
- Downloaded public model `Qwen/Qwen3-0.6B` at immutable revision
  `c1899de289a04d12100db370d81485cdf75e47ca`. The resolved snapshot is 1.5 GiB
  and contains no incomplete download files.
- Added a single `scripts/run_acceptance.sh` entrypoint that validates the
  model, performs offline inference, launches the loopback-only API server,
  runs API/concurrency checks, shuts it down, and verifies released resources.
- Result writers now remove stale success files before work, write atomically,
  preserve failure details, and record timestamps, source SHA, model revision,
  and runtime configuration.
- Both offline and online paths use `generation_config=vllm`, eager execution,
  FP16, 4096 maximum model length, and the native sampler baseline.
- Client URLs now derive from the same `LAB_HOST`/`LAB_PORT` config used by the
  server.

### Final runtime evidence

- Offline `LLM.chat()`: passed; load 13.844 seconds, generation 0.438 seconds,
  31 prompt tokens, and 54 completion tokens.
- `/health`, `/version`, `/v1/models`, and `/v1/chat/completions`: HTTP 200;
  model name and vLLM version matched the pinned baseline.
- Four long concurrent requests: 4/4 HTTP 200. Across 79 `/metrics` samples,
  the engine reported a peak of 4 running requests, 0 waiting requests, and
  peak KV-cache usage of 0.013708. This verifies server-side overlap, but is not
  presented as a throughput benchmark.
- Final counters after the API checks: 121 prompt tokens and 550 generation
  tokens; running, waiting, and KV-cache usage all returned to zero.
- Clean shutdown: port 8000 closed, no vLLM or EngineCore process remained,
  and GPU memory returned from the service allocation to 9 MiB idle.

### FlashInfer sampler boundary

The first exploratory run exposed a FlashInfer sampler JIT dependency on a
discoverable CUDA compiler. The supported `VLLM_USE_FLASHINFER_SAMPLER=0`
switch selects the native sampler for this architecture baseline. FlashInfer
JIT and source/kernel compilation are intentionally separate follow-up work;
neither the NVIDIA driver nor `infra_dev` was changed.

### Acceptance scope

The generated answers are intentionally not scored for semantic accuracy.
The accepted scope is model loading, GPU execution, API behavior, observed
concurrent engine requests, and clean shutdown. Comparative throughput and
model-quality evaluation require separate controlled benchmarks.

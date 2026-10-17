# Provenance cross-check

This note preserves the final read-only verification of the remote H100 run.

- Result directory: `/workspace/vllm_lab/results/serve/issue61991/h100_20261016T152655Z/`
- Run status: 27 `ok` rows (3 arms × 3 rounds × 3 concurrency settings).
- All 27 measured JSON files parse, contain the expected arm SHA, report 200 completed requests, and report 0 failed requests. One additional smoke JSON is present.
- Main SHA used by the matrix: `cffc0cfa311f4997be58cfe8f5f6473ab4898269`. This is present in the runner, all main result JSON metadata, the main worktree HEAD, and the installed main package version `0.32.0.dev0+gcffc0cfa31`.
- Parent worktree: `fa75d3456d60a5ffe923818c4b495295675b38e1`; installed package version `0.32.0rc1.dev284+gfa75d3456`.
- PR worktree: `b2625fd3fd7ea896940c075b8cc132d9e6bbaee3`.
- For the PR serving arm, the matrix runner copied `vllm/v1/worker/gpu/sample/logprob.py` from that PR worktree into the installed parent environment. Thus the PR SHA in each JSON is the arm/source label supplied by the runner; the process imported the parent-built package with the PR runtime file substituted, not a separately installed PR wheel. This isolates the only runtime file changed by the PR.
- Parent and main serving environments both report PyTorch `2.14.0+cu130`, CUDA runtime 13.0, Triton 3.8.0, and CUDA available.
- GPU at final check: H100 80 GB, 0 MiB allocated, 0% utilization. No benchmark or vLLM server process remained; the Pod itself was left running.
- Main-source version mismatch in the earlier `environment.txt` is described in `REPORT.md`; the original file was preserved as downloaded.
- Downloaded archive size at collection: about 21.8 MB. It contains raw benchmark JSON, logs, five-second GPU telemetry, environment snapshot, matrix/build scripts, build logs and smoke output.

`diagnostics/matrix-launch.log` records all 27 completed cases. The final `r3_pr_server.log` error occurs only after SIGTERM and the final benchmark JSON had already been written with 200 completed and 0 failed requests. Similar shutdown-time output-handler errors occur for other arms as well.

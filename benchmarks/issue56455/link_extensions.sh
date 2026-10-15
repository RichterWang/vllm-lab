#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TREE="${1:?Pass a vLLM worktree path}"
BINARY_ROOT="${LAB_ROOT}/src/vllm/vllm"

# Editable worktrees contain source only. Reuse the environment's compiled CUDA
# extensions in both arms; the links are ignored by vLLM's Git repository.
while IFS= read -r -d '' binary; do
  relative="${binary#"${BINARY_ROOT}/"}"
  destination="${TREE}/vllm/${relative}"
  mkdir -p "$(dirname "${destination}")"
  if [[ ! -e "${destination}" ]]; then
    ln -s "${binary}" "${destination}"
  fi
done < <(find "${BINARY_ROOT}" -name '*.so' -print0)

FLASHMLA_INTERFACE="${BINARY_ROOT}/third_party/flashmla/flash_mla_interface.py"
if [[ -f "${FLASHMLA_INTERFACE}" ]]; then
  destination="${TREE}/vllm/third_party/flashmla/flash_mla_interface.py"
  if [[ ! -e "${destination}" ]]; then
    ln -s "${FLASHMLA_INTERFACE}" "${destination}"
  fi
fi

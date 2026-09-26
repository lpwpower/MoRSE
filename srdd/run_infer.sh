#!/usr/bin/env bash
# ============================================================================
# SRDD MoLE inference — role + subtask routing over SRDD task graphs.
#
# Loads a trained MoLE checkpoint (router.pt / title_embedder.pt / lora_state.pt)
# produced by srdd/train.py (or run_train.sh) and runs the SRDD task-graph
# pipeline per sample, then evaluates each generated repo with srdd_evaluator.
#
# For a base-model baseline (no checkpoint), set NO_CKPT_LOAD=1 and omit
# CHECKPOINT_DIR.
# ============================================================================
set -euo pipefail

# REPO_ROOT is the directory that CONTAINS morse/, scicode/, srdd/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# ── Documented variables (edit as needed) ───────────────────────────────────
#   EXEC_MODEL_NAME  : HF model id / local snapshot for the MoLE execute backbone.
#   GRAPH_MODEL_NAME : HF model id for the task-graph generator LLM.
#   SRDD_CSV         : SRDD dataset CSV (category/name/description).
#   CHECKPOINT_DIR   : dir with router.pt/title_embedder.pt/lora_state.pt.
#   OUTPUT_ROOT      : where per-sample logs/repos/reports are written (required).
EXEC_MODEL_NAME="${EXEC_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
GRAPH_MODEL_NAME="${GRAPH_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
SRDD_CSV="${SRDD_CSV:-${REPO_ROOT}/srdd/data/SRDD.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/srdd/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/srdd/infer_runs}"

# Baseline (no checkpoint) toggle: set NO_CKPT_LOAD=1 to skip loading weights.
NO_CKPT_LOAD="${NO_CKPT_LOAD:-0}"

# MoLE architecture (must match the trained checkpoint).
NUM_SUBTASK_EXPERTS="${NUM_SUBTASK_EXPERTS:-4}"
SUBTASK_TOP_K="${SUBTASK_TOP_K:-2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_LAST_N_LAYERS="${LORA_LAST_N_LAYERS:-8}"

# Decoding / runtime.
DEVICE="${DEVICE:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ROUTER_MODE="${ROUTER_MODE:-greedy}"   # greedy | sample
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12288}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
SMOKE_TIMEOUT_S="${SMOKE_TIMEOUT_S:-10}"

echo "=== SRDD MoLE inference ==="
echo "REPO_ROOT       : ${REPO_ROOT}"
echo "EXEC_MODEL_NAME : ${EXEC_MODEL_NAME}"
echo "OUTPUT_ROOT     : ${OUTPUT_ROOT}"
echo "NO_CKPT_LOAD    : ${NO_CKPT_LOAD}"

CKPT_ARGS=()
if [[ "${NO_CKPT_LOAD}" == "1" ]]; then
  CKPT_ARGS+=(--no-ckpt-load)
else
  CKPT_ARGS+=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

python -u "${REPO_ROOT}/srdd/infer.py" \
  --srdd-csv             "${SRDD_CSV}" \
  --output-root          "${OUTPUT_ROOT}" \
  --skip-existing \
  \
  `# ── task-graph generator LLM ──────────────────────────────────` \
  --graph-model-name     "${GRAPH_MODEL_NAME}" \
  --graph-max-new-tokens 2048 \
  --graph-temperature    0.0 \
  --graph-device-map     none \
  --graph-device         -1 \
  --graph-torch-dtype    bfloat16 \
  --graph-retries        3 \
  \
  `# ── MoLE execute model + checkpoint ───────────────────────────` \
  "${CKPT_ARGS[@]}" \
  --exec-model-name      "${EXEC_MODEL_NAME}" \
  --torch-dtype          "${TORCH_DTYPE}" \
  --device               "${DEVICE}" \
  --num-subtask-experts  "${NUM_SUBTASK_EXPERTS}" \
  --subtask-top-k        "${SUBTASK_TOP_K}" \
  --max-new-tokens       "${MAX_NEW_TOKENS}" \
  --lora-rank            "${LORA_RANK}" \
  --lora-alpha           "${LORA_ALPHA}" \
  --lora-last-n-layers   "${LORA_LAST_N_LAYERS}" \
  --router-mode          "${ROUTER_MODE}" \
  --max-attempts         "${MAX_ATTEMPTS}" \
  --smoke-timeout-s      "${SMOKE_TIMEOUT_S}"

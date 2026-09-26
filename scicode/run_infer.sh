#!/usr/bin/env bash
# ============================================================================
# MoRSE / SciCode — inference / evaluation of a trained MoLE checkpoint.
#
# Runs the SciCode TaskGraph pipeline (Merger -> Executor) with the retry-mix
# router policy, then scores generated code against SciCode's step/general
# tests (requires eval/data/test_data.h5; see eval/download_test_data.sh).
#
# Portable launcher: plain bash, single GPU, no slurm / tmux / cluster paths.
# Edit the variables in the CONFIG block below.
# ============================================================================
set -euo pipefail

# Resolve the repo root as the parent of the directory containing this script
# (this file lives at <REPO_ROOT>/scicode/run_infer.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_USE_CHAT_TEMPLATE=off

# ============================ CONFIG (edit me) ==============================

# --- Model + checkpoint ----------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
# Path to a trained MoLE checkpoint, e.g. <CKPT_ROOT>/<RUN_NAME>/epoch_001.
# Leave empty to run the frozen backbone with no LoRA experts.
MOLE_CHECKPOINT="${MOLE_CHECKPOINT:-}"

# --- Data ------------------------------------------------------------------
# Evaluation problems. IID: data/mydev.jsonl (20). OOD: data/ood_test.jsonl (16).
DATASET="${DATASET:-${SCRIPT_DIR}/data/mydev.jsonl}"
# Precomputed (role, subtask) task graphs for the problems above.
GRAPH_ROOT="${GRAPH_ROOT:-${SCRIPT_DIR}/data/taskgraphs}"
# Numerical ground-truth tests (fetched via eval/download_test_data.sh).
H5PY_FILE="${H5PY_FILE:-${SCRIPT_DIR}/eval/data/test_data.h5}"

# --- Output ----------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/infer_runs}"
TIMESTAMP="${TIMESTAMP:-morse_scicode_infer_$(date +%Y%m%d_%H%M%S)}"

# --- GPU -------------------------------------------------------------------
GPU="${GPU:-0}"

# ============================================================================

mkdir -p "${OUTPUT_ROOT}"

echo "=== MoRSE / SciCode inference (retry-mix pipeline) ==="
echo "REPO_ROOT       : ${REPO_ROOT}"
echo "MODEL_NAME      : ${MODEL_NAME}"
echo "MOLE_CHECKPOINT : ${MOLE_CHECKPOINT:-<none / frozen backbone>}"
echo "DATASET         : ${DATASET}"
echo "GRAPH_ROOT      : ${GRAPH_ROOT}"
echo "OUTPUT_ROOT     : ${OUTPUT_ROOT}"
echo "TIMESTAMP       : ${TIMESTAMP}"

# Assemble args; only pass --mole-checkpoint when one is provided.
CKPT_ARGS=()
if [[ -n "${MOLE_CHECKPOINT}" ]]; then
  CKPT_ARGS+=(--mole-checkpoint "${MOLE_CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
python -u "${SCRIPT_DIR}/infer.py" \
  --dataset                  "${DATASET}" \
  --graph-root               "${GRAPH_ROOT}" \
  --h5py-file                "${H5PY_FILE}" \
  --output-root              "${OUTPUT_ROOT}" \
  --timestamp                "${TIMESTAMP}" \
  --skip-existing \
  --gpus                     "${GPU}" \
  --device                   0 \
  --device-map               auto \
  --torch-dtype              "${TORCH_DTYPE}" \
  --code-model-name          "${MODEL_NAME}" \
  --code-max-new-tokens      4096 \
  --code-temperature         0.0 \
  "${CKPT_ARGS[@]}" \
  `# ---- MoLE architecture (must match training) ----` \
  --mole-num-subtask-experts 4 \
  --mole-subtask-top-k       2 \
  --mole-lora-rank           8 \
  --mole-lora-alpha          16.0 \
  --mole-lora-last-n-layers  8 \
  `# ---- pipeline / scheduling ----` \
  --max-attempts             3 \
  --aggregate-mode           llm \
  --with-background \
  --prompt-style             strict \
  --prefill-python-fence \
  --keep-model-loaded

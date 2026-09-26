#!/usr/bin/env bash
# ============================================================================
# MoRSE / SciCode — HGRPO training (paper config).
#
# This is the "no-anchor" hierarchical-credit configuration reported in the
# paper: two-layer HGRPO credit (within-route advantage updates the LoRA
# experts; cross-route advantage updates the prototype router). It does NOT use
# a greedy "anchor" route — all executed routes are sampled.
#
# Portable launcher: plain bash, no slurm / tmux / cluster-specific paths.
# Edit the variables in the CONFIG block below for your model / data / GPUs.
# ============================================================================
set -euo pipefail

# Resolve the repo root as the parent of the directory containing this script
# (this file lives at <REPO_ROOT>/scicode/run_train.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Make the `morse` and `scicode` packages importable.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# The strict (non-chat) prompt template is used for both train and inference.
export HF_USE_CHAT_TEMPLATE=off

# ============================ CONFIG (edit me) ==============================

# --- Model -----------------------------------------------------------------
# HuggingFace model id or a local path to the frozen backbone.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# --- Data ------------------------------------------------------------------
# Training problems. Use the IID split (mytest.jsonl, 60 problems) or the OOD
# split (data/ood_train.jsonl, 64 problems = Physics+Math+Material Science).
DATASET="${DATASET:-${SCRIPT_DIR}/data/mytest.jsonl}"
# Difficulty metadata for the easy->hard curriculum.
CURRICULUM_FILE="${CURRICULUM_FILE:-${SCRIPT_DIR}/data/domain_difficulty_stats.tsv}"
# Precomputed (role, subtask) task graphs for the problems above.
# Point this at the directory holding the per-problem task-graph JSONs.
GRAPH_ROOT="${GRAPH_ROOT:-${SCRIPT_DIR}/data/taskgraphs}"
# Numerical ground-truth tests (fetched via eval/download_test_data.sh).
H5PY_FILE="${H5PY_FILE:-${SCRIPT_DIR}/eval/data/test_data.h5}"

# --- Output ----------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
CKPT_ROOT="${CKPT_ROOT:-${SCRIPT_DIR}/checkpoints}"
RUN_NAME="${RUN_NAME:-morse_scicode_$(date +%Y%m%d_%H%M%S)}"

# --- Distributed -----------------------------------------------------------
# Number of GPUs / data-parallel processes. Group structure: each process owns
# 1 route x 4 candidates, so GROUP_SIZE = 4 x NPROC.
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29713}"
GROUP_SIZE="${GROUP_SIZE:-16}"   # = 4 candidates x NPROC

# ============================================================================

mkdir -p "${OUTPUT_ROOT}" "${CKPT_ROOT}"

echo "=== MoRSE / SciCode HGRPO training (no-anchor, hierarchical credit) ==="
echo "REPO_ROOT  : ${REPO_ROOT}"
echo "MODEL_NAME : ${MODEL_NAME}"
echo "DATASET    : ${DATASET}"
echo "GRAPH_ROOT : ${GRAPH_ROOT}"
echo "RUN_NAME   : ${RUN_NAME}"
echo "NPROC      : ${NPROC}  (GROUP_SIZE=${GROUP_SIZE})"

python -u -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port "${MASTER_PORT}" \
  "${SCRIPT_DIR}/train.py" \
  \
  --dataset                    "${DATASET}" \
  --graph-root                 "${GRAPH_ROOT}" \
  --curriculum-difficulty-file "${CURRICULUM_FILE}" \
  --h5py-file                  "${H5PY_FILE}" \
  --run-name                   "${RUN_NAME}" \
  --output-root                "${OUTPUT_ROOT}" \
  --ckpt-root                  "${CKPT_ROOT}" \
  --model-name                 "${MODEL_NAME}" \
  --gpus                       "" \
  --torch-dtype                "${TORCH_DTYPE}" \
  \
  --epochs                     10 \
  --max-problems               60 \
  \
  `# ---- sampling ----` \
  --group-size                 "${GROUP_SIZE}" \
  --temperature                0.7 \
  --top-p                      0.95 \
  --max-new-tokens             3072 \
  \
  `# ---- MoLE architecture: 1 route x 4 subtask experts, top-2 ----` \
  --hierarchical-local-routes  1 \
  --num-subtask-experts        4 \
  --subtask-top-k              2 \
  --lora-rank                  8 \
  --lora-alpha                 16 \
  --lora-last-n-layers         8 \
  --lora-lr                    8e-5 \
  --router-lr                  3e-5 \
  \
  `# ---- expert diversity + router regularizers (baseline low values) ----` \
  --enforce-local-expert-diversity \
  --enforce-local-output-diversity \
  --diversity-max-resample     16 \
  --subtask-proto-l2           1e-4 \
  --subtask-proto-ortho        0.05 \
  --alpha-router               0.15 \
  \
  `# ---- reward: step-level rule-based verifier only ----` \
  --reward-w-step              1.0 \
  --reward-w-shape             0.0 \
  --reward-w-gt                0.0 \
  --reward-pass-bonus          0.0 \
  \
  `# ---- GRPO update policy ----` \
  --grpo-adv-normalize \
  --advantage-clip             5.0 \
  --grpo-skip-update-if-allzero \
  --grpo-skip-update-if-low-std \
  --grpo-min-reward-std        0.0005 \
  \
  `# ---- ground-truth signals fully disabled (no GT leakage) ----` \
  --disable-gt-code-signals \
  --no-enable-gt-stage-schedule \
  --no-enable-tf-ce \
  --no-enable-tf-reward \
  \
  `# ---- data + curriculum ----` \
  --include-execute-entries \
  --no-include-aggregate-entries \
  --shuffle-entries \
  --curriculum-mode            easy_to_hard \
  --router-text-source         title \
  \
  `# ---- prompt format (strict, non-chat) ----` \
  --prompt-style               strict \
  --prefill-python-fence \
  --hf-use-chat-template       off \
  --with-background \
  \
  `# ---- checkpointing + timeout ----` \
  --save-every-epochs          1 \
  --save-every-samples         10 \
  --no-save-candidate-snapshots \
  --dist-timeout-s             21600

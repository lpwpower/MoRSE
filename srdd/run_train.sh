#!/usr/bin/env bash
# ============================================================================
# SRDD HGRPO training — A2 NO-ANCHOR config (paper's main method)
#
#   credit_mode          = hierarchical  (LoRA: within-route adv, Router: across-route adv)
#   execute_route_policy = sample        (all sampled, no greedy anchor)
#   router_adv_mode      = mean_center   (route mean vs group route mean)
#   router_contrast_weight = 0.0
#   sampled_route_weight   = 1.0
#
#   SRDD-specific: both execute and aggregate experts are trained
#                  (--include-execute-entries + --include-aggregate-entries)
# ============================================================================
set -euo pipefail

# REPO_ROOT is the directory that CONTAINS morse/, scicode/, srdd/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# ── Documented variables (edit as needed) ───────────────────────────────────
#   MODEL_PATH     : HF model id or local snapshot dir for the MoLE backbone.
#   SRDD_CSV       : SRDD dataset CSV (category/name/description).
#   TASKGRAPH_ROOT : root containing pre-generated SRDD task_graph.json files.
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
SRDD_CSV="${SRDD_CSV:-${REPO_ROOT}/srdd/data/SRDD.csv}"
TASKGRAPH_ROOT="${TASKGRAPH_ROOT:-${REPO_ROOT}/srdd/data/taskgraphs}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/srdd/runs}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/srdd/checkpoints}"
RUN_NAME="${RUN_NAME:-srdd_exp2_A2_noanchor}"

# Distributed launch settings.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29721}"

# Training schedule.
EPOCHS="${EPOCHS:-10}"
GROUP_SIZE="${GROUP_SIZE:-16}"          # total candidates per GRPO update (across ranks)

echo "=== SRDD HGRPO Exp2: A2 no-anchor (hierarchical + all-sampled) — main method ==="
echo "REPO_ROOT      : ${REPO_ROOT}"
echo "RUN_NAME       : ${RUN_NAME}"
echo "MODEL_PATH     : ${MODEL_PATH}"
echo "credit_mode          = hierarchical"
echo "execute_route_policy = sample (no anchor)"
echo "router_adv_mode      = mean_center"
echo "include_aggregate    = True (SRDD: train both experts)"

python -u -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  "${REPO_ROOT}/srdd/train.py" \
  \
  --srdd-csv                   "${SRDD_CSV}" \
  --taskgraph-root             "${TASKGRAPH_ROOT}" \
  --run-name                   "${RUN_NAME}" \
  --output-root                "${OUTPUT_ROOT}" \
  --ckpt-root                  "${CKPT_ROOT}" \
  --model-name                 "${MODEL_PATH}" \
  --torch-dtype                bfloat16 \
  \
  --epochs                     "${EPOCHS}" \
  --max-samples                0 \
  \
  `# ── sampling ──────────────────────────────────────────────────` \
  --group-size                 "${GROUP_SIZE}" \
  --hierarchical-local-routes  1 \
  --temperature                0.7 \
  --top-p                      0.95 \
  --max-new-tokens             4096 \
  \
  `# ── MoLE architecture ─────────────────────────────────────────` \
  --num-subtask-experts        4 \
  --subtask-top-k              2 \
  --lora-rank                  8 \
  --lora-alpha                 16 \
  --lora-last-n-layers         8 \
  --lora-lr                    8e-5 \
  --router-lr                  3e-5 \
  \
  `# ── expert diversity & router regularization ──────────────────` \
  --enforce-local-expert-diversity \
  --no-enforce-local-output-diversity \
  --diversity-max-resample     16 \
  --router-diversity-reg       0.02 \
  --subtask-proto-l2           1e-4 \
  --subtask-proto-ortho        0.05 \
  \
  `# ── [SRDD] train both execute & aggregate experts ─────────────` \
  --include-execute-entries \
  --include-aggregate-entries \
  --aggregate-min-parents      2 \
  --smoke-timeout-s            30 \
  \
  `# ── [A2 no-anchor] all-sampled + mean_center ──────────────────` \
  --credit-mode                hierarchical \
  --execute-route-policy       sample \
  --router-adv-mode            mean_center \
  --router-contrast-weight     0.0 \
  --alpha-router               0.15 \
  --sampled-route-weight       1.0 \
  --no-sampled-positive-adv-only \
  --sampled-drop-below-anchor-delta -1.0 \
  \
  `# ── GRPO update policy ────────────────────────────────────────` \
  --grpo-adv-normalize \
  --advantage-clip             5.0 \
  --grpo-skip-update-if-allzero \
  --grpo-skip-update-if-low-std \
  --grpo-min-reward-std        0.0005 \
  \
  `# ── saving & timeout ──────────────────────────────────────────` \
  --save-every-epochs          1 \
  --dist-timeout-s             21600

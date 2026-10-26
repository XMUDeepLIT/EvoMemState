#!/usr/bin/env bash
# Train the EvoMemState Reasoning State policy using slime GRPO.
set +x
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

export PYTHONUNBUFFERED=1
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}/evomemstate-ray}"
export RAY_memory_monitor_refresh_ms=0

# Frozen answer (vLLM). Override if needed.
export ANSWER_MODEL="${ANSWER_MODEL:-${GEN_B_MODEL:-qwen3-8b}}"
export ANSWER_BASE_URL="${ANSWER_BASE_URL:-${GEN_B_BASE_URL:-http://127.0.0.1:8800/v1}}"
export ANSWER_API_KEY="${ANSWER_API_KEY:-${GEN_B_API_KEY:-EMPTY}}"
# Concurrent OpenAI calls to frozen answer during reward (AsyncOpenAI + semaphore).
export REASONING_STATE_ANSWER_CONCURRENCY="${REASONING_STATE_ANSWER_CONCURRENCY:-64}"
export REASONING_STATE_ANSWER_MAX_TOKENS="${REASONING_STATE_ANSWER_MAX_TOKENS:-512}"
# LLM judge (same ANSWER_* 8B): zero reward if reasoning_state contains answer statements.
export REASONING_STATE_ANSWER_JUDGE="${REASONING_STATE_ANSWER_JUDGE:-1}"
export REASONING_STATE_ANSWER_JUDGE_MAX_TOKENS="${REASONING_STATE_ANSWER_JUDGE_MAX_TOKENS:-32}"
export REASONING_STATE_ANSWER_JUDGE_PENALTY="${REASONING_STATE_ANSWER_JUDGE_PENALTY:-0.1}"
# Empty / unusable reasoning_state (after strip) → penalty_acc = -REASONING_STATE_FORMAT_EMPTY_PENALTY
export REASONING_STATE_FORMAT_EMPTY_PENALTY="${REASONING_STATE_FORMAT_EMPTY_PENALTY:-0.05}"
# Think-mode SFT/RL: extract <reasoning_state> before format gate / answer scoring.
export REASONING_STATE_THINK_MODE="${REASONING_STATE_THINK_MODE:-0}"
# Memory-only prompt: omit question/claim from reasoning_state messages.
export REASONING_STATE_MEMORY_ONLY="${REASONING_STATE_MEMORY_ONLY:-0}"
# Channel-needs prompt: inject fact_need/preference_need (typically with MEMORY_ONLY=1).
export REASONING_STATE_CHANNEL_NEEDS="${REASONING_STATE_CHANNEL_NEEDS:-0}"
# grpo: (delta - mean) / std. baseline_anchored: (S - S_0) / std, no mean.
export REASONING_STATE_ADVANTAGE="${REASONING_STATE_ADVANTAGE:-grpo}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "Frozen answer model: ${ANSWER_MODEL}"
echo "REASONING_STATE_ANSWER_CONCURRENCY=${REASONING_STATE_ANSWER_CONCURRENCY} REASONING_STATE_ANSWER_MAX_TOKENS=${REASONING_STATE_ANSWER_MAX_TOKENS}"
echo "REASONING_STATE_ANSWER_JUDGE=${REASONING_STATE_ANSWER_JUDGE} PENALTY=${REASONING_STATE_ANSWER_JUDGE_PENALTY} MAX_TOKENS=${REASONING_STATE_ANSWER_JUDGE_MAX_TOKENS}"
echo "REASONING_STATE_FORMAT_EMPTY_PENALTY=${REASONING_STATE_FORMAT_EMPTY_PENALTY}"
echo "REASONING_STATE_THINK_MODE=${REASONING_STATE_THINK_MODE} REASONING_STATE_MEMORY_ONLY=${REASONING_STATE_MEMORY_ONLY} REASONING_STATE_CHANNEL_NEEDS=${REASONING_STATE_CHANNEL_NEEDS}"
echo "REASONING_STATE_ADVANTAGE=${REASONING_STATE_ADVANTAGE}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/slime}"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/Megatron-LM}"

HF_CHECKPOINT="${HF_CHECKPOINT:-${REPO_ROOT}/models/Qwen3-4B-Instruct-2507}"
REF_TORCH_DIST="${REF_TORCH_DIST:-${REPO_ROOT}/models/Qwen3-4B-Instruct-2507_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/outputs/training/reasoning_state_train.jsonl}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-${REPO_ROOT}/outputs/training/reasoning_state_eval.jsonl}"

DATA_SAVED_PATH_DIR="${DATA_SAVED_PATH_DIR:-${REPO_ROOT}/outputs/slime_saved}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-evomemstate-reasoning-state-qwen3-4b}"
timestamp=$(date -u "+%Y-%m-%d-%H-%M-%S")
DATA_SAVED_PATH="${DATA_SAVED_PATH:-${DATA_SAVED_PATH_DIR}/${EXPERIMENT_NAME}}"

CKPTS_PATH="${CKPTS_PATH:-${DATA_SAVED_PATH}/ckpts}"
STEP_INFO_PATH="${STEP_INFO_PATH:-${DATA_SAVED_PATH}/step_info}"
LOG_DIR="${LOG_DIR:-${DATA_SAVED_PATH}/log}"
TB_ROOT="${TB_ROOT:-${DATA_SAVED_PATH_DIR}/tb_logs}"
TB_PATH="${TB_PATH:-${TB_ROOT}/${EXPERIMENT_NAME}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/logs-${timestamp}.log}"

# Default 2 GPUs: leave one card for frozen answer vLLM when only a few are free.
NUM_GPUS="${NUM_GPUS:-2}"
ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-${NUM_GPUS}}"

# train / eval schedule (env-overridable)
SAVE_FREQ="${SAVE_FREQ:-50}"
EVAL_FREQ="${EVAL_FREQ:-50}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
NUM_EPOCH="${NUM_EPOCH:-1}"
# Full train defaults: 512 prompts × 8 samples = 4096 global.
# Constraint: rollout_batch_size * n_samples_per_prompt == global_batch_size * num_steps_per_rollout
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-512}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
export REASONING_STATE_MAX_CHARS="${REASONING_STATE_MAX_CHARS:-600}"
export REASONING_STATE_MAX_LINES="${REASONING_STATE_MAX_LINES:-8}"
# IMPORTANT: no spaces inside JSON — bash array expansion would split on them.
if [ "${REASONING_STATE_THINK_MODE}" = "1" ]; then
  # Think-mode: long <think>+<reasoning_state>; enable native thinking unless user overrides.
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1536}"
  APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":true}}"
else
  # No-think: disable Qwen3 native CoT so response budget is not eaten.
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
  APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
fi

if [ ! -d "${HF_CHECKPOINT}" ]; then
    echo "Missing HF_CHECKPOINT=${HF_CHECKPOINT}"
    echo "Download Qwen3-4B-Instruct-2507 and convert to torch_dist first."
    exit 1
fi
if [ ! -d "${REF_TORCH_DIST}" ]; then
    echo "Missing REF_TORCH_DIST=${REF_TORCH_DIST}"
    echo "Example:"
    echo "  cd ${SLIME_DIR} && source scripts/models/qwen3-4B-Instruct-2507.sh"
    echo "  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=${MEGATRON_PATH} python tools/convert_hf_to_torch_dist.py \\"
    echo "See docs/reasoning_state_rl.md for checkpoint conversion."
    exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Set MEGATRON_PATH to your Megatron-LM checkout" >&2
    exit 1
fi
if [ ! -s "${PROMPT_DATA}" ]; then
    echo "Set PROMPT_DATA to a prepared training JSONL file" >&2
    exit 1
fi

mkdir -p "${CKPTS_PATH}" "${STEP_INFO_PATH}" "${LOG_DIR}" "${TB_PATH}"
export TENSORBOARD_DIR="${TB_PATH}"
export REASONING_STATE_ANSWER_CACHE_DIR="${REASONING_STATE_ANSWER_CACHE_DIR:-${DATA_SAVED_PATH}/answer_cache}"
mkdir -p "${REASONING_STATE_ANSWER_CACHE_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1
cp "$0" "${DATA_SAVED_PATH}/run_reasoning_state_rl.sh"

echo "PROMPT_DATA=${PROMPT_DATA}"
echo "EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA}"
echo "DATA_SAVED_PATH=${DATA_SAVED_PATH}"

if [ "${SMOKE:-0}" = "1" ]; then
   CKPTS_PATH="${CKPTS_PATH}_smoke"
   STEP_INFO_PATH="${STEP_INFO_PATH}_smoke"
fi

echo "CKPTS_PATH=${CKPTS_PATH}"
echo "STEP_INFO_PATH=${STEP_INFO_PATH}"
echo "LOG_PATH=${LOG_PATH}"
echo "NUM_GPUS=${NUM_GPUS} ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS}"
echo "batch: rbs=${ROLLOUT_BATCH_SIZE} n=${N_SAMPLES_PER_PROMPT} gbs=${GLOBAL_BATCH_SIZE} epoch=${NUM_EPOCH} save=${SAVE_FREQ} eval=${EVAL_FREQ}"

# Override for other sizes, e.g. MODEL_SCRIPT=qwen3-4B-Instruct-2507.sh
MODEL_SCRIPT="${MODEL_SCRIPT:-qwen3-4B-Instruct-2507.sh}"
source "${SLIME_DIR}/scripts/models/${MODEL_SCRIPT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_TORCH_DIST}"
   --save "${CKPTS_PATH}"
   --save-interval "${SAVE_FREQ}"
)

# Only resume when explicitly requested. Do NOT --load after a smoke run into a full run.
if [ "${RESUME:-0}" = "1" ]; then
   if [ -f "${CKPTS_PATH}/latest_checkpointed_iteration.txt" ]; then
      CKPT_ARGS+=(--load "${CKPTS_PATH}")
      echo "Resuming from ${CKPTS_PATH}"
      # When continuing on a *different* prompt set / shorter epoch, Megatron's
      # OptimizerParamScheduler asserts (old lr_decay_steps vs new train_iters).
      # CONTINUE_NEW_DATA=1 → load model weights only, fresh Adam + new schedule.
      #   (recommended for multiscope continue from delta-v4-f1)
      # OVERRIDE_OPT_SCHED=1 → keep Adam state but override schedule numbers.
      if [ "${CONTINUE_NEW_DATA:-0}" = "1" ]; then
         CKPT_ARGS+=(--no-load-optim --no-load-rng --finetune --override-opt-param-scheduler)
         echo "CONTINUE_NEW_DATA=1: load weights only (no optim/rng), override LR schedule"
      elif [ "${OVERRIDE_OPT_SCHED:-0}" = "1" ]; then
         CKPT_ARGS+=(--override-opt-param-scheduler)
         echo "OVERRIDE_OPT_SCHED=1: keep optim, override LR schedule for new train_iters"
      fi
   else
      echo "RESUME=1 but no checkpoint in ${CKPTS_PATH}; starting fresh."
   fi
else
   echo "Fresh training (no --load). Set RESUME=1 to continue from ${CKPTS_PATH}."
fi

if [ "${SMOKE:-0}" = "1" ]; then
   # Tiny connectivity check only (1 rollout). Do not use these for real training.
   mkdir -p "${CKPTS_PATH}" "${STEP_INFO_PATH}"
   ROLLOUT_ARGS=(
      --prompt-data "${PROMPT_DATA}"
      --input-key messages
      --label-key reward_spec
      --metadata-key metadata
      --apply-chat-template
      --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
      --rollout-shuffle
      --custom-rm-path evomemstate.reasoning_state_reward.reward_from_reasoning_state
      --custom-reward-post-process-path evomemstate.reasoning_state_reward.post_process_reasoning_state_rewards
      --group-rm
      --num-rollout 1
      --rollout-batch-size 4
      --n-samples-per-prompt 2
      --rollout-max-response-len "${MAX_RESPONSE_LEN}"
      --rollout-max-prompt-len 4096
      --rollout-temperature 1
      --global-batch-size 8
      --balance-data
   )
   EVAL_ARGS=()
else
   # ~69254 prompts / 512 ≈ 135 rollouts per epoch; each step trains on 4096 samples (512×8).
   ROLLOUT_ARGS=(
      --prompt-data "${PROMPT_DATA}"
      --input-key messages
      --label-key reward_spec
      --metadata-key metadata
      --apply-chat-template
      --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"
      --rollout-shuffle
      --custom-rm-path evomemstate.reasoning_state_reward.reward_from_reasoning_state
      --custom-reward-post-process-path evomemstate.reasoning_state_reward.post_process_reasoning_state_rewards
      --group-rm
      --num-epoch "${NUM_EPOCH}"
      --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
      --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
      --rollout-max-response-len "${MAX_RESPONSE_LEN}"
      --rollout-max-prompt-len "${MAX_PROMPT_LEN}"
      --rollout-temperature 1
      --global-batch-size "${GLOBAL_BATCH_SIZE}"
      --balance-data
   )
   if [ -f "${EVAL_PROMPT_DATA}" ]; then
      EVAL_ARGS=(
         --eval-interval "${EVAL_FREQ}"
         --eval-prompt-data eval_holdout "${EVAL_PROMPT_DATA}@[0:512]"
         --eval-input-key messages
         --eval-label-key reward_spec
         --n-samples-per-eval-prompt "${EVAL_N_SAMPLES}"
         --eval-max-response-len "${MAX_RESPONSE_LEN}"
         --eval-max-prompt-len "${MAX_PROMPT_LEN}"
         --eval-temperature 0.0
      )
   else
      echo "EVAL_PROMPT_DATA not found (${EVAL_PROMPT_DATA}), skip eval."
      EVAL_ARGS=()
   fi
fi

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # 0.6B can push high; 4B/Instruct should lower via MAX_TOKENS_PER_GPU.
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-evomemstate-reasoning_state
   # --wandb-group ${EXPERIMENT_NAME}
   # --wandb-key ${WANDB_KEY}
)

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name "${TB_ROOT}"
   --tb-experiment-name "${EXPERIMENT_NAME}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

STEP_INFO_ARGS=(
   --step-info-path "${STEP_INFO_PATH}"
   --rollout-all-samples-process-path evomemstate.slime_step_info.process_rollout_samples
   --rollout-function-path evomemstate.slime_step_info.generate_rollout_with_step_info
)

if [ "${DUMP_DETAILS:-0}" = "1" ]; then
   STEP_INFO_ARGS+=(--dump-details "${DATA_SAVED_PATH}/dump_details")
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
cd "${SLIME_DIR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8265

export REPO_ROOT MEGATRON_PATH HAS_NVLINK STEP_INFO_PATH
RUNTIME_ENV_FILE=$(mktemp "${TMPDIR:-/tmp}/evomemstate-runtime-XXXXXX.json")
chmod 600 "${RUNTIME_ENV_FILE}"
trap 'rm -f "${RUNTIME_ENV_FILE}"' EXIT
python3 - "${RUNTIME_ENV_FILE}" <<'PY_RUNTIME'
import json, os, sys
keys = ['ANSWER_MODEL', 'ANSWER_BASE_URL', 'ANSWER_API_KEY', 'REASONING_STATE_ANSWER_CACHE_DIR', 'REASONING_STATE_ANSWER_CONCURRENCY', 'REASONING_STATE_ANSWER_MAX_TOKENS', 'REASONING_STATE_ANSWER_JUDGE', 'REASONING_STATE_ANSWER_JUDGE_MAX_TOKENS', 'REASONING_STATE_ANSWER_JUDGE_PENALTY', 'REASONING_STATE_FORMAT_EMPTY_PENALTY', 'REASONING_STATE_THINK_MODE', 'REASONING_STATE_MEMORY_ONLY', 'REASONING_STATE_CHANNEL_NEEDS', 'REASONING_STATE_ADVANTAGE', 'REASONING_STATE_MAX_CHARS', 'REASONING_STATE_MAX_LINES']
values = {k: os.environ[k] for k in keys}
values.update(PYTHONPATH=os.pathsep.join([os.environ['MEGATRON_PATH'],os.environ['REPO_ROOT'],os.environ.get('PYTHONPATH','')]), CUDA_DEVICE_MAX_CONNECTIONS='1', NCCL_NVLS_ENABLE=os.environ['HAS_NVLINK'], EVOMEMSTATE_STEP_INFO_PATH=os.environ['STEP_INFO_PATH'])
with open(sys.argv[1], 'w') as handle:
    json.dump({'env_vars': values}, handle)
PY_RUNTIME

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env="${RUNTIME_ENV_FILE}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${TENSORBOARD_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${STEP_INFO_ARGS[@]}"

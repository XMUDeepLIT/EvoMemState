# Reasoning State RL

The RL implementation is adapted from the original project's training code. Package imports, environment variables, tags and metrics now use `evomemstate` / `REASONING_STATE_*` / `<reasoning_state>`. Legacy hint JSONL must be regenerated with the included converter before use. Model weights and training datasets are external inputs.

## Environment

Use Linux, NVIDIA GPUs and Python 3.12 with a mutually compatible CUDA/PyTorch, SGLang, Megatron-LM, Transformer Engine and FlashAttention installation. `pip install -e ./slime` installs the backend's Python requirements, but does not supply those GPU kernels or Megatron-LM. Install those components in your training environment before launching. The framework's original LICENSE is retained under `slime/`.

## Checkpoint conversion

From the repository root after setting `MEGATRON_PATH`, `HF_CHECKPOINT` and `REF_TORCH_DIST`:

```bash
REPO_ROOT="$PWD"
(
  cd slime
  source scripts/models/qwen3-4B-Instruct-2507.sh
  PYTHONPATH="$MEGATRON_PATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" --save "$REF_TORCH_DIST"
)
```

`MODEL_SCRIPT` defaults to `qwen3-4B-Instruct-2507.sh`. Override it with a matching file under `slime/scripts/models/` for other model sizes. Keep the HF checkpoint, model definition and converted checkpoint consistent.

## External training inputs

The runner requires an existing nonempty `PROMPT_DATA`; it never downloads or constructs datasets automatically. Each JSONL row uses `messages` (policy chat prompt), `reward_spec` (for example `{"type":"token_f1","target":"reference answer"}`) and `metadata` containing `memory_body`, `question`, `answer_system`, `answer_user_template` and optional `has_supersede`. The user template substitutes `{context}` and `{question}`. References are labels for reward computation, not part of the policy input. `verdict_match` is also supported.

The converter accepts externally prepared SciFact, FEVER and PersonaMem manifests:

```bash
python scripts/prepare_reasoning_state_rl_jsonl.py --help
```

Optional SFT warmup conversion is available in `scripts/prepare_reasoning_state_sft_jsonl.py`. No SFT trainer or source datasets are included.

## Training and reward

Use the command in the README. Set `EVAL_PROMPT_DATA` to an external held-out JSONL if evaluation is needed. `SMOKE=1` performs one small rollout; it still needs GPUs, full training dependencies and a frozen answer endpoint. `RESUME=1` resumes an existing checkpoint. `NUM_GPUS` defaults to two colocated actor/rollout GPUs; provision the frozen answer endpoint separately.

The default reward is `score(with_state) - score(without_state)`, gated by output format, plus penalties for empty states and answer leakage. GRPO whitens the improvement and penalty channels separately; `REASONING_STATE_ADVANTAGE=baseline_anchored` selects the original alternative normalization. This is retained from the source implementation.

No process-killing commands are included. Command tracing is disabled. Runtime API credentials are serialized with JSON escaping to a mode-600 temporary file removed when the launcher exits. The local Ray dashboard binds to loopback. Training outputs, answer caches and rollout logs contain prompts and model text; keep them local under the ignored `outputs/` directory.

The bundled slime source includes the original working-tree argument extension needed for rollout metrics. Its source revision is recorded in `slime/README.md`; local git metadata, agent settings, caches and unrelated examples are excluded.

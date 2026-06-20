# Skill: run-rl-training

**Trigger**: When asked to run RL training, GSPO training, or reinforcement learning on Tinker cloud.

**Scope**: Orchestrates `scripts/rl_train.py` on Tinker cloud via ServiceClient. Primary loss = GSPO (sequence-level IS via `forward_backward_custom` + RSPO stop-gradient floor). GRPO importance_sampling is the documented fallback only (`--grpo-fallback`/`--no-gspo`). See Deviations section.

---

## Dispatch Boundary Rule

`Agent(run_in_background=true)` is valid for exactly ONE purpose: the background telemetry monitor (Step 3).

Judge-consistency scoring runs INSIDE `rl_train.py` → `rl_judge_dispatch.py` → `claude_agent.py` subprocess (`claude --print`). This is NOT an `Agent()` call from the orchestrator. Do not spawn additional judge agents from this skill.

---

## Step 0a: Confirm base model

Check which model to train. Default: `Qwen/Qwen3-30B`.

```bash
rtk cat output/rl_checkpoints/checkpoint_manifest.json 2>/dev/null || echo "No manifest yet — fresh run."
```

Set `MODEL_ID` (default `Qwen/Qwen3-30B`) and `MODEL_SHORT` (e.g. `qwen3-30b`).

---

## Step 0b: Confirm dataset

```bash
rtk ls data/reasoning_dataset/
```

Required files:
- `openai_train.jsonl` (or augmented variant)
- `openai_val.jsonl`

Abort if missing.

---

## Step 0c: Set experiment name

```bash
DATE=$(date +%Y%m%d)
N=001   # increment from latest manifest entry if present
EXPERIMENT="${MODEL_SHORT}_rl_experiment_${N}_${DATE}"
echo "Experiment: $EXPERIMENT"
```

Used for telemetry directory and run summary. Not passed to the training script (it writes to fixed paths).

---

## Step 0d: Confirm configuration

Present to user:

| Parameter | Value |
|-----------|-------|
| Model | `$MODEL_ID` |
| Experiment | `$EXPERIMENT` |
| LoRA rank | 32 (default) |
| Batch size | 8 (default) |
| Total steps | 500 (default) |
| Checkpoint every | 50 steps |
| Jaccard check every | 20 steps |
| Loss mode | GSPO primary (use_gspo=True, default) |
| KL soft/hard | 0.1 / 0.3 |
| e_frac soft/hard | 0.7 / 0.5 |
| Mask path | output/profiling/reasoning-merged-v4/protected_expert_mask.npy |
| Metrics | output/rl_checkpoints/metrics/rl_metrics.jsonl |
| Manifest | output/rl_checkpoints/checkpoint_manifest.json |

Confirm with user before proceeding.

---

## Step 1: Configure (override defaults if requested)

Build the base command. Start with GSPO defaults — add `--grpo-fallback` only if user explicitly requests GRPO fallback mode:

```bash
BASE_CMD="python scripts/rl_train.py \
  --model-id $MODEL_ID \
  --lora-rank 32 \
  --lora-seed 42 \
  --total-steps 500 \
  --batch-size 8 \
  --checkpoint-every 50 \
  --jaccard-every 20 \
  --kl-soft 0.1 \
  --kl-hard 0.3 \
  --efrac-soft 0.7 \
  --efrac-hard 0.5"
# Add --grpo-fallback here ONLY if user explicitly requests it.
```

Flag reference:
- `--grpo-fallback` / `--no-gspo`: sets use_gspo=False → token-level IS via `forward_backward`. Use only when GSPO causes training instability and user authorizes fallback.
- `--protected-expert-mask PATH`: override default mask path if profiling artifacts moved.
- `--kl-soft`, `--kl-hard`: WARNING / HALT thresholds on kl_sample_train_v1.
- `--efrac-soft`, `--efrac-hard`: WARNING / HALT thresholds on e_frac_with_tokens_mean.

---

## Step 2: Validate preflight

Check Tinker credentials and protected-expert mask:

```bash
# Confirm Tinker credentials file exists (ServiceClient reads from ~/.tinker or env)
rtk ls ~/.tinker 2>/dev/null || echo "Check TINKER env vars"

# Confirm mask exists
test -f output/profiling/reasoning-merged-v4/protected_expert_mask.npy \
  && echo "Mask OK" \
  || echo "WARNING: mask missing — Jaccard monitor disabled"
```

If credentials are absent, surface auth gate (do not proceed).

---

## Step 3: Spawn background telemetry monitor

Before running, start a background agent to tail metrics:

```
Agent(
  model="sonnet",
  description="Tail rl_metrics.jsonl — surface KL autohalt / MoE violations / Jaccard drift",
  prompt="Tail output/rl_checkpoints/metrics/rl_metrics.jsonl every 60 seconds. For each new line, parse JSON and report: reward_mean, kl_sample_train_v1, kl_sample_train_v2, e_frac_with_tokens_mean, e_max_violation_mean, e_max_violation_max, jaccard_protected. Alert if kl_sample_train_v1 > 0.1 (soft) or > 0.3 (hard). Alert if e_frac_with_tokens_mean < 0.7 (soft) or < 0.5 (hard). Log to logs/monitor_${EXPERIMENT}.log. Continue until output/rl_checkpoints/_stop exists.",
  run_in_background=true
)
```

Note: metric keys in the JSONL file use underscore form (`kl_sample_train_v1`, `e_frac_with_tokens_mean`). The Tinker client step-dict uses colon form internally — `_log_step` converts to underscore for the JSONL sink.

Stop signal: `touch output/rl_checkpoints/_stop`

---

## Step 4: Dry run

Validate end-to-end wiring with a synthetic step before committing to a full run:

```bash
rtk python scripts/rl_train.py $BASE_CMD --dry-run
```

`--dry-run` runs one synthetic step with a mock Tinker client, writes a real metrics row, then exits 0. If this fails, fix the issue before proceeding. Do not bypass.

---

## Step 5: Run RL training

```bash
rtk $BASE_CMD 2>&1 | tee logs/rl_train_${EXPERIMENT}.log
TRAIN_EXIT=$?
```

### During training — what to watch

The background monitor (Step 3) surfaces these automatically. For manual inspection:

```bash
rtk tail -f output/rl_checkpoints/metrics/rl_metrics.jsonl
```

Key fields per step:
- `reward_mean`: primary learning signal
- `kl_sample_train_v1`: KL divergence — soft warning at 0.1, hard halt at 0.3
- `e_frac_with_tokens_mean`: expert utilization — soft warning below 0.7, hard halt below 0.5
- `e_max_violation_max`: expert violation headroom
- `jaccard_protected`: protected-expert set stability (monitor-only, no enforcement)

---

## Step 6: Handle auto-halt outcomes

Check exit code and last metrics entry:

```bash
LAST_METRIC=$(tail -1 output/rl_checkpoints/metrics/rl_metrics.jsonl 2>/dev/null)
echo "Final metric row: $LAST_METRIC"
```

| Exit code | Likely cause | Action |
|-----------|-------------|--------|
| 0 | Normal completion | Continue to Step 7 |
| 1 (KL) | kl_sample_train_v1 ≥ 0.3 hard halt | Inspect KL trend; consider reducing batch-size or lr before retry |
| 1 (e_frac) | e_frac_with_tokens_mean ≤ 0.5 hard halt | MoE expert collapse; report to user — do NOT auto-retry |
| Other | Tinker ServiceClient error / OOM | Check logs; surface to user |

Emergency checkpoint is written before KL/e_frac hard halt. Confirm in manifest:

```bash
rtk python -c "import json; m=json.load(open('output/rl_checkpoints/checkpoint_manifest.json')); print(list(m.keys())[-3:])"
```

---

## Step 7: Anti-hack regression gate

After a successful (exit 0) or checkpoint-halted run, run the regression gate to detect reward hacking (judge score inflated while quality degraded):

```bash
rtk python scripts/eval_judge.py --mode regression \
  --checkpoint-manifest output/rl_checkpoints/checkpoint_manifest.json \
  --baseline-dir output/baselines/ \
  2>&1 | tee logs/regression_${EXPERIMENT}.log
```

**CI-aware disposition**: bootstrap lower bound must clear the bar, measured identically on baseline and candidate. If judge-score increases while fix-correctness decreases, flag for human review — do NOT auto-promote.

Panickssery spot-check runs automatically inside `rl_train.py` every ~50 steps: rollouts where `|fix_correctness - judge_consistency| > 0.3` are logged to `rl_metrics.jsonl`. Review these entries for reward-signal alignment.

---

## Step 8: Stop telemetry monitor

```bash
touch output/rl_checkpoints/_stop
echo "Monitor stop signal sent."
```

Wait ~60 seconds for the background agent to observe the signal and exit cleanly.

---

## Step 9: Verify checkpoint and write summary

Confirm checkpoint manifest has a final entry:

```bash
rtk python -c "
import json
m = json.load(open('output/rl_checkpoints/checkpoint_manifest.json'))
keys = list(m.keys())
print('Checkpoint entries:', len(keys))
print('Latest:', keys[-1], '->', m[keys[-1]].get('sampler_path', 'NO sampler_path'))
"
```

For a full 500-step run, expect a `final-step-500` entry with a `sampler_path`.

Write run summary:

```bash
cat > logs/summary_${EXPERIMENT}.md << 'SUMMARY'
# RL Training Run Summary

Experiment: ${EXPERIMENT}
Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Loss mode: GSPO (use_gspo=True, default) — sequence-level IS via forward_backward_custom + RSPO stop-gradient floor

## Outcomes

- Training exit code: ${TRAIN_EXIT}
- Final reward_mean: (from last metrics row)
- Final kl_sample_train_v1: (from last metrics row)
- Checkpoint manifest entries: (count)
- Latest sampler_path: (from manifest)
- Anti-hack regression: (PASS / FLAG)

## Deviations / Notes

(document any auto-halts, fallback activations, or human interventions)
SUMMARY
```

Return path to user:
- Metrics: `output/rl_checkpoints/metrics/rl_metrics.jsonl`
- Manifest: `output/rl_checkpoints/checkpoint_manifest.json`
- Log: `logs/rl_train_${EXPERIMENT}.log`
- Summary: `logs/summary_${EXPERIMENT}.md`

---

## Deviations

### D-09-02: Router gates frozen on Tinker (monitor-only)

Router gates are FROZEN for this RL run. `create_lora_training_client` is called WITHOUT any `train_router` argument. Protected-expert Jaccard check (`jaccard_protected` field in metrics, checked every 20 steps via `--jaccard-every`) is monitor-only — it logs similarity against `protected_expert_mask.npy` (shape [48,128] bool) but does NOT enforce or halt training. Router-gate training is NOT available on this venue.

### D-09-03: GSPO is the PRIMARY/default loss (LOCKED)

GSPO (sequence-level importance sampling via `tc.forward_backward_custom` + RSPO stop-gradient floor `seq_ratio.clamp(min=1.0)`, use_gspo defaults True) is the primary loss, active by default with no flags required.

GRPO (token-level IS via `tc.forward_backward(data, loss_fn="importance_sampling")`, `use_gspo=False`) is the documented instability fallback only. Activate via `--grpo-fallback` or `--no-gspo` when GSPO causes observable training instability and the user explicitly authorizes fallback. GRPO is NOT a comparison target.

`save_weights_for_sampler(name=..., ttl_seconds=None)` is called on every checkpoint to persist weights with no expiry.

---

## Key Rules

1. **GSPO default**: Never add `--grpo-fallback` unless user explicitly requests it due to observed instability. GSPO (use_gspo=True) is always the starting point.

2. **Dry run is mandatory**: Always run `--dry-run` (Step 4) before the real training run. Never bypass.

3. **Router gates frozen**: Do not attempt to pass `train_router` args to `create_lora_training_client`. Monitor-only.

4. **Dispatch boundary**: `Agent(run_in_background=true)` = telemetry monitor only. Judge scoring = subprocess inside training script.

5. **Anti-hack gate**: Always run the regression gate (Step 7) after training. Do not auto-promote if judge-score rises while fix-correctness falls.

6. **Credential hygiene**: ServiceClient reads credentials from `~/.tinker` or env. Never hardcode tokens.

7. **Stop signal before exit**: Always send `touch output/rl_checkpoints/_stop` (Step 8) to terminate the background monitor cleanly.

8. **Metric key form**: JSONL sink uses underscore keys (`e_frac_with_tokens_mean`, `kl_sample_train_v1`). Tinker step-dicts use colon keys internally. Do not confuse the two.

9. **MO-GRPO normalization**: Reward pipeline applies per-signal within-group normalization before scalar combination. This is done inside `scripts/reward_pipeline.py` — do not re-normalize upstream.

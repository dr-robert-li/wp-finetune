---
status: partial
phase: 09-gspo-training
source: [09-VERIFICATION.md]
started: 2026-06-20
updated: 2026-06-20
---

## Current Test

[awaiting human testing — requires a credentialed live Tinker run]

## Tests

### 1. Live GSPO RL training run on Tinker
expected: `wp-finetune:run-rl-training` (or, equivalently, `.venv-tinker/bin/python scripts/rl_train.py --judge-base-url http://localhost:8000/v1 --judge-model wp_judge` without `--dry-run`) executes the GSPO sequence-level loop on Qwen3-30B-A3B over the assembled prompt pools (68 gen / 482 judge). Gradients flow to experts+attn+unembed (router frozen by design, D-09-02); per-step `rl_metrics.jsonl` emits; `forward_backward_custom` + RSPO floor is the active loss (use_gspo default True). Blocked only by Tinker cloud credentials (manual by design).
note (2026-06-21, phase9-live-rl-wiring): the live path was never runnable from the CLI — bare `python scripts/rl_train.py` (no `--dry-run`) raised `SystemExit` because `main()` could not construct a judge_client. RESOLVED by extracting `run_training(args)` and adding `--judge-base-url` (main() now builds the `openai.OpenAI` judge client itself); `--manifest-path`/`--metrics-path` added so a smoke can isolate outputs. Prereqs: `.venv-tinker` (tinker+openai+scipy), `TINKER_API_KEY` exported from `.env`, and the v4 judge served as `wp_judge` via `scripts/serve_v4_judge_vllm.sh`. The earlier `--judge-model`/`--n-votes`/hard-attribute guard fix (commit `06dcba7`) remains.
result: [pending — wiring complete + smoke; full live run gated on user go-ahead]

### 2. Per-step auto-halt fires on real divergence (GRPO-08)
expected: during the live run, a genuine KL (`kl_sample_train_v1` > 0.3) or MoE-routing-collapse (`e_frac_with_tokens:mean` < 0.5) breach halts training BEFORE the next `optim_step` and writes an emergency persistent checkpoint. (Synthetic-tensor path is already unit+integration tested; this confirms it on live metrics.)
result: [pending]

### 3. Dual-mode RLEV reward_breakdown in live metrics
expected: `rl_metrics.jsonl` from a live run contains both `<wp_gen>` (PHPCS+security+VeRPO) and `<wp_judge>` (capped consistency + fix-correctness) signals in `reward_breakdown`, confirming both pathways receive gradient and the judge≥gen budget holds — the fields Phase 10 (RLEV-01/02) consumes.
result: [pending]

### 4. Protected-expert Jaccard monitor vs Phase 7 mask (live routing)
expected: every-N-step Jaccard against `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` is logged (monitor-only, no enforcement — frozen-router consequence). Dry-run already shows the monitor wired (`jaccard=0.0020`); confirm against real routing distributions.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps

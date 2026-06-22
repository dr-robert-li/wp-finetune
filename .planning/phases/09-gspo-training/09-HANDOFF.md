# Phase 09 — RL Training HANDOFF (resume context)

**Written:** 2026-06-22 ~23:00 AEST · **Branch:** `phase10-execution`
**Status:** Zero-reward bug FIXED + validated (offline + live). Full RL run BLOCKED on one
decision: how to handle the judge-consistency scorer, which now bills paid API (policy change).
No processes running; judge vLLM stopped; no compute/$ burning.

---

## TL;DR — where things stand

1. **Warm-start RL is wired and works.** v4 `save_state` regenerated, warm-start confirmed live
   (`train_mlp=True attn=False unembed=False`), MoE-only per D-09-08 (signed off).
2. **The zero-reward bug is fixed.** Root cause: reward pipeline assumed instruct-style fenced/JSON
   output, but the v4 policy + served v4 judge emit bare code + prose scores → both reward parsers
   returned 0 → uniform reward → no gradient. Two parser fixes + a token-cap bump (commit `a1b98bb`).
   Proven: offline probe HARD-gate ALL PASS, and a **live 25-step run** showed `fix_correctness=1.0`
   and `reward_min != reward_max` on **25/25 steps**.
3. **BLOCKER (cost).** The judge-consistency reward (0.3 weight) is scored via `scripts/claude_agent.py`
   = `claude -p`. Per a recent Anthropic policy change (confirmed 2026-06-22), **`claude -p` / Agent SDK /
   managed agents ALWAYS bill direct API now — no subscription path.** This caused **~$90** direct API
   spend in the signal run. Must resolve before the full (multi-day) run. **3 options below.**

---

## What was done this session (commits, newest first)

- `652acb2` docs(09): record `claude -p` ALWAYS bills API + cost incident; corrected `claude_agent.py`
  docstring; global policy note in `~/.claude/CLAUDE.md`.
- `bdcd5bd` fix(09): scrub `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` from `claude` subprocess env
  (`_agent_env`). **NOTE: now only hygiene — does NOT make `claude -p` free** (policy makes it always paid).
- `a1b98bb` fix(09): **the zero-reward unblock** (the 2 parser fixes + token cap). See below.
- `db30563` docs(09): sign off RL init reconciliation — D-09-08 MoE-only warm-start RL.
- (earlier) warm-start arc: parser/extract/non-code-guard fixes `8e99619`, `d2773e2`, `39f10c7`.

---

## The zero-reward fix (commit `a1b98bb`) — what changed and why

Diagnosed with a custom probe: **`scripts/_probe_rl_reward.py`** (offline, $0, monkeypatches the judge —
no API). Run it anytime:
```bash
REWARD_SKIP_PHPCS_ASSERT=1 .venv-tinker/bin/python scripts/_probe_rl_reward.py \
  --completions output/rl_checkpoints/judge_failures.preguard.prefixfix-diag.jsonl
```

**Three mechanisms** (full writeup: `.planning/debug/09-rl-warmstart-zero-reward.md`):

- **Mechanism 1 (judge rollouts, fixed).** Reward = `fix_correctness` on the corrected code extracted from
  the judge completion. The pool prompt (`<wp_judge> Evaluate this code`) elicited only a prose score, no
  corrected code → `extract_php_code` found nothing → fix=0. **Fix:** `_augment_judge_prompt` appends a
  contract requiring "critique + corrected code in a ```php fenced block beginning with `<?php`";
  `_extract_corrected_php` accepts ```php OR `<corrected_code>` (the SFT delimiter); judge generation cap
  raised to `JUDGE_MAX_NEW_TOKENS=1536` (was 512 → truncated the fix). Files: `scripts/rl_rollouts.py`
  (`:254` cap, `:262` instruction, `:269` `_augment_judge_prompt`, `:574` call, `:613` cap plumb, `:626`
  `_extract_corrected_php`). Objective preserved = critique-then-fix (GRPO-05 / REVL-06), NOT redefined.
- **Mechanism 2 (frozen reward-judge parse, fixed).** Served v4 emits prose scores, but `parse_judge_response`
  only handled `<judge_output>` JSON → `None` → group-mean imputation bias. **Fix:** prose-score fallback
  `_parse_prose_dim_scores` → `_derive_overall_from_dims`, wired into `judge_score_single` (reward boundary
  ONLY; `parse_judge_response` kept pure for teacher-GT extraction, matching the existing
  `_derive_overall_from_dims` placement). Files: `eval/eval_judge.py` (`:162` label map area, `:207` helper,
  `:453` wiring). Proof: PROSE → overall 88.9 (was None).
- **Mechanism 3 (gen MO-GRPO normalization, by-design, secondary).** Composite uses within-group-normalized
  signals → collapses to 0 when a strong policy makes groups uniform. Left as-is per council; only add a
  small raw/non-normalized blend if uniform groups become common (they did NOT — 25/25 live steps non-uniform).

### Validation evidence (HARD gate — all met)
- Offline probe ALL PASS: judge shape → fix 1.000 (```php) / 0.997 (`<corrected_code>`); prose → 88.9; gen non-uniform.
- **Live** (run `b6e9qpx9a`, launched 18:46 AFTER the 18:37 fix commit): `fix_correctness=1.0` in log;
  **25/25 steps `reward_min != reward_max`** (live metrics archived at
  `output/rl_checkpoints/metrics/rl_metrics.prefixfix-archive.jsonl`). Reward noisy ~0.2, not collapsing.
- max_tokens: SFT judge ~900 tok + fix < 1536 cap → no truncation.

---

## THE DECISION TO MAKE — judge-consistency scorer ($0 paths)

The consistency reward is the ONLY paid piece. `fix_correctness` (0.7 weight) is deterministic/local ($0);
gen rewards (PHPCS + security + VeRPO) are local ($0). Consistency is capped 0.3 weight (D-09-05).
Pick one before relaunch:

### Option 1 — Local-model consistency (RECOMMENDED for unattended multi-day run)
Replace the `claude_agent` dispatch in consistency scoring with a LOCAL vLLM call (reuse the running
`wp_judge` server, or a second small local model). $0, fast, parallel, no session dependency, no IPC.
- Touch: `scripts/rl_judge_dispatch.py` (`score_judge_consistency_batch` — currently dispatches `claude_agent`;
  repoint to a vLLM chat completion with a consistency prompt: "does this critique match this code? 0–1").
  Parse a 0–1 score. Consistency prompt = new, small.
- Effort: moderate. Best robustness/throughput.

#### Recommended local model (GB10 / DGX Spark): `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
Clear pick: it's the only model with **explicit DGX Spark (aarch64) recipe guidance**, ships a validated NVFP4
variant, same 3B-active MoE footprint as the training model, and has `--reasoning-parser nemotron_v3` (CoT
`<think>` blocks work out of the box). Serve it on a **separate port (8001)** so it coexists with the
`wp_judge` fix-scoring server on :8000.

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --port 8001 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \       # DGX Spark-specific per recipe
  --max-model-len 32768 \               # reduced from 256K per recipe OOM guidance
  --max-num-seqs 8 \                    # DGX Spark-specific per recipe
  --moe-backend flashinfer_cutlass \    # required for NVFP4 at TP>=1 on Blackwell
  --reasoning-parser nemotron_v3 \
  --trust-remote-code
```

Model comparison for the consistency-judge role:

| Model | NVFP4 | GB10 explicit support | Active params | Judge quality |
|---|---|---|---|---|
| **Nemotron-3-Nano-Omni NVFP4** | ✅ | ✅ documented | 3B | High (reasoning + CoT) |
| Llama-4-Scout NVFP4 | ✅ | ⚠️ B200 only in recipe | 17B | Good |
| Nemotron-3-Super NVFP4 | ✅ | ⚠️ 120B total, tight | 12B | Very high, memory-risky |
| Qwen3.6-35B-A3B NVFP4 | ✅ | ❌ not in recipe hw list | 3B | Good |
| gemma-4-31B dense | ❌ BF16 only | ❌ | 31B | High but too heavy |

**Caveat — disable thinking for the consistency call.** Want a pure JSON score, not a reasoning chain: pass
`enable_thinking: false` (chat_template_kwargs), `temperature=0.2`, `max_tokens=256`. Maps to the recipe's
"Instruct" sampling row; keeps latency <1s/call at batch 8. Wire this in the consistency prompt builder when
repointing `score_judge_consistency_batch` at `http://localhost:8001/v1`.

**Memory note:** check :8001 (0.70 util) + the :8000 `wp_judge` server + Tinker client fit GB10 together; if
tight, lower one server's `--gpu-memory-utilization`, or serve the consistency model only during RL rollouts.

### Option 2 — Drop Claude-consistency (SIMPLEST, $0)
Run RL on `fix_correctness` (0.7, deterministic) + gen rewards only. Set consistency weight 0 / skip the
dispatch (add `--no-consistency` or force `consistency=neutral` so `combine_judge_reward` = fix only).
- Touch: `scripts/rl_rollouts.py` (skip `score_judge_consistency_batch`; combine with fix only) +
  `scripts/rl_train.py` (flag). Effort: small.
- Trade-off: loses the score-reasoning-consistency signal (Panickssery). But `fix_correctness=1.0` is proven
  to give real, non-uniform gradient → RL is NOT gradient-dead without consistency.

### Option 3 — File-queue + `/loop` main-agent (works, but slow/brittle; what was asked about)
RL appends judging requests to `judge_queue.jsonl`, blocks polling `judge_results.jsonl`. A `/loop` on the
**main interactive Claude Code thread** (subscription — NOT a spawned sub-agent, which would bill API) drains
the queue each tick and writes results.
- Touch: new `scripts/file_queue_judge.py` (write request {id, php_code, critique_text}; poll result by id;
  timeout→0.5 neutral), modify `rl_rollouts` consistency dispatch to use it, plus a `/loop` prompt that reads
  past a saved offset, scores 0–1, appends {id, score}.
- $0 (subscription) BUT: serial + Max rate-limited (RL crawls), session must stay alive for the whole run,
  consumes the main agent entirely, IPC fragile. Reserve for special cases.

---

## Relaunch runbook (after the consistency decision is wired)

1. **Serve judge** (currently stopped):
   ```bash
   bash scripts/serve_v4_judge_vllm.sh      # container wp-v4-judge-vllm, :8000, served name wp_judge
   # ready when: curl -s http://localhost:8000/v1/models | grep -q wp_judge   (~8 min load)
   ```
2. **Warm-start path** (already regenerated, loadable):
   `tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state`
   (read from `output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json` → `state_path`).
3. **Launch** as a HARNESS-TRACKED BACKGROUND TASK (`run_in_background:true`, NOT nohup — survives + notifies).
   ⚠️ Do NOT `set -a; . ./.env` an `ANTHROPIC_API_KEY` into the env if any `claude -p` path remains
   (it bills). For Options 1/2 there is no `claude -p` call, so it's moot.
   ```bash
   set -a; . ./.env; set +a   # for TINKER_API_KEY
   INIT=$(python3 -c "import json;print(json.load(open('output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json'))['state_path'])")
   env WP_JUDGE_DEBUG_DUMP=output/rl_checkpoints/judge_failures.jsonl \
     .venv-tinker/bin/python scripts/rl_train.py \
     --init-from "$INIT" --model-id Qwen/Qwen3-30B-A3B --lora-rank 32 --lora-seed 42 \
     --total-steps 500 --batch-size 8 --checkpoint-every 50 --jaccard-every 20 \
     --kl-soft 0.1 --kl-hard 0.3 --efrac-soft 0.7 --efrac-hard 0.5 \
     --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
     > output/rl_checkpoints/full_run.log 2>&1
   # (+ whatever flag the chosen consistency option adds, e.g. --no-consistency or a local consistency URL)
   ```
4. **Startup gate:** log must show `WARM START … train_mlp=True attn=False unembed=False`. COLD START → kill.
5. **Step-0 gate:** first `rl_metrics.jsonl` rows show `reward_min != reward_max` and judge rollouts parse
   (few/no entries in `judge_failures.jsonl`, no instruct-refusal phrasing).
6. **Signal-check first:** strongly consider `--total-steps 100` before the full 500 (~2-day) run; confirm
   reward stays non-uniform and doesn't collapse. (NOTE: no resume — each launch restarts from warm-start.)

---

## Open watch-items / known issues (carry forward)

- **Reward was flat ~0.2** over the 25-step signal run (not yet rising). Too few steps to judge; the full run
  shows the real trend. Non-uniform every step = healthy gradient. If it stays flat over 100+ steps, revisit
  LR / GSPO settings or Mechanism-3 blend.
- **Checkpoints empty** in `checkpoint_manifest.json` (run only reached ~step 23; first checkpoint at step 50).
  Verify the step-50 checkpoint actually WRITES on the next run; if empty past step ~60, investigate a
  checkpoint-write bug.
- **`kl_sample_train_v1 = 0.0000`** every step — on-policy artifact, NOT a bug (see reconciliation doc).
- **`.env` contains `ANTHROPIC_API_KEY`** (`sk-ant-…CQAA`). Sourcing `.env` exports it; the old code leaked it
  into `claude -p` → API billing. Options 1/2 remove the `claude -p` path entirely. Consider removing the key
  from `.env` or not sourcing it for Anthropic (TINKER_API_KEY is separate).
- **Pre-existing test failure:** `tests/test_rl_train.py::TestRLTrainUnit::test_lora_config` (MagicMock not
  JSON-serializable in `rl_train.py:140 build_training_client`) — fails WITHOUT my edits too (confirmed via
  stash). Not caused by this work; unrelated to fix.
- **`tests/test_preflight.py`** import-errors (`dotenv` not in `.venv-tinker`) — pre-existing env gap.

---

## Key files / pointers

- Reward probe: `scripts/_probe_rl_reward.py`
- Fixes: `eval/eval_judge.py` (Mechanism 2), `scripts/rl_rollouts.py` (Mechanism 1)
- Consistency dispatch: `scripts/rl_judge_dispatch.py` (`score_judge_consistency_batch`), `scripts/claude_agent.py`
- Diagnosis writeup: `.planning/debug/09-rl-warmstart-zero-reward.md`
- Init/warm-start decision (signed off, D-09-08 MoE-only): `.planning/phases/09-gspo-training/09-RL-INIT-RECONCILIATION.md`
- Judge serve: `scripts/serve_v4_judge_vllm.sh` (model `models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4`)
- v4 savestate manifest: `output/tinker/wp-reasoning-v4-r32-rp30-savestate-manifest.json`
- Live signal-run metrics (25 steps, post-fix): `output/rl_checkpoints/metrics/rl_metrics.prefixfix-archive.jsonl`
- Captured rollout shapes for the probe: `output/rl_checkpoints/judge_failures.preguard.prefixfix-diag.jsonl`
- Billing policy (global): `~/.claude/CLAUDE.md` "Billing — UNIVERSAL policy"

## Recommended next step
Wire **Option 1 (local-model consistency)** or **Option 2 (drop consistency)** → run a 100-step signal-check
→ then the full 500. Option 2 is the fastest route to a fully-$0 valid run; Option 1 keeps the consistency
signal at $0.

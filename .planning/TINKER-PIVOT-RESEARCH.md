# Pivot research — Thinking Machines **Tinker** (2026-06-07)

## Why this pivot exists
The GB10 (124.6 GiB unified) **cannot** load Qwen3-30B-A3B bf16 in-process: the load+adapter
transient is ~122 GiB (>total RAM), proven across Unsloth/transformers/4-bit/streaming/reshard
(see `output/format_stability/discriminator/MEMORY-INVESTIGATION-bf16.md`). The merge-vs-training
discriminator (04.3-03), the multi-user.target detached job, and the true-4bit-checkpoint build
were all workarounds for a venue that fundamentally can't host the model. **Tinker runs the model
in the cloud**, so the entire local-memory problem class disappears.

## What Tinker is
Managed **LoRA fine-tuning + sampling** API from Thinking Machines Lab (GA Dec 2025). You write the
training loop with low-level primitives; they handle distributed GPUs. OpenAI-compatible inference
from checkpoints via `tinker://` URIs. Key facts:
- **Install:** `uv pip install tinker-cookbook` (pulls the `tinker` SDK). Auth: env `TINKER_API_KEY` (✓ saved to `.env`, gitignored).
- **Primitives:** `ServiceClient` → `create_lora_training_client(base_model, rank)` →
  `forward_backward` / `optim_step` / `save_state` / `load_state` →
  `save_weights_and_get_sampling_client()` → `sampling_client.sample(...)`.
- **Weight export:** `rest_client.get_checkpoint_archive_url_from_tinker_path(model_path)` → download `.tar.gz`
  (tutorials `501_export_hf.py`, `502_lora_adapter.py`) → can serve locally/vLLM later if needed.
- **Eval:** built-in benchmark framework + inline evaluators; or sample via OpenAI-compatible client and run our own terse/rubric eval.
- **Cookbook:** `recipes/chat_sl` = supervised fine-tune on conversational data (our exact case);
  `supervised/` (`conversation_to_datum(messages, renderer, max_length, train_on_what)`); `renderers/qwen3.py`.

## DECISIVE: our base model is supported
`tinker_cookbook/model_info.py` lists **`Qwen3-30B-A3B`** (the thinking MoE — Phase 4.3's base),
plus `Qwen3-30B-A3B-Base` and `Qwen3-30B-A3B-Instruct-2507`. Exact architecture match.

## How our artifacts map
- **Dataset:** `data/reasoning_dataset/openai_train.jsonl` (341) + `openai_val.jsonl` (77) are already
  OpenAI `messages` chat format → a small `ChatDatasetBuilder` maps each `row["messages"]` via
  `conversation_to_datum` (same shape as the cookbook `Tulu3Builder`/`NoRobotsBuilder`).
- **Base:** `Qwen3-30B-A3B` (match Phase 4.3) — LoRA, rank per `hyperparam_utils` (Phase 4.3 used rank-heavy Unsloth; Tinker LR scales differently — use `hyperparam_utils` LR scaling, not the raw 2e-5).
- **Renderer:** `qwen3` (thinking) vs `qwen3_disable_thinking` — **this is the format-stability lever**
  that REVL-05 failed on. Re-training cleanly here is the chance to fix the 35% terse-JSON collapse.

## What this OBVIATES
- 04.3-03 merge-vs-training **discriminator** → MOOT. No local merge; sample the LoRA directly on Tinker.
- The GB10 memory wall, the multi-user.target job, the 4-bit checkpoint build → all unnecessary.
- 04.4 merge forensics + REVL-05 rejection → re-addressed by a clean cloud re-train + eval.
- Local `merged-v2` / `ckpt-72` become reference/fallback only (still READ-ONLY, not promoted).

## OPEN DECISIONS / RISKS (resolve before training)
1. **Special tokens.** Did Phase 4.3 add `<wp_gen>` / `<wp_judge>` (and rely on `[/REASONING]`/
   `<judge_output>` markers) to the tokenizer? Tinker uses the **stock Qwen3 tokenizer** — custom
   *added* tokens won't exist. If training depended on added special tokens, inline them as plain
   text or adjust the renderer. **Must verify in the train config + dataset before launch.**
2. **Base vs Instruct-2507 vs Base(-pretrain).** Phase 4.3 fine-tuned the thinking `Qwen3-30B-A3B`. Match it unless we deliberately switch.
3. **Thinking format.** Decide `qwen3` (thinking) vs `qwen3_disable_thinking` — drives whether the
   `[/REASONING]` prose CoT is in-band; tie this to the format-stability fix.
4. **Cost.** GA = usage-based pricing (early beta was free). Unknown $ for a 30B-A3B LoRA run — confirm before large sweeps.
5. **Account access.** Confirm our key has `Qwen3-30B-A3B` access (some large MoEs may be gated).

## Proposed pivot plan
- **P0 Connectivity:** `uv pip install tinker-cookbook`; `ServiceClient()`; list models; confirm `Qwen3-30B-A3B` available; tiny `forward_backward` smoke on a 1B to validate auth/loop.
- **P1 Data adapter:** `ChatDatasetBuilder` over `data/reasoning_dataset/openai_*.jsonl`; resolve special-token question; pick renderer.
- **P2 SFT:** LoRA on `Qwen3-30B-A3B`, `hyperparam_utils` LR scaling, checkpoints; small run first.
- **P3 Eval:** sample held-out val; measure **terse rate** (the 04.3-02 metric) + rubric; iterate on format stability — fast cloud loop, no memory wall.
- **P4 Decide:** if format-stable → this is the v1.2 reasoning model; export weights if downstream
  phases (RL / MoE-Sieve / packaging) need a local artifact.

## Sources
- thinkingmachines.ai/tinker, tinker-docs.thinkingmachines.ai (quickstart, model-lineup, rendering)
- github.com/thinking-machines-lab/tinker-cookbook (README, model_info.py, supervised/, recipes/chat_sl, renderers/, tutorials/)

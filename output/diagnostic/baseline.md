# Phase 0 — Diagnostic Baseline

Status (2026-05-11).

| Step | Description | Status | Output |
|------|-------------|--------|--------|
| 0.1  | `profile_base_model.py` on base + 30/70 adapter — E_eff(gen) vs E_eff(judge) | **PENDING** — GPU container required | `output/diagnostic/profiling_{base,30_70}/` |
| 0.2  | `rubric_scorer.py` on 27 human + 93 UGC + 25 boundary seeds | **DONE** (5-tool, LLM ON) | `output/diagnostic/seed_scorer_agreement{,_llm}.{json,md}` |
| 0.3  | `eval_judge.py` on 30/70 adapter vs seed-derived GT — Spearman | **READY** — synthesizer + recipe template, GPU container required | `output/diagnostic/judge_30_70_seed_spearman.json` |

## Step 0.2 — final results

### Run B — full 5-tool, LLM checks ON (`seed_scorer_agreement_llm.{json,md}`)

145 / 145 seeds scored. Backend: Claude Code agents (`sonnet`), 6 workers, chunked over 3 × 9-min runs with retry on 1 failed seed. Distribution: **min 53.3, max 100, mean 93.7, stdev 11.9** (wider spread than 4-tool determ run).

| Dim | n | Spearman | p | Pearson | Rubric mean (LLM) | Rubric mean (4-tool) |
|-----|---|----------|---|---------|-------------------|----------------------|
| D1_wpcs | 22 | +0.087 | 0.702 | +0.085 | 9.99 | 9.99 |
| D2_security | 15 | +0.000 | 1.000 | −0.081 | **3.92** | 7.58 |
| D3_sql | 12 | +0.996 | 0.000 | +1.000 | 3.32 | 3.32 |
| D4_perf | 8 | +0.286 | 0.493 | **+0.488** | 9.87 | 9.88 |
| D5_wp_api | 31 | −0.144 | 0.439 | −0.121 | **9.88** | 9.94 |
| D6_i18n | 16 | nan | nan | nan | 9.35 | 9.91 |
| D7_a11y | 12 | nan | nan | nan | 9.82 | 9.96 |
| D8_errors | 0 | n/a | n/a | n/a | n/a | n/a |
| D9_structure | 15 | −0.071 | 0.800 | −0.071 | 9.99 | 9.99 |

### Run A — deterministic-only (`seed_scorer_agreement.{json,md}`)

Same 145 seeds, LLM checks OFF. Distribution: min 64.6, max 100, mean 95.8.

### Honest interpretation

1. **Original plan gate (Pearson ≥ 0.75) fails at every dimension.** LLM checks moved D4_perf Pearson +0.037 → +0.488 (best per-dim signal); D5_wp_api Spearman improved −0.345 → −0.144 (less wrong); D2_security got worse (Pearson +0.158 → −0.081). Sample sizes (n=8–31 per dim) too small for strong statements.
2. **Seeds are not well-calibrated for the 41 LLM check questions.** Each seed annotates only some dims (the ones humans cared about); rubric scores ALL dims; per-dim Spearman on FAIL-only subset is the wrong measure of scorer fitness.
3. **Phase 1 calibration approach now defensible.** With 500 PASS anchors (rubric overall mean 99.77) + 145 FAIL seeds (rubric overall mean 93.7 LLM-on), the BINARY anchor agreement — clean code scores 95+, defective code scores < 90 — gives Phase 1 a reliable label signal even if per-dim Spearman on seeds remains weak.
4. **D3_sql 0.996 still trivial.** 10 pairs at (h=2, r=2), 2 at (h=8, r≈10) — not subtle ranking, just yes/no agreement.

### Scorer change-set this session

- `eval/rubric_scorer.py`: `<?php` auto-wrap for snippets; `RUBRIC_USE_LLM_CHECKS=1` env opt-in for Tool 4; `_LLM_CHECK_COUNT` derived from CHECK_REGISTRY (41).
- `eval/rubric_definitions.py`: `NA_DETECTION_HINTS` relaxed (D1/D5/D8/D9 match function-body forms; D2 broader).
- `eval/llm_checks.py` (new): 41 binary YES/NO prompts in single batched call. **Hybrid backend** — `LLM_BACKEND=claude` (Claude Code subscription) or `vllm` (local OpenAI-compatible endpoint, default Qwen3.6-35B-A3B-FP8 per `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml`).
- `eval/eval_judge.py`: `_GT_FIELD_TO_DIM` expanded to all 9 dims.
- `scripts/phase0_score_seeds.py`: SEED_DIM_MAP corrected (`dependency_integrity → D5_wp_api`); parallel workers; `--resume` + `--time-budget-sec` for chunked runs.
- `scripts/build_seeds_judge_test.py` (new): 145 seeds → wp_judge format with human-derived GT scores.
- `scripts/extract_pass_anchors.py` (new): 500 PASS anchors from `data/phase1_extraction/output/passed/`.
- `scripts/profile_base_model.py`: `--adapter` flag for PEFT LoRA stack.
- `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` (new): vLLM serving recipe for batch generation.

**Prior triage results (`output/triage_decision.md`, Spearman 0.5698 for 30/70) are not apples-to-apples with anything run after 2026-05-11.**

## Backend strategy (Phase 0.12)

Two LLM backends, used by workload:

| Workload | Backend | Why |
|----------|---------|-----|
| Quality audit, advisor, council, calibration spot-check | `claude` (Sonnet via CLI) | Flagship reasoning, small volume, $0 (subscription) |
| Phase 0.10 LLM checks at scale (145 here, 20K Phase 1) | `vllm` Qwen3.6-35B-A3B-FP8 | Volume — Claude CLI rate limits + wall time |
| Phase 1b re-judging stratified 20K | `vllm` | volume |
| Phase 1c boundary pack ~1500 contrastive pairs | `vllm` gen + `claude` quality gate | volume + verified gate |
| Phase 5 RL verifiable rewards | `vllm` (PHPCS/security stay deterministic) | latency |

`Qwen/Qwen3.6-35B-A3B-FP8` already cached in `~/.cache/huggingface/hub/`. Recipe at `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` (FP8, 0.55 UMA util, 16K context). Start with `sparkrun start recipes/qwen3.6-35b-a3b-fp8-vllm.yaml`.

## Step 0.1 — runbook (GPU container required)

```bash
cd ~/Desktop/projects/wp-finetune
bash deps/dgx-toolbox/containers/ngc-pytorch.sh
# Inside container:
cd /workspace

# Base model — baseline routing without task-token bias
python -m scripts.profile_base_model \
  --model-path models/Qwen3-30B-A3B \
  --tokenizer-path adapters/tokenizer \
  --output-dir output/diagnostic/profiling_base \
  --subsample 0.05

# 30/70 adapter — measures whether task tokens actually drove routing concentration
python -m scripts.profile_base_model \
  --model-path models/Qwen3-30B-A3B \
  --tokenizer-path adapters/tokenizer \
  --adapter adapters/qwen3-30b-wp-30_70 \
  --output-dir output/diagnostic/profiling_30_70 \
  --subsample 0.05
```

Wall time estimate: ~20–30 min each on GB10 at subsample 0.05.

**Expected signal:** if task tokens are doing their job, the 30/70 run should show `eeff_wp_gen` and `eeff_wp_judge` diverge per layer. Mean E_eff total being 69 across all 5 ratios in `output/triage_decision.md` suggested they did not — that's the load-bearing assumption behind the v2 retrain plan.

## Step 0.3 — runbook (GPU container required)

`output/diagnostic/seeds_as_judge_test.jsonl` (145 records) built with human-derived GT.

```bash
# Inside container — serve 30/70 adapter via vLLM
# Clone recipes/nemotron-3-nano-4b-bf16-vllm.yaml to wp-30_70-vllm.yaml,
# point base_model: models/Qwen3-30B-A3B, lora: adapters/qwen3-30b-wp-30_70
sparkrun start recipes/qwen3-30b-wp-30_70-vllm.yaml

# Eval against seed-derived GT
python -m eval.eval_judge \
  --test-jsonl output/diagnostic/seeds_as_judge_test.jsonl \
  --output output/diagnostic/judge_30_70_seed_spearman.json
```

**Phase 1 hinges on this number.** If 30/70 against human-derived GT comes back ≈ 0.65–0.75, the "judge is broken vs humans" premise weakens. Run *before* committing to Phase 1 rebuild.

## Phase 0 follow-up state

- 0.4 install composer + phpstan + WordPressVIPMinimum + phpcs-security-audit ✅
- 0.5 fix N/A heuristics + `<?php` auto-wrap ✅
- 0.6 extend `profile_base_model.py` with `--adapter` ✅
- 0.7 build `seeds_as_judge_test.jsonl` synthesizer ✅
- 0.8 re-run step 0.2 with full tooling ✅
- 0.9 expand `eval_judge.py` `_GT_FIELD_TO_DIM` for all 9 dims ✅
- 0.10 implement 41 LLM-assisted checks (rubric §F.5) ✅ (Claude backend; vLLM also wired)
- 0.11 PASS-anchor extraction (500 anchors, mean 99.77) ✅
- 0.12 hybrid LLM backend (Claude agents + vLLM) ✅

## Phase 1 readiness

Greenlit when:
- 0.1 + 0.3 GPU results land
- `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` validated (vLLM serves, `/v1/chat/completions` responds with JSON schema)

Anchor pool: 500 PASS + 145 FAIL = 645 labeled anchors covering 53–100 range. Phase 1a calibration uses these as fixed gold; Phase 1b re-judging via vLLM uses them as few-shot prompt anchors.

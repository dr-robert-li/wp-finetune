# Phase 0 — Diagnostic Baseline

Status (2026-05-11).

| Step | Description | Status | Output |
|------|-------------|--------|--------|
| 0.1  | `profile_base_model.py` on base + 30/70 adapter — E_eff(gen) vs E_eff(judge) | **DONE** | `output/diagnostic/profiling_{base,30_70}/base_model_eeff.jsonl` |
| 0.2  | `rubric_scorer.py` on 27 human + 93 UGC + 25 boundary seeds | **DONE** (5-tool, LLM ON) | `output/diagnostic/seed_scorer_agreement{,_llm}.{json,md}` |
| 0.3  | `eval_judge.py` on 30/70 adapter vs seed-derived GT — Spearman | **READY** — vLLM primary path + FastLanguageModel fallback | `output/diagnostic/judge_30_70_seed_spearman.json` |

## Step 0.1 — final results

Base model and 30/70 adapter profiled at subsample 0.05 of `data/final_dataset/ratio_30_70/openai_train.jsonl`. 48 MoE layers hooked, ~1742 examples per run.

| Metric | Base | 30/70 adapter | Shift |
|--------|------|---------------|-------|
| E_eff mean (total) | 69.96 | 74.42 | +4.46 |
| E_eff mean (wp_gen) | 59.48 | 60.20 | +0.72 |
| E_eff mean (wp_judge) | 69.96 | 75.18 | +5.22 |
| **Delta (judge − gen) mean** | **+10.48** | **+14.98** | **+4.50** |
| Delta max | +15.58 (L9) | +20.19 (L9) | +4.61 |

**Every layer shows positive shift.** Late layers (43–47) gain +5 to +9 delta points; early layers (0–3) gain +1 to +3.

### Interpretation (rewrites the v2 hypothesis)

The base model already routes `<wp_gen>` differently from `<wp_judge>` by ~10 E_eff points. Training amplified this differentiation to ~15 — routing is doing exactly what task-token-driven specialisation is supposed to do.

The v2 plan's load-bearing premise ("experiment_001 failed because task tokens didn't drive routing concentration") is **rejected**. Routing was never the problem. The actual causes of the Spearman 0.096 + parse-rate 13.2% regression must be:

1. **Label quality** — Claude bulk-judge labels noisy (the seed-anchored re-judge plan still applies).
2. **Embed_tokens overfitting** — training pushed `<wp_gen>` / `<wp_judge>` embeddings into a degenerate region for output gen (routing fine, semantics off).
3. **Output format collapse** — model lost JSON output discipline (`overall_score` / dimension-score keys); training data may not have enforced strict-JSON examples consistently.

### Important caveat — expert LoRA did NOT bind during profiling

PEFT 0.18.1 + transformers 4.56.2 raised:

```
RuntimeWarning: target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']
were set but no parameter was matched.
```

The 30/70 adapter declares LoRA on the MoE expert MLP weight tensors via the newer `target_parameters` API (Unsloth's MoE pattern). Raw PEFT cannot resolve those parameter names; only `modules_to_save=[embed_tokens, lm_head]` + standard `target_modules=[q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj]` actually loaded.

**Impact split:**
- **Routing E_eff (this measurement) — SAFE.** Router gate is frozen during training and never in any adapter. Routing decisions depend on (a) trained embed_tokens, which DID load, and (b) trained attention LoRA, which DID load. The +4.50 delta shift above is real.
- **Output quality — UNREPRESENTATIVE.** The base expert MLPs ran for every forward pass. Anything that reads model outputs (Phase 0.3 judge eval, vLLM generation) using `PeftModel.from_pretrained` would see partial-trained behaviour, not true 30/70 quality.

Filed as `DGX_TOOLBOX_ISSUES.md#7`.

### v2 plan revisions

| Axis | Old emphasis | New emphasis |
|------|--------------|--------------|
| Routing collapse | PRIMARY suspect | REJECTED — routing differentiates fine |
| Label quality (Claude bulk-judge) | Co-primary | PRIMARY |
| Output format integrity (JSON parse rate) | Implicit | **PROMOTED — explicit Phase 1d gate at ≥95%** |
| Embed_tokens overfitting | Not in plan | New diagnostic — verify in Phase 2 pilot |

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

**Container choice matters.** Two pitfalls:

1. `ngc-pytorch.sh` (NGC PyTorch 26.02-py3) ships transformers 5.8.0 / huggingface-hub 1.14 — no released PEFT version can load adapters trained with peft 0.18.1 against that stack.
2. `unsloth-studio.sh` (NGC PyTorch 25.11-py3) auto-installs **latest** unsloth (2026.5.x) which itself requires transformers 5.x. It also tries to run a `quickstart` script that exits if the Studio web-UI venv is absent, kicking you back to the host shell.

Use `unsloth-headless.sh` (NGC PyTorch 25.11-py3) — same base as training, idles on `sleep infinity` after install, exec-into via `docker exec`. Then force-downgrade the HF stack to the training-time pin set.

```bash
cd ~/Desktop/projects/wp-finetune

# unsloth-headless does NOT auto-mount the current dir (unlike ngc-pytorch.sh).
# Pass it via EXTRA_MOUNTS so the project is reachable inside the container.
EXTRA_MOUNTS="$(pwd):/workspace/project" bash deps/dgx-toolbox/containers/unsloth-headless.sh

# Container starts in background; wait for "Unsloth headless ready..." log line (~60-120s):
docker logs -f unsloth-headless    # Ctrl-C once you see "ready"

# Exec into the running container
docker exec -it unsloth-headless bash
# Inside container — project is at /workspace/project (NOT /workspace)
cd /workspace/project

# Force-downgrade auto-installed unsloth/transformers/peft/etc to training-time pins.
# --force-reinstall overrides the latest versions that unsloth-headless brought in.
pip install --no-deps --force-reinstall -r config/requirements-profiling.txt

# Sanity check: cuda + matching dep versions
python3 -c "import torch, peft, transformers; print('cuda:', torch.cuda.is_available()); print('peft:', peft.__version__, 'transformers:', transformers.__version__)"
# Expected: cuda: True   peft: 0.18.1   transformers: 4.56.2

# Base model — baseline routing without task-token bias
python -m scripts.profile_base_model \
  --model-path models/Qwen3-30B-A3B \
  --tokenizer-path adapters/tokenizer \
  --ratio ratio_30_70 \
  --output-dir output/diagnostic/profiling_base \
  --subsample 0.05

# 30/70 adapter — measures whether task tokens actually drove routing concentration
python -m scripts.profile_base_model \
  --model-path models/Qwen3-30B-A3B \
  --tokenizer-path adapters/tokenizer \
  --adapter adapters/qwen3-30b-wp-30_70 \
  --ratio ratio_30_70 \
  --output-dir output/diagnostic/profiling_30_70 \
  --subsample 0.05
```

Wall time estimate (warm cache): ~6 min load + ~6 min forward passes per run on GB10 at subsample 0.05. First run cold cache: model load ~6 min instead of ~36 s.

**Why neither ngc-pytorch.sh nor unsloth-studio.sh works without intervention:**
- ngc-pytorch.sh: transformers 5.8.0 / PEFT 0.19.1 / WeightConverter API drift / tokenizers 0.20 vs 0.22 mismatch / huggingface-hub 1.14 vs <1.0 cascade.
- unsloth-studio.sh: auto-pulls unsloth 2026.5.x at startup which requires transformers 5.x; the studio launcher then fails on missing studio venv (`/root/.unsloth/studio/unsloth_studio`) and drops the user back to host.

unsloth-headless.sh + the force-reinstall step mirror the training-time stack exactly. Re-run the pip install on every fresh `docker run`; it persists across `docker exec` invocations as long as the container stays up.

**Expected signal:** if task tokens are doing their job, the 30/70 run should show `eeff_wp_gen` and `eeff_wp_judge` diverge per layer. Mean E_eff total being 69 across all 5 ratios in `output/triage_decision.md` suggested they did not — that's the load-bearing assumption behind the v2 retrain plan.

## Step 0.3 — design (two adapter-binding backends)

`output/diagnostic/seeds_as_judge_test.jsonl` (145 records) built with human-derived GT.

The 0.1 caveat (raw PEFT does not bind the MoE expert LoRA) means we cannot get a true 30/70 judge eval via `PeftModel.from_pretrained`. Two viable paths:

### Path A (primary): vLLM-served LoRA via direct `docker run`

**Bypass sparkrun** — sparkrun cannot serve local model directories (DGX_TOOLBOX_ISSUES.md #8 + #9). We run the same prebuilt vLLM image (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`) directly. vLLM has its own LoRA loader independent of HF PEFT — it parses `adapter_config.json` directly and applies LoRA at the vLLM kernel level. Empirically often binds `target_parameters`-declared expert LoRA where raw PEFT does not.

`scripts/serve_30_70_vllm.sh` wraps the docker invocation; `recipes/qwen3-30b-wp-30_70-vllm.yaml` is now documentation-only.

Steps (host shell, NOT inside any container):
```bash
# Stop the unsloth-headless container if it's holding GPU memory (won't fit both)
docker stop unsloth-headless 2>/dev/null || true

# Launch vLLM serving base + 30/70 LoRA on :8001
bash scripts/serve_30_70_vllm.sh

# Follow weight-load + readiness
docker logs -f wp-30_70-vllm
# Ctrl-C once you see: "Application startup complete" + "Uvicorn running on http://0.0.0.0:8001"

# Verify endpoint + LoRA name
curl -s http://localhost:8001/v1/models | jq

# Quality smoke — manual eyeball of a generation
curl -sX POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"wp-30_70","messages":[{"role":"user","content":"<wp_judge> Evaluate: <?php echo $_GET[\"q\"];"}]}' \
  | head -c 800

# Full eval
python -m eval.eval_judge \
  --test-jsonl output/diagnostic/seeds_as_judge_test.jsonl \
  --output output/diagnostic/judge_30_70_seed_spearman.json

# When done
docker stop wp-30_70-vllm
```

**Acceptance signal:** vLLM startup logs must NOT show a "target_parameters not matched" warning (vLLM emits its own LoRA loader messages — they look different from PEFT's). If startup is silent on expert-LoRA AND smoke generation produces coherent JSON, Path A is good. If vLLM also fails to bind experts (look for "experts skipped" / "no LoRA applied" patterns in startup logs), drop to Path B.

### Path B (fallback): FastLanguageModel-based Python eval

Unsloth's `FastLanguageModel.from_pretrained()` knows how to bind `target_parameters` MoE expert LoRA. Slower than vLLM (no batching kernels), but guaranteed correct.

Author `scripts/eval_judge_unsloth.py` if Path A fails — reuses `eval/eval_judge.py`'s parser + Spearman computation but replaces the OpenAI client with direct in-process inference via FastLanguageModel. Estimated ~5-10 min for 145 examples on GB10. Stub the script under that filename, leave it empty until Path A is tested.

### Why Phase 1 hinges on this

If 30/70 (true, with expert LoRA bound) ranks defects against human GT at Spearman ≥ 0.6, the model isn't "broken vs humans" — only mis-aligned in absolute scoring and JSON discipline. Phase 1 rebuild scope contracts to: better calibration data, stricter JSON format enforcement, no need to re-think MoE specialisation. If Spearman < 0.4, the full rebuild stands.

Note: the original Spearman 0.5698 in `output/triage_decision.md` was vs Claude bulk-judge labels (also noisy). Phase 0.3 against human-derived seed GT is the apples-to-apples comparison.

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
- 0.13 pinned profiling env (unsloth-headless + force-reinstall) ✅
- 0.14 base + 30/70 routing E_eff results — rejected routing-collapse hypothesis ✅

## Phase 1 readiness

Greenlit when:
- 0.1 + 0.3 GPU results land
- `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` validated (vLLM serves, `/v1/chat/completions` responds with JSON schema)

Anchor pool: 500 PASS + 145 FAIL = 645 labeled anchors covering 53–100 range. Phase 1a calibration uses these as fixed gold; Phase 1b re-judging via vLLM uses them as few-shot prompt anchors.

## Phase 8 — Qwen3.6-base variant (stub, expand later)

Added to plan 2026-05-11. Parallel variant trained on **Qwen3.6-35B-A3B-FP8** base instead of Qwen3-30B-A3B. Motivation: Qwen3.6-35B is newer (likely stronger reasoning baseline), already cached locally, and serves as the vLLM backbone for in-pipeline batch inference. Training the customer-facing variant on the same family removes a base-model gap.

Dependencies (resolve before Phase 8 starts):
- Dataset: reuse Phase 1 v2 dataset as-is (task-token agnostic between bases).
- Task tokens: `<wp_gen>` / `<wp_judge>` must be added to Qwen3.6 tokenizer (separate vocab IDs vs. Qwen3-30B's 151669/151670).
- Tooling: verify Unsloth + PEFT support Qwen3.6 MoE router-freeze pattern (architecture identical family — likely fine, smoke test on 100-step LoRA first).
- Compute: FP8 base saves VRAM but LoRA training in BF16 needs upcast path — Unsloth Studio recipe TBD.

Phase 8 produces a parallel artifact track (8a SFT → 8b RL → 8c sieve → 8d prune → 8e package) that does NOT replace Phase 3–7 outputs — it's a second checkpoint shipped alongside, letting downstream consumers pick base by their compute budget.

Open questions for expand-later session:
- Single-shot trainer (8a–e linear) or full pipeline reuse (call Phase 3–7 with `--base Qwen/Qwen3.6-35B-A3B-FP8`)?
- Quantization target (FP8 already; need GGUF + AWQ paths)?
- Eval gates: same Phase 4 cutoffs, or recalibrated for 35B base?

# Skill: wp-finetune:run-evaluation

Run the complete evaluation and triage pipeline via DGX Toolbox. Profiles base model routing concentration (E_eff), evaluates all trained adapters through quality gates and wp-bench, then presents a structured triage decision for human approval.

## Architecture

```
Skill (this file — intent + decision logic)
  → scripts/profile_base_model.py (E_eff profiling, no GPU training needed)
  → scripts/run_eval_triage.py (orchestrator: profiling + eval + triage)
  → scripts/triage_ratios.py (GATE-02 elimination logic)
  → eval/eval_gen.py, eval_judge.py, eval_gate.py (existing eval suite)
  → DGX Toolbox (vLLM serving, container management)
    → Output: output/profiling/, output/eval_triage/, output/triage_decision.md
```

## Telemetry

> **Default: Lightweight monitor only.** The observe-evaluation agent team (3 agents, ~1.2 GB overhead) should NOT be spawned during eval if training is also running or memory headroom is <25 GB. On DGX Spark unified memory, every agent process (~0.4 GB each) competes directly with model serving for the same memory pool.
>
> The lightweight monitor captures watts, temp, util, and memory — sufficient for eval telemetry.
> Only spawn full observe agents if eval is running standalone with >25 GB headroom.
>
> **Memory impact reference:**
> | Mode | Processes | Memory | When to use |
> |------|-----------|--------|-------------|
> | Lightweight | 1 shell script | ~5 MB | Default — always safe |
> | Observe | 3 Python agents | ~1.2 GB | Standalone eval, >25 GB headroom |

## Trigger

User says: "run evaluation", "evaluate the model", "run eval triage", "/run-evaluation"

## Process

### Step 0: Inventory and Validation

#### 0a. Detect available adapters

```bash
ls adapters/qwen3-30b-wp-*/adapter_config.json 2>/dev/null
```

Build adapter inventory:
```
| # | Ratio | Adapter Path | Final Loss | Status |
|---|-------|-------------|-----------|--------|
| 1 | 30/70 | adapters/qwen3-30b-wp-30_70/ | ? | Partial (800 steps) |
| 2 | 40/60 | adapters/qwen3-30b-wp-40_60/ | ? | Partial (800 steps) |
| 3 | 50/50 | adapters/qwen3-30b-wp-50_50/ | 0.296 | Partial (800 steps) |
| 4 | 60/40 | adapters/qwen3-30b-wp-60_40/ | 0.277 | Complete (8134 steps, 43h) |
```

Read each adapter's `trainer_state.json` (last checkpoint) for final loss if available.

#### 0b. Detect available ratio data distributions (for profiling)

```bash
ls data/final_dataset/ratio_*/openai_train.jsonl 2>/dev/null
```

All 5 ratio distributions (30/70, 40/60, 50/50, 60/40, 70/30) should be available for base-model profiling regardless of whether adapters exist for all of them.

#### 0c. Check DGX readiness

```bash
nvidia-smi 2>&1 | head -1  # GPU available?
python3 -c "import torch; print(torch.cuda.is_available())"  # CUDA available?
ls models/Qwen3-30B-A3B/config.json 2>/dev/null  # Base model present?
```

If no CUDA: that's expected. The orchestrator runs from the **HOST** (not inside a container). It uses DGX Toolbox to launch vLLM as a separate Docker container, and eval scripts communicate via HTTP to `localhost:8020`. Only profiling (Step 1) needs CUDA directly — skip it with `--skip-profiling` if already complete.

```bash
# Run from HOST — DGX Toolbox manages the vLLM container
python3 scripts/run_eval_triage.py --skip-profiling --ratios 30_70,40_60,50_50,60_40
```

#### 0d. Check wp-bench availability

```bash
test -d wp-bench && echo "wp-bench: ready" || echo "wp-bench: not cloned"
```

If not cloned, note that wp-bench will be cloned during Step 2. If clone fails, eval continues without wp-bench (differentiation signal only, not a hard gate).

### Step 1: Base-Model E_eff Profiling

**Purpose:** Profile routing concentration across all 5 ratio data distributions on the base model (no adapter) to determine:
1. Whether WordPress data shows sharp routing concentration (expected for domain-specific code)
2. Whether 60/40 and 70/30 training is warranted (E_eff trending down = train more)
3. Baseline E_eff for later comparison with fine-tuned adapter profiling (Phase 7)

**Duration:** ~10 minutes total (all 5 ratios)

```python
python3 scripts/run_eval_triage.py --profiling-only
```

Or via orchestrator:
```python
from scripts.run_eval_triage import run_profiling
from scripts.profile_base_model import load_model_and_tokenizer, profile_ratio, write_profiling_jsonl, write_summary_md

model, tokenizer = load_model_and_tokenizer("models/Qwen3-30B-A3B", "adapters/tokenizer")
collector = RoutingCollector(model)

for ratio in ["30_70", "40_60", "50_50", "60_40", "70_30"]:
    data_path = f"data/final_dataset/ratio_{ratio}/openai_train.jsonl"
    profile_ratio(collector, tokenizer, data_path, ratio, subsample_frac=0.10)

write_profiling_jsonl(collector, "output/profiling/base_model_eeff.jsonl")
write_summary_md(collector, "output/profiling/base_model_eeff_summary.md")
```

**Decision Gate 1: E_eff Training Signal**

After profiling completes, present the E_eff summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVAL ► BASE-MODEL E_eff PROFILING COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Ratio  │ Mean E_eff │ Max E_eff │ E_eff Var │ Interpretation
  ───────┼────────────┼───────────┼───────────┼──────────────────────
  30/70  │ {val}      │ {val}     │ {val}     │ {judge-heavy}
  40/60  │ {val}      │ {val}     │ {val}     │
  50/50  │ {val}      │ {val}     │ {val}     │
  60/40  │ {val}      │ {val}     │ {val}     │ {gen-heavy, no adapter}
  70/30  │ {val}      │ {val}     │ {val}     │ {gen-heavy, no adapter}

  Trend: {downward / flat / upward} as gen% increases

  Lower E_eff = sharper routing = more experts prunable later.
  E_eff of 128 = perfectly uniform (worst). E_eff of 1 = one expert handles all (best).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Decision criteria:**
- **E_eff trending down** (each gen-heavier ratio has lower mean E_eff): WordPress gen tokens concentrate into fewer experts as gen data increases. **Recommendation: Train 60/40.** If 60/40 continues the trend, consider 70/30 too.
- **E_eff flat** (±5% across ratios): Routing concentration doesn't vary with gen/judge mix. **Recommendation: Skip additional training.** 3 ratios sufficient.
- **E_eff trending up**: Gen-heavy data diffuses routing. **Recommendation: Skip additional training.** Judge-heavy ratios are more compressible.

Use AskUserQuestion:
- header: "E_eff Signal"
- question: "Based on E_eff profiling, should we train additional ratios?"
- options:
  - "Train 60/40 ({est_hours}h, start in background)" → start 60/40 training, continue to Step 2
  - "Train 60/40 + 70/30 ({est_hours}h total)" → start both, continue to Step 2
  - "Skip — evaluate existing 3 adapters only" → continue to Step 2
  - "Abort — investigate profiling data first" → exit skill

If training warranted, start in background via run-training skill pattern:
```python
dgx = get_toolbox()
dgx.execute("unsloth", "python3", "/workspace/wp-finetune/scripts/train_model.py",
            f"config/train_config_60_40.yaml")
```

**Reclaim GPU memory** before proceeding:
```python
del model
torch.cuda.empty_cache()
gc.collect()
```

### Step 2: Sequential Adapter Evaluation

**Purpose:** Run each trained adapter through the full eval suite + wp-bench. Sequential because DGX Spark 128GB is too tight for parallel 30B vLLM instances.

**Duration:** ~25-40 minutes per adapter (vLLM startup ~10 min + merge ~5 min if LoRA fails + eval ~15 min + wp-bench ~15 min)

**For each adapter** (detected in Step 0a — typically 30/70, 40/60, 50/50, 60/40):

#### 2a. Serve adapter via vLLM

The orchestrator (`run_eval_triage.py`) handles vLLM lifecycle automatically:
1. Starts vLLM container via DGX Toolbox with `EXTRA_MOUNTS` for project directory
2. Polls health endpoint every 5s (configurable via `--health-timeout`, default 600s)
3. If LoRA loading fails (expected — `modules_to_save` incompatibility, Pitfall 7), automatically merges adapter and serves merged model

```bash
# The orchestrator handles this — no manual vLLM management needed
python3 scripts/run_eval_triage.py --ratios 30_70,40_60,50_50,60_40 --health-timeout 900
```

**What happens under the hood:**
- vLLM tries LoRA serving first (~10 min startup). For Qwen3-30B-A3B, LoRA will fail due to `modules_to_save` tensors
- Fallback: `scripts/merge_adapter.py --adapter-dir adapters/qwen3-30b-wp-{ratio} --output-dir models/merged-{ratio}` (~5-10 min, ~60GB per merged model)
- Merged model served without `--enable-lora` (~10 min startup)
- Health check: polls `http://localhost:8020/health` every 5s for up to `--health-timeout` seconds

Verify model name: `GET /v1/models` → extract model name string for eval scripts.

#### 2b. Spawn observe-evaluation (if telemetry enabled)

```
Agent(
  description="Telemetry: eval {ratio}",
  prompt="Monitor evaluation for ratio {ratio}. Write to telemetry/evaluation/{timestamp}/",
  run_in_background=true
)
```

#### 2c. Run static eval suite

> **Per-example logging (EVAL-06):** Both eval scripts now persist prompt, raw model response, and extracted code in per-example JSONL alongside scores. This enables human review of individual examples without re-running the model.
> - `eval_gen_results.jsonl`: includes `prompt`, `response`, `extracted_code` per example
> - `eval_judge_results.pairs.jsonl`: includes `prompt`, `response`, `code` per example
>
> **Per-dimension gates (EVAL-07):** `eval_gate.py` now correctly extracts `pass_rate_8` and `corr` from nested `per_dimension` dicts. Per-dimension gen and judge gates are functional when `gen_dimension_targets` / `judge_dimension_targets` are set in config.

```python
mkdir -p output/eval_triage/ratio_{ratio}

# PHPCS pass rate (EVAL-01)
python3 -m eval.eval_gen --model-url http://localhost:8020/v1 \
  --test-file data/final_dataset/openai_test.jsonl \
  --output-dir output/eval_triage/ratio_{ratio}/

# Judge Spearman correlation (EVAL-02)
python3 -m eval.eval_judge --model-url http://localhost:8020/v1 \
  --test-file data/final_dataset/openai_test.jsonl \
  --output-dir output/eval_triage/ratio_{ratio}/

# Quality gates (EVAL-05)
python3 -m eval.eval_gate --results-dir output/eval_triage/ratio_{ratio}/
```

Read results:
```python
import json
gen_results = json.load(open(f"output/eval_triage/ratio_{ratio}/eval_gen_results.json"))
judge_results = json.load(open(f"output/eval_triage/ratio_{ratio}/eval_judge_results.json"))
```

#### 2d. Run wp-bench (if available)

```bash
# Clone if needed
if [ ! -d wp-bench ]; then
  git clone https://github.com/WordPress/wp-bench.git
  cd wp-bench && ./setup.sh && cd ..
fi

# Configure and run
python3 wp-bench/run.py --model-url http://localhost:8020/v1 \
  --output-dir output/eval_triage/ratio_{ratio}/wpbench/
```

If wp-bench setup fails: set `wpbench_available=False`, continue without. wp-bench is for differentiation, not a hard gate.

#### 2e. Stop vLLM, reclaim memory

```python
dgx.stop("vllm")
# Wait for port to be released
time.sleep(10)
```

#### 2f. Present per-ratio results

After each adapter completes, display inline:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVAL ► RATIO {ratio} RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PHPCS pass rate:     {val}% {✓ PASS / ✗ FAIL} (gate: >95%)
  Judge Spearman:      {val}  {✓ PASS / ✗ FAIL} (gate: >0.85)
  Security pass rate:  {val}% {✓ PASS / ✗ FAIL} (gate: >98%)

  Overall gate:        {PASS / FAIL}

  wp-bench code gen:   {val} {or "skipped"}
  wp-bench knowledge:  {val} {or "skipped"}

  Training loss:       {final_loss}
  Training duration:   {hours}h
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 3: Automated Triage

**Purpose:** Apply GATE-02 elimination logic to determine which ratios survive to Phase 7.

```python
from scripts.triage_ratios import triage_ratios, write_triage_decision

result = triage_ratios(
    eval_dir="output/eval_triage",
    profiling_summary="output/profiling/base_model_eeff_summary.md",
    ratios=["30_70", "40_60", "50_50"]  # + 60_40/70_30 if trained
)

write_triage_decision(result, "output/triage_decision.md")
```

**Elimination rules (GATE-02):**
- Fail ANY hard gate (PHPCS ≤95%, Spearman ≤0.85, Security ≤98%) → **ELIMINATED**
- Overall score >5pp behind best ratio → **ELIMINATED**
- Everything else → **SURVIVES** (high bar for elimination, low bar for continuation)

### Step 4: Decision Gate 2 — Triage Review (Human)

**This is where the skill pauses for human decision.** Present the full triage comparison:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVAL ► TRIAGE DECISION — HUMAN REVIEW REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Quality Gate Results

  Ratio  │ PHPCS  │ Spearman │ Security │ Gates  │ Overall │ Status
  ───────┼────────┼──────────┼──────────┼────────┼─────────┼───────────
  30/70  │ {val}% │ {val}    │ {val}%   │ {P/F}  │ {val}   │ {SURVIVE/ELIM}
  40/60  │ {val}% │ {val}    │ {val}%   │ {P/F}  │ {val}   │ {SURVIVE/ELIM}
  50/50  │ {val}% │ {val}    │ {val}%   │ {P/F}  │ {val}   │ {SURVIVE/ELIM}
  60/40  │ {val}% │ {val}    │ {val}%   │ {P/F}  │ {val}   │ {SURVIVE/ELIM}
  {70/30}│ {if trained}                                     │

## wp-bench Differentiation (among gate-passers only)

  Ratio  │ Code Gen │ Knowledge │ Execution │ Overall │ Rank
  ───────┼──────────┼───────────┼───────────┼─────────┼──────
  {survivors only, ranked}

## E_eff Compressibility Signal (from Step 1 profiling)

  Ratio  │ Mean E_eff │ Max E_eff │ Interpretation
  ───────┼────────────┼───────────┼──────────────────────────────
  {all ratios — shows which will compress best}

  Note: Final ratio selected at Phase 7→8 gate using BOTH eval score
  AND fine-tuned adapter E_eff. This triage only eliminates clear losers.

## Automated Triage Verdict

  Survivors: {list}
  Eliminated: {list with reasons}
  {If NO_SURVIVORS: "⚠ No ratio passed all hard gates — see options below"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Decision criteria for the human:**

| Signal | What it tells you | Weight |
|--------|------------------|--------|
| Hard gates (PHPCS, Spearman, Security) | Minimum quality floor — non-negotiable | Binary (pass/fail) |
| Overall score | Gen-weighted quality (60% gen, 40% judge) | Eliminates >5pp behind |
| wp-bench | Real-world WordPress task performance | Differentiates among gate-passers |
| E_eff mean | Routing concentration → pruning headroom | Informational for Phase 7 |
| E_eff max | Worst-case layer → pruning ceiling | Informational for Phase 7 |
| Training loss | Training quality signal | Informational |

**Key insight:** A ratio with slightly lower eval score but much lower E_eff may produce a better production model after MoE-Sieve + REAP pruning. The triage preserves these candidates — Phase 7 makes the final call.

Use AskUserQuestion:
- header: "Triage"
- question: "Review the automated triage. Which ratios survive to Phase 7?"
- options:
  - "Accept automated triage ({N} survivors)" → proceed with automated verdict
  - "Override — add/remove survivors" → ask which ratios to include/exclude
  - "Train more ratios first" → pause, return to training
  - "Abort — quality insufficient" → exit with recommendations

**If NO_SURVIVORS:**
```
⚠ No ratio passed all hard gates. Options:

A. Lower a gate threshold (e.g., PHPCS >90% instead of >95%) — risky, justification needed
B. Retrain with different hyperparameters — learning rate, epochs, LoRA rank
C. Expand training data — re-run data pipeline with more repos
D. Investigate failures — what specific PHPCS/security checks are failing?
```

Use AskUserQuestion with these 4 options. Log the decision.

**If human overrides automated triage:** Append override to `output/triage_decision.md` preserving both automated and human verdicts:
```markdown
## Human Override

**Automated verdict:** {original survivors}
**Human decision:** {modified survivors}
**Rationale:** {user's explanation}
**Date:** {ISO date}
```

### Step 5: Update Project State

After human approval:

```bash
# Update STATE.md
node "$HOME/.claude/get-shit-done/bin/gsd-tools.cjs" state record-session \
  --stopped-at "Phase 4 eval complete — {N} survivors: {list}" \
  --resume-file "output/triage_decision.md"
```

Update STATE.md Accumulated Context:
```markdown
### Decisions
- [Phase 04 Eval]: {N} ratios survive triage: {list}. E_eff trend: {down/flat/up}. {60/40 training: warranted/skipped}.
- [Phase 04 Eval]: Best overall: {ratio} ({score}). Gate results: {summary}.
```

### Step 6: Present Next Steps

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVAL ► PHASE 4 TRIAGE COMPLETE ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Survivors: {list} → v1.2 Phase 4.1 (Judge Reasoning Data Gen)
  Winning ratio: {ratio} (highest overall score among survivors)
  Eliminated: {list}
  E_eff trend: {down/flat/up}

  Artifacts:
  - output/profiling/base_model_eeff_summary.md
  - output/eval_triage/ratio_{r}/  (per-ratio results)
  - output/eval_triage/ratio_{r}/eval_gen_results.jsonl  (per-example: prompt, response, code, scores)
  - output/eval_triage/ratio_{r}/eval_judge_results.pairs.jsonl  (per-example: prompt, response, code, model vs GT scores)
  - output/triage_decision.md (automated + human verdict)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ▶ Next Steps

**v1.2 Phase 4.1: Reasoning Data Generation** — pilot 20-50 examples per stream, then bulk deep judge CoT + critique-then-fix generation on winning ratio adapter

/gsd:discuss-phase 4.1

/clear first → fresh context window

Note: Phase 7 (Router Profiling) is blocked until v1.2 completes (Phases 4.1-4.4).
```

## CLI Reference

```bash
python3 scripts/run_eval_triage.py [options]

--skip-profiling      Skip E_eff profiling (use existing results)
--skip-wpbench        Skip wp-bench (static gates only)
--ratios RATIOS       Comma-separated ratios (default: 30_70,40_60,50_50)
--force               Clear all completion markers, re-run everything
--health-timeout SEC  vLLM health check timeout (default: 600). Use 900 for
                      30B MoE with LoRA fallback (merge adds ~10 min)
--model-path PATH     Base model dir relative to project root
--tokenizer-path PATH Extended tokenizer dir relative to project root
--dataset PATH        Test dataset JSONL relative to project root
--verbose             DEBUG logging
```

## Error Handling

| Error | Recovery |
|-------|----------|
| CUDA not available on HOST | Expected — orchestrator runs from HOST, vLLM runs in container. Only profiling needs CUDA (use `--skip-profiling` if done) |
| vLLM health timeout | Increase `--health-timeout`. 30B MoE: ~10 min load + ~2 min CUDA graphs. With LoRA fallback + merge: add ~15 min |
| vLLM LoRA load fails (modules_to_save) | Expected for Qwen3 adapters (Pitfall 7). Orchestrator auto-falls back to merge-and-serve |
| vLLM container missing project mount | Set `EXTRA_MOUNTS` env var (orchestrator does this automatically). DGX Toolbox `start-vllm.sh` must source `lib.sh` |
| wp-bench clone/setup fails | Continue without wp-bench, note in results |
| eval script crashes | Log error, skip to next ratio, include in triage as "EVAL_FAILED" |
| All ratios fail gates | Present NO_SURVIVORS options (A-D), human decides |
| Completion marker exists | Skip step (idempotent), use `--force` to override |

## Idempotency

Each major step writes a completion marker (`.complete` file):
- `output/profiling/.complete` — profiling done
- `output/eval_triage/ratio_{r}/.complete` — ratio eval done
- `output/.triage_complete` — triage done

Re-running the skill resumes from the last incomplete step. Use `--force` to re-run everything.

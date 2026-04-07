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

Build adapter inventory. Read each adapter's `trainer_state.json` (last checkpoint) for final loss if available.

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
# --limit 500 recommended for triage (representative sample, ~4h/ratio)
python3 scripts/run_eval_triage.py --skip-profiling --ratios 30_70,40_60,50_50,60_40 --limit 500
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

```bash
# Profiling is handled by the orchestrator (no separate --profiling-only flag)
# To run profiling + eval together:
python3 scripts/run_eval_triage.py --ratios 30_70,40_60,50_50,60_40

# To skip profiling (if already complete):
python3 scripts/run_eval_triage.py --skip-profiling --ratios 30_70,40_60,50_50,60_40
```

Profiling is handled internally by `run_eval_triage.py` which calls `profile_base_model.py`. Output lands in `output/profiling/`.

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
  60/40  │ {val}      │ {val}     │ {val}     │ {gen-heavy}
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

If training warranted, start via the `/wp-finetune:run-training` skill (handles container lifecycle and config).

**Reclaim GPU memory** before proceeding — the orchestrator handles this internally between profiling and vLLM serving.

### Step 2: Sequential Adapter Evaluation

**Purpose:** Run each trained adapter through the full eval suite + wp-bench. Sequential because DGX Spark 128GB is too tight for parallel 30B vLLM instances.

**Duration:** ~25-40 minutes per adapter (vLLM startup ~10 min + eval ~15 min + wp-bench ~15 min). Pre-merge runs once upfront before the eval loop.

**For each adapter** (detected in Step 0a — typically 30/70, 40/60, 50/50, 60/40):

#### 2a. Pre-merge adapters and serve via vLLM

The orchestrator pre-merges all adapters on HOST before the eval loop begins. This avoids the LoRA `modules_to_save` incompatibility that always causes vLLM LoRA serving to fail for Qwen3-30B-A3B.

**Pre-merge step (Step 1.5, runs automatically):**
- Calls `scripts/merge_adapter.py` on HOST with `device_map=cpu` — MoE models can't auto-offload to disk, and 30B bf16 (~60GB) fits in DGX Spark 128GB unified RAM. No training container required
- Idempotent: skips already-merged models after verifying special tokens
- Merged models written to `models/merged-{ratio}/` (e.g., `models/merged-30_70/`). Note: this differs from the run-training skill's merge path (`models/qwen3-30b-wp-{ratio}-merged/`). The orchestrator always uses its own path and does not reuse training's merged output
- Each merged model is ~60GB; check disk space before running all 4 ratios
- If a merge fails, that ratio is removed from the eval list

**Per-ratio vLLM serving:**
- Serves pre-merged model directly (no `--enable-lora`)
- Container name resolved from `dgx_toolbox.yaml` (not hardcoded)
- Health check polls every 5s for up to `--health-timeout` seconds (default 600s, use 900 for safety)

```bash
# The orchestrator handles all of this — no manual vLLM management needed
python3 scripts/run_eval_triage.py --ratios 30_70,40_60,50_50,60_40 --health-timeout 900
```

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

```bash
mkdir -p output/eval_triage/ratio_{ratio}

# PHPCS pass rate + per-dimension rubric scoring (EVAL-01)
# Note: vLLM endpoint resolved internally via dgx.vllm_endpoint(), not a CLI flag
python3 -m eval.eval_gen \
  --dataset data/final_dataset/openai_test.jsonl \
  --output output/eval_triage/ratio_{ratio}/eval_gen_results.json \
  --model <detected-model-name>

# Judge Spearman correlation (EVAL-02)
python3 -m eval.eval_judge \
  --dataset data/final_dataset/openai_test.jsonl \
  --output output/eval_triage/ratio_{ratio}/eval_judge_results.json \
  --model <detected-model-name>

# Quality gates (EVAL-05) — per-dimension gates now functional (EVAL-07)
python3 -m eval.eval_gate --results-dir output/eval_triage/ratio_{ratio}/
```

The orchestrator (`run_eval_triage.py`) handles all of this automatically — including model name detection from `/v1/models`.

#### 2d. Run wp-bench (if available)

The orchestrator clones wp-bench if needed, then runs it via `config/wp-bench.yaml` with a temp config override per ratio:

```python
# Under the hood (orchestrator handles this automatically):
# 1. Reads config/wp-bench.yaml, overrides output_path per ratio
# 2. Writes temp config to output/eval_triage/ratio_{ratio}/wp_bench_config_tmp.yaml
# 3. Runs: wp-bench run --config <tmp_config>  (cwd=wp-bench/)
# 4. Output: output/eval_triage/ratio_{ratio}/wp_bench_results.json
```

If wp-bench clone/setup fails or returns non-zero: continues without wp-bench and writes error detail to result JSON. wp-bench is for differentiation, not a hard gate.

#### 2e. Stop vLLM, reclaim memory

The orchestrator's internal `_stop_vllm()` handles container cleanup (container name resolved from `dgx_toolbox.yaml`) and port release between ratios.

#### 2f. Present per-ratio results

After each adapter completes, display inline:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVAL ► RATIO {ratio} RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PHPCS pass rate:     {val}% {✓ PASS / ✗ FAIL} (gate: >95%)
  Security pass rate:  {val}% or null {✓ PASS / ✗ FAIL / N/A} (gate: >98%)
  Judge Spearman:      {val}  {✓ PASS / ✗ FAIL} (gate: >0.85)

  Gen quality score:   {avg(phpcs, security)} (used for 5pp elimination)
  Judge calibration:   {spearman} (separate axis, not blended)

  N/A transparency:    {n_applicable_dims_mean}/9 dims tested
  Hard gates:          {PASS / FAIL}

  wp-bench:            {val} {or "skipped"}
  Training loss:       {final_loss}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 3: Automated Triage

**Purpose:** Apply GATE-02 elimination logic to determine which ratios survive to Phase 7.

```python
from scripts.triage_ratios import load_eval_results, triage_ratios, write_triage_decision

eval_results = load_eval_results("output/eval_triage")
result = triage_ratios(eval_results)
write_triage_decision(result, None, "output/triage_decision.md")
```

The orchestrator (`run_eval_triage.py`) calls this automatically after all adapters are evaluated.

**Elimination rules (GATE-02):**
- Fail ANY hard gate (PHPCS ≤95%, Spearman ≤0.85, Security ≤98%) → **ELIMINATED**
- Gen quality score >5pp behind best ratio → **ELIMINATED** (gen quality = avg of PHPCS + security pass rates; Spearman is a separate axis, not blended)
- Everything else → **SURVIVES** (high bar for elimination, low bar for continuation)

### Step 4: Decision Gate 2 — Triage Review (Human)

**This is where the skill pauses for human decision.** Present the full triage comparison:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVAL ► TRIAGE DECISION — HUMAN REVIEW REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Quality Gate Results

  Ratio  │ PHPCS  │ Security │ Gen Quality │ Spearman │ Gates  │ Status
  ───────┼────────┼──────────┼─────────────┼──────────┼────────┼───────────
  30/70  │ {val}% │ {val}%   │ {avg}       │ {val}    │ {P/F}  │ {SURVIVE/ELIM}
  40/60  │ {val}% │ {val}%   │ {avg}       │ {val}    │ {P/F}  │ {SURVIVE/ELIM}
  50/50  │ {val}% │ {val}%   │ {avg}       │ {val}    │ {P/F}  │ {SURVIVE/ELIM}
  60/40  │ {val}% │ {val}%   │ {avg}       │ {val}    │ {P/F}  │ {SURVIVE/ELIM}
  {70/30}│ {if trained}                                        │

  Gen Quality = avg(PHPCS, Security) — used for 5pp elimination
  Spearman = judge calibration — separate ranking axis, not blended

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
| Gen quality score | avg(PHPCS, Security) — code generation quality | Eliminates >5pp behind best |
| Judge calibration (Spearman) | How well model scores match GT — separate axis | Ranked independently |
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
--limit N             Max examples for eval_gen/eval_judge per ratio (default: all).
                      Does NOT affect wp-bench (always runs full canonical suite).
                      Recommended: 500 for triage (~2.8h/ratio), full for final eval.
--force               Clear all completion markers, re-run everything
--health-timeout SEC  vLLM health check timeout (default: 600). Use 900 for
                      30B MoE (model load ~10 min + CUDA graphs ~2 min)
--model-path PATH     Base model dir relative to project root
--tokenizer-path PATH Extended tokenizer dir relative to project root
--dataset PATH        Test dataset JSONL relative to project root
--verbose             DEBUG logging
```

**Time estimates by --limit:**

| --limit | eval_gen | eval_judge | wp-bench | Total/ratio | x4 ratios |
|---------|----------|------------|----------|-------------|-----------|
| 500 | ~1.4h | ~1.4h | ~1h | ~4h | ~16h |
| 1000 | ~2.8h | ~2.8h | ~1h | ~6.6h | ~26h |
| (all 10,166) | ~29h | ~29h | ~1h | ~59h | ~237h |

**Recommendation:** Use `--limit 500` for triage (statistically representative at 5% of test set). Run full eval only on the winning ratio after triage.

## Error Handling

| Error | Recovery |
|-------|----------|
| CUDA not available on HOST | Expected — orchestrator runs from HOST, vLLM runs in container. Pre-merge uses `device_map=cpu` (GPU if available, CPU fallback). Only profiling needs CUDA (use `--skip-profiling` if done) |
| vLLM health timeout | Increase `--health-timeout`. 30B MoE: ~10 min load + ~2 min CUDA graphs |
| Pre-merge fails | Check `peft` and `transformers` versions on HOST match training container. Merge uses `device_map=cpu` (GPU if available, CPU fallback). Failed ratios are automatically skipped |
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

# Phase 4: Base-Model Profiling & Evaluation (Triage) - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Profile base model with all 5 ratio data distributions to determine whether 60/40 and 70/30 warrant training. Then eval existing adapters (30/70, 40/60, 50/50) through quality gates and wp-bench. Triage eliminates clearly failing ratios; survivors carried to Phase 7 for fine-tuned adapter profiling.

This is a TRIAGE phase, not a winner-selection phase. The final ratio is chosen at the Phase 7→8 gate based on combined eval quality + routing compressibility (E_eff).

</domain>

<decisions>
## Implementation Decisions

### Evaluation Architecture
- **D-01:** Phase 4 is triage — high bar for elimination (>5pp behind or fails hard gates), low bar for continuation. 1-2pp differences may invert after pruning.
- **D-02:** Phase 4 Step 1 is base-model profiling of all 5 ratio data distributions (~minutes) to produce E_eff per layer — determines whether to train 60/40 and 70/30 before spending 100+ hours of GPU time.
- **D-03:** If E_eff trending down as gen% increases, start 60/40 training in background while eval runs on existing 3 adapters. If E_eff flat or trending up, skip additional training.
- **D-04:** All 3 existing adapters (30/70, 40/60, 50/50) evaluated through full static eval + wp-bench.

### Ratio Comparison
- **D-05:** Eval comparison is gen-weighted — PHPCS pass rate + security matter more since gen is the user-facing output and judge can be refined via GRPO. But this is triage, not final selection.
- **D-06:** The metric that matters for production is post-compression quality-per-VRAM, not pre-compression eval score. Phase 7 E_eff completes the picture.

### Eval Infrastructure (from Phase 3 context)
- **D-07:** wp-bench is the canonical WordPress AI benchmark (user confirmed in Phase 3)
- **D-08:** No Claude in the eval loop — solves circularity
- **D-09:** Eval scripts already exist: eval/eval_gen.py (PHPCS), eval/eval_judge.py (Spearman), eval/eval_gate.py (quality gates)
- **D-10:** Test data: data/final_dataset/openai_test.jsonl (597 held-out examples)
- **D-11:** Model serving via DGX Toolbox vLLM + LiteLLM proxy

### E_eff Profiling
- **D-12:** E_eff = exp(entropy) per MoE layer — directly predicts pruning headroom. Lower = sharper routing = more experts prunable.
- **D-13:** Report includes mean E_eff, max E_eff (bottleneck layer), E_eff variance (predicts uniform vs layer-adaptive pruning need)
- **D-14:** Profile hooks into Qwen3MoeSparseMoeBlock gating output, count-based routing per expert per layer, separate counts for <wp_gen> and <wp_judge> tokens

### Claude's Discretion
- Eval execution strategy (sequential vs parallel adapter serving)
- wp-bench task subset selection
- Profiling script implementation details

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Eval Suite
- `eval/eval_gen.py` — PHPCS pass rate measurement on held-out gen tasks
- `eval/eval_judge.py` — Spearman correlation scorer for judge calibration
- `eval/eval_gate.py` — Quality gate aggregator (PHPCS >95%, Spearman >0.85, Security >98%)
- `eval/rubric_scorer.py` — 9-dimension rubric scoring engine
- `eval/rubric_definitions.py` — 45KB rubric dimension definitions

### Training Adapters
- `adapters/qwen3-30b-wp-30_70/` — Completed 30/70 adapter
- `adapters/qwen3-30b-wp-40_60/` — Completed 40/60 adapter (OOM run but final adapter saved, validated)
- `adapters/qwen3-30b-wp-50_50/` — Completed 50/50 adapter

### Test Data
- `data/final_dataset/openai_test.jsonl` — 597 held-out test examples
- `data/final_dataset/ratio_30_70/` — 30/70 ratio dataset
- `data/final_dataset/ratio_40_60/` — 40/60 ratio dataset
- `data/final_dataset/ratio_50_50/` — 50/50 ratio dataset
- `data/final_dataset/ratio_60_40/` — 60/40 ratio dataset (for profiling, not yet trained)
- `data/final_dataset/ratio_70_30/` — 70/30 ratio dataset (for profiling, not yet trained)

### Config
- `config/train_config.yaml` — Base training config
- `config/train_config_30_70.yaml` — 30/70 ratio config
- `config/train_config_40_60.yaml` — 40/60 ratio config
- `config/train_config_50_50.yaml` — 50/50 ratio config

### Prior Context
- `.planning/phases/03-model-prep-and-training/03-CONTEXT.md` — wp-bench decision, eval suite design, DGX serving approach

### MoE-Sieve Paper
- `https://arxiv.org/html/2603.24044v1` — Routing concentration analysis, Jaccard stability, E_eff methodology

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/eval_gen.py` — Ready to run PHPCS pass rate on any adapter
- `eval/eval_judge.py` — Ready to run Spearman correlation
- `eval/eval_gate.py` — Ready to aggregate gates
- `eval/rubric_scorer.py` + `rubric_definitions.py` — Full 9-dimension scoring
- `scripts/train_model.py` — Training script with GPUSampler, Unsloth detection, failure classification (Phase 6 additions)

### Established Patterns
- Model serving: DGX Toolbox vLLM + LiteLLM proxy pattern (from Phase 3)
- Telemetry: canonical JSONL thermal logs in `telemetry/training/`
- Adapter isolation: each ratio in `adapters/qwen3-30b-wp-{ratio}/`

### Integration Points
- Profiling script needs to hook `Qwen3MoeSparseMoeBlock` — same hook pattern as MoE-Sieve Phase 7 profiling, but on base model with raw data distributions
- vLLM serving for eval: `dgx.run("vllm")` with adapter loaded via `--lora-modules`

</code_context>

<specifics>
## Specific Ideas

- Base-model profiling produces the first E_eff data for this project — establishes whether WordPress data shows the sharp routing concentration expected from domain-specific code tasks
- The base-model vs fine-tuned adapter E_eff comparison (Phase 4 vs Phase 7) will show how much LoRA training shifts routing — novel data point for MoE-Sieve research
- 40/60 adapter was saved despite OOM — validated as intact (12,674 tensors, zero NaN/Inf, final adapter_model.safetensors present)

</specifics>

<deferred>
## Deferred Ideas

- Full 5-ratio comparison through entire pipeline (too expensive — E_eff-guided triage is the practical alternative)
- Eval on 60/40 and 70/30 adapters — deferred pending base-model E_eff signal

</deferred>

---

*Phase: 04-evaluation*
*Context gathered: 2026-04-03*

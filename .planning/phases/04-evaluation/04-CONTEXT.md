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

### Base-Model Profiling (Step 1 — runs first, ~minutes)
- **D-01:** Profile base model (no adapter) with all 5 ratio data distributions (30/70, 40/60, 50/50, 60/40, 70/30) by hooking `Qwen3MoeSparseMoeBlock` gating output
- **D-02:** 10% subsample per ratio for stability (MoE-Sieve paper: Jaccard ≥0.94 at 10%)
- **D-03:** Output format: JSONL per-layer raw data (machine-readable for Phase 7 comparison) + markdown summary table with E_eff mean/max/variance per ratio
- **D-04:** E_eff = exp(entropy) per layer. Count-based routing per expert per layer. Separate counts for `<wp_gen>` and `<wp_judge>` tokens.
- **D-05:** E_eff training trigger: ANY downward trend in E_eff as gen% increases → train 60/40. Cost is ~2 days GPU vs weeks of uncertainty.

### Eval Execution (Step 2 — runs in parallel with any new training)
- **D-06:** Adapters served sequentially via vLLM `--lora-modules` — one at a time, eval fully before loading next. DGX Spark 128GB too tight for parallel 30B instances.
- **D-07:** All 3 existing adapters (30/70, 40/60, 50/50) get full eval: static suite + full wp-bench run per adapter.
- **D-08:** If 60/40 training starts, its eval runs when training completes (~2 days). Triage decision waits for all available data.

### Eval Quality Gates
- **D-09:** Static eval gates first, wp-bench for differentiation. A ratio must pass hard gates before wp-bench scores matter. Among gate-passers, wp-bench differentiates.
- **D-10:** Hard gates: PHPCS pass rate >95%, Judge Spearman >0.85, Security pass rate >98%
- **D-11:** Triage is gen-weighted — PHPCS + security matter more since gen is user-facing output. Judge refined via GRPO later.

### Triage Elimination Rules
- **D-12:** Elimination = fails ANY hard gate OR >5pp behind best ratio on overall score. No other rules.
- **D-13:** High bar for elimination, low bar for continuation — 1-2pp differences may invert after pruning if routing concentration differs. Unless clearly failing, carry to Phase 7.
- **D-14:** The metric that matters for production is post-compression quality-per-VRAM, not pre-compression eval score.

### Timing & Parallelism
- **D-15:** Step 1 (base-model profiling) completes first (~minutes). If E_eff trending down → start 60/40 training immediately.
- **D-16:** Step 2 (eval on existing 3 adapters) runs in parallel with any 60/40 training.
- **D-17:** Triage decision made when all warranted adapters are evaluated.

### From Prior Phases (locked)
- **D-18:** wp-bench is the canonical WordPress AI benchmark (Phase 3 — user confirmed)
- **D-19:** No Claude in the eval loop — solves eval circularity (Phase 3)
- **D-20:** Eval scripts exist and are ready: `eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py`
- **D-21:** Test data: `data/final_dataset/openai_test.jsonl` (597 held-out examples)
- **D-22:** Model serving via DGX Toolbox vLLM + LiteLLM proxy

### Claude's Discretion
- Profiling script implementation details (hook registration, data loading)
- Eval execution ordering among the 3 adapters (which ratio first doesn't matter)
- wp-bench task category weighting within the full run
- Markdown summary table formatting

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
- `adapters/qwen3-30b-wp-40_60/` — Completed 40/60 adapter (OOM run but final adapter saved and validated: 12,674 tensors, zero NaN/Inf)
- `adapters/qwen3-30b-wp-50_50/` — Completed 50/50 adapter (final loss 0.296, 2 epochs, 39.6 hours)

### Test & Profiling Data
- `data/final_dataset/openai_test.jsonl` — 597 held-out test examples
- `data/final_dataset/ratio_30_70/` — 30/70 ratio dataset (profiling + training data)
- `data/final_dataset/ratio_40_60/` — 40/60 ratio dataset
- `data/final_dataset/ratio_50_50/` — 50/50 ratio dataset
- `data/final_dataset/ratio_60_40/` — 60/40 ratio dataset (for profiling; training only if E_eff warrants)
- `data/final_dataset/ratio_70_30/` — 70/30 ratio dataset (for profiling; training only if E_eff warrants)

### Config
- `config/train_config.yaml` — Base training config
- `config/train_config_30_70.yaml` — 30/70 ratio training config
- `config/train_config_40_60.yaml` — 40/60 ratio training config
- `config/train_config_50_50.yaml` — 50/50 ratio training config

### Model
- `models/Qwen3-30B-A3B/` — Base model for profiling (router weights in `Qwen3MoeSparseMoeBlock`)

### Prior Context
- `.planning/phases/03-model-prep-and-training/03-CONTEXT.md` — wp-bench decision, eval suite design, DGX serving approach

### Research
- MoE-Sieve paper (arxiv 2603.24044) — E_eff methodology, Jaccard stability, routing concentration analysis
- AIMER paper (arxiv 2603.18492) — Weight-based pruning scoring, tested on Qwen3-30B-A3B
- REAP (Cerebras) — Calibration-based pruning, native Qwen3 support

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval/eval_gen.py` — Ready to run PHPCS pass rate on any adapter
- `eval/eval_judge.py` — Ready to run Spearman correlation
- `eval/eval_gate.py` — Ready to aggregate quality gates
- `eval/rubric_scorer.py` + `rubric_definitions.py` — Full 9-dimension scoring
- `scripts/train_model.py` — Training script with GPUSampler, Unsloth detection, failure classification

### Established Patterns
- Model serving: DGX Toolbox vLLM + LiteLLM proxy (from Phase 3)
- Adapter isolation: each ratio in `adapters/qwen3-30b-wp-{ratio}/`
- Telemetry: canonical JSONL in `telemetry/training/`
- Training configs: `config/train_config_{ratio}.yaml`

### Integration Points
- NEW: Profiling script hooks `Qwen3MoeSparseMoeBlock` gating output — same pattern as Phase 7 but on base model
- vLLM serving: `dgx.run("vllm")` with `--lora-modules` for adapter loading
- wp-bench: clone from `github.com/WordPress/wp-bench`, configure via `wp-bench.example.yaml`

</code_context>

<specifics>
## Specific Ideas

- Base-model profiling produces the first E_eff data for this project — establishes whether WordPress data shows sharp routing concentration expected from domain-specific code tasks
- The base-model vs fine-tuned adapter E_eff comparison (Phase 4 vs Phase 7) quantifies how much LoRA training shifts routing — novel data point
- 40/60 adapter validated as intact despite OOM (12,674 tensors, zero NaN/Inf, final adapter_model.safetensors present)
- Phase 4 profiling JSONL is directly consumed by Phase 7 for base-vs-adapter comparison — format consistency is critical

</specifics>

<deferred>
## Deferred Ideas

- Full 5-ratio comparison through entire pipeline (too expensive — E_eff-guided triage is the alternative)
- 70/30 training — only if E_eff trend is strongly downward AND 60/40 continues the trend
- Layer-adaptive pruning ratio analysis — deferred to Phase 12, but E_eff variance data from Phase 4 profiling feeds it

</deferred>

---

*Phase: 04-evaluation*
*Context gathered: 2026-04-03 (updated with profiling + eval strategy decisions)*

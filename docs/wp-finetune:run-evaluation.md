# Skill: run-evaluation

Run the Phase 4 evaluation and triage pipeline. Profiles base model routing concentration (E_eff), evaluates trained adapters through quality gates and wp-bench, then presents a structured triage decision for human approval.

## Architecture

```
Skill (SKILL.md — intent + decision logic)
  -> scripts/run_eval_triage.py (orchestrator: profiling + eval + triage)
  -> scripts/profile_base_model.py (E_eff profiling, hooks Qwen3MoeTopKRouter)
  -> scripts/triage_ratios.py (GATE-02 elimination logic)
  -> eval/eval_gen.py, eval_judge.py, eval_gate.py (existing eval suite)
  -> DGX Toolbox (vLLM serving, container management)
     Output: output/profiling/, output/eval_triage/, output/triage_decision.md
```

## Telemetry

Embeds observe-evaluation (3 agents) during long-running eval steps.
Output: `telemetry/evaluation/{timestamp}/`

## Trigger

User says: "run evaluation", "evaluate the model", "run eval triage", "/run-evaluation"

## Scripts

### `scripts/profile_base_model.py`

Hooks all 48 Qwen3MoeTopKRouter gate layers to capture per-expert routing counts split by `<wp_gen>` vs `<wp_judge>` tokens. Computes E_eff (effective number of experts via Shannon entropy) per layer per ratio across all 5 data distributions (30/70 through 70/30).

```bash
python -m scripts.profile_base_model
python -m scripts.profile_base_model --model-path models/Qwen3-30B-A3B
```

Output:
- `output/profiling/base_model_eeff.jsonl` (Phase 7-compatible JSONL)
- `output/profiling/base_model_eeff_summary.md`

### `scripts/run_eval_triage.py`

Full orchestrator: setup, base-model profiling, sequential adapter eval (vLLM LoRA per ratio), triage decision. Idempotent via `.complete` marker files.

```bash
python scripts/run_eval_triage.py                    # full pipeline
python scripts/run_eval_triage.py --skip-wpbench      # skip wp-bench
python scripts/run_eval_triage.py --ratios 30_70,50_50 # subset of ratios
python scripts/run_eval_triage.py --skip-profiling     # skip E_eff profiling
python scripts/run_eval_triage.py --force              # re-run everything
```

### `scripts/triage_ratios.py`

GATE-02 elimination logic. Reads per-ratio eval results and applies:

1. **Hard gates** (strict `>`, value AT threshold FAILS):
   - PHPCS pass rate > 0.95
   - Judge Spearman > 0.85
   - Security pass rate > 0.98
2. **5pp rule:** eliminated if `best_overall - ratio_score > 0.05`
   (exactly 5pp behind survives — low bar for continuation)

Returns `NO_SURVIVORS` status if zero ratios pass (does not crash).

```bash
python -m scripts.triage_ratios
python -m scripts.triage_ratios --eval-dir output/eval_triage
```

## Tests

- `tests/test_eeff.py` — E_eff computation, routing collector, summary generation
- `tests/test_triage.py` — GATE-02 elimination logic, edge cases, NO_SURVIVORS handling

## Process Summary

| Step | What | Duration | Output |
|------|------|----------|--------|
| 0 | Inventory adapters + DGX readiness | Seconds | Adapter table |
| 1 | Base-model E_eff profiling (5 ratios) | ~10 min | `output/profiling/` |
| 2 | Sequential adapter eval (vLLM + eval suite + wp-bench) | ~30-45 min/adapter | `output/eval_triage/ratio_{r}/` |
| 3 | Automated triage (GATE-02 elimination) | Seconds | `output/triage_decision.md` |
| 4 | Human review + decision gate | Manual | Override appended to triage decision |
| 5 | Update project state | Seconds | STATE.md |

## Key Decisions

- **E_eff trending down** -> train more gen-heavy ratios (60/40, 70/30)
- **E_eff flat/up** -> skip additional training, evaluate existing adapters
- **Triage preserves ambiguous candidates** — Phase 7 makes the final call using both eval score and fine-tuned adapter E_eff

## Idempotency

Completion markers:
- `output/profiling/.complete`
- `output/eval_triage/ratio_{r}/.complete`
- `output/eval_triage/.triage_complete`

Re-running resumes from last incomplete step. Use `--force` to re-run everything.

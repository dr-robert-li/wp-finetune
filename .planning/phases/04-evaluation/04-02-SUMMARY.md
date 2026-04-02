---
phase: 04-evaluation
plan: 02
subsystem: eval
tags: [vllm, lora, profiling, triage, wp-bench, eval-orchestration, moe]

# Dependency graph
requires:
  - phase: 04-evaluation-01
    provides: "profile_base_model.py and triage_ratios.py scripts created in Plan 01"
  - phase: 03-model-prep-and-training
    provides: "Trained LoRA adapters at adapters/qwen3-30b-wp-{ratio}/ and base model at models/Qwen3-30B-A3B"
provides:
  - "scripts/run_eval_triage.py -- full Phase 4 pipeline orchestrator"
  - "Execution commands for DGX Spark GPU environment"
affects: [04-evaluation-03, 07-router-profiling]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Idempotency markers: .complete files per pipeline step, skip-if-exists, --force to bypass"
    - "Sequential vLLM LoRA serving: start -> health poll -> eval -> stop -> repeat for next ratio"
    - "GPU memory reclamation: del model, gc.collect(), torch.cuda.empty_cache() after profiling"
    - "Fallback pattern: LoRA load failure -> merge_adapter.py -> serve merged model"
    - "Retry-once: eval_gen/eval_judge retry once after 30s on exception"

key-files:
  created:
    - scripts/run_eval_triage.py
  modified: []

key-decisions:
  - "Task 2 execution deferred: CUDA unavailable in current Python env (cpu-only torch 2.10.0); GPU execution requires running inside DGX Toolbox vLLM container or activating CUDA-enabled env"
  - "Script is importable without GPU -- all heavy imports inside function bodies using lazy loading"
  - "run_eval_and_wpbench_for_ratio() keeps vLLM alive between eval_gen, eval_judge, and wp-bench for the same ratio to avoid restart overhead"
  - "Triage marker is stale-checked: if any eval marker is newer, triage re-runs automatically"

patterns-established:
  - "Completion marker pattern: each step writes output/{step}/.complete on success"
  - "Force bypass: --force arg clears markers and re-runs all steps"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03, EVAL-04]

# Metrics
duration: 15min
completed: 2026-04-02
---

# Phase 4 Plan 02: Eval Triage Orchestrator Summary

**Idempotent eval triage orchestrator with vLLM LoRA sequential serving, GPU memory reclamation, merge fallback, and NO_SURVIVORS handling -- ready to execute on DGX Spark**

## Performance

- **Duration:** 15 min
- **Started:** 2026-04-02T22:26:25Z
- **Completed:** 2026-04-02T22:41:00Z
- **Tasks:** 1 of 2 (Task 1 complete; Task 2 requires DGX GPU execution)
- **Files modified:** 1

## Accomplishments

- Created `scripts/run_eval_triage.py` (1135 lines) -- orchestrates the full Phase 4 pipeline
- All 16 acceptance criteria verified (function definitions, CLI flags, GPU cleanup, idempotency, NO_SURVIVORS handling)
- Script is importable without GPU (all heavy imports are lazy inside function bodies)
- Documented execution commands for DGX Spark GPU session (see below)

## Task Commits

1. **Task 1: Create eval orchestrator script** - `99591e7` (feat)

## Files Created/Modified

- `scripts/run_eval_triage.py` -- 1135-line orchestrator with 5 exported functions, idempotency markers, vLLM lifecycle management, wp-bench integration, merge fallback, and full triage pipeline

## Decisions Made

- Task 2 execution is deferred to user's DGX GPU session. The local Python environment has `torch 2.10.0+cpu` (no CUDA), and the Docker container also has CUDA unavailable (NVML init error). The GB10 GPU is present (`nvidia-smi` shows it) but the Python CUDA runtime needs the correct environment activation.
- `run_eval_and_wpbench_for_ratio()` keeps vLLM alive between eval_gen, eval_judge, and wp-bench to avoid the cost of stopping and restarting vLLM for the same ratio.
- Triage marker stale-check: if any eval marker's mtime is newer than triage's mtime, triage automatically re-runs to incorporate new results.

## Deviations from Plan

### Task 2 Not Executed (Environment Constraint)

Task 2 requires running `python scripts/run_eval_triage.py` on DGX Spark with CUDA-enabled torch. The current worktree environment has CPU-only torch (`torch 2.10.0+cpu`). The `nvidia-smi` shows GB10 is present but CUDA is not accessible from the current Python environment.

**Action required:** Execute in the DGX GPU session (see Execution Commands below).

This is not a deviation from the script design -- it is an environment constraint documented in the objective ("If the DGX environment is not available from this worktree, document the execution commands in SUMMARY.md").

## Execution Commands for DGX Spark

Run these commands in a terminal with CUDA-enabled Python (inside the training container or with the correct conda env):

```bash
cd /home/robert_li/Desktop/projects/wp-finetune

# Step 1: Full pipeline (profiling + eval + triage)
python scripts/run_eval_triage.py

# Step 2a: Skip profiling if already done
python scripts/run_eval_triage.py --skip-profiling

# Step 2b: Skip wp-bench if Docker WP runtime not ready
python scripts/run_eval_triage.py --skip-wpbench

# Step 2c: Run only specific ratios
python scripts/run_eval_triage.py --ratios 30_70,40_60,50_50

# Step 2d: Force full re-run (bypass all .complete markers)
python scripts/run_eval_triage.py --force

# Step 2e: Debug mode
python scripts/run_eval_triage.py --verbose
```

**Inside DGX Toolbox container:**
```bash
# Activate CUDA env (or run inside the unsloth-headless container)
EXTRA_MOUNTS="/home/robert_li/Desktop/projects/wp-finetune:/workspace/wp-finetune" \
  bash ~/dgx-toolbox/containers/unsloth-headless-sync.sh \
  python -m scripts.run_eval_triage

# Or exec into running container:
docker exec -it unsloth-headless bash
cd /workspace/wp-finetune
python scripts/run_eval_triage.py
```

**Expected outputs after execution:**
- `output/profiling/base_model_eeff.jsonl` -- 240+ records (5 ratios x 48 layers)
- `output/profiling/base_model_eeff_summary.md` -- E_eff summary table
- `output/profiling/.complete` -- profiling completion marker
- `output/eval_triage/ratio_30_70/eval_gen_results.json`
- `output/eval_triage/ratio_30_70/eval_judge_results.json`
- `output/eval_triage/ratio_40_60/eval_gen_results.json`
- `output/eval_triage/ratio_40_60/eval_judge_results.json`
- `output/eval_triage/ratio_50_50/eval_gen_results.json`
- `output/eval_triage/ratio_50_50/eval_judge_results.json`
- `output/eval_triage/ratio_*/.complete` -- per-ratio markers (3 files)
- `output/triage_decision.md` -- STATUS: OK or STATUS: NO_SURVIVORS
- `output/.triage_complete` -- triage completion marker

**Timing estimate:** Profiling ~5-10 min, each adapter eval ~30-60 min, total ~2-4 hours.

**If vLLM LoRA fails to load (Pitfall 7: modules_to_save tensors):**
The script automatically falls back to `scripts/merge_adapter.py` to create merged checkpoints (~60GB each) and serves them as full models. Monitor logs for fallback activation.

**If E_eff downward trend detected:**
The script prints: `*** E_eff DOWNWARD TREND: 60/40 training warranted (D-05) ***`
Start 60/40 training separately. Its eval runs when training completes (~2 days). Per D-08, triage waits for all warranted adapters.

**If NO_SURVIVORS:**
The script prints a prominent warning. Do NOT panic -- Plan 03 human review handles this contingency.

## Issues Encountered

- CUDA unavailable in local Python environment (`torch 2.10.0+cpu`). Docker container also lacks working CUDA despite GB10 GPU being detected by `nvidia-smi`. This is an environment issue requiring the correct Python runtime activation, not a script defect.

## Next Phase Readiness

- `scripts/run_eval_triage.py` is ready to execute on DGX Spark
- All dependencies verified: `scripts/profile_base_model.py`, `scripts/triage_ratios.py`, `eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py`
- Once executed, outputs feed directly into Plan 03 (04-03: human review of triage decision)
- Phase 7 (Router Profiling) is blocked on this plan completing -- triage result needed for winning gen/judge ratio

---
*Phase: 04-evaluation*
*Completed: 2026-04-02*

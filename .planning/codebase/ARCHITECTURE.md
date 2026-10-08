---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
<!-- refreshed: 2026-10-09 -->

# Architecture

**Analysis Date:** 2026-10-09

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                 Operator layer (Claude Code skills)                   │
│  `.claude/skills/wp-finetune:*/SKILL.md`  (run-*, observe-*, review-*)│
└───────────┬──────────────────────────┬────────────────────────────────┘
            │ invokes                  │ spawns Claude Code agents for LLM work
            ▼                          ▼
┌────────────────────────────┐  ┌────────────────────────────────────────┐
│ Orchestration / drivers    │  │ Stage scripts (one job per file)        │
│ `scripts/pipeline_         │  │ `scripts/phase1_*`, `phase2_*`, `phase3_*`│
│   orchestrator.py`         │  │ `scripts/build_*`, `assemble_*`,        │
│ `scripts/orchestrator.py`  │  │ `scripts/tinker_*`, `train_model.py`,   │
│ `scripts/rl_train.py`      │  │ `scripts/merge_*`, `scripts/sieve_*`,   │
│ `eval/eval_gate.py`        │  │ `scripts/aimer_prune*`, `scripts/prune_*`│
└─────────────┬──────────────┘  └───────────────┬────────────────────────┘
              │ imports                          │ imports
              ▼                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Shared libraries                                                      │
│ `scripts/utils.py` (JSON extract, checkpoints)                        │
│ `scripts/dgx_toolbox.py` (container resolve/validate/execute)         │
│ `scripts/claude_agent.py` (agent invocation)                          │
│ `scripts/reward_pipeline.py` (RL reward, VeRPO-style shaping)         │
│ `scripts/judge_functions.py` (heuristic 9-dimension scorers)          │
└──────────────┬───────────────────────────────────┬───────────────────┘
               ▼                                   ▼
┌───────────────────────────────┐   ┌──────────────────────────────────┐
│ Evaluation package `eval/`    │   │ External compute                  │
│ `rubric_scorer.py` (PHPCS,    │   │ DGX Spark (GB10) containers via   │
│   regex, floor rules)         │   │   `deps/dgx-toolbox` (vLLM, etc.) │
│ `eval_judge.py` (LLM judge)   │   │ Tinker cloud (LoRA SFT / GSPO)    │
│ `output_parsers.py`           │   │ llama.cpp / GGUF (packaging)      │
└──────────────┬────────────────┘   └──────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Artifacts (file-based, no database)                                   │
│ `config/` (YAML/MD)  `data/` (JSONL/JSON)  `output/` (receipts/models)│
│ `data/checkpoints/*.json` (phase state)                               │
└──────────────────────────────────────────────────────────────────────┘
```

## Pattern Overview

**Overall:** Staged, file-based ML pipeline (offline batch) with a Claude Code skill layer on top. There is no long-running service, web server, or database owned by this repo. Each stage reads files produced by earlier stages and writes new JSON/JSONL artifacts plus receipts.

**Key Characteristics:**
- Stages communicate only through files under `data/` and `output/`. Stage boundaries are the file formats, not function calls.
- Configuration is externalized to `config/*.yaml` (e.g. `config/train_config_v4.yaml`, `config/repos.yaml`, `config/dgx_toolbox.yaml`, `config/adaptive_planning.yaml`). Thresholds are meant to live in config, not in code (`config/adaptive_planning.yaml` header: "Do not hardcode in skills or scripts").
- Most stage scripts are standalone CLIs with `argparse` and an `if __name__ == "__main__":` block, run as `python -m scripts.<name>` or `python scripts/<name>.py`.
- LLM-heavy labelling steps run through Claude Code agents, not direct paid API loops (`README.md`, `PIPELINE.md` prerequisites).
- Compute is split across three backends: DGX Spark containers (via `scripts/dgx_toolbox.py`), Tinker cloud (`scripts/tinker_*.py`), and local llama.cpp/GGUF for packaging.
- Gates are first-class: every stage has a pass/fail gate, and "no winner" is a recorded valid outcome (`PIPELINE.md`, conditional gates A/B/C).

## Layers

**Operator skills (Claude Code):**
- Purpose: Turn a pipeline stage into a repeatable invocation with telemetry.
- Location: `.claude/skills/wp-finetune:*/SKILL.md` (13 skills). Mirror copies of several live in `docs/wp-finetune:*.md`.
- Contains: Markdown instructions, architecture notes, agent team assessments.
- Depends on: `scripts/` entry points, `config/`.
- Used by: The operator via `/wp-finetune:*` commands.
- Note: `docs/wp-finetune:run-training.md` and `.claude/skills/wp-finetune:run-training/SKILL.md` already differ, so the mirrors can drift from the live skill.

**Orchestration and drivers:**
- Purpose: Decide what to run next, or drive a multi-step run.
- Location: `scripts/pipeline_orchestrator.py` (status/plan for the data pipeline, `get_status`, `get_plan`, `compute_targets`); `scripts/orchestrator.py` (CoT/CTF batch spawn, merge, validate); `scripts/rl_train.py` (GSPO RL driver, 1767 lines, `run_training_step`, `check_halt`); `eval/eval_gate.py` (`run_gate`, `check_gates`).
- Depends on: Stage scripts, `scripts/utils.py`, `scripts/dgx_toolbox.py`.
- Used by: Operator skills and direct CLI.

**Stage scripts:**
- Purpose: One unit of pipeline work each (clone, extract, judge, generate, train, merge, profile, prune, eval).
- Location: `scripts/` (177 Python files, 14 shell scripts). Key families:
  - Data: `scripts/phase1_clone.py`, `phase1_extract.py`, `phase1_judge.py`, `phase2_gap_analysis.py`, `phase2_generate.py`, `phase2_judge.py`, `phase2_mutate.py`, `phase3_cot.py`, `generate_cot*.py`, `assemble_reasoning_dataset.py`, `build_*.py`.
  - Training: `scripts/train_model.py` (local transformers/Unsloth path), `scripts/tinker_reasoning_sft.py` (+ `_v4`), `scripts/tinker_rl_data.py`, `scripts/rl_train.py`.
  - Merge and compression: `scripts/merge_adapter.py`, `merge_tinker_v3.py`, `aimer_prune.py`, `aimer_prune_v4.py`, `reap_prune.py`, `prune_*.py`, `scripts/sieve_*.py`.
  - Eval drivers: `scripts/run_eval_reasoning.py`, `run_eval_triage.py`, `run_grid_eval.py`, `scripts/relabel/eval_relabel.py`, `bootstrap_gate.py`, `tost_gate.py`.
- Depends on: `eval/`, `scripts/utils.py`, `scripts/dgx_toolbox.py`, `scripts/reward_pipeline.py`.
- Used by: Orchestrators, skills, and other stage scripts (e.g. `scripts/rl_train.py` imports `scripts/rl_rollouts.py` and `scripts/rl_codegen_tripwire.py`).

**Shared libraries:**
- Purpose: Cross-cutting helpers.
- Location: `scripts/utils.py` (`extract_json` at line 19, `load_checkpoint`/`save_checkpoint` at lines 85/120), `scripts/dgx_toolbox.py` (`get_toolbox()` resolver, `ValidationResult`, container execute), `scripts/claude_agent.py`, `scripts/judge_functions.py` (`score_wpcs`, `score_sql`, `score_security`, … one per rubric dimension), `scripts/reward_pipeline.py` (`compute_reward` line 627, `compute_group_rewards` line 425).
- Depends on: `eval/` (reward_pipeline imports `eval.rubric_scorer` and `eval.eval_judge`).
- Used by: Nearly all stage scripts (`scripts.dgx_toolbox` is imported by 13 modules; `scripts.utils` by 12).

**Evaluation package:**
- Purpose: Score WordPress PHP against the 9-dimension rubric and gate model quality.
- Location: `eval/` (9 files).
  - `eval/rubric_scorer.py`: `score_code` (line 723) combines PHPCS (`run_phpcs`, line 197), PHPStan (`run_phpstan`, line 235), regex checks (`run_regex_checks`, line 441), and floor rules (`apply_floor_rules`, line 594) into a `RubricScore`.
  - `eval/eval_judge.py`: `judge_score_single` (line 386), `run_eval` (line 556); talks to an OpenAI-compatible judge endpoint.
  - `eval/output_parsers.py`: `parse_judge_scores` (line 156), `strip_think` (line 78). Handles judge output that may be JSON or prose.
  - `eval/rubric_definitions.py`: `CHECK_REGISTRY`, `CRITICAL_FLOOR_RULES`.
  - `eval/eval_gate.py`: threshold gates from YAML.
  - `eval/dim_map.json`: dimension mapping.
- Depends on: `scripts.dgx_toolbox` (`eval/eval_gate.py` line 24 and `eval/eval_judge.py` `_get_dgx`). This is a two-way package dependency with `scripts/` (see Architectural Constraints).
- Used by: `scripts/reward_pipeline.py`, `scripts/run_eval_*.py`, `scripts/relabel/*`, and `tests/`.

**Container and serving shims:**
- Purpose: Patch behaviour inside serving or eval containers without touching upstream code.
- Location: `scripts/_sieve_vllm_patch/sitecustomize.py` (applies the expert mask as a forward hook inside vLLM, duplicating math from `scripts/sieve_expert_mask_inference.py`), `scripts/_sieve_profile_vllm_patch/sitecustomize.py`, `scripts/_wpbench_pth/usercustomize.py` (forces `enable_thinking: false` for wp-bench), `scripts/_wpbench_shim/npx`.
- Depends on: Being on `PYTHONPATH` inside the container.
- Used by: `scripts/serve_*.sh`, wp-bench runs.

**Data and artifacts:**
- Purpose: Persist everything between stages.
- Location: `data/` (148 tracked files: seeds, relabel labels, reasoning datasets, RL prompt pools, calibration), `output/` (281 tracked files: eval receipts, packaging, prune and sieve results, `output/packaging/MODEL_CARD.md`), `data/checkpoints/` (runtime phase-state JSON, gitignored).
- Used by: All stages.

## Data Flow

### Primary request path: data pipeline to judge model

1. Clone and extract: `scripts/phase1_clone.py` reads `config/repos.yaml`; `scripts/phase1_extract.py` runs a PHP tokenizer helper (`scripts/php_extract_functions.php`) to pull function boundaries.
2. Heuristic and agent judging: `scripts/judge_functions.py` (`judge_function`, line 538) produces per-dimension scores; `scripts/phase1_judge.py` dispatches Claude agents for LLM judgement. WordPress Core is auto-passed (`config/repos.yaml`, `quality_tier: core`).
3. Synthetic fill: `scripts/phase2_gap_analysis.py` writes a coverage report, `scripts/phase2_generate.py` generates examples for gaps, `scripts/phase2_judge.py` judges them with the same bar.
4. Chain of thought: `scripts/phase3_cot.py` and `scripts/generate_cot*.py` attach reasoning. `scripts/orchestrator.py` batches this (`spawn`, `merge`, `validate`).
5. Assembly gate: `scripts/assemble_reasoning_dataset.py` and `scripts/validate_reasoning_consistency.py` reject examples whose reasoning contradicts the numeric score, then build the mix (`scripts/build_replay_mix.py`).
6. Train: `scripts/tinker_reasoning_sft.py --stage full --epochs 3 --seed {1,0,2}` runs LoRA SFT on Tinker (`config/train_config_v4.yaml` for v4 settings), or `scripts/train_model.py` runs the local path.
7. Merge and package: `scripts/merge_adapter.py` or `merge_tinker_v3.py` fuse adapters; packaging follows `scripts/run_packaging_recipe.md`.
8. Evaluate: `eval/eval_gate.py` runs thresholds from config; `scripts/run_eval_reasoning.py` and `eval/eval_judge.py` score outputs; receipts land in `output/`.

Stage progress is tracked by `scripts/pipeline_orchestrator.py`, which counts artifacts (`count_json_items`, `count_jsonl_lines`, line 58/72) and emits an ordered `actions` list from `get_plan` (line 432).

### RL reward path (conditional gate A)

1. `scripts/rl_train.py` `run_training_step` (line 946) calls `scripts/rl_rollouts.py` `collect_rollouts`.
2. Each rollout is scored by `scripts/reward_pipeline.py` `compute_group_rewards` (line 425) and `compute_reward` (line 627).
3. `compute_reward` calls `eval.rubric_scorer.score_code` for verifiable signals (PHPCS/regex) and `eval.eval_judge.judge_score_single` for the judge component.
4. `scripts/rl_codegen_tripwire.py` checks for codegen regression (Goodhart guard). `check_halt` (`scripts/rl_train.py` line 491) stops the run on the pre-registered kill criterion.

### Evaluation and triage path

1. `scripts/profile_base_model.py` and `scripts/compute_concentration.py` profile routing (E_eff).
2. `scripts/sieve_ksweep_run.py` / `sieve_ksweep_v4_run.py` sweep expert counts k. `scripts/bootstrap_gate.py` and `scripts/tost_gate.py` decide equivalence.
3. `scripts/run_eval_triage.py` produces a triage decision for human approval (`output/triage_decision.md`).

**State management:**
- Stateless between runs except for files. `scripts/utils.py` keeps per-phase JSON state in `data/checkpoints/` (`load_checkpoint`, `save_checkpoint`), and the RL driver writes manifests (`_write_manifest`, `scripts/rl_train.py` line 91).
- Calibration caches are module-level and loaded lazily (`eval/rubric_scorer.py` `_load_calibration`, line 115).

## Key Abstractions

**`RubricScore`:**
- Purpose: Result of scoring one PHP snippet across the 9 dimensions.
- Examples: `eval/rubric_scorer.py` line 69.
- Pattern: Dataclass returned by `score_code`; consumed by reward and eval code.

**`RewardBreakdown` / `RewardResult`:**
- Purpose: Per-rollout reward and its components.
- Examples: `scripts/reward_pipeline.py` lines 229 and 302.
- Pattern: Dataclasses. Reward is computed per group (`_verpo_group`, line 336), then normalised (`_mo_grpo_norm`, line 205).

**`DgxToolbox` / `ValidationResult` / `CheckResult`:**
- Purpose: Resolve a named component (e.g. `unsloth_studio`, `vllm`) from `config/dgx_toolbox.yaml`, validate preconditions, execute inside a container.
- Examples: `scripts/dgx_toolbox.py` (`get_toolbox`, dataclasses at lines near the top).
- Pattern: Facade. Callers use `dgx.validate([...])`, `dgx.ensure_ready(...)`, `dgx.execute(...)`, `dgx.status_report()`. Scripts never hard-code container commands.

**Dimension scorers:**
- Purpose: One deterministic heuristic per rubric dimension.
- Examples: `scripts/judge_functions.py` `score_wpcs` (line 65), `score_sql` (136), `score_security` (191), `score_performance` (265), `score_wp_api` (305), `score_code_quality` (348), `score_dependency` (383), `score_i18n` (401), `score_accessibility` (451).
- Pattern: Each returns `(score, list_of_reasons)`. `detect_training_tags` (line 484) maps findings to training-set tags.

**Gate:**
- Purpose: Decide ship or stop at each stage.
- Examples: `eval/eval_gate.py` `check_gates` (line 96), `scripts/bootstrap_gate.py`, `scripts/tost_gate.py`, `scripts/reward_validity_gate.py`.
- Pattern: Thresholds read from YAML (`load_thresholds`, line 45). Gate results are recorded as receipts, and a failing gate is a valid recorded outcome.

## Entry Points

**Data pipeline status and plan:**
- Location: `scripts/pipeline_orchestrator.py` (`__main__` at line 734).
- Triggers: Operator runs `python scripts/pipeline_orchestrator.py`.
- Responsibilities: Report counts against targets (`print_status`, `print_plan`), emit next steps.

**Stage scripts:**
- Location: `scripts/phase1_clone.py`, `scripts/phase1_extract.py`, `scripts/phase1_judge.py`, `scripts/phase2_*.py`, `scripts/phase3_cot.py`, and others.
- Triggers: `python -m scripts.<name>` as printed by `get_plan`.
- Responsibilities: One stage each.

**Training:**
- Location: `scripts/tinker_reasoning_sft.py` (cloud LoRA SFT), `scripts/rl_train.py` (`_parse_args` line 1235, `_dry_run` line 1495), `scripts/train_model.py` (`train` line 499, `_build_parser` line 634).
- Triggers: Operator skills `wp-finetune:run-training` and `wp-finetune:run-rl-training`.

**Evaluation:**
- Location: `eval/eval_gate.py` (`main` line 294), `eval/eval_judge.py` (`run_eval` line 556), `scripts/run_eval_reasoning.py`, `scripts/run_eval_triage.py`.
- Triggers: Operator skill `wp-finetune:run-evaluation`.

**Batch CoT driver:**
- Location: `scripts/orchestrator.py` (`main` line 304): subcommands for spawn, merge, validate.

**Tests:**
- Location: `tests/conftest.py` (fixtures, adds project root to `sys.path`), `tests/phase4_4/conftest.py`.

## Architectural Constraints

- **Threading:** Single-process Python scripts. Concurrency is at the level of containers, Tinker jobs, and Claude Code agent batches, not threads inside the repo. `scripts/rl_rollouts.py` collects rollouts in-process. No threading model is documented.
- **Global state:** Module-level caches and env handling.
  - `scripts/utils.py` `_DEFAULT_CHECKPOINT_DIR` (lazily set global).
  - `eval/rubric_scorer.py` `_load_calibration` (module-level calibration cache).
  - `scripts/reward_pipeline.py` pops environment variables at import time before importing `eval.*` ("Lazy imports after env-var suppression"). Import order matters here.
  - `scripts/reward_pipeline.py` `_RECALIB_PATH` and `_load_score_offset` (line 47).
- **Package dependency direction:** Not one-way. `eval/eval_gate.py` and `eval/eval_judge.py` import `scripts.dgx_toolbox`, and `scripts/reward_pipeline.py` imports `eval.*`. Keep `scripts/dgx_toolbox.py` free of `eval` imports to avoid a real cycle.
- **Duplicated logic:** The expert-mask math is duplicated on purpose between `scripts/sieve_expert_mask_inference.py` and `scripts/_sieve_vllm_patch/sitecustomize.py` (the container cannot import host modules). Keep the two in sync and run `tests/test_sieve_vllm_patch.py` after changes.
- **Frozen router:** The MoE router is kept frozen during SFT (`scripts/train_model.py` `assert_router_frozen_and_report`, line 459). Changes to training must preserve this.
- **Config paths:** Several scripts resolve paths from `config/*.yaml` via `resolve_path` (`scripts/train_model.py` line 45). Hard-coded paths in older scripts bypass this.

## Anti-Patterns

### Versioned sibling scripts instead of parameters

**What happens:** Behaviour changes are shipped as new files with version suffixes, e.g. `scripts/aimer_prune.py` and `scripts/aimer_prune_v4.py`, `scripts/tinker_reasoning_sft.py` and `scripts/tinker_reasoning_sft_v4.py`, `scripts/tinker_reasoning_data.py` and `scripts/tinker_reasoning_data_v4.py`, `scripts/prune_apply_physical.py` and `..._v4.py`, `scripts/eval4_ext_*` alongside `scripts/eval_judge_unsloth.py`.
**Why it's wrong:** Duplicated logic drifts. A fix in one copy does not reach the other, and it is unclear which copy the skills call.
**Do this instead:** Add a parameter or a config key in `config/` for the new behaviour, and keep one implementation. Where a sibling must stay for reproducibility, say so in its module docstring and point the skill at the current one.

### Underscore-prefixed one-off drivers inside the live `scripts/` tree

**What happens:** Single-experiment drivers sit beside pipeline code, e.g. `scripts/_p0_*.py`, `scripts/_pkg_ens8192_run.sh`, `scripts/_pub03_upload.sh`, `scripts/_probe_rl_reward.py`. `deprecated/scripts/` holds the older versions of the same kind of script.
**Why it's wrong:** The live pipeline and experiment scaffolding are indistinguishable by location, and `PIPELINE.md` says `deprecated/` is off the pipeline path.
**Do this instead:** Put new one-off experiments in `deprecated/` once finished, or in a dated `output/` receipt directory, not in `scripts/`.

### Hard-coded container commands

**What happens:** Scripts call `docker` or vLLM commands directly rather than going through the toolbox.
**Why it's wrong:** Bypasses precondition checks and telemetry in `scripts/dgx_toolbox.py`.
**Do this instead:** `dgx = get_toolbox(); dgx.validate([...]); dgx.execute("vllm", ...)`, as in the docstring of `scripts/dgx_toolbox.py`.

## Error Handling

**Strategy:** Fail closed at gates, fail soft at parse. Gates return a pass/fail result and a receipt. Parsers return `None` instead of raising when judge output is malformed, and the caller counts it as a parse failure.

**Patterns:**
- `scripts/utils.py` `extract_json` returns `Optional[dict]`.
- `eval/output_parsers.py` `parse_judge_scores` returns `None` on failure; parse-failure rates are reported (for example "0/121 parse failures" in `README.md`).
- Judge failures are dumped for inspection by `_dump_judge_failure` (`eval/eval_judge.py` line 228).
- Checkpointing: `scripts/utils.py` `save_checkpoint` lets a stage resume after a failure.

## Cross-Cutting Concerns

**Logging:** `print` and `logging` mix. `scripts/rl_train.py` uses `logging` with step metrics written to files (`_log_step`, line 759). Stage scripts mostly use `print`.

**Validation:** Config validation in `scripts/preflight.py` and `scripts/sieve_env_precheck.py`. Dataset validation in `scripts/validate_reasoning_consistency.py` and `scripts/validate_ctf_batch.py`. Memory and hardware checks in `scripts/train_model.py` `check_memory` (line 59) and `scripts/dgx_toolbox.py`.

**Authentication:** Judge LLM access uses an OpenAI-compatible endpoint configured by environment (`.env.example`). Credentials are never read into docs. Tinker access goes through the Tinker client. Claude Code agents use the operator's own subscription.

**Telemetry:** Operator skills spawn observer agents that write to `telemetry/<stage>/{timestamp}/` (per `.claude/skills/wp-finetune:observe-*/SKILL.md`). `wp-finetune:review-telemetry` consolidates those files. This layer is not represented in the repository tree; the directory is produced at run time.

---

*Architecture analysis: 2026-10-09*

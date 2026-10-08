---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
<!-- refreshed: 2026-10-09 -->

# Codebase Structure

**Analysis Date:** 2026-10-09

## Directory Layout

```
wp-finetune/
├── scripts/            # Pipeline stages, drivers, shared libs (177 .py, 14 .sh)
│   ├── relabel/        # Human-relabel prep, aggregation, eval (16 .py)
│   ├── _sieve_vllm_patch/          # sitecustomize inside vLLM container
│   ├── _sieve_profile_vllm_patch/  # sitecustomize for routing profile
│   ├── _wpbench_pth/               # usercustomize: disables Qwen thinking for wp-bench
│   └── _wpbench_shim/npx           # shim for wp-bench runner
├── eval/               # Rubric scorer, judge, parsers, gates (9 files)
├── tests/              # pytest suites (77 .py) + fixtures
│   ├── phase4_4/       # Subpackage with its own conftest.py
│   └── fixtures/       # PHP cases + sample CSVs
├── config/             # YAML and markdown configuration (18 files)
├── recipes/            # Serving recipes (vLLM YAML, 3 files)
├── data/               # Pipeline data artifacts (148 tracked files)
│   ├── calibration/    # Calibration train/holdout + audit
│   ├── checkpoints/    # Runtime phase state JSON (gitignored contents)
│   ├── phase1b/        # Rejudge runs
│   ├── phase4_4/       # Smoke prompts and baselines
│   ├── phase4_reasoning/  # CoT and CTF batches, pilots, manifests
│   ├── reasoning_dataset/ # SFT dataset splits and metadata
│   ├── relabel_v1/     # Human relabel labels and GT dims
│   ├── rl_probe/       # RL reward probe corpus
│   ├── rl_prompts/     # RL prompt pools + PROVENANCE.md
│   └── seeds/          # UGC seed sets and summaries
├── output/             # Receipts, eval results, packaging (281 tracked files)
│   ├── base20/, base21/   # Base-model bring-up receipts
│   ├── bench17/           # wp-bench results
│   ├── eval3/, eval4/     # Evaluation rounds
│   ├── packaging/         # GGUF/AWQ packaging, MODEL_CARD inputs, hf_cards
│   ├── pkg-v4/            # v4 packaging receipts (f16, q5, q6, q8)
│   ├── prune/, prune-v4/  # Expert prune gates and masks
│   ├── sieve/, sieve-v4/  # MoE-Sieve k-sweeps and protected masks
│   ├── relabel/           # Relabel eval results
│   └── rl_checkpoints/    # RL metrics
├── docs/               # Operator docs, rubric, v5 labelling kit (34 files)
│   ├── eval/              # Rubric and research notes
│   ├── v5-labelling/      # Extraction guide, labelling plan, corpus manifest
│   │   ├── labelled/              # Drop folder for labelled output
│   │   └── v4-judge-training-dataset/
│   └── wp-finetune:*.md   # Mirror copies of operator skill docs
├── deps/
│   ├── dgx-toolbox/       # Git submodule (empty in working tree)
│   └── wp-finetune-data/  # Seed data mirror and seed-collection prompts
├── deprecated/         # Frozen earlier campaigns (114 tracked files, off pipeline path)
│   ├── scripts/           # Old experiment drivers
│   ├── planning-handoffs/ # Old continue-here notes
│   └── planning-reviews/  # Old review docs
├── .planning/          # GSD planning state (phases, quick tasks, research, debug notes)
│   ├── phases/         # 20-27 phase folders with PLAN/SUMMARY/VERIFICATION
│   ├── quick/          # Quick-task folders
│   ├── research/       # Stack, features, architecture research
│   ├── debug/          # Debug knowledge base and resolved notes
│   ├── todos/pending/  # Pending todos
│   └── codebase/       # These mapping documents
├── .claude/skills/     # Operator skills: wp-finetune:* (13 SKILL.md)
├── README.md           # Project overview, model card links, quickstart
├── PIPELINE.md         # Frozen end-to-end method, gate by gate
├── PROJECT.md          # Project spec and status
├── JOURNAL.md          # Build log (decision record)
├── CHANGELOG.md        # Versioned changes
├── .env.example        # Environment variable template (no secrets)
├── .gitmodules         # Declares deps/dgx-toolbox submodule
└── .gitignore          # Ignores checkpoints, .env, models/, adapters/, caches
```

## Directory Purposes

**`scripts/`:**
- Purpose: Everything executable in the pipeline: stage CLIs, orchestrators, shared libraries, container shims.
- Contains: `.py` modules (most with `argparse` and `__main__`), `.sh` launchers for serving and eval (`serve_*_vllm.sh`, `eval4_ext_*.sh`, `_pkg_*_run.sh`), one PHP helper (`php_extract_functions.php`), and a prompt-ish recipe `run_packaging_recipe.md`.
- Key files:
  - `scripts/pipeline_orchestrator.py`: data pipeline status and next actions.
  - `scripts/orchestrator.py`: batch CoT/CTF driver.
  - `scripts/utils.py`: `extract_json`, checkpoint load/save.
  - `scripts/dgx_toolbox.py`: container resolver and executor.
  - `scripts/reward_pipeline.py`: RL reward.
  - `scripts/rl_train.py`: GSPO RL driver.
  - `scripts/train_model.py`: local training path.
  - `scripts/tinker_reasoning_sft.py`: cloud SFT driver.
  - `scripts/judge_functions.py`: heuristic dimension scorers.

**`eval/`:**
- Purpose: The judge's scoring contract. Used by evaluation, RL reward, and relabel.
- Contains: `rubric_scorer.py`, `eval_judge.py`, `output_parsers.py`, `rubric_definitions.py`, `eval_gate.py`, `eval_gen.py`, `llm_checks.py`, `dim_map.json`, `__init__.py`.
- Key files: `eval/rubric_scorer.py` (`score_code` line 723), `eval/eval_judge.py` (`run_eval` line 556), `eval/eval_gate.py` (`main` line 294).

**`tests/`:**
- Purpose: pytest suites mirroring `scripts/` and `eval/` modules by name.
- Contains: `tests/test_<module>.py` (about 65 top-level), `tests/phase4_4/test_*.py` (13 modules with their own conftest), shared fixtures `tests/_rl_fixtures.py`, `tests/conftest.py`, and `tests/fixtures/`.
- Key files: `tests/conftest.py` (adds project root to `sys.path`), `tests/test_reward_pipeline.py`, `tests/test_rl_train.py`, `tests/test_eval_gate.py`, `tests/test_sieve_vllm_patch.py`.

**`config/`:**
- Purpose: All tunables. Scripts read these instead of hard-coding values.
- Contains: Train configs, eval thresholds, repo list, rubric, taxonomy, prompts, DGX toolbox component map.
- Key files:
  - `config/train_config_v4.yaml`: current training config (LoRA r=32, alpha 64, eval targets).
  - `config/train_config_{30_70,40_60,50_50,60_40,70_30}.yaml`: task-mix sweep configs.
  - `config/train_config_reasoning.yaml`, `config/train_config_qwen3-30b_experiment_001_20260411.yaml`: experiment-specific.
  - `config/repos.yaml`: repos to clone with `quality_tier` (core, assessed, poor).
  - `config/judge_system.md`: 9-dimension rubric the judge follows.
  - `config/rubric_calibration.yaml`: calibration parameters.
  - `config/taxonomy.yaml`, `config/synthetic_prompts.yaml`: data generation inputs.
  - `config/dgx_toolbox.yaml`: component paths for containers.
  - `config/adaptive_planning.yaml`: power, thermal, and memory thresholds.
  - `config/wp-bench.yaml`: wp-bench dataset, model endpoint, output paths.
  - `config/requirements-profiling.txt`: profiling dependencies.

**`recipes/`:**
- Purpose: Serving recipes for vLLM.
- Contains: `qwen3-30b-wp-30_70-vllm.yaml`, `qwen3.6-35b-a3b-vllm.yaml`, `qwen3.6-35b-a3b-fp8-vllm.yaml`.

**`data/`:**
- Purpose: Inputs and intermediate datasets. Large or regenerable outputs are gitignored at `data/phase1_extraction/`, `data/phase2_synthetic/`, `data/phase3_cot/`, `data/final_dataset/`.
- Contains: JSONL and JSON splits, labels, seeds, and audit transcripts.
- Key files: `data/reasoning_dataset/openai_train.jsonl`, `openai_val.jsonl`, `holdout_v1.jsonl`, `metadata.json`; `data/relabel_v1/labels.json`, `gt_dims_train.json`; `data/rl_prompts/PROVENANCE.md`.
- Note: several sibling variants exist (`openai_train.augmented.jsonl`, `openai_train.pre_vendorfilter.jsonl`, `openai_train_relabel_v1.augmented.replay30.jsonl`). Check `metadata.json` before choosing one.

**`output/`:**
- Purpose: Evidence of what each run produced: receipts, gate verdicts, model cards, packaged artifacts.
- Contains: Per-campaign directories (`base20`, `bench17`, `eval3`, `eval4`, `pkg-v4`, `prune-v4`, `sieve-v4`, `rl_eval`, `packaging`, `tinker`), plus top-level markdown decision notes (`output/triage_decision.md`, `output/phase1b_*_review*.md`).
- Convention: Each campaign directory holds its own receipts. Do not overwrite a prior campaign's directory. Create a new one.

**`docs/`:**
- Purpose: Operator and method docs that are not the top-level project files.
- Key files: `docs/AGENT_PIPELINE.md` (agent execution model), `docs/eval/wp_code_quality_rubric.md`, `docs/eval/research_wpcs_standards.md`, `docs/eval/research_wp_security_sql_perf.md`, `docs/v5-labelling/EXTRACTION_GUIDE.md`, `docs/v5-labelling/LABELING_PLAN.md`, `docs/v5-labelling/corpus_manifest.jsonl` (+ `.schema.json`), `docs/v5-labelling/overlap_receipt.json` (gate file, currently pending).
- Note: `docs/wp-finetune:*.md` are colon-named mirrors of the skill files. Edit the `.claude/skills/` copy first.

**`deps/`:**
- Purpose: External or vendored inputs.
- `deps/dgx-toolbox/`: git submodule (`.gitmodules`). The working-tree directory is currently empty, so `scripts/dgx_toolbox.py` resolves to a checkout at `dgx_toolbox_path` (default `~/dgx-toolbox`, `config/dgx_toolbox.yaml`) or `DGX_TOOLBOX_PATH`. Run `git submodule update --init` before expecting `deps/dgx-toolbox` contents.
- `deps/wp-finetune-data/`: seed JSON/CSV mirror and seed-collection prompts (`UGC_COLLECTION_PROMPT.md`).
- `deps/DGX_TOOLBOX_ISSUES.md`: issues filed against the toolbox.

**`deprecated/`:**
- Purpose: Frozen scaffolding from earlier campaigns and superseded specs. Off the pipeline path. Keep for provenance.
- Contains: `deprecated/scripts/` (`_04.4_*`, `_p0_*`, `_rlev01_*`, `bench17_*`), `deprecated/planning-handoffs/`, `deprecated/planning-reviews/`, and `deprecated/README.md`.

**`.claude/skills/`:**
- Purpose: Claude Code operator skills for each stage.
- Contains: 13 directories named `wp-finetune:<name>/SKILL.md`.
- Naming: Directory name includes a colon (`wp-finetune:run-training`). Skill names match these directory suffixes.

**`.planning/`:**
- Purpose: GSD workflow state. Not part of the ML pipeline.
- Contains: `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `config.json`, `phases/` (per-phase `NN-PLAN.md`, `NN-SUMMARY.md`, `NN-VERIFICATION.md`, `CONTEXT.md`, `RESEARCH.md`), `quick/`, `research/`, `debug/`, `todos/`, `codebase/`.
- Phase folders are named `NN-<slug>` (`20-base-bring-up`, `27-packaging-publication-refresh`).

## Key File Locations

**Entry Points:**
- `scripts/pipeline_orchestrator.py`: `python scripts/pipeline_orchestrator.py` prints status and plan.
- `scripts/phase1_clone.py`, `scripts/phase1_extract.py`, `scripts/phase1_judge.py`: data pipeline phase 1 (`python -m scripts.<name>`).
- `scripts/tinker_reasoning_sft.py`: cloud SFT (`--stage`, `--epochs`, `--seed`).
- `scripts/rl_train.py`: RL (`_parse_args` line 1235; `--dry-run` path `_dry_run` line 1495).
- `scripts/train_model.py`: local training.
- `eval/eval_gate.py`: gate CLI.

**Configuration:**
- `config/*.yaml`: see Directory Purposes above.
- `.env.example`: lists required environment variables. Copy to `.env` locally. Never commit `.env`.

**Core Logic:**
- `eval/rubric_scorer.py`: scoring.
- `scripts/reward_pipeline.py`: RL reward.
- `scripts/judge_functions.py`: heuristics.
- `scripts/assemble_reasoning_dataset.py`: dataset assembly and gate.
- `scripts/merge_adapter.py`, `scripts/merge_tinker_v3.py`: LoRA merge.

**Testing:**
- `tests/conftest.py`: shared fixtures.
- `tests/test_<module>.py`: unit tests named after the module under test.
- `tests/phase4_4/`: REVL and fidelity tests.

## Naming Conventions

**Files (Python):**
- Pipeline phases: `phase<N>_<step>.py` (`phase1_clone.py`, `phase2_judge.py`, `phase3_cot.py`). Phase 4 and later use other prefixes.
- Build outputs: `build_<artifact>.py` (`build_reasoning_negatives.py`, `build_rl_prompts.py`).
- Audits: `audit_<thing>.py` (`audit_dropped_seeds.py`).
- Eval gates: `*_gate.py` (`eval_gate.py`, `bootstrap_gate.py`, `tost_gate.py`, `reward_validity_gate.py`).
- Provider-specific drivers: `tinker_*.py` for Tinker cloud, `*_vllm.py` or `serve_*_vllm.sh` for vLLM, `eval4_ext_*` for the v4 extension evaluation.
- Versioned siblings: `*_v4.py` marks the v4 variant (`aimer_prune_v4.py`). Prefer parameters over new siblings (see ARCHITECTURE.md Anti-Patterns).
- Experiment-scoped: `_p0_*`, `_rlev*`, `revl*_*`, `_pkg_*`, `_pub*_upload.sh`. Leading underscore means one-off driver.
- Tests: `test_<module>.py` matching the module name.

**Files (other):**
- Config: `<purpose>.yaml`, `train_config_<mix>.yaml` where `<mix>` is `30_70`, `50_50`, etc. (task-token ratio).
- Data: `openai_<split>.jsonl` for chat-format splits. `*.augmented.*`, `*.replay30.*`, `*.pre_vendorfilter.*` suffix variants.
- Docs: UPPERCASE.md for top-level and method docs (`PIPELINE.md`, `README.md`). `docs/v5-labelling/` uses UPPER_SNAKE (`EXTRACTION_GUIDE.md`, `LABELING_PLAN.md`).
- Phase artifacts: `NN-MM-PLAN.md`, `NN-MM-SUMMARY.md`, `NN-VERIFICATION.md` under `.planning/phases/NN-slug/`.

**Directories:**
- Output campaigns: lowercase, hyphen or underscore (`pkg-v4`, `prune-v4`, `eval_reasoning_v4_winner`). Version tag in the name.
- Skill directories: `wp-finetune:<kebab-name>`.

## Where to Add New Code

**New pipeline stage:**
- Implementation: `scripts/<stage_name>.py` with an `argparse` `main()` and `if __name__ == "__main__":`. Read paths from `config/` and use `scripts/utils.py` for checkpoints.
- Config: add a block to the appropriate `config/*.yaml`.
- Tests: `tests/test_<stage_name>.py`.
- Operator entry: add or update the matching `.claude/skills/wp-finetune:<name>/SKILL.md`.

**New evaluation or rubric logic:**
- Scorer or check: `eval/rubric_scorer.py` (`score_code`), with check definitions in `eval/rubric_definitions.py` (`CHECK_REGISTRY`).
- Judge parsing: `eval/output_parsers.py`.
- Tests: `tests/test_eval_*.py` (name-matched).

**New RL reward component:**
- Implementation: `scripts/reward_pipeline.py` (add to `compute_reward` or `_verpo_group`).
- Tests: `tests/test_reward_pipeline.py` and `tests/test_reward_pipeline_integration.py`.
- Note: keep the env-var suppression block above the `eval` imports.

**New training run or config variant:**
- Config: `config/train_config_<name>.yaml`. Do not edit `train_config_v4.yaml` for an experiment.
- Driver: reuse `scripts/tinker_reasoning_sft.py` or `scripts/train_model.py`. Add a flag before creating a new `*_v<N>.py`.

**New dataset build:**
- Implementation: `scripts/build_<artifact>.py`.
- Output: `data/<purpose>/` for inputs and `output/<campaign>/` for receipts.

**New experiment or campaign:**
- Write receipts to a new `output/<campaign-name>/` directory. Do not reuse an existing one.
- Finished one-off drivers go to `deprecated/scripts/` once the campaign closes.

**Utilities:**
- Shared JSON and checkpoint helpers: `scripts/utils.py`.
- Container resolution: `scripts/dgx_toolbox.py`. Do not hard-code container paths.

**Tests for a new module:**
- `tests/test_<module>.py`. Use `tests/_rl_fixtures.py` for RL fixtures and `tests/fixtures/` for PHP and CSV samples.

## Special Directories

**`data/checkpoints/`:**
- Purpose: Runtime phase state written by `scripts/utils.py` `save_checkpoint`.
- Generated: Yes (at run time).
- Committed: Contents are gitignored (`data/checkpoints/*.json`, `*.tmp`). `.gitkeep` is tracked.

**`deps/dgx-toolbox/`:**
- Purpose: DGX Toolbox execution engine (git submodule).
- Generated: No.
- Committed: Submodule pointer only. Working-tree content is empty until initialised.

**`deprecated/`:**
- Purpose: Frozen earlier campaigns.
- Generated: No.
- Committed: Yes. Do not import from here.

**`output/`:**
- Purpose: Run receipts and packaged artifacts.
- Generated: Yes, by pipeline runs.
- Committed: Yes, for receipts. Model weights are gitignored (`models/`, `adapters/`, `merged_model/`).

**`.planning/codebase/`:**
- Purpose: Architecture and mapping docs for GSD commands.
- Generated: Yes, by `/gsd-map-codebase`.
- Committed: Yes.

**Gitignored at root (run-time, not in tree):** `models/`, `adapters/`, `merged_model/`, `data/phase1_extraction/`, `data/phase2_synthetic/`, `data/phase3_cot/`, `data/final_dataset/`, `graphify-out/`. Their absence from the tree is expected.

---

*Structure analysis: 2026-10-09*

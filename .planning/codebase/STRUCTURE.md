# Codebase Structure

**Analysis Date:** 2026-03-26

## Directory Layout

```
wp-finetune/
├── config/                              # Configuration files (YAML + prompts)
├── scripts/                             # Python pipeline scripts (10 phases total)
├── phase1_extraction/                   # Phase 1 repository cloning + extraction output
│   ├── repos/                           # Cloned WordPress repos (git directories)
│   └── output/
│       ├── extracted/                   # Raw extracted functions (JSON per repo)
│       ├── passed/                      # Quality-assessed passed functions
│       └── failed/                      # Functions that failed assessment
├── phase2_synthetic/                    # Phase 2 gap analysis, generation, mutation
│   ├── gap_report.json                  # Coverage analysis snapshot
│   └── output/
│       ├── generated/                   # Claude-generated synthetic examples
│       ├── judged/                      # Judged synthetic code (passed/failed)
│       ├── mutated/                     # Automated contrastive pairs (bad→good)
│       └── judge_training/              # Rubric-scored judge training data
├── phase3_cot/                          # Phase 3 chain-of-thought output
│   └── output/                          # CoT checkpoints before final merge
├── final_dataset/                       # Final training dataset (all formats)
│   ├── metadata.json                    # Dataset statistics + composition
│   ├── openai_train.jsonl               # OpenAI finetuning format
│   ├── openai_val.jsonl
│   ├── openai_test.jsonl
│   ├── alpaca_train.json                # Alpaca/Llama-MoE format
│   ├── alpaca_val.json
│   ├── alpaca_test.json
│   ├── raw_train.jsonl                  # Full metadata format
│   ├── raw_val.jsonl
│   └── raw_test.jsonl
├── .planning/                           # Planning & documentation (output of /gsd commands)
│   └── codebase/                        # Codebase analysis documents
├── PROJECT.md                           # Full project specification (phases A-E)
├── README.md                            # Quick start guide
└── wp-moe.md                            # Model architecture specification
```

## Directory Purposes

**config/**
- Purpose: Configuration files controlling pipeline behavior
- Contains: YAML repository list, judge criteria, concept taxonomy, prompt templates
- Key files: `repos.yaml` (user-editable), `judge_system.md` (9-dimension rubric), `taxonomy.yaml` (concept coverage), `synthetic_prompts.yaml` (generation templates)

**scripts/**
- Purpose: Python 3.10+ pipeline scripts implementing all phases
- Contains: Repository cloning, extraction, judgment, gap analysis, synthesis, mutation, export
- Key files: `phase1_*.py`, `phase2_*.py`, `phase3_*.py`, `export_dataset.py`, `php_extract_functions.php` (PHP helper)

**phase1_extraction/**
- Purpose: Repository sources and extracted function outputs
- Contains: Git repositories (shallow-cloned), raw extracted JSON, assessed functions
- Key files: `output/extracted/` (per-repo JSON), `output/passed/` (training-ready), `output/failed/` (analysis)

**phase2_synthetic/**
- Purpose: Gap analysis results, generated/mutated code, judge training data
- Contains: Gap report, synthetic generated code, mutation pairs, rubric-scored examples
- Key files: `gap_report.json` (drives generation), `output/generated/` (Claude-produced), `output/mutated/` (automated), `output/judge_training/` (judge examples)

**phase3_cot/**
- Purpose: Intermediate checkpoint before final merge and export
- Contains: CoT reasoning outputs, instruction synthesis results
- Key files: `output/` (before merging into final_dataset)

**final_dataset/**
- Purpose: Training-ready dataset in multiple formats
- Contains: 3 formats × 3 splits = 9 files total, plus metadata
- Key files: `openai_train.jsonl` (OpenAI API), `alpaca_train.json` (Llama-MoE), `raw_train.jsonl` (full metadata), `metadata.json` (statistics)

## Key File Locations

**Entry Points:**
- `scripts/phase1_clone.py`: Begin repository curation
- `scripts/phase1_extract.py`: Extract functions from cloned repos
- `scripts/phase1_judge.py`: Assess code quality and assign taxonomy tags
- `scripts/phase2_gap_analysis.py`: Analyze coverage gaps
- `scripts/phase2_mutate.py`: Generate contrastive mutation pairs
- `scripts/phase2_generate.py`: Generate synthetic code filling gaps
- `scripts/phase2_judge.py`: Judge synthetic examples
- `scripts/phase2_judge_dataset.py`: Create judge training data with rubric scores
- `scripts/phase3_cot.py`: Generate instructions and CoT reasoning
- `scripts/export_dataset.py`: Final formatting, task tokens, train/val/test split

**Configuration:**
- `config/repos.yaml`: User-editable list of WordPress core + plugins/themes (quality_tier, path filters)
- `config/judge_system.md`: 9-dimension judge criteria (WPCS, SQL safety, security, performance, API, code quality, dependencies, i18n, accessibility)
- `config/taxonomy.yaml`: Hierarchical concept taxonomy (11 categories: sql_patterns, security, hooks, data_modeling, rest_api, admin, theme, performance, plugin_architecture, multisite, cron)
- `config/synthetic_prompts.yaml`: Prompt templates for generation, keyed by gap tag

**Core Data Pipelines:**
- `scripts/phase1_extract.py`: Reads from `config/repos.yaml`, writes to `phase1_extraction/output/extracted/`
- `scripts/phase1_judge.py`: Reads extracted JSON, consults `config/judge_system.md`, writes passed/failed
- `scripts/phase2_generate.py`: Reads Phase 1 passed, gap report, `config/synthetic_prompts.yaml`, generates via Claude
- `scripts/phase2_mutate.py`: Reads Phase 1 passed, applies deterministic mutations, verifies with PHPCS
- `scripts/phase2_judge_dataset.py`: Reads judged examples, generates rubric-scored judge training data
- `scripts/phase3_cot.py`: Reads all Phase 1 + 2 outputs, generates instruction/reasoning, merges data
- `scripts/export_dataset.py`: Reads from `final_dataset/wordpress_finetune.jsonl`, converts formats, splits 80/10/10

**Testing (Implicit):**
- Each phase outputs JSON with consistent schema
- PHPCS verification for mutation pairs (bad versions must fail, good versions must pass)
- Judge output sanity checks (verdict is PASS/FAIL, scores 0-10, critical_failures list)

## Naming Conventions

**Files:**
- Python scripts: `phase{N}_{operation}.py` (e.g., `phase1_extract.py`, `phase2_generate.py`)
- Config files: `{purpose}.yaml` or `{purpose}.md` (e.g., `repos.yaml`, `judge_system.md`)
- Output directories: `{operation}/` matching script name or phase (e.g., `extracted/`, `passed/`, `generated/`)
- Data files: `{format}_{context}.{ext}` (e.g., `openai_train.jsonl`, `alpaca_val.json`, `raw_test.jsonl`)

**Directories:**
- Phase directories: `phase{N}_{operation}/` (e.g., `phase1_extraction`, `phase2_synthetic`, `phase3_cot`)
- Data organization: `input/`, `output/`, `repos/`
- Semantic grouping: Pass/fail/extracted/generated/mutated/judged

**Variables (in Python scripts):**
- Paths: `SCREAMING_SNAKE_CASE` for module-level constants (e.g., `PROJECT_ROOT`, `REPOS_DIR`, `EXTRACTED_DIR`)
- Functions: `snake_case` (e.g., `extract_repo()`, `load_style_anchors()`, `build_prompt()`)
- Config dicts: `snake_case` keys from YAML (e.g., `quality_tier`, `skip_paths`)
- Rate limiting: `REQUESTS_PER_MINUTE`, `REQUEST_INTERVAL`

## Where to Add New Code

**New Pipeline Phase:**
- Create `scripts/phase{N}_{operation}.py`
- Use existing pattern: load config/previous output, process in batches, write to `phase{N}_*/output/{operation}/`
- Implement rate limiting if making API calls (`REQUESTS_PER_MINUTE`, `time.sleep()`)
- Add entry point to README.md quick start

**New Judge Dimension:**
- Edit `config/judge_system.md` (add dimension 10, 11, etc.)
- Update `scripts/phase1_judge.py` and `scripts/phase2_judge.py` response schema validation
- Update `scripts/phase3_cot.py` if dimension affects CoT reasoning

**New Taxonomy Category:**
- Add to `config/taxonomy.yaml` under `categories:`
- Update auto-tagging logic in `phase1_judge.py` `auto_tag_function()`
- Create prompt templates in `config/synthetic_prompts.yaml` for new category

**New Generation/Mutation Type:**
- Mutation: Add function like `mutate_remove_prepare()` in `scripts/phase2_mutate.py`, register in `MUTATIONS` list
- Generation: Add template to `config/synthetic_prompts.yaml`, Claude picks appropriate templates based on gap_tag
- Update `phase2_judge.py` to apply same quality criteria to new types

**Output Format Addition:**
- Edit `scripts/export_dataset.py`
- Implement `to_{format}_format()` function following OpenAI/Alpaca patterns
- Call in `main()` loop with condition on format type
- Document in README.md

## Special Directories

**phase1_extraction/repos/**
- Purpose: Git repositories (cloned shallow, depth=1)
- Generated: Yes (by `phase1_clone.py`)
- Committed: No (`.gitignore` excludes `/repos/`)
- Cleanup: Safe to delete; will be re-cloned on next `phase1_clone.py` run

**phase1_extraction/output/extracted/**
- Purpose: Raw extracted function JSON (per-repo files)
- Generated: Yes (by `phase1_extract.py`)
- Committed: No (intermediate output)
- Checkpointing: Contains function metadata used to seed judgment phase

**phase1_extraction/output/{passed,failed}/**
- Purpose: Quality-assessed functions (terminal Phase 1 output)
- Generated: Yes (by `phase1_judge.py`)
- Committed: No (training data, too large)
- Checkpointing: Used by Phase 2 as style anchors + base for mutations

**phase2_synthetic/gap_report.json**
- Purpose: Coverage analysis snapshot (which taxonomy tags have insufficient examples)
- Generated: Yes (by `phase2_gap_analysis.py`)
- Committed: No (intermediate report)
- Checkpointing: Seeds `phase2_generate.py` template selection

**phase2_synthetic/output/{generated,mutated,judge_training}/**
- Purpose: Synthetic code outputs (Phase 2 results)
- Generated: Yes (by `phase2_generate.py`, `phase2_mutate.py`, `phase2_judge_dataset.py`)
- Committed: No (training data)
- Checkpointing: Inputs to Phase 3 merging

**phase3_cot/output/**
- Purpose: CoT reasoning checkpoints before final export
- Generated: Yes (by `phase3_cot.py`)
- Committed: No (intermediate)
- Checkpointing: Inspection point for quality control before export

**final_dataset/**
- Purpose: Training-ready dataset (terminal output for all phases)
- Generated: Yes (by `export_dataset.py`)
- Committed: No (dataset too large for git)
- Checkpointing: Ready for Unsloth Studio fine-tuning or OpenAI API

**.planning/codebase/**
- Purpose: Codebase analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by `/gsd:map-codebase` command)
- Committed: Yes (documentation)

---

*Structure analysis: 2026-03-26*

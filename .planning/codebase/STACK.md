# Technology Stack

**Analysis Date:** 2026-03-26

## Languages

**Primary:**
- Python 3.10+ - Data pipeline orchestration, all phases of extraction, judging, generation, and export
- PHP - Code analysis via PHP tokenizer for function extraction and PHPCS compliance checking
- JSON - Structured data interchange between pipeline stages
- YAML - Configuration files (repos, taxonomy, prompts, judge system)

**Secondary:**
- Markdown - Documentation and system prompts

## Runtime

**Environment:**
- Python 3.10+ (required by core scripts and SDK dependencies)
- PHP CLI (required for tokenization and PHPCS execution)
- Composer (for managing PHP_CodeSniffer dependencies)

**Package Manager:**
- pip (Python package manager)
- Composer (PHP package manager)

## Frameworks

**Core:**
- No traditional web frameworks - this is a data pipeline, not a web service
- anthropic Python SDK - Claude API integration for code judging and generation
- pyyaml - YAML configuration parsing

**Extraction & Analysis:**
- PHP tokenizer extension (built-in to PHP) - function boundary detection, dependency extraction
- PHP_CodeSniffer - WordPress Coding Standards compliance checking via `phpcs` CLI
- WordPress-Coding-Standards (WPCS) - Rulesets for `phpcs`

**Build/Dev:**
- Standard Python subprocess for Git operations and external tool execution

## Key Dependencies

**Critical (required):**
- anthropic - Python SDK for Claude API access (`scripts/phase1_judge.py`, `phase2_generate.py`, `phase2_judge.py`, `phase2_judge_dataset.py`, `phase3_cot.py`)
- pyyaml - YAML parsing for `config/repos.yaml`, `config/taxonomy.yaml`, `config/synthetic_prompts.yaml` (`scripts/phase1_clone.py`, `phase1_extract.py`, `phase1_judge.py`, `phase2_gap_analysis.py`, `phase2_generate.py`)

**Infrastructure (external tools, not Python packages):**
- git (shallow cloning of repositories in `scripts/phase1_clone.py`)
- phpcs (PHP_CodeSniffer CLI for WPCS compliance checking in `scripts/phase1_judge.py`)
- php (CLI executable with tokenizer extension enabled)

## Configuration

**Environment:**
- ANTHROPIC_API_KEY - Required environment variable for Claude API access
  - Set via shell before running scripts
  - Used by `anthropic.Anthropic()` initialization in phase1_judge.py, phase2_generate.py, phase2_judge.py, phase2_judge_dataset.py, phase3_cot.py

**Build:**
- No traditional build system
- Python scripts executed directly via `python scripts/phase*.py`
- Installation of dependencies documented in README.md:
  ```bash
  pip install anthropic pyyaml
  composer global require squizlabs/php_codesniffer wp-coding-standards/wpcs
  ```

## Platform Requirements

**Development:**
- Linux/macOS/Windows with Python 3.10+ and PHP 7.4+ (with tokenizer extension)
- PHPCS and WordPress-Coding-Standards installed and accessible in PATH
- 500MB+ disk space for cloned repositories (phase1_extraction/repos/)
- Network access to GitHub for repository cloning
- Network access to Anthropic API (api.anthropic.com)

**Production (Target Infrastructure):**
- DGX Spark (Blackwell GB10, 128GB unified memory) via DGX Toolbox
- Runs data pipeline to generate training datasets for subsequent model training phases
- No external service dependencies beyond Anthropic API during pipeline execution

## Data Formats

**Input:**
- YAML files for configuration (`config/repos.yaml`, `config/taxonomy.yaml`, `config/synthetic_prompts.yaml`)
- Markdown for judge system instructions (`config/judge_system.md`)
- PHP source code from cloned repositories

**Intermediate (Pipeline States):**
- JSON files for extracted functions (`phase1_extraction/output/extracted/`)
- JSON files for assessment results (`phase1_extraction/output/passed/`, `phase1_extraction/output/failed/`)
- JSON files for gap analysis (`phase2_synthetic/gap_report.json`)
- JSON files for generated synthetic examples (`phase2_synthetic/output/generated/`)
- JSON files for mutated code pairs (`phase2_synthetic/output/mutated/`)
- JSON files for judge training data (`phase2_synthetic/output/judge_training/`)

**Output:**
- JSONL (line-delimited JSON) - Final training data with CoT (`final_dataset/wordpress_finetune.jsonl`)
- JSONL (OpenAI format) - `final_dataset/openai_{train,val,test}.jsonl`
- JSON (Alpaca format) - `final_dataset/alpaca_{train,val,test}.json`
- JSONL (raw with metadata) - `final_dataset/raw_{train,val,test}.jsonl`
- JSON metadata - `final_dataset/metadata.json`

## API Clients & Endpoints

**Anthropic Claude API:**
- Endpoint: api.anthropic.com (implicit in SDK)
- Models used:
  - `claude-sonnet-4-6-20250514` - Most judgments, instruction synthesis, basic generation
  - `claude-opus-4-6-20250514` - Chain-of-thought reasoning (more expensive, better quality)
- Rate limiting: 40-50 requests per minute enforced in code via `REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE`

**Git (GitHub):**
- Shallow clones from GitHub URLs specified in `config/repos.yaml`
- Read-only access (no push/PR operations)
- Uses git CLI via subprocess

## Caching & Storage

**Local Filesystem:**
- `phase1_extraction/repos/` - Cloned source repositories
- `phase1_extraction/output/` - Extracted and assessed functions
- `phase2_synthetic/output/` - Generated, judged, and mutated code
- `phase3_cot/output/` - CoT processing checkpoints (every 500 examples)
- `final_dataset/` - Final training dataset in multiple formats

**No Database:**
- All data persisted as JSON/JSONL files
- No SQL database, cache server, or external storage service

---

*Stack analysis: 2026-03-26*

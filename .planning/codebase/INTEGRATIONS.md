# External Integrations

**Analysis Date:** 2026-03-26

## APIs & External Services

**Anthropic Claude API:**
- Primary integration for code quality assessment, generation, and reasoning
- SDK: `anthropic` Python package
- Auth: Environment variable `ANTHROPIC_API_KEY`
- Usage locations:
  - `scripts/phase1_judge.py` - Code quality assessment (9-dimension rubric)
  - `scripts/phase2_generate.py` - Synthetic code generation for taxonomy gaps
  - `scripts/phase2_judge.py` - Quality judgment of synthetic examples
  - `scripts/phase2_judge_dataset.py` - Rubric scoring for judge training data
  - `scripts/phase3_cot.py` - Instruction synthesis and chain-of-thought reasoning

**Models Used:**
- claude-sonnet-4-6-20250514 (primary) - Most tasks (judging, basic generation, instruction synthesis)
- claude-opus-4-6-20250514 (premium) - Chain-of-thought reasoning where quality is critical

**GitHub (Repository Hosting):**
- Source: Repository URLs in `config/repos.yaml`
- Purpose: Clone WordPress plugins, themes, and WordPress Core
- Method: Git CLI shallow clone (--depth=1)
- Auth: Public repositories (no authentication required); private repos would need SSH key setup
- Usage: `scripts/phase1_clone.py` clones all configured repositories

## Data Storage

**Databases:**
- Not used - all state persisted to local filesystem

**File Storage:**
- Local filesystem only
- Directory structure:
  - `phase1_extraction/repos/` - Cloned repositories (max ~1GB+ depending on repo count)
  - `phase1_extraction/output/extracted/` - Raw extracted functions (JSON)
  - `phase1_extraction/output/passed/` - Assessed passing functions (JSON)
  - `phase1_extraction/output/failed/` - Assessed failing functions (JSON)
  - `phase2_synthetic/output/generated/` - Claude-generated synthetic examples (JSON)
  - `phase2_synthetic/output/judged/` - Judged synthetic examples (JSON)
  - `phase2_synthetic/output/mutated/` - Automated mutation contrastive pairs (JSON)
  - `phase2_synthetic/output/judge_training/` - Rubric-scored judge training data (JSON)
  - `phase3_cot/output/` - CoT processing checkpoints (JSONL, every 500 examples)
  - `final_dataset/` - Final training dataset (JSONL + JSON in multiple formats)

**Caching:**
- No explicit caching layer
- Processed data written to files for reuse across pipeline stages
- CoT processing saves checkpoints every 500 examples to `phase3_cot/output/checkpoint_*.jsonl`

## Authentication & Identity

**Auth Provider:**
- Anthropic API Key only
  - Env var: `ANTHROPIC_API_KEY`
  - Scope: All Claude API calls
  - Set before running scripts (no in-code secrets)

**Public Access:**
- Git repository cloning uses public HTTPS URLs (no auth needed)

## Monitoring & Observability

**Error Tracking:**
- Not integrated (would be DGX Toolbox responsibility in future phases)

**Logs:**
- Console/stderr output from Python scripts
- Structured JSON output in intermediate files
- Progress reporting every 10-100 examples depending on phase

**Token Usage Tracking:**
- Logged per request in Claude API responses
- Accumulated and reported in script output (e.g., "Total tokens used: X")
- Example: `phase2_generate.py` prints "Estimated cost: $Y.ZZ"

## CI/CD & Deployment

**Hosting:**
- No web application - data pipeline only
- Target deployment: DGX Spark via DGX Toolbox (future phases)

**CI Pipeline:**
- Not currently implemented
- Planned: DGX Toolbox infrastructure for training and evaluation

## Environment Configuration

**Required env vars:**
- `ANTHROPIC_API_KEY` - Mandatory, must be set before script execution

**Optional env vars:**
- None explicit in current codebase
- Python logging/warnings controlled via standard Python env vars if needed

**Secrets location:**
- ANTHROPIC_API_KEY should be set from:
  - Shell environment: `export ANTHROPIC_API_KEY=sk-...`
  - Or read from secure credential store before running scripts
  - NOT committed to any config files

**Git/Repository Access:**
- Public HTTPS URLs in `config/repos.yaml` - no auth needed
- If private repos are added, would require SSH key in ~/.ssh/

## Webhooks & Callbacks

**Incoming:**
- None - this is a batch data pipeline

**Outgoing:**
- None - no callbacks or webhooks to external services

## External Tools (CLI Dependencies)

**PHP_CodeSniffer (phpcs):**
- Purpose: WordPress Coding Standards compliance pre-filtering
- Invocation: `scripts/phase1_judge.py` line 50
  ```bash
  phpcs --standard=WordPress-Extra --report=json <file>
  ```
- Installation: `composer global require squizlabs/php_codesniffer wp-coding-standards/wpcs`
- Output: JSON report of PHPCS violations

**Git:**
- Purpose: Clone repositories with `--depth=1` shallow clone
- Invocation: `scripts/phase1_clone.py` line 29-31
- No special auth setup needed for public repos

**PHP CLI:**
- Purpose: Run PHP tokenizer script for function extraction
- Invocation: `scripts/phase1_extract.py` line 54-58
  ```bash
  php php_extract_functions.php <file>
  ```
- Requirements: PHP 7.4+ with tokenizer extension enabled

## Data Flow with External Services

**Phase 1 (Clone & Extract):**
```
GitHub (repos.yaml URLs)
    ↓ (git clone)
Local filesystem (phase1_extraction/repos/)
    ↓ (php_extract_functions.php via PHP CLI)
Extracted functions JSON
    ↓ (PHPCS pre-filter + Claude judge)
Anthropic Claude API
    ↓
Assessed functions (passed/failed)
```

**Phase 2 (Synthetic Generation & Judge Data):**
```
Gap analysis
    ↓ (gap_report.json)
Prompt templates + style anchors
    ↓
Anthropic Claude API (generate + judge)
    ↓
Judged synthetic examples
    ↓ (automated mutations)
Contrastive pairs
    ↓ (Claude scoring)
Anthropic Claude API
    ↓
Judge training data (rubric scores)
```

**Phase 3 (CoT & Export):**
```
All passed + judge examples
    ↓ (instruction synthesis + CoT)
Anthropic Claude API
    ↓
Training data with reasoning
    ↓ (format conversion)
OpenAI/Alpaca/Raw JSONL format
    ↓ (train/val/test split)
final_dataset/
```

## Rate Limiting

**Claude API:**
- Enforced in code with `REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE`
- Phase 1: 50 requests/minute (0.005 second pause per request)
- Phase 2: 40 requests/minute (0.015 second pause per request)
- Phase 3: 40 requests/minute (0.015 second pause per request)
- Purpose: Stay within API rate limits while maximizing throughput

## Future DGX Toolbox Integrations (Phase B-E)

Not implemented in this pipeline, but documented in PROJECT.md:
- Unsloth Studio (:8000) - Interactive fine-tuning UI
- eval-toolbox - lm-eval benchmarks, W&B tracking
- data-toolbox - Dataset curation, deduplication
- vLLM (:8020) - Batch inference for DPO candidate generation
- LiteLLM (:4000) - Unified API for cross-model evaluation (Claude, GPT-4, local)
- Label Studio / Argilla - Human annotation for DPO preference data
- Safety harness (:5000) - Guardrails, red-teaming, PII redaction

---

*Integration audit: 2026-03-26*

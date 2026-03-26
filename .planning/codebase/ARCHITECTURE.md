# Architecture

**Analysis Date:** 2026-03-26

## Pattern Overview

**Overall:** Multi-phase data pipeline for training dataset generation

**Key Characteristics:**
- Sequential processing across 3 major phases (extraction, synthesis, export)
- Quality-gated filtering at each phase (PHPCS + Claude judgment)
- Taxonomy-driven gap analysis for synthetic data generation
- Contrastive learning pairs (bad→good) generated via mutation and AI
- Multi-format output for different training frameworks

## Layers

**Phase 1: Repository Curation & Extraction**
- Purpose: Extract and assess code from WordPress core + curated plugins/themes
- Location: `scripts/phase1_*.py`, config stored in `config/repos.yaml`
- Contains: Repository cloning, PHP function extraction via tokenizer, PHPCS pre-filtering, Claude quality judgment
- Depends on: PHP CLI with tokenizer extension, PHP_CodeSniffer, Anthropic API, external git repositories
- Used by: Phase 2 (provides style anchors + quality baseline)

**Phase 2: Synthetic Generation & Judge Data Creation**
- Purpose: Fill coverage gaps in extracted code, create contrastive pairs, generate judge training data
- Location: `scripts/phase2_*.py`, gap analysis config in `config/taxonomy.yaml` and `config/synthetic_prompts.yaml`
- Contains: Gap analysis, automated mutation of real code, Claude synthetic generation, judgment of generated code, rubric scoring
- Depends on: Phase 1 output (passed/failed functions), taxonomy definitions, prompt templates, Anthropic API
- Used by: Phase 3 (provides synthetic + judge training examples)

**Phase 3: Chain-of-Thought & Final Export**
- Purpose: Generate instruction-response pairs with reasoning, merge all sources, add task tokens, export multi-format
- Location: `scripts/phase3_cot.py`, `scripts/export_dataset.py`
- Contains: Instruction synthesis (reverse-engineer prompts), CoT reasoning generation, data merging, format conversion
- Depends on: Phase 1 + Phase 2 output, Anthropic API
- Used by: Final training (produces training-ready dataset in `final_dataset/`)

## Data Flow

**Phase 1 Extraction Flow:**

1. User curates repository list in `config/repos.yaml` (WordPress core + plugins/themes with quality_tier)
2. `phase1_clone.py` shallow-clones all repos to `phase1_extraction/repos/`
3. `phase1_extract.py` runs PHP tokenizer on each file via `php_extract_functions.php`
   - Outputs: Raw function metadata (name, body, docblock, SQL patterns, hooks used) → `phase1_extraction/output/extracted/`
4. `phase1_judge.py` filters + judges extracted functions
   - WordPress Core (quality_tier: "core") → auto-passed, tagged only
   - Everything else (quality_tier: "assessed") → PHPCS pre-filter (< 5 errors/100 lines) → Claude 9-dimension judgment (all scores ≥ 7)
   - Outputs: `phase1_extraction/output/passed/` (training-ready) and `phase1_extraction/output/failed/` (analysis)

**Phase 2 Synthesis Flow:**

1. `phase2_gap_analysis.py` compares tag coverage in Phase 1 output against `config/taxonomy.yaml` minimums
   - Outputs: `phase2_synthetic/gap_report.json` (which tags need more examples)
2. Two parallel synthesis paths:
   - **Path A (Mutation):** `phase2_mutate.py` creates contrastive pairs from Phase 1 passed code
     - Automated mutations: remove prepare(), strip nonces, strip escaping, remove capability checks, remove sanitization, inject SELECT *
     - Each mutation verified detectable by PHPCS
     - Outputs: `phase2_synthetic/output/mutated/` (bad→good pairs)
   - **Path B (Generation):** `phase2_generate.py` generates new synthetic examples grounded in Phase 1 style anchors
     - Uses real code snippets as few-shot style references
     - Fills identified gaps based on `config/synthetic_prompts.yaml`
     - Outputs: `phase2_synthetic/output/generated/`
3. `phase2_judge.py` judges synthetic code (same criteria as Phase 1)
   - Failed examples get 1 revision attempt, then discarded
   - Outputs: `phase2_synthetic/output/judged/` (passed/failed synthetic)
4. `phase2_judge_dataset.py` generates `<wp_judge>` training data
   - Scores passed code (high), failed code (low), mutated code (controlled defects) on 0-100 rubric across 6 dimensions
   - Sanity-checked against expected quality tier
   - Outputs: `phase2_synthetic/output/judge_training/`

**Phase 3 Synthesis & Export Flow:**

1. `phase3_cot.py` processes all Phase 1 + Phase 2 examples
   - **Instruction synthesis:** Reverse-engineer prompts for each code unit
   - **CoT reasoning:** Generate step-by-step explanations for complex examples (SQL, performance, architecture)
   - **Contrastive reasoning:** Enhance mutation pairs with CoT explanations
   - Merges judge training data from Phase 2
   - Outputs: `phase3_cot/output/` (CoT checkpoints)
2. `export_dataset.py` final formatting + splitting
   - Infers task type (generation vs. judgment) from metadata/content
   - Adds `<wp_gen>` or `<wp_judge>` task tokens to user messages
   - Exports 3 formats: OpenAI JSONL, Alpaca JSON, Raw JSONL with metadata
   - Splits: 80/10/10 train/validation/test
   - Outputs: `final_dataset/` with `{openai,alpaca,raw}_{train,val,test}.*`

**State Management:**
- Passed/failed functions stored as JSON per-file to enable resumability
- Gap report is JSON snapshot used to seed synthetic generation
- Judge training data includes explicit scores + defect annotations for contrastive learning
- CoT output stored before final export to allow checkpoint inspection

## Key Abstractions

**Quality Assessment:**
- Purpose: Multi-stage filtering to ensure training data quality
- Examples: `phase1_judge.py`, `phase2_judge.py`
- Pattern: PHPCS pre-filter (cheap, static) → Claude judgment (expensive, semantic) → separate passed/failed outputs

**Style Anchoring:**
- Purpose: Ground synthetic generation in real-world code patterns
- Examples: `phase2_generate.py` → `load_style_anchors()`
- Pattern: Query Phase 1 passed code for matching taxonomy tags, extract 3 representative functions as few-shot examples

**Taxonomy-Driven Gap Analysis:**
- Purpose: Ensure coverage of WordPress concept space
- Examples: `config/taxonomy.yaml`, `phase2_gap_analysis.py`
- Pattern: Organize concepts hierarchically (sql_patterns, security, hooks, data_modeling, etc.), tag extracted/generated code, compute coverage deltas

**Contrastive Learning Pairs:**
- Purpose: Train both good patterns and error detection
- Examples: `phase2_mutate.py` (automated mutation) + `phase2_judge_dataset.py` (rubric scoring)
- Pattern: Transform passed code into detectable violations (prepared statements→concatenation, nonce checks→removed, etc.), keep mutation metadata for explanation

**Multi-Task Training Tokens:**
- Purpose: Route examples to specialized MoE expert pathways
- Examples: `<wp_gen>` (code generation), `<wp_judge>` (rubric scoring/critique)
- Pattern: Inference in `export_dataset.py`, assignment based on task_type metadata or content inspection

## Entry Points

**Repository Cloning & Configuration:**
- Location: `scripts/phase1_clone.py`
- Triggers: Manual invocation by user
- Responsibilities: Shallow-clone repositories from `config/repos.yaml`, pull updates if already present

**Function Extraction:**
- Location: `scripts/phase1_extract.py`
- Triggers: After `phase1_clone.py`
- Responsibilities: Tokenize PHP, extract function boundaries + metadata, output JSON per repo

**Quality Judgment (Phase 1 & 2):**
- Location: `scripts/phase1_judge.py`, `scripts/phase2_judge.py`
- Triggers: After extraction/generation respectively
- Responsibilities: Filter code via PHPCS + Claude, assign pass/fail + taxonomy tags, rate-limit API calls (50 req/min)

**Gap Analysis:**
- Location: `scripts/phase2_gap_analysis.py`
- Triggers: After Phase 1 complete
- Responsibilities: Compare tag coverage against taxonomy minimums, produce gap report

**Synthetic Generation:**
- Location: `scripts/phase2_generate.py`
- Triggers: After gap analysis
- Responsibilities: Load style anchors, generate targeted examples using Claude, write to `phase2_synthetic/output/generated/`

**Automated Mutation:**
- Location: `scripts/phase2_mutate.py`
- Triggers: After Phase 1 complete (parallel to phase2_generate.py)
- Responsibilities: Apply deterministic transformations to passed code, verify violations detectable by PHPCS, output bad→good pairs

**Judge Dataset Creation:**
- Location: `scripts/phase2_judge_dataset.py`
- Triggers: After synthetic code is judged
- Responsibilities: Generate rubric-scored examples for judge training, sanity-check scores against quality tier

**CoT & Instruction Synthesis:**
- Location: `scripts/phase3_cot.py`
- Triggers: After all Phase 2 output complete
- Responsibilities: Reverse-engineer prompts, generate reasoning explanations, merge all sources

**Final Export:**
- Location: `scripts/export_dataset.py`
- Triggers: After `phase3_cot.py`
- Responsibilities: Add task tokens, convert to OpenAI/Alpaca/Raw formats, apply train/val/test split

## Error Handling

**Strategy:** Graceful degradation with detailed reporting

**Patterns:**
- PHP extraction timeouts (30s per file) → return empty list, continue with next file
- PHPCS not installed → skip pre-filter, defer to Claude (higher API cost)
- Claude API failures → retry with exponential backoff, log failures for manual review
- Failed synthetic code → attempt 1 revision, then discard if still failing
- Phase execution → each phase can be re-run independently; outputs overwritten, but can resume from checkpoints

## Cross-Cutting Concerns

**Logging:** Print-based with operation labels (e.g., `[repo_name] operation description`)

**Validation:**
- JSON pre/post-write validation in data processing
- PHPCS pass/fail as cheap gating before expensive Claude calls
- Judge output sanity checks (scores in 0-10 range, verdict is PASS/FAIL, critical_failures list)

**Authentication:** Anthropic API key via `ANTHROPIC_API_KEY` environment variable

**Rate Limiting:**
- Phase 1 judge: 50 requests/minute
- Phase 2 generate: 40 requests/minute
- Phase 3 CoT: 40 requests/minute
- Implemented via `time.sleep(REQUEST_INTERVAL)` between calls

**Resumability:**
- Phase operations check for existing output directories
- If output exists, scripts append rather than overwrite (for judgment phases)
- Gap analysis + generation/mutation can be re-run to fill new gaps

---

*Architecture analysis: 2026-03-26*

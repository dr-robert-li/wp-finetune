# Codebase Concerns

**Analysis Date:** 2026-03-26

## Tech Debt & Design Issues

### API Rate Limiting Not Enforced on All Scripts

**Issue:** Rate limiting constants defined (`REQUESTS_PER_MINUTE = 50`) but only partially enforced across scripts.

**Files:**
- `scripts/phase1_judge.py` (line 30-31): Rate limiting implemented via `time.sleep(REQUEST_INTERVAL)`
- `scripts/phase2_judge.py` (line 20-21): Rate limiting implemented
- `scripts/phase2_generate.py` (line 25-26): Rate limiting implemented
- `scripts/phase3_cot.py` (line 27-28): Rate limiting implemented, but may not apply to all API calls
- `scripts/phase2_judge_dataset.py` (line 27-28): Rate limiting NOT implemented between scoring calls

**Impact:** `phase2_judge_dataset.py` could exceed rate limits when scoring many code samples, causing API errors or account throttling. Phase 3 CoT generation uses both Sonnet and Opus models without per-model rate limiting.

**Fix approach:**
1. Add `time.sleep(REQUEST_INTERVAL)` after each Claude API call in `phase2_judge_dataset.py` line 74
2. Consider separate rate limits for different model tiers (Opus vs Sonnet)
3. Add retry logic with exponential backoff for 429 errors across all scripts

---

### JSON Parsing Fragility in Judge Response Extraction

**Issue:** Response parsing uses brittle string splitting to extract JSON from markdown blocks.

**Files:**
- `scripts/phase1_judge.py` (lines 195-200): Split by markdown markers
- `scripts/phase2_judge.py` (lines 47-51): Identical parsing logic
- `scripts/phase2_judge_dataset.py` (lines 70-74): Identical parsing logic
- `scripts/phase3_cot.py` (line 152): Returns raw text, no JSON parsing

**Problem Code:**
```python
if "```json" in text:
    text = text.split("```json")[1].split("```")[0]
elif "```" in text:
    text = text.split("```")[1].split("```")[0]
return json.loads(text.strip())
```

**Impact:** If Claude wraps JSON in different markers (e.g., `"```"` without `json`), parsing fails. Error handling returns empty/stub responses (`verdict: "FAIL"`) which pollutes training data. No logging of parse failures for debugging.

**Fix approach:**
1. Extract JSON parsing to shared utility function in `utils.py`
2. Add robust JSON extraction: try JSON at string boundaries, handle markdown variants
3. Log all parse failures with sample text for debugging
4. Implement fallback: if parsing fails, reject the example rather than returning stub
5. Add test cases for common response variations

---

### Dependency Extraction May Miss Complex Patterns

**Issue:** PHP tokenizer-based extraction (`scripts/php_extract_functions.php`) uses regex-based dependency detection that has blind spots.

**Files:**
- `scripts/php_extract_functions.php` (lines 159-188): `extract_dependencies()` function
  - Uses simple regex: `/\b([a-z_][a-z0-9_]*)\s*\(/i`
  - Static method detection uses: `/([A-Z][a-zA-Z0-9_]*)::\s*([a-z_][a-z0-9_]*)\s*\(/i`
  - Hardcoded PHP builtins list (lines 163-174)

**Missing patterns:**
- Variable function calls: `$func()` won't be detected
- Dynamic class instantiation: `new $class_name()` won't be detected
- Method calls on variables: `$obj->method()` won't be detected
- Indirect dependencies through closures or callbacks

**Impact:** Functions with hidden dependencies may pass validation but fail at runtime during training/inference. Contrastive mutation pairs may break dependencies unexpectedly.

**Fix approach:**
1. Document limitations in comments: "This extraction is conservative; complex dependency patterns are not captured"
2. Add validation check: if function calls extracted functions that aren't in the dataset, flag for manual review
3. Consider more conservative approach: only accept functions with zero or fully-resolvable dependencies
4. Add dependency resolution pass in Phase 1 to verify all dependencies exist

---

### Mutation Detection Relies on PHPCS Availability

**Issue:** `phase2_mutate.py` uses PHPCS to verify mutations are detectable, but gracefully degrades if PHPCS is unavailable.

**Files:**
- `scripts/phase2_mutate.py` (lines 156-177): `verify_mutation_detectable()` function

**Problem Code:**
```python
except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
    # If PHPCS isn't available, accept the mutation anyway.
    return True
```

**Impact:** If PHPCS isn't installed (line 166 catches `FileNotFoundError`), ALL mutations are accepted as "detectable" without verification. This means undetectable mutations could enter training data, training the model on invisible defects.

**Fix approach:**
1. Fail fast: require PHPCS at script start, exit with clear error message if missing
2. Add pre-flight check: `verify_phpcs_available()` before processing begins
3. Log all PHPCS skips with sample code for audit trail
4. Alternatively: add fallback detection using regex-based error patterns if PHPCS unavailable

---

### Incomplete Configuration Halts Pipeline

**Issue:** `config/repos.yaml` contains minimal example repos; production run requires manual curation.

**Files:**
- `config/repos.yaml` (lines 30-86): Only `woocommerce` is uncommented; plugins and themes sections are empty

**Impact:** Pipeline requires users to manually add repos to `config/repos.yaml` before running. Current state only has WooCommerce configured. Phase A1.1 is documented as "waiting on curated repos.yaml" in PROJECT.md (line 210).

**Fix approach:**
1. Add more example repos with explanatory comments showing quality criteria
2. Create optional `config/repos-examples.yaml` with pre-vetted repository list
3. Add validation in `phase1_clone.py`: warn if fewer than N repos configured (line 45-47 only checks existence)
4. Document minimum viable set: suggest at least 5-10 quality plugins for meaningful training

---

### Judge Criteria Ambiguity in Accessibility & i18n Scoring

**Issue:** Judge system instructions allow "N/A" (score of 10) for accessibility and i18n if code doesn't produce HTML or user-facing strings.

**Files:**
- `config/judge_system.md` (lines 73-85): Accessibility and i18n scoring rules

**Problem:**
- Line 74: "Score N/A (10) if the function has no user-facing strings"
- Line 84: "Score N/A (10) if the function produces no HTML output"

**Impact:** Backend functions that pass a/11y and i18n criteria by non-applicability inflate average scores. Model learns to associate avoiding HTML/i18n with high quality, which is backwards. Judge training data becomes skewed toward backend examples.

**Fix approach:**
1. Redefine N/A handling: score functions without a/11y applicability as separate category, don't inflate to 10
2. Track applicability separately: `{ "dimension": "a11y", "applicable": false, "score": 0, "reason": "no_html_output" }`
3. Adjust overall score calculation to NOT count N/A dimensions, use proportional weighting
4. Document: accessibility/i18n are ALWAYS applicable to user-facing code; if function produces user output, MUST score

---

### SQL Safety Tag Auto-Tagging Misses Indirect Violations

**Issue:** `phase1_judge.py` auto-tagging for WordPress Core uses presence detection (e.g., "if 'prepared_query' in sql") but doesn't verify correctness.

**Files:**
- `scripts/phase1_judge.py` (lines 84-164): `auto_tag_function()` function
  - Lines 92-100: Tags based on string presence in extracted SQL patterns
  - Example (line 93-94): If "prepared_query" in sql patterns, tags as `sql:prepared_statements`

**Problem:** This only tags Core code; Claude judge has full responsibility for assessed code. But the process doesn't validate that prepared statements are used CORRECTLY (e.g., placeholder types, variable ordering).

**Impact:** Core code may contain subtle SQL vulnerabilities (e.g., using %s for integers) that aren't caught because auto-tagging only checks string presence. These become reference implementations in style anchors.

**Fix approach:**
1. Don't auto-tag Core; either judge Core with same Claude criteria or disable style anchors from Core
2. Or: add validation step for Core-originating functions: sample them through judge system to verify
3. Document: "Core code should be verified; auto-tagging is not validation"
4. Add note in `phase2_generate.py` style anchors: exclude Core code or mark as "reference only" in prompts

---

### Training Data Imbalance: Judge Examples May Dominate

**Issue:** Judge training dataset sourcing (`phase2_judge_dataset.py`) scores high-quality, low-quality, AND mutation examples, but target dataset composition (PROJECT.md lines 175-185) allocates 3000 judge examples vs. 7500 generation examples.

**Files:**
- `scripts/phase2_judge_dataset.py` (lines 100-165): Generates judge training data from passed, failed, and mutated code
- `PROJECT.md` (lines 175-185): Target composition table

**Problem:**
- Judge training comes from Phase 1 passed (good), Phase 1 failed (bad), and Phase 2 mutations (controlled defects)
- Current plan: ~1500 high-quality + ~1000 low-quality + ~1500 mutations = 4000 judge examples
- But Phase 1 only extracts ~9000 functions total; if 50% pass, only ~4500 high-quality examples exist
- Judge training will consume ~40% of all passed examples, starving generation training of diverse examples

**Impact:** Generation model undertrained on diverse real examples, may overfit to synthetic/mutated patterns. Judge model may have insufficient high-quality examples to learn nuanced scoring.

**Fix approach:**
1. Adjust target composition: increase generation examples, reduce judge duplicates
2. Stratify judge training: prioritize high-confidence passed examples, use low-quality/mutations sparingly
3. Implement deduplication: if code example used in generation training, don't reuse in judge training
4. Track split in metadata: add `used_for: ["generation", "judgment"]` to avoid reuse

---

### Phase 3 CoT Generation Uses Expensive Model Without Budget Tracking

**Issue:** `phase3_cot.py` uses Claude Opus (most expensive) for CoT generation without tracking or warning about cost.

**Files:**
- `scripts/phase3_cot.py` (line 145): `model="claude-opus-4-6-20250514"` for CoT
- `scripts/phase2_generate.py` (line 98): Uses Sonnet for standard generation, Opus for contrastive

**Problem:**
- Opus costs ~3x Sonnet
- Phase 3 applies CoT to examples matching COT_TAGS (lines 75-88) + long functions (line 164)
- Line 221 in `phase2_generate.py` estimates cost: "~${total_tokens / 1_000_000 * 5:.2f}" (hardcoded $5M rate, doesn't account for Opus)
- No warning if CoT generation will be expensive
- No option to skip CoT for cost-conscious runs

**Impact:** Users may be surprised by high API bills. Development/testing runs with full dataset could cost $500+. No way to do test run with small subset to estimate total cost.

**Fix approach:**
1. Add cost estimation before starting Phase 3: estimate token usage for CoT examples, warn user
2. Add `--dry-run` flag to estimate without executing
3. Implement `--max-cot` flag: limit CoT generation to top N examples by importance
4. Track and report actual costs at end of each phase
5. Use cheaper model (Sonnet) for CoT on lower-priority examples (line_count < 50)

---

### Export Script Doesn't Validate Task Token Placement

**Issue:** `export_dataset.py` infers task tokens (`<wp_gen>` vs `<wp_judge>`) but relies on metadata that may be missing or inconsistent.

**Files:**
- `scripts/export_dataset.py` (lines 33-51): `infer_task_type()` function

**Problem Code:**
```python
def infer_task_type(example: dict) -> str:
    metadata = example.get("metadata", {})
    if metadata.get("task_type") == "judge":
        return "judge"
    # Fall back to searching user message content
    # ...
    # Default: generation
    return "gen"
```

**Impact:**
- If metadata is missing, wrong task type is inferred
- Judge examples might be marked as generation examples and vice versa
- No validation that `<wp_judge>` examples actually have structured JSON output
- No check that `<wp_gen>` examples have well-formed PHP code blocks

**Fix approach:**
1. Add strict validation: require `metadata.task_type` to be present and explicit
2. Validate example structure: judge examples must have JSON assistant output, gen examples must have code
3. Fail with clear error if task type cannot be determined, don't default to "gen"
4. Add `--validate` flag to scan exported dataset for task token consistency

---

### CLI Scripts Have No Progress Checkpointing

**Issue:** Multi-hour scripts (Phase 1 judge, Phase 2 generate) have no way to resume after failure.

**Files:**
- `scripts/phase1_judge.py` (lines 220-310): Processes all functions in one pass
- `scripts/phase2_generate.py` (lines 148-216): Generates all gaps in one pass
- `scripts/phase3_cot.py` (lines 170-300): Full dataset processing

**Problem:**
- If script crashes at example 3000/5000, user must restart from beginning
- API errors during processing can't be resumed
- No checkpoint file to track progress
- No option to process subset for testing

**Impact:** Development is slow; cost/time wasteful on large datasets. Users can't easily test with small repo subsets first.

**Fix approach:**
1. Add `--start-from <N>` flag to resume processing at example N
2. Write progress checkpoint every 100 examples (phase, example_num, timestamp)
3. Implement `--sample <N>` flag to process only first N examples for testing
4. Add checkpoint file validation: warn if resuming after long gap (API schema may have changed)

---

## Security & Data Quality Issues

### Claude Judge Model Version Hardcoded

**Issue:** Model identifiers hardcoded to specific Claude version.

**Files:**
- `scripts/phase1_judge.py` (line 189): `model="claude-sonnet-4-6-20250514"`
- `scripts/phase2_judge.py` (line 41): `model="claude-sonnet-4-6-20250514"`
- `scripts/phase2_judge_dataset.py` (line 64): `model="claude-sonnet-4-6-20250514"`
- `scripts/phase2_generate.py` (lines 98-100): Hardcoded both Sonnet and Opus
- `scripts/phase3_cot.py` (lines 135, 145): Hardcoded both models

**Impact:**
- Model versions become outdated/deprecated
- Can't easily test with different models
- If model is deprecated, entire pipeline breaks with cryptic error
- No way to systematically upgrade all scripts at once

**Fix approach:**
1. Move model names to config file: `config/models.yaml`
2. Define: `judge_model`, `generation_model`, `reasoning_model`
3. Load at script start, fail with clear error if model unavailable
4. Add `--model <name>` CLI flag to override

---

### No Validation of Required Environment Variables

**Issue:** Scripts require `ANTHROPIC_API_KEY` but don't validate at startup.

**Files:**
- `scripts/phase1_judge.py` (line 225): `client = anthropic.Anthropic()` — relies on env var
- Same pattern in all scripts

**Impact:** Script may run for hours before failing on first API call due to missing key, wasting time and disk I/O.

**Fix approach:**
1. Add startup check in each script (or shared utility):
```python
if not os.environ.get('ANTHROPIC_API_KEY'):
    print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
    sys.exit(1)
```
2. Or better: shared `check_env_vars()` function called first thing

---

## Testing & Validation Gaps

### No Integration Tests for Full Pipeline

**Issue:** No test suite. Can't validate that pipeline produces valid output before running expensive phases.

**Files:** No test directory or test files found

**Impact:**
- Can't catch bugs until running full pipeline
- Difficult to debug failures that only occur at scale
- No regression detection when code changes

**Fix approach:**
1. Create `tests/` directory with unit tests for:
   - JSON parsing robustness
   - Dependency extraction accuracy
   - Mutation detection effectiveness
   - Task token inference correctness
2. Create integration test: small repo (100 functions), run phases 1-3, validate output
3. Add `pytest` to requirements, CI pipeline (GitHub Actions) to run tests on commits

---

### No Validation of Final Dataset Quality

**Issue:** Export script doesn't validate output dataset before completion.

**Files:**
- `scripts/export_dataset.py` (lines 120-182): No validation at end

**Missing checks:**
- Train/val/test split actually 80/10/10
- All examples have required fields
- Task tokens present and consistent
- Code examples parse as valid PHP
- JSON outputs in judge examples are valid
- No duplicate examples across splits

**Impact:** Invalid dataset silently exported. Training will fail or produce poor models.

**Fix approach:**
1. Add `validate_dataset()` function at end of export
2. Check: all examples have `messages`, task tokens present, no malformed JSON
3. Check splits: verify counts match expected ratios
4. Sample validation: parse 100 code examples to ensure PHP validity
5. Fail export if validation fails, report specific issues

---

### Taxonomy Minimum Coverage Not Enforced

**Issue:** Minimum coverage targets defined in `config/taxonomy.yaml` but not enforced; gap analysis reports gaps but pipeline continues anyway.

**Files:**
- `config/taxonomy.yaml`: Defines concept tags but no minimum_coverage section visible in read output
- `scripts/phase2_gap_analysis.py` (lines 27-56): Loads minimums from taxonomy, reports gaps
- `scripts/phase2_generate.py` (line 134-136): Skips generation if no gaps found, but continues if gaps exist

**Problem:**
- No way to fail pipeline if gaps aren't filled
- `phase2_generate.py` line 136: "Proceeding to contrastive pair generation only" if no gaps
- Final dataset may have poor tag coverage despite planning

**Impact:** Training data may not cover all intended concepts. Model gaps undetected until evaluation phase.

**Fix approach:**
1. Add `--enforce-coverage` flag to exit if gaps remain after generation
2. Calculate coverage percentage, fail if < 95%
3. In `export_dataset.py`: validate final dataset coverage matches taxonomy targets
4. Report actual coverage percentage in exported metadata

---

## Known Limitations

### Docker/Container Support Missing

**Issue:** Pipeline requires PHP, PHPCS, and multiple Python dependencies but no Docker container provided.

**Impact:** Setup is manual and error-prone. PHPCS installation via Composer is not well-documented.

**Fix approach:**
1. Create `Dockerfile` with all dependencies
2. Create `docker-compose.yml` for convenience
3. Add `.dockerignore` to exclude cloned repos from image
4. Document in README: "Quick start with Docker"

---

### Repos Configuration Example Incomplete

**Issue:** `config/repos.yaml` comments mention example entries but don't uncomment them for users to see.

**Files:**
- `config/repos.yaml` (lines 42-60, 74-85): Example entries are commented out

**Impact:** New users aren't sure what good configuration looks like. Pipeline has no real examples to work with out-of-the-box.

**Fix approach:**
1. Uncomment at least 2-3 examples: a well-known plugin, a theme
2. Or: create separate `config/repos-example.yaml` with full set of pre-vetted repos

---

## Performance & Scaling Concerns

### No Batching for Synthetic Generation

**Issue:** `phase2_generate.py` generates examples one at a time, sleeping between requests.

**Files:**
- `scripts/phase2_generate.py` (lines 162-180): Loop processes gap_tag sequentially

**Impact:** If 100 gaps exist, requires 100+ API calls with 1.5s sleep between each = 150+ seconds minimum. Scales poorly with number of gaps.

**Fix approach:**
1. Batch generation: collect prompt templates, send to API in parallel jobs using threading/asyncio
2. Or: use Claude batch API for cost optimization (10x cheaper for async workloads)
3. Add `--max-parallel <N>` flag for concurrency control

---

### Judge Scoring on Large Code Blocks Expensive

**Issue:** Judge system designed to score individual functions, but if functions are large (>4000 chars), scoring is inefficient.

**Files:**
- `scripts/phase1_judge.py` (line 137): `code[:3000]` truncates long functions for prompt
- `scripts/phase2_judge_dataset.py` (line 57): `code[:4000]` truncation
- `scripts/phase3_cot.py` (line 149): `code[:4000]` truncation

**Impact:**
- Large functions are truncated, potentially losing important context
- Inconsistent truncation across scripts (3000 vs 4000)
- If function is critical and truncation removes security-critical code, judge may pass when should fail

**Fix approach:**
1. Define maximum function size: reject functions > 5000 chars in Phase 1 extraction
2. Or: split large functions into logical sub-functions for extraction
3. Use consistent truncation everywhere (pick 4000 as standard)
4. Add warning log if function truncated, include function_name in alert

---

## Documentation Issues

### PROJECT.md Phase Status Out of Date

**Issue:** Project status (lines 207-215) shows incomplete status but no dates or next-step clarity.

**Files:**
- `PROJECT.md` (lines 207-215)

**Impact:** Unclear what's blocking forward progress. "Waiting on curated repos.yaml" is vague.

**Fix approach:**
1. Update status with specific blockers: "Phase A1.1: Needs user to add 5+ plugins to config/repos.yaml (see config/repos-example.yaml)"
2. Add estimated timeline per phase
3. Track actual vs. estimated

---

### Missing Environment Setup Documentation

**Issue:** README.md mentions dependencies but doesn't show installation steps for PHPCS.

**Files:**
- `README.md` (lines 34-35): Composer install command without error handling
- No troubleshooting guide

**Impact:** Users may not install PHPCS correctly, causing silent failures (prefilter gracefully degrades if PHPCS unavailable).

**Fix approach:**
1. Expand README with step-by-step PHPCS setup and verification
2. Add troubleshooting section: "If extraction fails, check: PHP CLI available, tokenizer extension, PHPCS installed"
3. Create `scripts/verify_setup.py` to validate all dependencies at startup

---

*Concerns audit: 2026-03-26*

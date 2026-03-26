# Testing Patterns

**Analysis Date:** 2026-03-26

## Test Framework

**Status:** No automated testing framework detected

**Found:**
- No pytest, unittest, vitest, or similar test runner configuration
- No `*_test.py`, `test_*.py`, or `*.spec.py` files in repository
- No `pytest.ini`, `setup.cfg`, or test configuration files
- No CI/CD pipeline configuration (no `.github/workflows/`, `.gitlab-ci.yml`, etc.)

## Current Testing Approach

All validation occurs **within the pipeline** as runtime quality gates:

### Phase 1: PHPCS Pre-filter
**Location:** `scripts/phase1_judge.py`, lines 34-71

```python
def phpcs_prefilter(code: str, max_errors_per_100_lines: float = 5.0) -> dict:
    """Run PHPCS as a cheap pre-filter before sending to Claude."""
    # Wrap bare function in <?php for PHPCS parsing.
    if not code.strip().startswith("<?php"):
        code = f"<?php\n{code}"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["phpcs", "--standard=WordPress-Extra", "--report=json", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        report = json.loads(result.stdout)
        file_report = list(report["files"].values())[0]
        line_count = max(code.count("\n"), 1)
        error_density = file_report["errors"] / line_count * 100

        return {
            "passed": error_density <= max_errors_per_100_lines,
            "errors": file_report["errors"],
            "warnings": file_report["warnings"],
            "error_density": round(error_density, 2),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        # PHPCS not installed or failed — skip pre-filter, let Claude decide.
        return {"passed": True, "errors": -1, "warnings": -1, "error_density": 0, "skipped": True}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

**Purpose:** Reject high-error-density code before spending API tokens on Claude assessment

### Phase 1: Claude Judge Assessment
**Location:** `scripts/phase1_judge.py`, lines 173-220 (sketch of call; full logic in `config/judge_system.md`)

Assesses code on 9 dimensions:
1. WPCS (WordPress Coding Standards)
2. SQL safety (prepared statements, escaping)
3. Security (nonces, capability checks, output escaping, input sanitization)
4. Performance (caching, query efficiency)
5. WordPress API correctness
6. Code quality (clarity, maintainability)
7. Dependency management
8. i18n (internationalization)
9. Accessibility

**Structure:** Returns JSON assessment:
```json
{
  "verdict": "PASS|FAIL",
  "scores": {
    "wpcs": 95,
    "sql_safety": 98,
    "security": 99
  },
  "critical_failures": [],
  "notes": "..."
}
```

### Phase 2: Synthetic Code Validation
**Location:** `scripts/phase2_judge.py`, lines 29-53

Same judge criteria as Phase 1. Failed synthetics get **one revision attempt**:

```python
if assessment.get("verdict") == "PASS":
    example["training_tags"] = assessment.get("training_tags", [example["gap_tag"]])
    passed.append(example)
else:
    # One revision attempt.
    issues = assessment.get("critical_failures", []) + [assessment.get("notes", "")]
    if issues:
        revised_code = revise_with_feedback(example, issues, client)
        time.sleep(REQUEST_INTERVAL)
        re_assessment = judge_synthetic(revised_code, example["gap_tag"], client, judge_system)

        if re_assessment.get("verdict") == "PASS":
            example["body"] = revised_code
            passed.append(example)
        else:
            failed.append(example)
    else:
        failed.append(example)
```

### Phase 2: Mutation Validation
**Location:** `scripts/phase2_mutate.py`, lines 145-244+

Mutations are verified to be detectable:
- Apply mutation (e.g., remove `$wpdb->prepare()`)
- Run PHPCS on mutated code
- Verify it fails PHPCS
- Store bad->good pair only if mutation is verifiably incorrect

**Mutation Types:**
- `sql_injection`: Remove `$wpdb->prepare()`, concatenate variables directly
- `csrf`: Remove nonce verification
- `xss`: Remove output escaping functions
- `authorization`: Remove capability checks
- `input_validation`: Remove sanitization functions
- `i18n`: Remove translation wrappers
- `performance`: Replace targeted SELECT with `SELECT *`

### Phase 2: Judge Training Data Generation
**Location:** `scripts/phase2_judge_dataset.py`

Generates training data for `<wp_judge>` task:
- Passed code: 100/100 quality score
- Failed code: 0/100
- Mutated code: Controlled quality (depends on mutation type)

Rubric categories (6 dimensions):
1. WPCS compliance
2. SQL safety
3. Security (escaping, nonces, capabilities)
4. Performance
5. API correctness
6. Maintainability

## Coverage & Validation

**Manual Validation:**
- Gap analysis (`phase2_gap_analysis.py`): Compares tag coverage against `config/taxonomy.yaml` minimums
- Reports deficit in categories, drives synthetic generation targets
- No automated test suite validates coverage programmatically

**Example Gap Report Output:**

```
Phase 1 Coverage Report
============================================================
Total passed functions: 847
Unique tags present: 34

Tag                                  Have   Need    Gap    %
---------------------------------------- ------ ------ ------  ------
sql:prepared_statements               45    50      5   90%
security:output_escaping             123   150     27   82% <-- GAP
perf:query_caching                    18    25      7   72% <-- GAP
rest:permission_callbacks             12    20      8   60% <-- GAP
```

## Test Data Organization

**Training Data Sources:**

1. **Phase 1 Real Code:** `phase1_extraction/output/passed/` (JSON files, one per repo)
2. **Phase 2 Synthetic:** `phase2_synthetic/output/judged/` (*_passed.json, *_failed.json)
3. **Phase 2 Mutations:** `phase2_synthetic/output/mutated/contrastive_mutations.json`
4. **Judge Training:** `phase2_synthetic/output/judge_training/judge_training.json`

**Final Dataset Exports:**

```
final_dataset/
├── openai_train.jsonl   # OpenAI format, 80%
├── openai_val.jsonl     # Validation split, 10%
├── openai_test.jsonl    # Test split, 10%
├── alpaca_train.json    # Alpaca/Llama-MoE format
├── alpaca_val.json
├── alpaca_test.json
├── raw_train.jsonl      # Metadata preserved
├── raw_val.jsonl
├── raw_test.jsonl
└── metadata.json        # Statistics
```

**Split Strategy** (`scripts/export_dataset.py`, lines 114-132):

```python
random.seed(42)  # Deterministic split
random.shuffle(dataset)

n = len(dataset)
train_end = int(n * TRAIN_SPLIT)           # 0.80
val_end = train_end + int(n * VAL_SPLIT)   # 0.80 + 0.10

train_set = dataset[:train_end]
val_set = dataset[train_end:val_end]
test_set = dataset[val_end:]
```

## Defect Testing (Mutation Testing)

**Pattern:** Intentional introduction of controlled defects to validate judge quality

**From `phase2_mutate.py`:**

```python
def mutate_remove_prepare(code: str) -> tuple[str, str]:
    """Remove $wpdb->prepare() and inline values directly (SQL injection)."""
    pattern = r'\$wpdb->prepare\(\s*(["\'])(.*?)\1\s*,\s*(.*?)\)'
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        return None, None

    query_str = match.group(2)
    vars_str = match.group(3)
    vars_list = [v.strip() for v in vars_str.split(",")]
    mutated_query = query_str
    for var in vars_list:
        mutated_query = re.sub(r'%[sdf]', f"' . {var} . '", mutated_query, count=1)

    bad_code = code[:match.start()] + f'"{mutated_query}"' + code[match.end():]
    return bad_code, "sql_injection: Removed $wpdb->prepare(), concatenating variables directly into SQL"
```

**Verification Workflow:**
1. Mutate real code (create intentional defect)
2. Verify mutation is detected by PHPCS
3. Store mutation in contrastive pair dataset
4. Judge must score mutated code as lower quality than original

## Quality Gates

**Phase 1 Real Code:**
1. PHPCS pre-filter (error_density ≤ 5.0 per 100 lines)
2. Claude judge assessment (9 dimensions, pass/fail)

**Phase 2 Synthetic Code:**
1. Claude judge assessment
2. If failed: One revision attempt with feedback
3. If still failed: Discard

**Phase 2 Mutated Code:**
1. Verify PHPCS detects mutation
2. Store bad->good pair

**Phase 2 Judge Training:**
1. Passed code: High score (100)
2. Failed code: Low score (0)
3. Mutated code: Controlled score (60-80 depending on mutation severity)

## Missing Test Infrastructure

**Not Yet Implemented:**
- Unit tests for helper functions (extraction, tokenization, tagging)
- Integration tests for phase-to-phase data flow
- Regression tests for judge consistency
- Property-based testing (e.g., "all mutations must be detectable")
- Performance benchmarks for API calls, PHPCS execution

**Recommended Future Testing:**
- Pytest with fixtures for test data loading
- Mocked Claude API responses for deterministic testing
- PHPCS command mocking to test prefilter logic
- Generator/judge correlation tests on held-out examples

---

*Testing analysis: 2026-03-26*

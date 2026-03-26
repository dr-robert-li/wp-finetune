# Coding Conventions

**Analysis Date:** 2026-03-26

## Naming Patterns

**Files:**
- Snake case with phase prefix: `phase1_extract.py`, `phase2_generate.py`, `phase3_cot.py`
- Helper scripts: descriptive names like `php_extract_functions.php`
- Export/utility scripts: action-based names like `export_dataset.py`

**Functions:**
- Snake case universally: `load_config()`, `extract_functions_from_file()`, `judge_synthetic()`, `revise_with_feedback()`
- Type hint suffixed names: `load_judge_system()`, `load_taxonomy()`, `load_style_anchors()`, `load_dataset()`
- Action-first naming: `mutate_remove_prepare()`, `mutate_remove_nonce()`, `mutate_remove_escaping()`, `mutate_remove_capability_check()`
- Verb-noun pairs: `synthesize_instruction()`, `generate_cot()`, `infer_task_type()`, `add_task_token()`

**Variables:**
- Snake case throughout: `tag_counts`, `total_functions`, `error_density`, `brace_depth`
- Descriptive plurals for collections: `all_functions`, `all_repos`, `all_examples`, `gaps`, `mutations`
- Context-prefixed for clarity: `function_name`, `current_class`, `current_docblock`, `gap_tag`
- Status/state variables: `in_function`, `in_class`, `passed`, `failed`, `processed`

**Types (Python):**
- Return type hints used: `-> dict`, `-> list[dict]`, `-> str`, `-> tuple[str, str]`
- Parameter type hints: `code: str`, `gap_tag: str`, `client: anthropic.Anthropic`
- No return type hints for `main()` functions (implicitly None)

**Constants:**
- UPPER_CASE: `PROJECT_ROOT`, `REPOS_DIR`, `CONFIG_PATH`, `REQUESTS_PER_MINUTE`, `REQUEST_INTERVAL`
- Path constants fully qualified: `EXTRACTED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "extracted"`

## Code Style

**Formatting:**
- No explicit linter detected; follows Python PEP 8 conventions
- Indentation: 4 spaces
- Line length: appears to target ~100 characters (some lines exceed for readability)
- Blank lines: 2 between functions, 1 for logical separation within functions

**Linting:**
- No `.eslintrc`, `.pylintrc`, or similar configuration file detected
- No explicit linting configuration — assumes Python developers follow PEP 8 by convention

**Comments:**
- Docstring format: Triple-quoted descriptions at function start
- Docstring style: Single-line for simple functions, multi-line for complex operations
- Inline comments sparse, reserved for non-obvious logic (e.g., token type matching, regex patterns)
- Section separators: `# ─── [Section Name] ────────────────────────────────` (decorative style in `phase2_mutate.py`)

**Docstring Examples:**

```python
def extract_functions_from_file(file_path: Path) -> list[dict]:
    """Use PHP tokenizer to extract functions from a PHP file."""
    # One-liner for simple utilities

def phpcs_prefilter(code: str, max_errors_per_100_lines: float = 5.0) -> dict:
    """Run PHPCS as a cheap pre-filter before sending to Claude.

    Returns dict with 'passed', 'errors', 'warnings', 'error_density'.
    Functions that fail PHPCS badly are rejected without spending API tokens.
    """
    # Multi-line for complex functions with side effects/rationale

def revise_with_feedback(original: dict, issues: list[str],
                         client: anthropic.Anthropic) -> str:
    """Attempt to revise a failed synthetic example using the judge's feedback."""
    # Includes context about what the function does
```

## Import Organization

**Order (Python):**
1. Standard library imports: `json`, `subprocess`, `sys`, `tempfile`, `time`, `pathlib`, `collections`, `random`, `re`, `itertools`
2. Third-party imports: `anthropic`, `yaml`

**Specific Example from `phase1_judge.py`:**

```python
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import anthropic
import yaml
```

**Path Aliases:**
- No path aliases (`@` style) used; standard `pathlib.Path` throughout
- Relative path construction via Path operations: `PROJECT_ROOT / "phase1_extraction" / "output" / "extracted"`
- Project root always resolved from script: `PROJECT_ROOT = Path(__file__).resolve().parent.parent`

## Error Handling

**Patterns Observed:**

1. **Try-Except with Specific Exceptions:**
   ```python
   try:
       result = subprocess.run(["php", str(PHP_EXTRACTOR), str(file_path)], ...)
       if result.returncode != 0:
           return []
       return json.loads(result.stdout)
   except (subprocess.TimeoutExpired, json.JSONDecodeError):
       return []
   ```
   - Catch specific exception types, not bare `except`
   - Return empty container (empty list/dict) on error to allow pipeline continuation

2. **JSON Parsing with Graceful Degradation:**
   ```python
   try:
       text = response.content[0].text
       if "```json" in text:
           text = text.split("```json")[1].split("```")[0]
       return json.loads(text.strip())
   except (json.JSONDecodeError, IndexError) as e:
       return {"verdict": "FAIL", "notes": f"Parse error: {e}", "scores": {}}
   ```
   - Extract from markdown code blocks before parsing
   - Return structured failure responses rather than raising

3. **Subprocess Error Handling:**
   ```python
   try:
       result = subprocess.run(["phpcs", ...], ...)
   except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
       return {"passed": True, "errors": -1, "warnings": -1, "error_density": 0, "skipped": True}
   finally:
       Path(tmp_path).unlink(missing_ok=True)
   ```
   - Use `check=False` or catch `CalledProcessError` rather than propagating
   - Cleanup in `finally` blocks; use `.unlink(missing_ok=True)` for safe deletion

4. **API Error Handling:**
   ```python
   try:
       revised_code = revise_with_feedback(example, issues, client)
       time.sleep(REQUEST_INTERVAL)
       re_assessment = judge_synthetic(revised_code, example["gap_tag"], client, judge_system)
   except anthropic.APIError:
       failed.append(example)
       total_failed += 1
   ```
   - Catch `anthropic.APIError` for API-specific failures
   - Treat as pipeline failure (append to `failed` list)

5. **Early Returns for Validation:**
   ```python
   if not repo_root.exists():
       print(f"  [{name}] Not cloned, skipping. Run phase1_clone.py first.")
       return []
   ```
   - Validate preconditions early
   - Use informative messages in stdout rather than raising

## Logging

**Framework:** `print()` — no dedicated logging library used

**Patterns:**
- Progress logging with repo/file context: `print(f"  [{name}] Found {len(php_files)} PHP files")`
- Bracketed context for clarity: `[phase_name]`, `[repo_name]`, `[file_name]`
- Milestone messages: `print("Done. Run phase1_extract.py next.")`
- Tabular reports for summary data (phase2_gap_analysis.py):
  ```python
  print(f"{'Tag':<40} {'Have':>6} {'Need':>6} {'Gap':>6} {'%':>6}")
  print(f"{'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
  ```
- Exit with `sys.exit(1)` for fatal errors and informative message

## Rate Limiting

**Pattern (constants at module level):**
```python
REQUESTS_PER_MINUTE = 50
REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE

# In loop:
time.sleep(REQUEST_INTERVAL)
```

Observed constants:
- `phase1_judge.py`: 50 req/min
- `phase2_judge.py`: 50 req/min
- `phase2_generate.py`: 40 req/min
- `phase3_cot.py`: 40 req/min

**Application:** Unconditional sleep after API calls in loops (not adaptive backoff)

## Function Design

**Size:** Most functions 10-30 lines; longest (e.g., `auto_tag_function()` in `phase1_judge.py`) ~80 lines due to many conditional tag assignments

**Parameters:**
- Prefer explicit params over kwargs: `def mutate_remove_prepare(code: str) -> tuple[str, str]`
- Complex operations pass client object: `def judge_synthetic(code: str, gap_tag: str, client: anthropic.Anthropic, system: str) -> dict`
- Optional parameters with defaults: `max_errors_per_100_lines: float = 5.0`

**Return Values:**
- Return tuples for paired values: `-> tuple[str, str]` (bad_code, explanation)
- Return dicts for structured results: `-> dict` (assessment, metadata)
- Return lists for collections: `-> list[dict]` (functions, examples)
- Return None implicitly for `main()` functions

**Mutation Functions Pattern (phase2_mutate.py):**
```python
def mutate_remove_prepare(code: str) -> tuple[str, str]:
    """Remove $wpdb->prepare() and inline values directly (SQL injection)."""
    pattern = r'\$wpdb->prepare\(\s*(["\'])(.*?)\1\s*,\s*(.*?)\)'
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        return None, None  # Signal no mutation applied
    # ... apply mutation ...
    return bad_code, "sql_injection: Removed $wpdb->prepare(), concatenating variables directly into SQL"
```

## Module Design

**Entry Points:**
- All scripts use `if __name__ == "__main__": main()` pattern
- `main()` function orchestrates entire script logic

**Exports:**
- No explicit exports; scripts meant to be run directly (not imported)
- Helper functions defined at module level (e.g., `extract_dependencies()`, `extract_sql_patterns()`)

**Barrel Files:**
- Not used; no `__init__.py` or aggregation patterns

**Configuration Loading:**
- Always at module start via constants: `CONFIG_PATH = PROJECT_ROOT / "config" / "repos.yaml"`
- Loaded lazily on demand: `config = load_config()` inside `main()`

**Directory Structure:**
```
scripts/
├── phase1_clone.py          # Data collection
├── phase1_extract.py        # Parsing/tokenization
├── phase1_judge.py          # Quality assessment
├── phase2_gap_analysis.py   # Coverage analysis
├── phase2_generate.py       # Synthetic generation
├── phase2_mutate.py         # Contrastive pairs
├── phase2_judge.py          # Synthetic assessment
├── phase2_judge_dataset.py  # Judge training data
├── phase3_cot.py            # Reasoning synthesis
├── export_dataset.py        # Format export
└── php_extract_functions.php # Tokenizer (called via subprocess)
```

---

*Convention analysis: 2026-03-26*

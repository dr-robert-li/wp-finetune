# Testing Patterns

**Analysis Date:** 2026-03-31

## Test Framework

**Runner:**
- pytest (no version pinned -- installed via pip)
- Config: No `pytest.ini`, `pyproject.toml`, or `setup.cfg` -- pytest discovers tests via naming convention

**Assertion Library:**
- Built-in `assert` statements (pytest rewrites)
- No additional assertion libraries (no `assertpy`, `hamcrest`, etc.)

**Run Commands:**
```bash
pytest tests/                    # Run all tests
pytest tests/test_utils.py       # Run single module
pytest -v tests/                 # Verbose output
pytest -k "test_gate"            # Filter by name
```

## Test File Organization

**Location:**
- All tests in `tests/` directory at project root (separate from source, not co-located)

**Naming:**
- Files: `test_{module_name}.py` matching source module
- Classes: `TestPascalCase` grouping related tests
- Functions: `test_descriptive_name` with snake_case

**Structure:**
```
tests/
├── __init__.py                    # Empty namespace package
├── test_config.py                 # Config file correctness (judge rubric, thresholds)
├── test_csv_to_repos.py           # CSV-to-repos conversion
├── test_eval_gate.py              # Quality gate pass/fail logic
├── test_eval_gen.py               # Generation evaluation (PHPCS, security)
├── test_eval_judge.py             # Judge evaluation (Spearman, score inversion)
├── test_export.py                 # Dataset export (ratios, dedup, weights)
├── test_phase2_judge_dataset.py   # Judge dataset patterns (backoff, checkpoint usage)
├── test_phase2_mutate.py          # PHPCS hard-fail guard
├── test_pipeline_integration.py   # Phase 1 checkpoint skip behavior
├── test_preflight.py              # Preflight check pass/fail
├── test_prepare_tokenizer.py      # Special token addition, embedding init
├── test_train_model.py            # Training config, dataset schema, model checks
└── test_utils.py                  # JSON extraction, checkpoints, backoff, routing
```

**Fixtures directory:**
- `tests/fixtures/` contains sample CSV data for `test_csv_to_repos.py`

## Test Structure

**Suite Organization:**
```python
# Class-based grouping (test_train_model.py):
class TestLoraConfigParams:
    """test_lora_config_params -- assert key hyperparameters are set correctly."""

    def test_lora_r(self):
        cfg = load_train_config()
        assert cfg["lora"]["r"] == 32

    def test_bf16_enabled(self):
        cfg = load_train_config()
        assert cfg["training"]["bf16"] is True

# Function-based (test_utils.py):
def test_extract_json_raw():
    """Strategy 1: raw JSON string."""
    result = extract_json('{"verdict":"PASS"}')
    assert result == {"verdict": "PASS"}
```

**Patterns:**
- **Wave 0 / TDD stubs:** Tests are written before implementation. Module docstrings explicitly state this: `"""Wave 0 tests for training config -- run before implementation."""`
- **No shared setup/teardown:** Each test is self-contained. `tmp_path` fixture from pytest used for temp directories.
- **Descriptive docstrings on every test:** Each test function has a docstring explaining what it validates.

## Mocking

**Framework:** `unittest.mock` (stdlib)

**Patterns:**

1. **Subprocess mocking** (most common pattern, used in `test_preflight.py`, `test_eval_gen.py`, `test_phase2_mutate.py`):
```python
def _make_run(returncode=0, stdout=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    return result

def test_missing_phpcs():
    def side_effect(cmd, **kwargs):
        if "phpcs" in cmd:
            return _make_run(returncode=1)
        return _make_run(returncode=0)

    with patch("subprocess.run", side_effect=side_effect):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight()
    assert exc_info.value.code == 1
```

2. **API client mocking** (`test_utils.py`):
```python
def test_backoff_retries():
    mock_client = MagicMock()
    rate_limit_error = anthropic.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    mock_client.messages.create.side_effect = [
        rate_limit_error, rate_limit_error, rate_limit_error, success_response,
    ]
    with patch("time.sleep"):
        result = call_with_backoff(mock_client, max_retries=5, ...)
    assert mock_client.messages.create.call_count == 4
```

3. **Module-level patching** for pipeline integration tests (`test_pipeline_integration.py`):
```python
with (
    patch.object(clone_mod, "load_config", return_value=mock_config),
    patch("scripts.phase1_clone.load_checkpoint", return_value=CHECKPOINT_COMPLETED),
    patch("scripts.phase1_clone.save_checkpoint"),
    patch("scripts.phase1_clone.clone_repo") as mock_clone,
):
    clone_mod.main()
mock_clone.assert_not_called()  # Verify skipped
```

4. **Tokenizer mocking** (`test_prepare_tokenizer.py`) -- custom fixture that simulates vocab behavior:
```python
@pytest.fixture()
def small_tokenizer():
    vocab = {f"tok{i}": i for i in range(100)}
    tok = MagicMock()
    def add_special_tokens(d: dict) -> int:
        new_tokens = [t for t in d.get("additional_special_tokens", []) if t not in vocab]
        for t in new_tokens:
            vocab[t] = len(vocab)
        return len(new_tokens)
    tok.add_special_tokens.side_effect = add_special_tokens
    return tok, vocab
```

**What to Mock:**
- External processes: `subprocess.run` (phpcs, php, docker)
- API clients: `anthropic.Client`, `anthropic.RateLimitError`
- Environment variables: `patch.dict(os.environ, ...)`
- Heavy ML imports: Not mocked -- tests that need them use `pytest.mark.skipif`
- `time.sleep`: Always mock in backoff tests

**What NOT to Mock:**
- Pure functions: `extract_json()`, `batch_or_direct()`, `infer_task_type()`, `enforce_ratio()`
- Config file reading: Tests read actual `config/train_config.yaml` for config validation tests
- `pathlib.Path` operations: Use `tmp_path` fixture instead of mocking

## Fixtures and Factories

**Test Data:**
```python
# Factory functions (test_export.py):
def _make_gen_example(idx: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a WordPress expert."},
            {"role": "user", "content": f"Write a WordPress function #{idx}."},
            {"role": "assistant", "content": f"<?php function example_{idx}() {{ return {idx}; }}"},
        ],
        "metadata": {"source": "phase1_real", "task_type": "gen", "training_tags": []},
    }

def _make_judge_example(idx: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a WordPress expert."},
            {"role": "user", "content": f"Review this WordPress code #{idx}."},
            {"role": "assistant", "content": f'{{"overall_score": 8, "verdict": "PASS"}}'},
        ],
        "metadata": {"source": "phase2_judge", "task_type": "judge", "training_tags": []},
    }
```

**Location:**
- Fixtures defined at module top or as pytest fixtures
- CSV fixture files in `tests/fixtures/`
- No shared conftest.py

## Coverage

**Requirements:** None enforced (no coverage config or CI gate)

**View Coverage:**
```bash
pytest --cov=scripts --cov=eval tests/   # Requires pytest-cov
```

## Test Types

**Unit Tests (majority):**
- Pure function testing: `extract_json()`, `batch_or_direct()`, `compute_pass_rate()`, `invert_phpcs_errors()`
- Config validation: Verify YAML config has expected keys and values
- Static analysis: Check source code contains required patterns (e.g., `output_router_logits=True` string present)

**Integration Tests:**
- Checkpoint skip behavior: `test_pipeline_integration.py` -- verifies that `main()` of clone/extract scripts skips already-completed repos
- Pattern enforcement: `test_phase2_judge_dataset.py` -- verifies module uses `call_with_backoff` not `time.sleep(REQUEST_INTERVAL)`

**Conditional/Skip Tests:**
- Dataset-dependent tests use `@pytest.mark.skipif`:
```python
@pytest.mark.skipif(not TRAIN_JSONL.exists(), reason="openai_train.jsonl not present")
def test_messages_key_present(self):
    with open(TRAIN_JSONL) as f:
        record = json.loads(f.readline().strip())
    assert "messages" in record
```

**E2E Tests:** Not used (no end-to-end test framework)

## Common Patterns

**Async Testing:** Not used (no async code)

**Error Testing:**
```python
# SystemExit testing (preflight, mutate):
def test_missing_phpcs():
    with patch("subprocess.run", side_effect=FileNotFoundError("phpcs not found")):
        with pytest.raises(SystemExit):
            _require_phpcs()

# Return None for parse failures:
def test_extract_json_failure_no_json():
    result = extract_json('no json here at all')
    assert result is None
```

**Config Assertion Pattern:**
```python
# Verify config files contain required content:
def test_judge_threshold_v2():
    text = JUDGE_SYSTEM_PATH.read_text()
    assert ">= 8" in text, "judge_system.md must require ALL dimensions >= 8"

def test_security_auto_fail():
    text = JUDGE_SYSTEM_PATH.read_text()
    assert "SECURITY AUTO-FAIL" in text
    assert "< 5" in text
```

**Source Inspection Pattern** (behavioral enforcement via `inspect.getsource()`):
```python
def test_rate_limiting_uses_backoff():
    source = inspect.getsource(jd)
    assert "call_with_backoff" in source, (
        "phase2_judge_dataset.py must use call_with_backoff for rate limiting"
    )

def test_no_time_sleep_request_interval_pattern():
    source = inspect.getsource(jd)
    assert "REQUEST_INTERVAL" not in source
```

**Temporary Path Pattern:**
```python
def test_checkpoint_roundtrip(tmp_path):
    state = {"completed": ["a", "b"], "failed": ["c"], "batch_job_ids": []}
    save_checkpoint("test", state, checkpoint_dir=tmp_path)
    loaded = load_checkpoint("test", checkpoint_dir=tmp_path)
    assert loaded["completed"] == ["a", "b"]
```

## Runtime Quality Gates (In-Pipeline)

Beyond unit tests, the pipeline has runtime validation:

**Pre-training gates:**
- `scripts/preflight.py`: ANTHROPIC_API_KEY, php, phpcs, WordPress-Extra standard
- `scripts/train_model.py:check_memory()`: Minimum 70 GB free RAM
- `scripts/train_model.py:load_model_and_tokenizer()`: Special token verification via assertions

**Post-training gates:**
- `eval/eval_gate.py`: Reads thresholds from `config/train_config.yaml` eval section
- Gates: overall_mean >= 75.0, overall_spearman >= 0.80, phpcs_pass >= 0.95, security_pass >= 0.98
- Per-dimension gen targets: D2_security >= 0.95, D3_sql >= 0.90
- Per-dimension judge targets: D2_security >= 0.75, D3_sql >= 0.75
- Exits non-zero if any gate fails

**Mutation verification:**
- `scripts/phase2_mutate.py:verify_mutation_detectable()`: Every mutation must be PHPCS-detectable
- `scripts/phase2_mutate.py:_require_phpcs()`: Hard exit if PHPCS unavailable (no silent fallback)

**Adaptive resource monitoring:**
- `MemoryWatchdogCallback`: Monitors `/proc/meminfo` every training step
- Thermal telemetry: logged to `telemetry/training/*.jsonl` for adaptive planning between runs

## Adding New Tests

When adding a new pipeline script or utility:

1. Create `tests/test_{module_name}.py`
2. Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at top
3. Import the functions under test directly
4. Write Wave 0 tests first (before implementation) using mocks
5. Use `tmp_path` for any filesystem operations
6. Mock `subprocess.run` for external tool calls
7. Mock API clients with `MagicMock()` and `side_effect`
8. Use `pytest.raises(SystemExit)` for hard-exit validation
9. Use factory functions like `_make_gen_example()` for test data

---

*Testing analysis: 2026-03-31*

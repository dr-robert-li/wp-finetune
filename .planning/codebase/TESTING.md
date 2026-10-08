---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
# Testing Patterns

**Analysis Date:** 2026-10-09

## Test Framework

**Runner:**
- pytest (version not pinned; no `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini` in the repo)
- Config: none. Discovery uses pytest defaults. Only `tests/conftest.py` and `tests/phase4_4/conftest.py` configure fixtures.
- `unittest.mock` is used alongside pytest (`MagicMock`, `mock.patch.object`) in about 10 test files.

**Assertion Library:**
- Plain `assert` statements (pytest rewriting). `pytest.raises` for exceptions (22 uses across `tests/`).

**Run Commands:**

```bash
python -m pytest tests/ -q                          # Run the whole suite (from repo root)
python -m pytest tests/test_utils.py -x -q          # Single module, stop on first failure
python -m pytest tests/phase4_4/ -q                 # One sub-package (has its own conftest)
python -m pytest tests/ -k gb10                     # Keyword filter
python -m pytest scripts/test_assemble_reasoning_dataset.py -q   # Tests that live inside scripts/
```

Run from the repo root. Every test module inserts the project root into `sys.path` itself, so imports like `from scripts.utils import ...` and `from eval.eval_gate import ...` resolve without installing the package.

No watch-mode, coverage, or parallel-run command is configured in the repo.

## Test File Organization

**Location:**
- Separate tree. All tests live in `tests/`, not next to source.
- Two exceptions: `scripts/test_assemble_reasoning_dataset.py` and `scripts/test_validate_reasoning_consistency.py` sit beside the code they test.
- `tests/phase4_4/` is a sub-package (with `__init__.py` and its own `conftest.py`) for the Phase 4.4 REVL-series tests (about 14 modules).

**Naming:**
- Module: `tests/test_<source_module>.py` mirrors the source it covers. Example: `scripts/utils.py` is tested by `tests/test_utils.py`; `eval/eval_gate.py` by `tests/test_eval_gate.py`; `scripts/rl_judge_dispatch.py` by `tests/test_rl_judge_dispatch.py`.
- Phase or ticket prefixes appear in some names: `tests/test_reward_fix1_judge.py`, `tests/test_reward_fix2_gen.py`, `tests/phase4_4/test_revl03_evaluator_output.py`.
- Functions: `test_<behaviour>` or `test_<function>_<scenario>`. Examples: `test_extract_json_fenced_json`, `test_extract_json_failure_empty`, `test_checkpoint_atomic`, `test_gate_fail_phpcs`.

**Structure:**

```
tests/
├── __init__.py
├── conftest.py                  # Shared session/function fixtures (reward pipeline, mocks)
├── _rl_fixtures.py              # Helper module (underscore prefix = not a test file)
├── fixtures/
│   └── reward_integration_cases/
│       ├── known_good_php/      # ~25 PHP files the reward gate must accept
│       └── known_bad_php/       # ~24 PHP files the reward gate must reject
├── phase4_4/
│   ├── __init__.py
│   ├── conftest.py              # Synthetic judge-response builders + tmp fixtures
│   └── test_*.py
└── test_*.py                    # ~60 flat modules
```

## Test Structure

**Suite Organization:**
Two styles coexist. Flat module-level functions dominate (211 top-level `def test_*`). Grouped `class Test*` suites appear in about 38 files (563 indented test methods in total). Group by class when a module has several related behaviours that share setup, such as `TestSpecialTokensAdded` or `TestEmbeddingsMeanInit`.

```python

# tests/test_utils.py (module-level style, with section banners)

"""Unit tests for scripts/utils.py. ..."""
import sys
from pathlib import Path
import pytest

# Add project root to path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import extract_json, load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------

# extract_json tests

# ---------------------------------------------------------------------------

def test_extract_json_fenced_json():
    """Strategy 2: ```json fenced block."""
    result = extract_json('```json\n{"verdict":"PASS"}\n```')
    assert result == {"verdict": "PASS"}
```

**Patterns:**
- Each test has a one-line docstring stating the case, often naming the strategy or branch it exercises (`"""Strategy 1: raw JSON string."""`). Keep this habit.
- Section banners (`# ----` with a title) separate groups of tests inside a module.
- Helper builders are prefixed with `_` and placed above the tests that use them. Example: `_full_thresholds()` and `_full_results()` in `tests/test_eval_gate.py`; `_judge_response()` in `tests/phase4_4/conftest.py`.
- Setup happens inside the test or in a fixture. There are no `setUp`/`tearDown` methods in the pytest-style files.
- Teardown is handled by `tmp_path` and `monkeypatch`, which clean up automatically. No explicit `finally` cleanup is needed.

**Module-level skip guards:**

```python

# tests/test_reward_fix2_gen.py

pytestmark = pytest.mark.skipif(
    shutil.which("php") is None,
    reason="php binary required for php -l parseability tiers",
)
```

## Fixtures and Factories

**Shared fixtures in `tests/conftest.py`:**
- `recalib_json` (session scope): writes a `judge_recalibration.json` into a `tmp_path_factory` directory.
- `php_fixture_dir` (session scope): returns `tests/fixtures/reward_integration_cases`.
- `mock_judge_client` (function scope): `MagicMock` standing in for `openai.OpenAI`, returning `{"overall_score": 75}` from `chat.completions.create`.
- `sample_rollout_group` (function scope): four small PHP snippets for rollout-group tests.
- `mock_tinker_client` (function scope): mock Tinker training client. It explicitly sets both `forward_backward` (GRPO token-level path) and `forward_backward_custom` (GSPO sequence-level path) to the same output object.

```python

# tests/conftest.py

@pytest.fixture
def mock_judge_client():
    """MagicMock openai.OpenAI client that returns a parseable JSON overall_score."""
    from unittest.mock import MagicMock

    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"overall_score": 75})
    client.chat.completions.create.return_value = mock_resp
    return client
```

Note that the fixtures import `MagicMock` inside the function body, so `conftest.py` still collects when optional dependencies are absent.

**Fixture scope rules used here:**
- `session` scope for read-only or expensive values: fixture-file paths, one-time tmp artifacts.
- `function` scope for anything a test may mutate: mock clients, sample lists.

**Phase 4.4 fixtures (`tests/phase4_4/conftest.py`):**
- `_judge_response(idx, with_fix=False)` builds a realistic close-only judge response: prose, a `[/REASONING]` tag (no opening tag), and a `<judge_output>{...}</judge_output>` block. The docstring warns against adding a paired `[REASONING]` open tag because that would pass tests while the real format extracts nothing. Keep fixtures matched to the real data format.
- `tmp_pairs_jsonl(tmp_path)` writes 20 captured-response rows in the same format.

**Test data location:**
- Inline strings for small inputs (JSON, PHP snippets).
- `tests/fixtures/` for PHP corpora used by the reward gate.
- Real data under `data/`, `output/`, and `models/` is referenced by absolute path from `PROJECT_ROOT` in some tests. These tests are guarded with `skipif` (see Test Types).

## Mocking

**Frameworks:**
- `monkeypatch` (pytest) is the dominant tool, with about 130 uses. It is used to replace module attributes, environment variables, and fake functions.
- `unittest.mock` (`MagicMock`, `mock.patch.object`) is used for client objects and selective attribute patches. About 10 files use it.

**Patterns:**

```python

# tests/test_rl_judge_dispatch.py: replace a module-level function with a fake

def fake_generate_json(*args, **kwargs):
    ...
monkeypatch.setattr(rl_judge_dispatch, "generate_json", fake_generate_json)
```

```python

# tests/test_rl_rollouts_reward_shape.py: context-managed patch of a module constant

with mock.patch.object(rp, "_REWARD_SEC_TRIGGERS", set()):
    ...
```

```python

# tests/test_rl_codegen_tripwire.py: monkeypatch.setattr(...) on a module attribute

monkeypatch.setattr(...)   # used repeatedly to swap collaborators in a single test
```

**What to Mock:**
- External LLM/HTTP clients (`openai`, Tinker training client, vLLM endpoints). Use `mock_judge_client` and `mock_tinker_client` from `tests/conftest.py`.
- Slow or GPU-bound model calls. Replace the function that calls the model with a fake returning canned JSON.
- Environment variables that control behaviour: use `monkeypatch.setenv` / `monkeypatch.delenv`.
- Subprocess or external binaries when the test is about logic, not the binary. Example: `shutil.which("php")` is checked, then the test is skipped if PHP is absent.

**What NOT to Mock:**
- Pure helpers such as `extract_json`, `load_checkpoint`, and `save_checkpoint` (`tests/test_utils.py` exercises the real file system with `tmp_path`).
- The checkpoint atomicity contract. `test_checkpoint_atomic` checks the real `.json` file and asserts that no `.tmp` file remains.
- Gate logic in `eval/eval_gate.py`. The tests call `check_gates()` directly with full dicts from `_full_results()` and `_full_thresholds()`.

## Test Data and Real-Artifact Guards

Many tests read real project artifacts. Each such test must be guarded so the suite still runs on a machine that lacks the data:

```python

# tests/test_calib_join.py

@pytest.mark.skipif(not SIDECAR.exists(), reason="sidecar data file not present")
def test_real_pool_join_coverage():
    ...
```

```python

# tests/test_launch_validated_smoke.py

@pytest.mark.skipif(
    not REAL_SWEEP.exists(),
    reason="Real sweep_results.json not present in test env",
)
```

```python

# tests/test_train_model.py (method-level)

@pytest.mark.skipif(not TRAIN_JSONL.exists(), reason="openai_train.jsonl not present")
```

Rule: a skip reason must name the missing artifact. Do not write a bare `skip` without a reason.

## Coverage

**Requirements:** None enforced. No coverage threshold, no `.coveragerc`, no CI gate. `.coverage` and `htmlcov/` are gitignored, so coverage reports are optional local output.

**View Coverage:**

```bash
python -m pytest tests/ --cov=scripts --cov=eval --cov-report=term-missing   # requires pytest-cov, not pinned in repo
```

## Test Types

**Unit Tests:**
- The majority of the suite. Pure functions and gate logic, using `tmp_path`, `monkeypatch`, and inline data.
- Examples: `tests/test_utils.py`, `tests/test_eval_gate.py`, `tests/test_bootstrap_ci.py`, `tests/test_concentration.py`, `tests/test_jaccard_stability.py`.

**Integration Tests:**
- Modules named `*_integration.py` or `test_*_integration*` exercise several components together, usually with the fixture corpora: `tests/test_pipeline_integration.py`, `tests/test_eval_integration.py`, `tests/test_rl_train_integration.py`, `tests/test_reward_pipeline_integration.py`.
- The PHP reward integration uses `tests/fixtures/reward_integration_cases/known_good_php/` and `known_bad_php/`. Tests run `php -l` on these, so they need the `php` binary.
- Tests needing a GPU, vLLM server, or model weights are skipped or do not exist in the default run. The skip pattern above keeps them from failing on dev machines.

**Source-Guard Tests (static checks on code):**
- `tests/test_gb10_load_safety.py` parses every `scripts/*.py` file with `ast` and fails if any call passes `device_map="auto"`. It is parametrized over the file list:

```python

# tests/test_gb10_load_safety.py

@pytest.mark.parametrize("py_file", sorted(SCRIPTS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_no_bare_device_map_auto(py_file: Path) -> None:
    """No script may load a model with the GB10-unsafe device_map="auto"."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    hits = _auto_device_map_calls(tree)
    assert not hits, ...
```

- `tests/test_antihack.py` and `tests/test_rl_judge_dispatch.py` also use `ast.parse` for structural checks.
- Prefer this style when a rule must hold across many files: an AST or grep-based test over the source tree catches regressions a unit test would miss.

**E2E Tests:**
- Not used. No browser or full-pipeline run is part of the suite. Full pipeline runs are done manually through `scripts/` entry points.

## Common Patterns

**Temporary files:**

```python
def test_checkpoint_roundtrip(tmp_path):
    """save_checkpoint then load_checkpoint returns identical state."""
    state = {"completed": ["a", "b"], "failed": ["c"], "batch_job_ids": []}
    save_checkpoint("test", state, checkpoint_dir=tmp_path)
    loaded = load_checkpoint("test", checkpoint_dir=tmp_path)
    assert loaded["completed"] == ["a", "b"]
```

Use `tmp_path` (about 177 uses) rather than `tempfile` or hard-coded paths. The source functions accept an explicit directory override (`checkpoint_dir=`) so tests never write into `data/checkpoints/`.

**Async Testing:**
- Rare. Only one test module under `tests/` uses asyncio (`grep -l "async def test_\|asyncio.run" tests/`). Write sync tests by default; wrap async calls with `asyncio.run(...)` inside the test instead of adding a plugin.

**Error Testing:**

```python

# tests/test_sieve_vllm_patch.py

with pytest.raises(RuntimeError, match="no MoE-block class resolved"):
    ...
```

- Assert on the exception type and a `match=` substring. Do not use bare `pytest.raises(Exception)`.
- Project exceptions subclass `RuntimeError` (for example `VllmBootTimeout` in `scripts/_p0_vllm_smoke_serve.py` and `PreconditionError` in `scripts/fidelity_gate_v3.py`), so tests can catch either the subclass or `RuntimeError`.

**Output capture:**
- `capsys` (3 uses) for asserting on `print` output from scripts. Prefer asserting on return values; use `capsys` only when the output itself is the contract.

**Parametrization:**
- `@pytest.mark.parametrize` is rare (1 use). Where several cases share one assertion, write the cases as separate small test functions with clear names, or parametrize with `ids=` for readable failures, as `test_gb10_load_safety.py` does.

## Adding a New Test

1. Create `tests/test_<module>.py` matching the source file name.
2. Add the `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` line before importing project modules, unless the module is in `tests/phase4_4/`, which handles it in its `conftest.py`.
3. Use `tmp_path` for all file output and `monkeypatch` for swapping collaborators.
4. Put reusable builders in `tests/conftest.py` only if two or more modules need them. Otherwise keep them local with a `_` prefix.
5. Guard any test that needs real artifacts with `skipif(..., reason="<artifact name> not present")`.
6. For a rule that must hold across `scripts/`, write an AST or grep-based guard test, following `tests/test_gb10_load_safety.py`.

---

*Testing analysis: 2026-10-09*

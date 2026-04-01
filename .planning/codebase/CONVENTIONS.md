# Coding Conventions

**Analysis Date:** 2026-03-31

## Naming Patterns

**Files:**
- Use `snake_case.py` for all Python modules: `train_model.py`, `pipeline_orchestrator.py`, `eval_gate.py`
- Pipeline phase scripts follow `phase{N}_{action}.py` pattern: `phase1_clone.py`, `phase1_extract.py`, `phase2_generate.py`, `phase2_mutate.py`, `phase3_cot.py`
- Test files mirror source: `test_{module_name}.py` in `tests/` directory
- Config files use `snake_case.yaml`: `train_config.yaml`, `repos.yaml`, `taxonomy.yaml`

**Functions:**
- Use `snake_case` for all functions: `load_config()`, `extract_json()`, `check_memory()`
- Private/internal functions prefixed with underscore: `_available_mb()`, `_get_checkpoint_dir()`, `_build_parser()`, `_require_phpcs()`
- Entry-point functions named `main()` for pipeline scripts, `train()` for training script
- Action-first naming: `mutate_remove_prepare()`, `mutate_remove_nonce()`, `verify_mutation_detectable()`

**Variables:**
- Use `snake_case` for local variables and parameters: `train_dataset`, `config_path`, `resume_checkpoint`
- Use `UPPER_SNAKE_CASE` for module-level constants: `PROJECT_ROOT`, `CONFIG_PATH`, `MIN_FREE_MEMORY_GB`, `BATCH_THRESHOLD`, `OOM_WATCHDOG_THRESHOLD_MB`
- Path constants use `_DIR` or `_PATH` suffix: `PASSED_DIR`, `EXTRACTED_DIR`, `CONFIG_PATH`, `TRAIN_JSONL`

**Types/Classes:**
- Use `PascalCase` for classes: `MemoryWatchdogCallback`, `DGXToolbox`, `CheckResult`, `ValidationResult`, `ExecResult`
- Dataclasses use `@dataclass` decorator with type annotations: see `scripts/dgx_toolbox.py`
- Frozen dataclasses for immutable data: `@dataclass(frozen=True)` in `eval/rubric_definitions.py`

## Code Style

**Formatting:**
- No formatter config file (no `pyproject.toml`, `ruff.toml`, or `black` config)
- Indentation: 4 spaces (standard Python)
- Line length: ~100-120 characters observed, no enforced limit
- Use f-strings for string interpolation throughout: `f"Loading model from {local_dir} ..."`
- 2 blank lines between top-level functions, 1 for logical separation within functions

**Linting:**
- No linting config file (no `.flake8`, `ruff.toml`, or `pylintrc`)
- Inline `noqa` comments used for intentional suppressions: `# noqa: PLC0415` (deferred imports), `# noqa: F401` (imported but unused), `# noqa: ARG001` (unused argument)
- No CI linting pipeline detected

## Import Organization

**Order:**
1. `from __future__ import annotations` (when used, always first)
2. Standard library imports: `json`, `sys`, `time`, `re`, `subprocess`, `pathlib`, `argparse`
3. Third-party imports: `yaml`, `torch`, `anthropic`, `pytest`
4. Local/project imports: `from scripts.utils import ...`, `from eval.rubric_scorer import ...`

**Deferred Imports:**
- Heavy ML libraries are imported inside functions to avoid startup cost:
  ```python
  def load_model_and_tokenizer(config: dict):
      from unsloth import FastLanguageModel  # noqa: PLC0415
      from transformers import AutoTokenizer  # noqa: PLC0415
  ```
- This pattern is used consistently in `scripts/train_model.py` for `unsloth`, `transformers`, `mlflow`, `trl`, `datasets`

**Path Aliases:**
- No path aliases or import shortcuts. All imports use full dotted paths: `from scripts.utils import extract_json`
- `sys.path.insert(0, ...)` used in test files to ensure project root is on path

## Error Handling

**Patterns:**

1. **Fail-open for non-critical checks:** Return safe defaults when checks cannot run. Used in `MemoryWatchdogCallback._available_mb()` (`scripts/train_model.py`):
   ```python
   except Exception:
       pass
   return 999_999  # fail-open: if we can't read, don't block training
   ```

2. **Hard exit for critical preconditions:** Use `sys.exit(1)` with descriptive error for must-have requirements. Used in `scripts/preflight.py`, `scripts/phase2_mutate.py`:
   ```python
   except FileNotFoundError:
       print("ERROR: phpcs not found...", file=sys.stderr)
       sys.exit(1)
   ```

3. **Graceful degradation with fallback chain:** Try primary method, then fallback, then skip. Used in `scripts/train_model.py` memory check:
   ```python
   try:
       meminfo = Path("/proc/meminfo").read_text()
   except Exception:
       try:
           import psutil
       except ImportError:
           print("Warning: Cannot check memory. Proceeding anyway.")
           return
   ```

4. **Exponential backoff for API calls:** Centralized in `scripts/utils.py:call_with_backoff()` with jitter, retry-after header support, and max retries. All API-calling scripts must use this instead of raw `client.messages.create()`.

5. **Atomic writes for state:** Checkpoint saves use write-to-tmp-then-rename pattern in `scripts/utils.py:save_checkpoint()` to prevent partial reads.

6. **Structured validation results:** `scripts/dgx_toolbox.py` uses `CheckResult` and `ValidationResult` dataclasses for composable validation:
   ```python
   result = dgx.validate(["toolbox", "training_data", "config", "memory:70"])
   if not result.ok:
       print(result.report())
       sys.exit(1)
   ```

## Logging

**Framework:** `print()` statements (no logging framework)

**Patterns:**
- Use `print()` for all output; no `logging` module usage
- Errors go to stderr: `print("ERROR: ...", file=sys.stderr)`
- Progress indicators use formatted banners:
  ```python
  print(f"\n{'=' * 60}")
  print(f"  MEMORY PRE-CHECK")
  print(f"{'=' * 60}")
  ```
- Status lines use fixed-width formatting for alignment: `f"  Total system memory: {total_gb:.1f} GB"`
- Unicode check/cross marks for pass/fail: `"✓"` and `"✗"`

## Comments

**When to Comment:**
- Module-level docstrings describe full pipeline context, usage, and invocation: every `scripts/*.py` and `eval/*.py` file has a comprehensive module docstring
- Section separators use `# ── Section Name ──` or `# ---------------------------------------------------------------------------`
- Inline comments explain non-obvious decisions, especially locked hyperparameters:
  ```python
  load_in_4bit=False,  # LOCKED — no QLoRA for MoE
  modules_to_save=lora_cfg["modules_to_save"],  # ["embed_tokens", "lm_head"] — LOCKED
  ```
- Reference ticket IDs in comments: `# TRNG-04`, `# TRNG-05`, `# PIPE-03`

**Docstrings:**
- Use triple-quoted docstrings on all public functions
- Include `Args:` and `Returns:` sections for complex functions (Google style)
- Class docstrings describe purpose and design rationale (see `MemoryWatchdogCallback`, `DGXToolbox`)

## Function Design

**Size:** Functions are generally focused and under 50 lines. Larger functions (like `get_status()` in `scripts/pipeline_orchestrator.py`) aggregate data and are still single-purpose.

**Parameters:**
- Use `config: dict` pattern -- load YAML config once, pass dict through pipeline
- Use `Optional[Path]` with default `None` for overridable paths (useful for testing): `checkpoint_dir: Optional[Path] = None`
- CLI args use `argparse.Namespace`

**Return Values:**
- Return `dict` for structured data (status, config, results)
- Return `tuple` for multiple values: `(successes, failures)` from `parse_batch_results()`
- Return `None` for failure/not-found cases: `extract_json()` returns `None` on parse failure
- Return `bool` for checks: `verify_mutation_detectable()` returns `True`/`False`
- Return `(None, None)` tuple to signal "no mutation applied" in mutation functions

## Module Design

**Exports:** No `__all__` lists. Import specific names from modules: `from scripts.utils import extract_json, call_with_backoff, load_checkpoint, save_checkpoint`

**Barrel Files:** `scripts/__init__.py`, `eval/__init__.py`, and `tests/__init__.py` exist but are empty (namespace packages)

**Singleton Pattern:** Used in `scripts/dgx_toolbox.py`:
```python
_instance: DGXToolbox | None = None

def get_toolbox() -> DGXToolbox:
    global _instance
    if _instance is None:
        _instance = DGXToolbox()
    return _instance
```

**Dataclass Pattern:** Used for structured results in `scripts/dgx_toolbox.py`:
```python
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
```

## Idempotency Pattern

Every pipeline script checks whether its output already exists before running. Follow this pattern for new scripts:

```python
# At start of train():
adapter_config = output_dir / "adapter_config.json"
if adapter_config.exists() and not args.resume and not args.dry_run:
    print(f"Trained adapter already exists at {output_dir}/adapter_config.json")
    return
```

Checkpoint-based idempotency for multi-item processing (see `scripts/phase1_extract.py`, `scripts/phase1_clone.py`):
```python
checkpoint = load_checkpoint("phase1_extract")
completed = set(checkpoint["completed"])
for repo in repos:
    if repo["name"] in completed:
        continue  # skip already-processed
    # ... process ...
    checkpoint["completed"].append(repo["name"])
    save_checkpoint("phase1_extract", checkpoint)
```

DGX Toolbox also supports idempotency via the `idempotency_check` parameter:
```python
dgx.execute("unsloth_studio", "python", "-m", "scripts.train_model",
            idempotency_check="/workspace/wp-finetune/adapters/qwen3-wp/adapter_config.json")
```

## Config-Driven Design

All hyperparameters and targets live in YAML config files, never hardcoded in scripts:

- Training hyperparameters: `config/train_config.yaml`
- Eval thresholds: `config/train_config.yaml` (eval section)
- Pipeline percentage targets: `scripts/pipeline_orchestrator.py` constants (derived from data)
- Repo list and filters: `config/repos.yaml`
- DGX container config: `config/dgx_toolbox.yaml`
- Judge rubric: `config/judge_system.md`
- Taxonomy: `config/taxonomy.yaml`

When adding new configurable values, add them to the appropriate YAML file and load via `yaml.safe_load()`.

## CLI Pattern

Pipeline scripts use `if __name__ == "__main__":` with `argparse` or simple `sys.argv` parsing:

```python
# Simple CLI (pipeline_orchestrator.py):
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

# Full argparse (train_model.py):
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--resume", nargs="?", const=True, ...)
    return parser

if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args)
```

Scripts are also importable as modules: `python -m scripts.train_model` or `from scripts.train_model import load_config`.

## Callback Pattern (Training)

The `MemoryWatchdogCallback` in `scripts/train_model.py` inherits `transformers.TrainerCallback` with a fail-open design:

```python
class MemoryWatchdogCallback(_TrainerCallback):
    def __init__(self, threshold_mb: int = OOM_WATCHDOG_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self._triggered = False

    @staticmethod
    def _available_mb() -> int:
        try:
            # read /proc/meminfo
        except Exception:
            return 999_999  # fail-open

    def on_step_end(self, args, state, control, **kwargs):
        if self._triggered:
            return
        avail = self._available_mb()
        if avail < self.threshold_mb:
            self._triggered = True
            control.should_save = True
            control.should_training_stop = True
```

Key design: reads `/proc/meminfo` every step, triggers graceful save before OOM killer strikes, fires only once (`_triggered` flag).

---

*Convention analysis: 2026-03-31*

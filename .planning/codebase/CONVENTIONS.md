---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
# Coding Conventions

**Analysis Date:** 2026-10-09

## Naming Patterns

**Files:**
- Python modules: `snake_case.py`. Examples: `scripts/utils.py`, `scripts/rl_judge_dispatch.py`, `eval/eval_gate.py`.
- Pipeline scripts carry a phase or family prefix that groups related work: `scripts/phase1_extract.py`, `scripts/phase2_judge.py`, `scripts/sieve_ksweep_run.py`, `scripts/eval4_ext_verdict.py`, `scripts/relabel/prep_batches.py`, `scripts/revl07_classification.py`. Name a new script after its family prefix if one already exists.
- Verb-first names for build and generation steps: `build_*.py`, `generate_*.py`, `merge_*.py`, `check_*.py`, `audit_*.py`, `serve_*.sh`.
- Helper or internal-only modules and scripts start with an underscore: `scripts/_p0_smoke_common.py`, `scripts/_pkg_gguf_eval_run.sh`, `scripts/_wpbench_pth/usercustomize.py`. Keep the underscore for anything not meant to be run directly.
- Shell runners end in `.sh` and use `serve_*` for server launchers, `_*_run.sh` for one-off eval/packaging runs.
- Test modules: `tests/test_<module>.py` (see `TESTING.md`).
- Documentation: UPPERCASE for top-level docs (`README.md`, `PIPELINE.md`, `PROJECT.md`, `CHANGELOG.md`). Config and recipe YAML use lowercase with underscores or hyphens (`config/train_config_v4.yaml`, `recipes/qwen3.6-35b-a3b-vllm.yaml`). Match the neighbouring files.

**Functions:**
- `snake_case`. Public: `extract_json`, `load_checkpoint`, `save_checkpoint`, `check_gates`, `load_thresholds`.
- Private helpers: leading underscore. Examples: `_get_checkpoint_dir` in `scripts/utils.py`, `_auto_device_map_calls` in `tests/test_gb10_load_safety.py`, `_judge_response` in `tests/phase4_4/conftest.py`.
- Verbs first: `load_*`, `save_*`, `check_*`, `build_*`, `parse_*`.

**Variables:**
- `snake_case`. Local names are short but descriptive: `stripped`, `tmp_path`, `gate_rows`, `hits`.
- Module-level defaults are `UPPER_SNAKE_CASE`: `DEFAULT_CONFIG_PATH = "config/train_config.yaml"` in `eval/eval_gate.py`; `BASE_MODEL`, `MAX_LENGTH`, `TRAIN_PATH` in `scripts/`.
- Module-level "private" constants start with an underscore: `_FALLBACK_THRESHOLDS` in `eval/eval_gate.py`.
- Mutable module-level cache: `_DEFAULT_CHECKPOINT_DIR: Optional[Path] = None` with a `global` statement in `scripts/utils.py`. Keep this pattern for lazy one-time values only.
- Environment-variable names are `UPPER_SNAKE` and are read with `os.environ.get(NAME, default)`: `VLLM_BASE_URL`, `VLLM_MODEL`, `SIEVE_MASK_NPY`, `XDG_STATE_HOME`.

**Types and classes:**
- `PascalCase`. Exceptions end in `Error` or describe the failure: `PreconditionError(RuntimeError)` in `scripts/fidelity_gate_v3.py`, `VllmBootTimeout(RuntimeError)` in `scripts/_p0_vllm_smoke_serve.py`.
- Use `@dataclass` for structured records (5 files in `scripts/` and `eval/`).
- Test helper classes are named `Test*` and grouped by behaviour (see `TESTING.md`).

## Code Style

**Formatting:**
- No formatter config is committed (no `.prettierrc`, `pyproject.toml` with black/ruff settings, or `.editorconfig`). Match the surrounding file.
- Indentation: 4 spaces for Python, 2 spaces in YAML.
- Quotes: double quotes dominate in Python (about 103 scripts use `print("...")`). Single quotes appear inside PHP strings and in test fixture content, not as the general style.
- Line length: most lines are under 100 characters, but the codebase is not strict. About 1,100 of roughly 71,700 Python lines exceed 100 characters and about 330 exceed 120. Keep new lines under 100 and wrap long function calls across lines with a trailing comma.
- Long f-strings for prompts are normal. Multiline prompt text lives in triple-quoted strings (`f"""You are generating ..."""` in `scripts/orchestrator.py`). Keep them readable rather than wrapping them mechanically.

**Linting:**
- No linter is configured or enforced in CI. Do not assume `ruff`/`flake8` passes.
- The code still signals intent with `# noqa: E402` on imports that follow a `sys.path.insert(...)` line (see Import Organization).
- Rule that is actually enforced: `tests/test_gb10_load_safety.py` fails the suite on `device_map="auto"` in any `scripts/*.py`. Treat that as a hard rule, not style.

**Type hints:**
- Present on function signatures in the utility and gate modules: `def extract_json(text: str) -> Optional[dict]:` in `scripts/utils.py`.
- Two spellings are in use: `Optional[X]` (21 files) and `X | None` (37 files). The `X | None` form needs `from __future__ import annotations` on Python versions before 3.10. Use `X | None` with the future import in new code; keep `Optional` in a file that already uses it.
- `from __future__ import annotations` is in 143 of 228 Python files under `scripts/`, `eval/`, and `tests/`. Add it to new files.
- Keep signatures simple: no `TypedDict`, `Protocol`, or generics are needed for the existing patterns.

## Import Organization

**Order:**
1. `from __future__ import annotations` (when present)
2. Standard library (`import json`, `import re`, `from pathlib import Path`, `from typing import Optional`)
3. Third-party (`import yaml`, `import pytest`, `import torch`)
4. Project modules (`from scripts.dgx_toolbox import get_toolbox`, `from eval.eval_gate import ...`)

Keep the blank lines between groups. Example from `eval/eval_gate.py`:

```python
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from scripts.dgx_toolbox import get_toolbox
```

**Path bootstrap for scripts run directly:**
Most scripts are run as `python scripts/<name>.py` or `python -m scripts.<name>`. They add the project root to `sys.path` before importing project modules, then mark the late import with `# noqa: E402`:

```python

# scripts/check_verdict_confusion.py

sys.path.insert(0, str(ROOT))
from eval.output_parsers import parse_judge_scores  # noqa: E402
```

Use this pattern (`sys.path.insert(0, str(PROJECT_ROOT))` with `PROJECT_ROOT = Path(__file__).resolve().parent.parent`) in new scripts that import from `scripts.` or `eval.`. Appears in about 122 files.

**Path Aliases:**
- None. Import by package path: `scripts.<module>`, `eval.<module>`. Both `scripts/` and `eval/` have an `__init__.py`.
- Local imports inside functions are used to defer heavy or optional dependencies: `from scripts._p0_vllm_smoke_serve import boot_vllm` inside `main()` in `scripts/build_gen03_merge.py`; `from unittest.mock import MagicMock` inside a fixture. Use this when a module is expensive to import or may be absent.

## Module Layout

**Top of file:**
1. Module docstring describing purpose, provided functions, and a `Usage:` block when runnable. Example from `eval/eval_gate.py`:
   ```
   Usage:
       python -m eval.eval_gate [--results-dir PATH] [--config PATH]
   ```
2. `from __future__ import annotations`
3. Imports (order above)
4. Constants (`UPPER_SNAKE`), including default paths
5. Section banners (see below)
6. Functions, then `if __name__ == "__main__":` (present in about 148 files)

**Section banners:**
Separate groups of related functions with a comment banner (used in `scripts/utils.py` and `eval/eval_gate.py`):

```python

# ---------------------------------------------------------------------------

# Threshold loading

# ---------------------------------------------------------------------------

```

**Entry points:**
- `main()` function plus `if __name__ == "__main__": main()` (or `sys.exit(main())` when the exit code matters).
- Command-line parsing uses `argparse` (74 files). Name the parser variable `ap` or `parser`. Give every option a default that can be read from an environment variable when it is a service URL or model name:
  ```python
  ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL))
  ```
- Exit codes: `sys.exit(1)` for gate or validation failure, `sys.exit(2)` for usage or precondition errors. 83 files call `sys.exit`. Quality gates must exit non-zero on failure (see `eval/eval_gate.py`).

## Documentation Comments

**Docstrings:**
- Google style: `Args:` and `Returns:` sections with indented descriptions. Used in `scripts/utils.py` and `eval/eval_gate.py` (`Args:` / `Returns:` in about 39 files).
- Module docstrings are required for runnable scripts and describe the phase, inputs, outputs, and usage.
- Keep the first line as a short imperative summary: `"""Load evaluation thresholds from config/train_config.yaml eval section."""`.
- Tests have one-line docstrings stating the case.

```python
def load_checkpoint(phase: str, checkpoint_dir: Optional[Path] = None) -> dict:
    """Load checkpoint state for a given phase name.

    Returns a fresh empty state if no checkpoint file exists.

    Args:
        phase: Phase identifier used as the filename stem
        checkpoint_dir: Override directory (for testing)
    """
```

**Inline comments:**
- Use comments to explain why, especially for numeric thresholds, model-specific quirks, and hardware or environment traps. Examples: the GB10 unified-memory note in `tests/test_gb10_load_safety.py`, and the close-only `[/REASONING]` format note in `tests/phase4_4/conftest.py`.
- Reference the decision or ticket ID when the reason is recorded elsewhere (for example `D-09-03`, `PKG-03`, `PUB4-01`, `REVL-03` appear in comments and test names). Keep these IDs when editing those sections so the trail is preserved.
- Numbered strategies or steps are written as comments: `# Strategy 1: direct parse`, `# Strategy 2: ```json fenced block`.

**TODO/FIXME:**
- Rare. Only 5 `TODO`/`FIXME`/`HACK`/`XXX` markers in `scripts/`, `eval/`, `tests/` combined. Record open work in `PROJECT.md` or `.planning/`, and avoid leaving bare TODOs.

## Error Handling

**Patterns:**

- Catch narrow exceptions when the failure is expected. Examples: `json.JSONDecodeError` and `ValueError` in `scripts/utils.py:extract_json`, `FileNotFoundError`, `subprocess.TimeoutExpired`, `ImportError` for optional dependencies.
- Broad `except Exception` appears often (about 120 occurrences) in pipeline and serving code, where a single bad item must not stop a batch. Log the error and continue, or record it in a failed list. Do not use bare `except:` (0 occurrences). Do not silently swallow exceptions.
- Fallback-with-default is the typical recovery path. Example: `load_thresholds` returns `dict(_FALLBACK_THRESHOLDS)` when the config file is missing or has no `eval` section.
- Raise `RuntimeError` for environment or runtime failures (35 files) and `ValueError` for bad input (16 files). Define a subclass of `RuntimeError` for a failure callers may want to catch by name (`VllmBootTimeout`, `PreconditionError`).
- Use `raise ... from exc` or include the original message when re-raising; preserve the cause.
- Subprocesses: wrap `subprocess.run` with an explicit `timeout=` and catch `subprocess.TimeoutExpired`. 28 files use `subprocess.run`.

```python
try:
    return json.loads(stripped)
except (json.JSONDecodeError, ValueError):
    pass
```

**Atomic writes:**
Persist state with write-to-temp then rename so readers never see a partial file. The pattern in `scripts/utils.py:save_checkpoint`:

```python
tmp_path = directory / f"{phase}_checkpoint.tmp"
with tmp_path.open("w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
tmp_path.rename(path)
```

Use this for any new checkpoint or result file that another process may read while it is being written.

**Validation at the boundary:**
Validate LLM output and external JSON with `extract_json` (from `scripts/utils.py`) rather than calling `json.loads` directly. It tolerates fenced and embedded JSON in model responses.

## Logging

**Framework:** Mixed. Standard `logging` in library-style modules and `print` for operator-facing progress in scripts.

**Counts:**
- `print(...)` is used by about 167 files, mostly with a bracketed tag prefix:
  ```python
  print(f"[sft] ...")
  print(f"[probe] ...")
  print(f"[vllm] ...")
  print("[merge] subprocess exited 0", flush=True)
  ```
- `logging.getLogger(__name__)` is used in 12 files. Calls are mostly `logger.info` (113), `logger.warning` (54), `logger.error` (38), `logger.debug` (6).
- `logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")` is the standard setup in entry points (4 files).

**Patterns:**
- For a new script, use the bracketed-tag `print(f"[tag] ...", flush=True)` style when it is a long-running pipeline step, so output is readable in `tee`d logs.
- For a new library module used by several scripts, define `logger = logging.getLogger(__name__)` at module top and log through it.
- Do not log secrets, API keys, or full model responses. Log counts, paths, and scores.

## Comments

**When to Comment:**
- Explain non-obvious constraints, hardware or model quirks, and the reason for an unusual choice.
- Do not restate what the code does.
- Put a short banner above a group of related functions (see Module Layout).

**JSDoc/TSDoc:**
- Not applicable. Python project. Use Google-style docstrings (see Documentation Comments).

## Function Design

**Size:**
- Helpers are small and single-purpose. Long pipeline functions exist (orchestrators, `main()` functions with argparse) and are acceptable for CLI glue. Extract the testable logic into a named function so `tests/` can import it without running `main()`.

**Parameters:**
- Accept an explicit override for any directory or file path that would otherwise default into `data/` or `output/`. Example: `save_checkpoint(phase, state, checkpoint_dir: Optional[Path] = None)`. Tests use this override so they never write into real project data.
- Use keyword arguments for options with defaults. Callers in `tests/` pass them explicitly by name.

**Return Values:**
- Return `Optional[...]` for a lookup that may fail (`extract_json` returns `None`). Return a fresh empty state, not `None`, for a missing checkpoint (`load_checkpoint`).
- Return tuples for a pass/fail plus detail: `check_gates(...) -> (bool, list[dict])`.

## Module Design

**Exports:**
- No `__all__` declarations (0 in `scripts/` and `eval/`). Public names are those without a leading underscore.
- `eval/__init__.py` and `scripts/__init__.py` are empty.

**Barrel Files:**
- Not used. Import the concrete module: `from eval.output_parsers import parse_judge_scores`, not from a package re-export.

**Shared utilities:**
- Shared helpers go in `scripts/utils.py` (JSON extraction, checkpoints), `scripts/dgx_toolbox.py` (hardware and environment queries via `get_toolbox()`), and `eval/` (evaluation and parsing). Check these before writing a new helper; several one-off copies exist, so search with `grep -rn "def <name>" scripts eval` first.

**Hardware-safe loading (project rule):**
- Never pass `device_map="auto"` to `from_pretrained`. On the GB10 unified-memory host this causes an OOM. Use `scripts.sieve_arch.gb10_load_kwargs()` instead. Enforced by `tests/test_gb10_load_safety.py`.

## Shell Scripts

- `scripts/*.sh` files: 8 of 14 set `set -e` (or `set -uo pipefail`). Add `set -euo pipefail` to new runners.
- Put a header comment after the shebang that states the phase ID, purpose, and usage (`# Usage: _pkg_gguf_eval_run.sh <gguf_path> <alias> <out_dir> <port>`).
- Read configuration from positional arguments or environment variables, with defaults, not hard-coded values.
- Known deviation: `scripts/_pkg_ens8192_run.sh` hard-codes an absolute home path (`cd /home/robert_li/...`). Do not copy this. Resolve the project root relative to the script (`cd "$(dirname "$0")/.."`).

## Known Deviations

- Multiple styles of CLI logging coexist (`print` vs `logging`). Follow the file you are editing.
- `Optional[X]` and `X | None` both appear. Follow the file you are editing.
- Phase- and ticket-numbered names (`phase2_*`, `revl07_*`, `eval4_*`) make the codebase larger to navigate. Put new work under the existing family prefix instead of creating a new one.

---

*Convention analysis: 2026-10-09*

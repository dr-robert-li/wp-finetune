---
phase: 20-base-bring-up
plan: 01
subsystem: infra
tags: [huggingface_hub, transformers, peft, moe, qwen3.6, base-model-download]

# Dependency graph
requires: []
provides:
  - "config/train_config_v4.yaml — v4 base config sibling (model.name=Qwen/Qwen3.6-35B-A3B, local_dir=./models/Qwen3.6-35B-A3B), config/train_config.yaml unchanged"
  - "scripts/download_model.py --config-path flag — download any train_config.yaml-shaped config's model into its local_dir"
  - "scripts/smoke_load_base20.py — reusable BASE-01-style load smoke (trust_remote_code load + class assert + forward pass + JSON receipt)"
  - "models/Qwen3.6-35B-A3B/ — 26 safetensors shards, 67.0 GB, downloaded and load-verified"
  - "output/base20/load_smoke.json — BASE-01 gate receipt (status=pass)"
affects: [20-02-token-alignment, 20-03-deltanet-smoke, 20-04-vl-merge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "config/train_config_v4.yaml as a non-destructive sibling to config/train_config.yaml — v4.0 scripts take --config-path rather than mutating the v3.x default"
    - "output/base20/*.json flat-JSON gate receipts (status field + asserted fields), following output/tinker/PROMOTED_*.json and output/merge_v4_winner/merge_report.json convention"

key-files:
  created:
    - config/train_config_v4.yaml
    - tests/test_download_model_v4.py
    - scripts/smoke_load_base20.py
    - output/base20/load_smoke.json
    - .planning/phases/20-base-bring-up/deferred-items.md
  modified:
    - scripts/download_model.py

key-decisions:
  - "Upgraded torchvision 0.25.0 -> 0.27.1 (exact match for installed torch==2.12.1) — pre-existing version-mismatch broke every `from transformers import PreTrainedModel`-triggered import chain (peft, AutoModelForCausalLM instantiation), not something introduced by this plan"
  - "Upgraded pytest 6.0.0rc2.dev33 -> 9.1.1 — pre-existing broken dev build could not collect ANY test file (Python 3.13 ast-rewrite incompatibility: 'required field lineno missing from alias'), blocking the entire tests/ suite, not just this plan's new tests"
  - "Force-added output/base20/load_smoke.json despite output/ being gitignored by default — same precedent as tracked output/bench17/*.json gate receipts (small JSON audit artifacts, not large eval output)"

requirements-completed: [BASE-01]

coverage:
  - id: D1
    description: "config/train_config_v4.yaml exists (model.name=Qwen/Qwen3.6-35B-A3B, local_dir=./models/Qwen3.6-35B-A3B) and config/train_config.yaml is byte-for-byte unchanged"
    requirement: "BASE-01"
    verification:
      - kind: unit
        ref: "tests/test_download_model_v4.py::TestV4Config -- 2 tests"
        status: pass
      - kind: other
        ref: "git diff --exit-code config/train_config.yaml"
        status: pass
    human_judgment: false
  - id: D2
    description: "download_model.py accepts --config-path (default CONFIG_PATH), threading the override through load_config/download_model; bare no-arg invocation still defaults to the v3 config"
    requirement: "BASE-01"
    verification:
      - kind: unit
        ref: "tests/test_download_model_v4.py::TestDownloadIdempotency, TestConfigPathFlag -- 5 tests"
        status: pass
      - kind: other
        ref: "python scripts/download_model.py --help"
        status: pass
    human_judgment: false
  - id: D3
    description: "Qwen3.6-35B-A3B downloaded (idempotent, resumable snapshot_download) to models/Qwen3.6-35B-A3B/ as 26 .safetensors shards, 67.0 GB"
    requirement: "BASE-01"
    verification:
      - kind: other
        ref: "python scripts/download_model.py --config-path config/train_config_v4.yaml (logs/base20/download_v4.log: 'Download complete: 26 safetensors shards, 67.0 GB')"
        status: pass
    human_judgment: false
  - id: D4
    description: "Base loads via transformers with trust_remote_code=True, resolves to Qwen3_5MoeForConditionalGeneration, single forward pass executes; output/base20/load_smoke.json records status=pass + transformers/peft/huggingface_hub versions + model class"
    requirement: "BASE-01"
    verification:
      - kind: other
        ref: "python scripts/smoke_load_base20.py (logs/base20/smoke_load.log: 'BASE-01 SMOKE PASSED'); output/base20/load_smoke.json"
        status: pass
    human_judgment: false

duration: 11min
completed: 2026-07-13
status: complete
---

# Phase 20 Plan 01: v4 Base Bring-Up — Config + Download + Load Smoke Summary

**Qwen3.6-35B-A3B downloaded (67 GB, 26 shards) and load-verified via a new config-driven `--config-path` download path, resolving to `Qwen3_5MoeForConditionalGeneration` with a passing forward pass — v3.x pipeline config untouched.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-07-13T01:42:11Z
- **Completed:** 2026-07-13T01:52:53Z
- **Tasks:** 2
- **Files modified:** 6 (2 created config/test in Task 1's RED commit, 1 modified in GREEN, 2 created in Task 2)

## Accomplishments
- `config/train_config_v4.yaml` added as a non-destructive sibling of `config/train_config.yaml` (same structure, only `model.name`/`model.local_dir` changed to the v4 base)
- `scripts/download_model.py` gained a `--config-path` CLI flag (via new `build_arg_parser()`), threaded through `load_config()`/`download_model()`, with the bare no-arg invocation still defaulting to the v3 config
- `Qwen/Qwen3.6-35B-A3B` downloaded via the new config path: 26 `.safetensors` shards, 67.0 GB, idempotent/resumable via `huggingface_hub.snapshot_download`
- `scripts/smoke_load_base20.py` (new) loads the base with `trust_remote_code=True`, asserts the resolved architecture contains `Qwen3_5MoeForConditionalGeneration`, runs a single CPU bf16 forward pass, and writes `output/base20/load_smoke.json` (status=pass, all three library versions recorded)

## Task Commits

Each task was committed atomically (Task 1 is `tdd="true"`, RED then GREEN):

1. **Task 1 RED: failing v4-config-download tests** - `febb194` (test)
2. **Task 1 GREEN: --config-path flag + direct-execution sys.path fix** - `0f2b4c6` (feat)
3. **Task 2: download Qwen3.6-35B-A3B + load smoke** - `2a1f3ce` (feat)

**Plan metadata:** pending (docs: complete plan, this commit)

## Files Created/Modified
- `config/train_config_v4.yaml` - v4 base config (model.name/local_dir changed; tokenizer/training/eval/lora blocks copied unchanged from v3)
- `scripts/download_model.py` - `build_arg_parser()` + `--config-path` threading; direct-execution `sys.path` fix for the pre-existing `scripts.dgx_toolbox` absolute import
- `tests/test_download_model_v4.py` - 7 Wave-0 mock-only tests (config values, idempotency skip/call, argparse flag)
- `scripts/smoke_load_base20.py` - BASE-01 load-smoke script + `output/base20/load_smoke.json` gate receipt
- `output/base20/load_smoke.json` - BASE-01 receipt: `status=pass`, `model_class="Qwen3_5MoeForConditionalGeneration"`, `transformers_version=5.3.0`, `peft_version=0.18.1`, `huggingface_hub_version=1.23.0`, `shard_count=26`, `total_size_gb=67.0`
- `models/Qwen3.6-35B-A3B/` (gitignored, not committed) - downloaded base weights, consumed by plans 20-02/20-03/20-04
- `.planning/phases/20-base-bring-up/deferred-items.md` - 7 pre-existing, out-of-scope test failures logged (not fixed)

## Decisions Made
- Upgraded `torchvision` 0.25.0 -> 0.27.1 to match the installed `torch==2.12.1` exactly (PyPI metadata confirmed `torchvision==0.27.1` requires `torch==2.12.1`). `pip check` had already flagged the mismatch (`torchvision 0.25.0 has requirement torch==2.10.0`); every `peft`/`transformers.PreTrainedModel` import chain was broken by it, independent of this plan's own changes.
- Upgraded `pytest` 6.0.0rc2.dev33 -> 9.1.1. The installed dev build could not collect ANY test file in the repo (`TypeError: required field "lineno" missing from alias` — a Python 3.13 `ast`-rewrite incompatibility in that old pytest build), confirmed via `git stash` that this predates this plan's changes.
- Force-added `output/base20/load_smoke.json` to git despite `output/` being gitignored by default, matching the existing `output/bench17/*.json` precedent for small JSON gate-receipt artifacts (as opposed to large regenerable eval output).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] torchvision/torch version mismatch broke every peft/transformers.PreTrainedModel import**
- **Found during:** Task 1 setup (verifying environment before writing tests)
- **Issue:** `import peft` (and any `from transformers import PreTrainedModel`-triggering import) raised `RuntimeError: operator torchvision::nms does not exist` — `torchvision==0.25.0` requires `torch==2.10.0` but the host has `torch==2.12.1`. This blocks Task 2's `smoke_load_base20.py`, which needs `AutoModelForCausalLM`/`peft.__version__`.
- **Fix:** `pip install torchvision==0.27.1` (verified via PyPI JSON metadata that this exact version requires `torch==2.12.1`, an exact match — no torch reinstall triggered).
- **Files modified:** none (host environment change, no repo file; no requirements.txt exists to pin)
- **Verification:** `from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer` and `import peft` both succeed after the fix; `pip check` no longer flags torchvision.
- **Committed in:** N/A (environment-only change; documented here per deviation rules, not a repo commit)

**2. [Rule 3 - Blocking] Broken pytest dev build could not collect any test file**
- **Found during:** Task 1 RED-phase test run
- **Issue:** `python -m pytest tests/test_prepare_tokenizer.py` (pre-existing, unmodified test file) failed to even collect: `TypeError: required field "lineno" missing from alias` inside pytest's assertion-rewrite AST pass — `pytest==6.0.0rc2.dev33+g7f7a36478.d20260625` is incompatible with Python 3.13's `ast` module. Confirmed via `git stash` that this is not caused by any Task 1 change.
- **Fix:** `pip install pytest==9.1.1` (matches `pytest-asyncio`'s own declared floor `pytest<10,>=8.2`, already flagged by `pip check`).
- **Files modified:** none (host environment change)
- **Verification:** `pytest tests/test_prepare_tokenizer.py` (7/7) and the new `tests/test_download_model_v4.py` (7/7) both collect and pass; full `pytest tests/` now runs (678 passed, 22 skipped, 7 pre-existing unrelated failures — see deferred-items.md).
- **Committed in:** N/A (environment-only change)

**3. [Rule 3 - Blocking] `python scripts/download_model.py --help` failed on direct execution (pre-existing import bug)**
- **Found during:** Task 1 acceptance-criteria verification
- **Issue:** `from scripts.dgx_toolbox import get_toolbox` (pre-existing line, unmodified by this task) raised `ModuleNotFoundError: No module named 'scripts.dgx_toolbox'` when the script is run directly (`python scripts/download_model.py`) because Python puts `scripts/` — not the repo root — on `sys.path[0]` under direct execution. `python -m scripts.download_model` was unaffected. Confirmed pre-existing via `git stash` on the file before Task 1's edits.
- **Fix:** Insert `PROJECT_ROOT` onto `sys.path` before the `scripts.dgx_toolbox` import when `__package__` is unset (i.e., only under direct execution; no-op under `-m` or import-as-module).
- **Files modified:** `scripts/download_model.py`
- **Verification:** `python scripts/download_model.py --help` now prints usage including `--config-path`; `python -m scripts.download_model --help` still works; all 7 tests in `tests/test_download_model_v4.py` still pass.
- **Committed in:** `0f2b4c6` (Task 1 GREEN commit)

---

**Total deviations:** 3 auto-fixed (all Rule 3 - blocking issues, all pre-existing environment/import bugs surfaced while satisfying this plan's own acceptance criteria)
**Impact on plan:** All three fixes were necessary to make Task 1/2's stated acceptance criteria (pytest collection, `--help` output, `peft`/`transformers` model loading) achievable at all. No scope creep — no unrelated code was changed. Two of the three (torchvision, pytest) are host-environment package upgrades with no repo diff; the third (`sys.path` fix) is a 6-line addition to a file already in this task's `files_modified` list.

## Issues Encountered
- The full `tests/` suite has 7 pre-existing failures unrelated to Phase 20 (Phase 8/8.2/9 reward-calibration/RL-judge-dispatch/RL-train tests — NaN confidence intervals in reward-validity gates, and `ModuleNotFoundError: No module named 'tinker'`). These only became visible once the pytest environment bug (deviation #2) was fixed — previously `pytest tests/` could not collect anything at all. Logged in `.planning/phases/20-base-bring-up/deferred-items.md`, not fixed (out of scope: none of these files are touched by this plan).

## User Setup Required

None - no external service configuration required. (HuggingFace Hub auth already configured on this host, verified via `huggingface_hub.HfApi().whoami()` before the download.)

## Next Phase Readiness

- `models/Qwen3.6-35B-A3B/` is present, load-verified, and ready for plan 20-02 (eos/pad token-ID alignment) and 20-03 (DeltaNet-aarch64 serving smoke) to consume directly.
- `config/train_config_v4.yaml` and `scripts/download_model.py --config-path` are reusable by any later v4.0 script that needs `model.local_dir`.
- No blockers. The torchvision/pytest environment fixes are host-level and persist for all subsequent plans in this phase (20-02/20-03/20-04 no longer need to rediscover them).

---
*Phase: 20-base-bring-up*
*Completed: 2026-07-13*

## Self-Check: PASSED

All created files verified present on disk; all 3 task commit hashes (febb194, 0f2b4c6, 2a1f3ce) verified present in git log.

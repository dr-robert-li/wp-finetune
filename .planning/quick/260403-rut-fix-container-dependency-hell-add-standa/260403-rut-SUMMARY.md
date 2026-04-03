---
phase: quick
plan: 260403-rut
subsystem: infra
tags: [transformers, peft, merge, lora, unsloth]

# Dependency graph
requires: []
provides:
  - scripts/merge_adapter.py runnable as python3 scripts/merge_adapter.py in any peft+transformers container
affects: [phase-12-lora-merge]

# Tech tracking
tech-stack:
  added: []
  patterns: ["AutoModelForCausalLM.from_pretrained for base model loading instead of Unsloth"]

key-files:
  created: []
  modified:
    - scripts/merge_adapter.py

key-decisions:
  - "Use AutoModelForCausalLM.from_pretrained (bfloat16, device_map=auto) instead of Unsloth FastLanguageModel — identical output, no pip-install side effects"
  - "Remove dgx_toolbox import entirely — it was unused in _verify_merged_model; fallback vLLM command constructs path from config directly"

patterns-established:
  - "LoRA merge pattern: AutoModelForCausalLM.from_pretrained -> PeftModel.from_pretrained -> merge_and_unload"

requirements-completed: []

# Metrics
duration: 5min
completed: 2026-04-03
---

# Quick Task 260403-rut: Fix Container Dependency Hell — Standardize merge_adapter.py

**LoRA merge script de-Unslothed: uses AutoModelForCausalLM + PeftModel only, runnable as python3 scripts/merge_adapter.py in any peft+transformers NGC container**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-03T00:00:00Z
- **Completed:** 2026-04-03T00:05:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Removed `from unsloth import FastLanguageModel` — eliminates Unsloth pip install which destroys CUDA-enabled torch in NGC containers
- Replaced `FastLanguageModel.from_pretrained` (which returned model+tokenizer tuple) with `AutoModelForCausalLM.from_pretrained` (returns model only — tokenizer loaded separately from extended tokenizer dir as before)
- Removed `from scripts.dgx_toolbox import get_toolbox` and the unused `dgx = get_toolbox()` call in `_verify_merged_model`
- Updated docstring to reflect new strategy and show `python3 scripts/merge_adapter.py` as primary invocation pattern

## Task Commits

1. **Task 1: Replace Unsloth with AutoModelForCausalLM and fix dgx_toolbox import** - `e8ae427` (fix)

## Files Created/Modified
- `scripts/merge_adapter.py` - Unsloth and dgx_toolbox imports removed; base model loaded via AutoModelForCausalLM.from_pretrained with bfloat16 and device_map=auto

## Decisions Made
- `AutoModelForCausalLM.from_pretrained` with `torch_dtype=torch.bfloat16, device_map="auto"` is the correct drop-in: Unsloth's FastLanguageModel returned `(model, tokenizer)` but only model was used; the tokenizer was always loaded from the extended tokenizer dir anyway
- No sys.path fixup needed: the only thing requiring module-style invocation (`python -m scripts.merge_adapter`) was the `scripts.dgx_toolbox` import; with it removed, `python3 scripts/merge_adapter.py` works directly

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- scripts/merge_adapter.py is ready for Phase 12 LoRA merge execution in any standard NGC container with peft+transformers
- No Unsloth required; no dgx_toolbox required at import time

---
*Phase: quick*
*Completed: 2026-04-03*

## Self-Check: PASSED
- scripts/merge_adapter.py: FOUND
- Commit e8ae427: FOUND

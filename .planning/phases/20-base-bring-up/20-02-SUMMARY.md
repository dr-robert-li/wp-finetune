---
phase: 20-base-bring-up
plan: 02
subsystem: infra
tags: [transformers, eos-pad-alignment, moe, qwen3.6, token-gate]

# Dependency graph
requires:
  - phase: 20-base-bring-up
    provides: "models/Qwen3.6-35B-A3B/ downloaded and load-verified (20-01, BASE-01)"
provides:
  - "scripts/check_token_alignment.py — align_and_check/classify_stopped_naturally/build_receipt (unit-testable) + run_gate() (real CPU load, JSON-surgery persistence, stop-token smoke)"
  - "models/Qwen3.6-35B-A3B/config.json — text_config.eos_token_id/pad_token_id aligned to tokenizer's (248046/248044), rest of the VL wrapper structure byte-identical to the backup"
  - "models/Qwen3.6-35B-A3B/config.json.orig — pre-fix backup (gitignored, local only)"
  - "output/base20/token_alignment.json — BASE-02 gate receipt (status=pass), the canonical_ids block Phase 21 must consume"
affects: [20-03-deltanet-smoke, 20-04-vl-merge, 21-sft-loss-masking]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gate scripts that mutate on-disk model config MUST JSON-surgery the original file, not call config_object.save_pretrained() — composite/VL config objects can silently unwrap to a flattened sub-config after model loading, and save_pretrained() on that unwrapped object drops sibling fields (vision_config, architectures, image/video token ids)"
    - "output/base20/*.json flat-JSON gate receipts (status field + asserted fields), consistent with 20-01's load_smoke.json and output/tinker/PROMOTED_*.json"

key-files:
  created:
    - scripts/check_token_alignment.py
    - tests/test_check_token_alignment.py
    - output/base20/token_alignment.json
  modified:
    - models/Qwen3.6-35B-A3B/config.json (gitignored, local only)

key-decisions:
  - "model.config, as returned by AutoModelForCausalLM.from_pretrained() on this VL checkpoint, is NOT the composite Qwen3_5MoeConfig the raw config.json describes — it unwraps to the plain text-only sub-config (get_text_config() returns self on it). Verified empirically both ways: AutoConfig.from_pretrained() alone gives the composite object with .text_config/.vision_config; the loaded model's .config does not."
  - "Persistence is done via direct JSON read-modify-write against config.json.orig (mutate only text_config.eos_token_id/pad_token_id), not model.config.save_pretrained() — the latter was caught (Rule 1, before commit) silently stripping vision_config/architectures/image_token_id/video_token_id/vision_*_token_id from the persisted file, which would have broken 20-04's VL merge path"
  - "generation_config.json needed no fix — it already carried Qwen's documented multi-stop eos_token_id=[248046, 248044] and pad_token_id=248044 (non-null), consistent with PITFALLS.md Pitfall 1's 'working as intended' framing; only model.config.text_config was misaligned"

requirements-completed: [BASE-02]

coverage:
  - id: D1
    description: "check_token_alignment.py exposes align_and_check (mismatched pair -> aligned, tokenizer-pad-None fallback to tokenizer-eos, already-aligned no-op) and classify_stopped_naturally (run-to-length -> False, natural-stop -> True) as pure/mockable functions, with a passing Wave-0 test suite"
    requirement: "BASE-02"
    verification:
      - kind: unit
        ref: "tests/test_check_token_alignment.py -- 11 tests (TestAlignAndCheck x4, TestClassifyStoppedNaturally x5, TestBuildReceipt x2)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The alignment gate ran against the real downloaded models/Qwen3.6-35B-A3B: confirmed the documented pre-fix mismatch (text_config.eos_token_id=248044, pad_token_id=None, tokenizer.eos_token_id=248046) exactly, applied the fix, and persisted it to config.json while preserving the VL wrapper structure byte-for-byte elsewhere (verified via full JSON diff against config.json.orig)"
    requirement: "BASE-02"
    verification:
      - kind: other
        ref: "logs/base20/check_token_alignment_run2.log ('BASE-02 GATE PASSED'); diff config.json.orig vs config.json shows exactly 2 changed lines (eos_token_id, pad_token_id)"
        status: pass
    human_judgment: false
  - id: D3
    description: "A real stop-token smoke generation (CPU, greedy, max_new_tokens=64) on models/Qwen3.6-35B-A3B stopped naturally at 19 tokens, strictly before the budget, ending on the aligned eos boundary — not a coincidental run-to-length pass"
    requirement: "BASE-02"
    verification:
      - kind: other
        ref: "output/base20/token_alignment.json: stopped_naturally=true, stop_gen_len=19 < max_tokens_budget=64, aligned_eos_id==tokenizer_eos_id (248046)"
        status: pass
    human_judgment: false
  - id: D4
    description: "output/base20/token_alignment.json records status=pass plus orig/aligned eos+pad IDs and a canonical_ids block — the Stage 1.5 gate receipt Phase 21 must consume, blocking Stage 2/3 on failure (gate fails closed: status=fail + exit 1 on any misalignment or non-natural-stop, per run_gate()'s exception/assertion handling)"
    requirement: "BASE-02"
    verification:
      - kind: other
        ref: "python -c \"import json; d=json.load(open('output/base20/token_alignment.json')); assert d['status']=='pass'; assert d['stopped_naturally'] is True; assert d['aligned_eos_id']==d['tokenizer_eos_id']; assert d['aligned_pad_id'] is not None\" (exits 0)"
        status: pass
    human_judgment: false

duration: 8min
completed: 2026-07-13
status: complete
---

# Phase 20 Plan 02: v4 Base Bring-Up — eos/pad Token Alignment Gate Summary

**New `scripts/check_token_alignment.py` gate detected and fixed the documented eos/pad mismatch on Qwen3.6-35B-A3B (text_config.eos_token_id 248044→248046, pad_token_id None→248044), persisted the fix via JSON surgery that preserves the VL wrapper's `vision_config`/`architectures` (avoiding a `model.config.save_pretrained()` corruption bug caught mid-execution), and proved the fix with a real CPU generation that stopped naturally at 19/64 tokens — `output/base20/token_alignment.json` records status=pass as the Stage 1.5 gate Phase 21 must consume.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-13T01:55:00Z
- **Completed:** 2026-07-13T02:03:18Z
- **Tasks:** 2
- **Files modified:** 4 (2 created in Task 1's RED/GREEN commits, 1 fixed + 1 receipt in Task 2)

## Accomplishments
- `scripts/check_token_alignment.py` (new): `align_and_check(model_config, tokenizer)` mutates + asserts the post-fix invariant (tokenizer-authoritative per Pitfall 1, with tokenizer-pad-None fallback to tokenizer-eos); `classify_stopped_naturally(output_ids, max_tokens, eos_token_ids)` distinguishes a real natural stop from a run-to-length false pass; `build_receipt(...)` emits the flat-JSON gate-receipt shape; `run_gate()` wires these against the real model
- `tests/test_check_token_alignment.py` (new, Wave-0): 11 mock-only tests, no GPU/model load, covering mismatched-pair alignment, tokenizer-pad-None fallback, already-aligned no-op, run-to-length vs natural-stop classification (list and bare-int eos forms), and receipt field completeness
- Ran the gate against `models/Qwen3.6-35B-A3B`: confirmed the exact documented pre-fix state (`text_config.eos_token_id=248044`, `text_config.pad_token_id=None`, `tokenizer.eos_token_id=248046`) via CPU bf16 load (`AutoModelForCausalLM.from_pretrained(..., device_map="cpu", trust_remote_code=True)`), applied the fix, and ran a real greedy generation ("Reply with exactly one word: OK", `max_new_tokens=64`) that stopped naturally at 19 tokens on the aligned eos boundary
- `output/base20/token_alignment.json`: `status=pass`, `aligned_eos_id=248046`, `aligned_pad_id=248044`, `stopped_naturally=true`, plus a `canonical_ids` block for Phase 21 to consume directly

## Task Commits

Task 1 (`tdd="true"`, RED then GREEN), Task 2 (auto, includes an in-flight bug fix before final commit):

1. **Task 1 RED: failing eos/pad alignment tests** - `93fd831` (test)
2. **Task 1 GREEN: check_token_alignment.py implementation** - `e99f539` (feat)
3. **Task 2: ran gate against real model, fixed persistence bug, wrote receipt** - `55ff811` (feat)

**Plan metadata:** pending (docs: complete plan, this commit)

## Files Created/Modified
- `scripts/check_token_alignment.py` - `align_and_check`/`classify_stopped_naturally`/`build_receipt` (unit-testable) + `run_gate()`/`resolve_text_config()` (real-model plumbing, JSON-surgery persistence)
- `tests/test_check_token_alignment.py` - 11 Wave-0 mock-only tests mirroring `tests/test_prepare_tokenizer.py`'s style
- `output/base20/token_alignment.json` - BASE-02 gate receipt: `status=pass`, orig/aligned eos+pad IDs, `stopped_naturally=true`, `stop_gen_len=19`, `canonical_ids` block
- `models/Qwen3.6-35B-A3B/config.json` (gitignored, not committed) - `text_config.eos_token_id`/`pad_token_id` aligned in place; every other field byte-identical to the backup (verified by full JSON diff)
- `models/Qwen3.6-35B-A3B/config.json.orig` (gitignored, not committed) - pre-fix backup, created before any mutation

## Decisions Made
- `model.config` on the loaded model is NOT the composite VL config the raw `config.json` describes — `AutoModelForCausalLM.from_pretrained()` unwraps it to the plain text-only sub-config (`get_text_config()` returns `self` on it). Confirmed by contrasting `AutoConfig.from_pretrained()` alone (returns the composite `Qwen3_5MoeConfig` with `.text_config`/`.vision_config`) against the loaded model's `.config` (flat, no vision fields, same key set as `text_config`'s content).
- Persistence uses direct JSON read-modify-write against `config.json.orig`'s content (mutate only `text_config.eos_token_id`/`pad_token_id`, or top-level if no `text_config` key), not `model.config.save_pretrained()`. This sidesteps the object-identity ambiguity entirely and is verifiably safe (full-diff shows exactly the 2 target lines changed).
- `generation_config.json` needed no change — it already had `eos_token_id=[248046, 248044]` and `pad_token_id=248044`, Qwen's documented multi-stop-token design (Pitfall 1: "working as intended," not a bug). Only `model.config.text_config` was actually misaligned.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `model.config.save_pretrained()` corrupted config.json — caught before commit, fixed with JSON surgery**
- **Found during:** Task 2, first gate run
- **Issue:** The plan's action text called `model.config.save_pretrained(local_dir)` to persist the aligned eos/pad fix. Running this against the real model revealed that the loaded model's `.config` object is not the composite VL config the file describes — it silently unwraps to the plain text-only sub-config. Calling `save_pretrained()` on it overwrote `config.json` with ONLY the flattened text fields, dropping `vision_config`, `architectures`, `image_token_id`, `video_token_id`, `vision_start_token_id`, `vision_end_token_id` entirely. This would have broken 20-04's VL merge path, which depends on the intact wrapper structure.
- **Fix:** Restored `config.json` from `config.json.orig` immediately. Rewrote the persistence step to JSON-surgery the original file directly (`json.load` the backup, mutate only `text_config.eos_token_id`/`pad_token_id`, `json.dump` back), bypassing the model object's serialization entirely.
- **Files modified:** `scripts/check_token_alignment.py`, `models/Qwen3.6-35B-A3B/config.json` (restored then correctly re-fixed)
- **Verification:** Re-ran the gate end-to-end; `diff` between `config.json.orig` and the re-persisted `config.json` shows exactly 2 changed lines (`eos_token_id`, `pad_token_id`), full key-set diff empty, `architectures`/`vision_config` intact.
- **Committed in:** `55ff811` (Task 2 commit — the bug was caught and fixed before any commit was made, so no separate fix commit exists; the committed script already has the correct JSON-surgery approach)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug caught and fixed before commit)
**Impact on plan:** The plan's literal `model.config.save_pretrained()` instruction turned out to be unsafe for this specific VL-wrapped model; the fix (JSON surgery against the original file) achieves the same stated goal (persist eos/pad alignment to config.json) more safely and is the version that shipped. No scope creep — same two files (`config.json`, `config.json.orig`) as planned.

## Issues Encountered
None beyond the deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `output/base20/token_alignment.json` is the Stage 1.5 gate receipt Phase 21 (SFT) must read for canonical eos/pad IDs — `status=pass`, `canonical_ids.eos_token_id=248046`, `canonical_ids.pad_token_id=248044`.
- `models/Qwen3.6-35B-A3B/config.json` now serves correct stop tokens for any local transformers/vLLM load — ready for 20-03 (DeltaNet-aarch64 serving smoke) and 20-04 (VL merge-path check), both of which depend on the VL wrapper structure (`architectures`, `vision_config`) staying intact — confirmed intact by this plan's fix.
- No blockers. The `model.config.save_pretrained()` pitfall (composite VL configs unwrapping on load) is now documented in `tech-stack.patterns` above for any future script that touches this model's config object.

---
*Phase: 20-base-bring-up*
*Completed: 2026-07-13*

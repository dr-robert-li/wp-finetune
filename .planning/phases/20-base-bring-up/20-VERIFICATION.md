---
phase: 20-base-bring-up
verified: 2026-07-13T18:10:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 20: Base Bring-Up Verification Report

**Phase Goal:** Qwen3.6-35B-A3B is downloaded, loads correctly, and every architecture-specific
serving/training risk (token alignment, DeltaNet kernel, VL merge path) is smoke-tested and
resolved before any SFT run starts.
**Verified:** 2026-07-13T18:10:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Qwen3.6-35B-A3B downloads and loads on GB10 with `trust_remote_code`, transformers 5.x import succeeds, `Qwen3_5MoeForConditionalGeneration` resolves (BASE-01) | ✓ VERIFIED | `output/base20/load_smoke.json`: `status=pass`, `model_class="Qwen3_5MoeForConditionalGeneration"`, `transformers_version=5.3.0`, `peft_version=0.18.1`, `huggingface_hub_version=1.23.0`, `shard_count=26`, `total_size_gb=67.0`. `config/train_config_v4.yaml` exists (model.name/local_dir correct); `git diff --exit-code config/train_config.yaml` is clean (v3 untouched, re-confirmed). |
| 2 | eos/pad token-ID alignment gate passes — config matches tokenizer, confirmed by a real stop-token smoke generation; blocks Stage 2/3 on failure (BASE-02) | ✓ VERIFIED | `output/base20/token_alignment.json`: `status=pass`, `orig_eos_id=248044→aligned_eos_id=248046` (matches `tokenizer_eos_id`), `orig_pad_id=null→aligned_pad_id=248044`, `stopped_naturally=true`, `stop_gen_len=19 < max_tokens_budget=64` with real `decoded_output`. `models/Qwen3.6-35B-A3B/config.json.orig` backup confirmed present on disk; persisted `config.json` re-read directly shows `text_config.eos_token_id=248046`, `pad_token_id=248044`, `vision_config`/`architectures` intact. |
| 3 | DeltaNet aarch64 serving smoke passes with vLLM CUDA-graph capture enabled (vLLM ≥0.19.0); fallback documented if capture fails; `use_kernels` decision recorded (BASE-03) | ✓ VERIFIED | `output/base20/deltanet_smoke.json`: `status=pass`, `vllm_version="0.20.2rc1.dev196+g84f7a5534.d20260510"` (≥0.19.0), `cuda_graph_capture="enabled"`, `fallback_used=false`, `warm_gen_ok=true` with non-empty `warm_gen_sample`, `use_kernels=false` with a non-empty rationale. |
| 4 | VL merge-path round-trip succeeds: Tinker LoRA export → merge onto `model.language_model.*` keys → vLLM serve `--language-model-only` → real, observably-adapter-influenced generation, dual key-prefix risk explicitly checked (BASE-04) | ✓ VERIFIED (see carry-forward note below) | `output/base20/lora_target_modules.json`: `source=tinker`, `confidence=full`, 190 attached modules logged, `prefix_observed` references `language_model`. `output/base20/vl_merge_roundtrip.json`: `status=pass`, `served_ok=true`, `merged_target_module_count=100==expected_target_module_count=100`, `raw_expected_module_count=190`, `dropped_module_count=90` (documented, not silent), `base_vs_merged_differs=true` with real `base_output`/`merged_output` text shown. `merge_adapter.py --help` lists `--config-path`; `trust_remote_code=True` confirmed at both `from_pretrained` call sites. |

**Score:** 4/4 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `config/train_config_v4.yaml` | v4 config sibling | ✓ VERIFIED | model.name/local_dir correct; lora.target_modules/target_parameters carry the CR-01 fix (see below) |
| `scripts/download_model.py` | `--config-path` flag, idempotent/resumable | ✓ VERIFIED | argparse flag present; WR-01 fix (shard-count-aware idempotency) committed `6947148` |
| `scripts/smoke_load_base20.py` | load smoke + receipt | ✓ VERIFIED | reads `model.local_dir` via shared `load_config()`; writes `load_smoke.json` |
| `tests/test_download_model_v4.py` | Wave-0 tests | ✓ VERIFIED | 9 tests, all pass |
| `output/base20/load_smoke.json` | BASE-01 receipt | ✓ VERIFIED | `status=pass` |
| `scripts/check_token_alignment.py` | align_and_check + stop classifier + gate | ✓ VERIFIED | JSON-surgery persistence (not `save_pretrained()`, avoiding VL-config corruption); WR-06 atomic-write fix applied |
| `tests/test_check_token_alignment.py` | Wave-0 tests | ✓ VERIFIED | 11 tests, all pass |
| `output/base20/token_alignment.json` | BASE-02 receipt | ✓ VERIFIED | `status=pass`, canonical_ids block present |
| `models/Qwen3.6-35B-A3B/config.json.orig` | pre-fix backup | ✓ VERIFIED | present on disk |
| `recipes/qwen3.6-35b-a3b-vllm.yaml` | bf16 vLLM recipe | ✓ VERIFIED | `gpu_memory_utilization: 0.80`, no fp8 quantization line; WR-09 fix aligned `max_model_len` to the script's actual default |
| `scripts/serve_base20_vllm.sh` | v4 serve script | ✓ VERIFIED | `bash -n` clean; `LANGUAGE_MODEL_ONLY`/`ENFORCE_EAGER` env gating present, no `wp-30_70`/SIEVE carryover |
| `scripts/_p0_vllm_smoke_serve.py` | `boot_vllm(serve_script=, extra_env=)` | ✓ VERIFIED | additive params confirmed; WR-07/WR-08 fixes (captured launch diagnostics, exact liveness match) committed `91c818b` |
| `scripts/smoke_deltanet_base20.py` | BASE-03 smoke | ✓ VERIFIED | reuses `serve_base20_vllm.sh` via `SERVE_SCRIPT` constant |
| `output/base20/deltanet_smoke.json` | BASE-03 receipt | ✓ VERIFIED | `status=pass` |
| `scripts/build_base20_probe_adapter.py` | probe adapter builder | ✓ VERIFIED | WR-10 fix (tar member-type validation) committed `0161ebd` |
| `scripts/merge_adapter.py` | prefix-aware merge | ✓ VERIFIED (see carry-forward note) | `--config-path` + `trust_remote_code=True` + module-count guard confirmed in current source; WR-02/WR-03/WR-04/WR-05/WR-06 fixes committed but not re-exercised end-to-end post-fix (see below) |
| `scripts/smoke_vl_merge_base20.py` | BASE-04 round-trip smoke | ✓ VERIFIED | reuses `serve_base20_vllm.sh` via `SERVE_SCRIPT` constant |
| `output/base20/vl_merge_roundtrip.json` | BASE-04 receipt | ✓ VERIFIED | `status=pass` (generated pre-review-fix, see carry-forward note) |
| `output/base20/lora_target_modules.json` | attached-module log | ✓ VERIFIED | 190 modules logged, `source=tinker`/`confidence=full` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `scripts/download_model.py` | `config/train_config_v4.yaml` | `--config-path` CLI flag threaded through `load_config()` | ✓ WIRED | `--config-path` present in argparse; docstring example shows the v4 invocation |
| `scripts/smoke_load_base20.py` | `config/train_config_v4.yaml` | shared `load_config()` → `model.local_dir` | ✓ WIRED | imports `load_config` from `scripts.download_model`, resolves `local_dir` from the parsed config |
| `output/base20/token_alignment.json` | Phase 21 SFT (future consumer) | `canonical_ids` block | ✓ WIRED (producer side) | receipt contains `canonical_ids.eos_token_id=248046`/`pad_token_id=248044` — the contract Phase 21 must read |
| `scripts/smoke_deltanet_base20.py` | `scripts/serve_base20_vllm.sh` | `boot_vllm(serve_script=SERVE_SCRIPT, extra_env=...)` | ✓ WIRED | `SERVE_SCRIPT` constant points at `serve_base20_vllm.sh`; `extra_env` passes `LANGUAGE_MODEL_ONLY` |
| `scripts/smoke_vl_merge_base20.py` | `scripts/serve_base20_vllm.sh` | same `boot_vllm(serve_script=, extra_env=)` reuse from 20-03 | ✓ WIRED | same `SERVE_SCRIPT` constant pattern confirmed |
| `scripts/merge_adapter.py` | `config/train_config_v4.yaml` | `--config-path` flag → `load_config()` | ✓ WIRED | `--config-path` present, help text confirms |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Wave-0 tests for download/token-alignment logic pass on current (post-review-fix) code | `pytest tests/test_download_model_v4.py tests/test_check_token_alignment.py -x -q` | 20 passed | ✓ PASS |
| v3 config untouched | `git diff --exit-code config/train_config.yaml` | exit 0, no diff | ✓ PASS |
| CR-01 fix present in current config | `grep target_parameters config/train_config_v4.yaml` | `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` present | ✓ PASS |
| Debt-marker scan (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER) across all 11 phase-modified script/config files | `grep -n -E "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER"` | 0 matches | ✓ PASS |

GPU/vLLM smokes themselves were NOT re-run (per task instructions — receipts + process evidence
suffice); the receipts above are treated as ground-truth evidence of the actual GPU/DeltaNet/vLLM
runs.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| BASE-01 | 20-01-PLAN.md | Qwen3.6-35B-A3B downloads and loads on GB10 | ✓ SATISFIED | `load_smoke.json` status=pass |
| BASE-02 | 20-02-PLAN.md | eos/pad token-ID alignment gate | ✓ SATISFIED | `token_alignment.json` status=pass |
| BASE-03 | 20-03-PLAN.md | DeltaNet aarch64 serving smoke, CUDA-graph capture | ✓ SATISFIED | `deltanet_smoke.json` status=pass |
| BASE-04 | 20-04-PLAN.md | VL merge-path round-trip | ✓ SATISFIED | `vl_merge_roundtrip.json` status=pass |

No orphaned requirements — REQUIREMENTS.md maps exactly BASE-01..04 to Phase 20, and all four
appear in a plan's `requirements` frontmatter field.

### Anti-Patterns Found

None. Scanned all 11 phase-created/modified script/config files (`config/train_config_v4.yaml`,
`scripts/download_model.py`, `scripts/smoke_load_base20.py`, `scripts/check_token_alignment.py`,
`scripts/serve_base20_vllm.sh`, `scripts/_p0_vllm_smoke_serve.py`,
`scripts/smoke_deltanet_base20.py`, `recipes/qwen3.6-35b-a3b-vllm.yaml`,
`scripts/merge_adapter.py`, `scripts/build_base20_probe_adapter.py`,
`scripts/smoke_vl_merge_base20.py`) for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` — zero
hits.

### Carry-Forward Items (not phase-blocking, recorded for Phase 21)

**1. CR-01 fix (`config/train_config_v4.yaml` LoRA `target_modules`/`target_parameters`) needs a
GPU/CPU dry-run confirmation.** The fix was verified statically against
`model.safetensors.index.json` (module-name presence only, no weights loaded) — per the task's
explicit instruction, this does not block Phase 20 but should be confirmed by
`train_model.py --dry-run` (`assert_router_frozen_and_report`) as part of Phase 21's SFT
bring-up.

**2. `scripts/merge_adapter.py`'s WR-02/WR-03/WR-04 review-fix commits post-date the passing
`vl_merge_roundtrip.json` receipt and have not been re-exercised end-to-end.** Timestamp check:
`output/base20/vl_merge_roundtrip.json` was written at `2026-07-13T13:35:09+10:00`; the review
fixes to `merge_adapter.py` (`558d783` target_modules full-path fix, `fa60ea2` unexpected-module
abort, `660130b` idempotency-probe reuse) were committed at `13:59:50`–`14:00:46`, roughly 25-90
minutes AFTER the receipt that certifies BASE-04. There is no unit test file for
`merge_adapter.py` (`find . -iname "*test*merge_adapter*"` returns nothing), so the only
empirical evidence for BASE-04 (the real merge → serve → base-vs-merged-diff round trip) reflects
the PRE-fix code path, not the code currently in the repo. Risk is assessed as low — WR-02/WR-03
are the reviewer's own stated "currently produces correct results, but not guarded for future
cases" characterization (i.e., behaviorally a no-op for this exact run), and WR-04 only affects
an idempotency fast-path that wasn't exercised on this run (the run did a full merge, not a
skip). No FAIL is raised. Recommend a fresh `python scripts/smoke_vl_merge_base20.py` run before
Phase 21's real SFT adapters depend on `merge_adapter.py`'s guard logic, to confirm the fixed code
still produces `merged_target_module_count == expected_target_module_count` and
`base_vs_merged_differs == true`.

### Human Verification Required

None. All four BASE-0x truths have direct empirical receipt evidence from real GPU/CPU runs
(not just code presence/wiring) — this satisfies the behavior-dependent-truth bar without
needing further human testing. See the two carry-forward items above for lower-priority
follow-up recommended before Phase 21 relies on the fixed code paths.

### Gaps Summary

No gaps. All four Success Criteria (BASE-01 through BASE-04) are backed by passing gate receipts
with real, non-fabricated data (actual generation text, actual shard counts, actual vLLM
version strings, actual module counts). The Phase 20 code review (16 findings: 1 critical, 10
warning, 5 info) had all 11 in-scope findings fixed and committed; `pytest
tests/test_download_model_v4.py tests/test_check_token_alignment.py -x` is green post-fix (20/20).
Two carry-forward items are recorded above for Phase 21 awareness — neither blocks Phase 20
sign-off per the task's explicit scoping instructions.

---

*Verified: 2026-07-13T18:10:00Z*
*Verifier: Claude (gsd-verifier)*

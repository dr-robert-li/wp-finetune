---
phase: 18-production-sweep-huggingface-publication
plan: 02
subsystem: packaging-publication
tags: [huggingface, publication, gguf, safetensors, model-card]
requires: [18-01]
provides:
  - "PUBLIC HF repo iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2 (gen bf16 safetensors + tokenizer)"
  - "PUBLIC HF repo iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf (3x judge Q8_0 GGUF)"
  - output/packaging/pub03_upload_manifest.json
  - output/packaging/pub03_validation_receipt.json
affects: []
tech-stack:
  added: []
  patterns:
    - "sequential per-file hf upload with io-stall watchdog (upload-large-folder deadlocks on this host)"
    - "HF_HUB_DISABLE_XET=1 for all Hub transfers on this aarch64 host"
key-files:
  created:
    - output/packaging/hf_cards/gen_README.md
    - output/packaging/hf_cards/judge_gguf_README.md
    - output/packaging/pub03_upload_manifest.json
    - output/packaging/pub03_validation_receipt.json
    - output/packaging/pub03_smoke/ (judge + gen smoke responses)
    - scripts/_pub03_upload.sh
  modified: []
decisions:
  - "Two model repos (one per model), HF-loader-friendly, cross-linked cards — not a combined repo"
  - "Judge ships GGUF-only: models/tinker_export/v1.3 is empty, so the safetensors-if-complete discretion resolves to GGUF-only"
  - "Xet transfer backend disabled host-wide for HF transfers (wedges on aarch64); classic HTTP LFS multipart works"
  - "upload-large-folder abandoned for sequential per-file hf upload (worker-pool deadlock x2 on this host)"
metrics:
  duration: "~13h wall (dominated by 148 GB upload at ~8.7 MB/s + 88 GB validation download)"
  completed: 2026-07-12
status: complete
---

# Phase 18 Plan 02: HuggingFace Publication (PUB-02/PUB-03) Summary

Two-model WordPress pair published PUBLIC under iamchum — gen v1.2 bf16 safetensors (57 GB, 13 shards + task-token tokenizer) and judge v1.3 Q8_0 GGUF (3 ensemble seeds, 32.5 GB each) — with MODEL_CARD-derived cross-linked cards, and the round-trip proven by downloading both artifacts back from HF and passing gen+judge smokes on the downloaded copies.

## Published Repos

| Repo | Contents | Size |
|---|---|---|
| [iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-gen-v1.2) | 13 bf16 safetensors shards + index + config + generation_config + chat_template + tokenizer.json + tokenizer_config + README (20 files) | 57.0 GB |
| [iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf](https://huggingface.co/iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf) | wp-v1.3-judge-s{0,1,2}.Q8_0.gguf + README (4 files) | 90.7 GB |

Total uploaded: 147.64 GiB across 24 files. Exclusions honored per PUB-02 LOCKED: the three bf16 judge GGUFs and qwen3-30b-base.bf16.gguf were never staged; judge bf16 safetensors export not shipped (models/tinker_export/v1.3 is empty — GGUF-only judge per CONTEXT discretion clause).

## Task Commits

| Task | Name | Commit(s) |
|---|---|---|
| 1 | HF cards + ship manifest | `72771a9` |
| 2 | Repo creation + 148 GB upload | `13c4570` (runner v1), `218bc00` (Xet fix), `0b72d6a` (sequential v3) |
| 3 | Post-upload download validation | `35067a6` |

## Validation (from DOWNLOADED artifacts — pub03_validation_receipt.json)

- **API listing:** both repos PUBLIC; file sets and byte sizes match the manifest allowlist exactly; no extras, no bf16/base GGUFs.
- **GGUF load:** judge s1 Q8_0 (32,483,931,840 bytes) downloaded from HF loads in llama.cpp — GGUF v3, arch qwen3moe, 128 experts, 48 blocks, file_type Q8_0.
- **Judge smoke:** `<wp_judge>` prompt (smoke_prompts.json idx 0) against the downloaded GGUF returns a critique covering all 9 rubric dimensions; `<judge_output>` JSON parses via the project's own parser (eval/output_parsers.parse_judge_scores), overall 74, verdict PASS.
- **Gen smoke:** the downloaded gen repo served under vLLM (serve_30_70_vllm.sh, enable_thinking=false, temp 0.0); `<wp_gen>` prompt (idx 121) returns WPCS-shaped PHP (snake_case, tab indentation, wc_get_order/WP_Error, i18n `__()` with translators comments) — routing correct.
- Scratch downloads (106 GB) cleaned after validation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] upload-large-folder deadlocks on this host — switched to sequential per-file upload**
- **Found during:** Task 2
- **Issue:** `hf upload-large-folder` wedged twice: 10 workers stuck "pre-uploading" with zero io for 1h+ each run (first with the Xet backend, then again with Xet disabled and hub 1.23).
- **Fix:** Single-file `hf upload` probe committed a 4.98 GB shard end-to-end (~11 min), so the runner was rewritten to iterate the manifest allowlist sequentially with per-file 3-attempt retry and a 5-minute io-stall watchdog. Sustained ~8.7 MB/s to completion.
- **Files modified:** scripts/_pub03_upload.sh
- **Commits:** `218bc00`, `0b72d6a`

**2. [Rule 3 - Blocking] Xet transfer backend disabled**
- **Found during:** Task 2 (first wedge diagnosis)
- **Issue:** Session env carried `HF_XET_HIGH_PERFORMANCE=1`; the Xet path never moved a byte on this aarch64 host. huggingface_hub was also outdated (1.11.0).
- **Fix:** `HF_HUB_DISABLE_XET=1` + unset `HF_XET_HIGH_PERFORMANCE` in the runner; hub upgraded to 1.23.0 (hf_transfer is deprecated in 1.23 — classic HTTP LFS multipart is the working path). Also applied to the validation downloads.
- **Commit:** `218bc00`

## Authentication Gates / Interventions

1. **HF write-token gate (checkpoint, resolved by user):** the cached token authenticated as iamchum but lacked write scope — `hf repos create` returned 403. Execution paused at a human-action checkpoint; the user minted a write-scope token ("wp-finetune", role: write) and re-authenticated. No repos or bytes had been created before the gate.
2. **upload-large-folder deadlock (auto-fixed):** see Deviations above — strategy switch to sequential uploads.
3. **Background-process reaping (orchestrator-assisted):** the sequential upload script died silently at shard 8 when the spawning task ended (harness reaped the process group); the orchestrator relaunched it detached (`setsid nohup`, session-owned) and it ran to completion. The same pattern hit the validation download (killed at 19/57 GB) — relaunched detached from this session; `hf download` resumed from the partial files. Lesson recorded: long transfers must be launched detached, never tied to an agent task's lifetime.

## Known Stubs

None — both repos carry complete, validated artifacts.

## Threat Flags

None beyond the plan's threat model. T-18-02-SP (whoami == iamchum asserted before create/upload), T-18-02-ID (allowlist staging dir, listing gate confirms no excluded weights), T-18-02-INT (round-trip from downloaded artifacts) all mitigated as planned.

## Self-Check: PASSED

- output/packaging/hf_cards/gen_README.md: FOUND
- output/packaging/hf_cards/judge_gguf_README.md: FOUND
- output/packaging/pub03_upload_manifest.json: FOUND
- output/packaging/pub03_validation_receipt.json: FOUND
- Commits 72771a9, 13c4570, 218bc00, 0b72d6a, 35067a6: FOUND in git log

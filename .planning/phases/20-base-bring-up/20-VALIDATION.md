---
phase: 20
slug: base-bring-up
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-07-13
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (project standard; `tests/` dir, `tests/conftest.py` shared fixtures) |
| **Config file** | none at repo root — tests run via bare `pytest tests/` |
| **Quick run command** | `pytest tests/test_<new_file>.py -x` |
| **Full suite command** | `pytest tests/` |
| **Estimated runtime** | ~5-15 s for the two new unit files (mock-only, no GPU/download) |

The two testable pieces (BASE-01 config/path logic, BASE-02 alignment logic) are mock-only
pytest units. BASE-03 (vLLM DeltaNet serving) and BASE-04 (VL merge round-trip) are GPU/Docker/
Tinker-dependent smoke scripts — consistent with existing repo convention (`_p0_vllm_smoke_serve.py`,
`merge_adapter.py` have no pytest suites); each is gated by a deterministic automated
receipt-status assertion (`output/base20/*.json` status==pass), not a pytest test.

---

## Sampling Rate

- **After every task commit:** Run the task's `<automated>` verify (quick pytest for the two
  unit tasks; the receipt-status assertion for the smoke tasks).
- **After every plan wave:** Run `pytest tests/` (full suite — confirms no regression to the
  20+ existing v3.x test files, which Phase 20 must not touch).
- **Before `/gsd-verify-work`:** Full suite green AND all four `output/base20/*.json` receipts
  show status=pass.
- **Max feedback latency:** ~15 s for unit tasks; smoke tasks are bounded by GB10 vLLM boot
  (wait_healthy >=1200 s timeout) but the proof (receipt assertion) is instant.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 20-01-01 | 01 | 1 | BASE-01 | T-20-01 | HF-hash download integrity; v3 config untouched | unit (mock) | `pytest tests/test_download_model_v4.py -x && git diff --exit-code config/train_config.yaml` | ❌ W0 (authored test-first in this tdd task) | ⬜ pending |
| 20-01-02 | 01 | 1 | BASE-01 | T-20-02 | trust_remote_code only on allowlisted Qwen repo | smoke + receipt | `python -c "import json;d=json.load(open('output/base20/load_smoke.json'));assert d['status']=='pass'"` | receipt at runtime | ⬜ pending |
| 20-02-01 | 02 | 2 | BASE-02 | T-20-02b | gate fails CLOSED on unaligned/non-natural-stop | unit (mock) | `pytest tests/test_check_token_alignment.py -x` | ❌ W0 (authored test-first in this tdd task) | ⬜ pending |
| 20-02-02 | 02 | 2 | BASE-02 | T-20-02a/b | canonical IDs verified by a real natural-stop generation; orig config backed up | smoke + receipt | `python -c "import json;d=json.load(open('output/base20/token_alignment.json'));assert d['status']=='pass' and d['stopped_naturally']"` | receipt at runtime | ⬜ pending |
| 20-03-01 | 03 | 3 | BASE-03 | T-20-03a | serve script defaults use_kernels/Atlas kernel OFF; 0.80 mem-util | unit (static) | `bash -n scripts/serve_base20_vllm.sh` + yaml/signature assert (see plan verify) | ✅ (inline assert) | ⬜ pending |
| 20-03-02 | 03 | 3 | BASE-03 | T-20-03a/b | CUDA-graph capture ON first; use_kernels=False recorded; eager fallback documented | smoke + receipt | `python -c "import json;d=json.load(open('output/base20/deltanet_smoke.json'));assert d['status']=='pass' and d['warm_gen_ok']"` | receipt at runtime | ⬜ pending |
| 20-04-01 | 04 | 4 | BASE-04 | T-20-04b | log ACTUAL attached modules; assert language_model.* prefix | smoke + receipt | `python -c "import json;d=json.load(open('output/base20/lora_target_modules.json'));assert d['attached_modules']"` | receipt at runtime | ⬜ pending |
| 20-04-02 | 04 | 4 | BASE-04 | T-20-04a | base-vs-merged diff + merged-target-module-count guard (no silent partial-load) | smoke + receipt | `python -c "import json;d=json.load(open('output/base20/vl_merge_roundtrip.json'));assert d['status']=='pass' and d['served_ok']"` | receipt at runtime | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Both new unit test files are authored **test-first within their own `tdd="true"` tasks** (not a
separate Wave 0 plan) — the failing test is written from the `<behavior>` block before the logic:

- [ ] `tests/test_download_model_v4.py` — BASE-01 config/path logic (mirror `tests/test_prepare_tokenizer.py`: mock `snapshot_download`, idempotency-skip, correct kwargs, v4 config values) — authored in task 20-01-01
- [ ] `tests/test_check_token_alignment.py` — BASE-02 alignment logic (MagicMock config/tokenizer: mismatch→fix invariant, no-op on aligned, run-to-length=fail classifier, receipt fields) — authored in task 20-02-01

No pytest scaffolding needed for BASE-03/BASE-04 — GPU/Docker/Tinker smoke scripts are
consistently not unit-tested elsewhere in this repo; they carry automated receipt-status gates instead.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 67 GiB download + bf16 load + forward | BASE-01 | Network + 67 GiB unified-RAM load not CI-automatable | Run task 20-01-02; confirm `output/base20/load_smoke.json` status=pass, model_class=Qwen3_5MoeForConditionalGeneration |
| vLLM DeltaNet serving smoke under CUDA-graph capture | BASE-03 | Requires GB10 GPU + `dgx-vllm-eugr-nightly` Docker container | Run task 20-03-02; confirm `output/base20/deltanet_smoke.json` status=pass, cuda_graph_capture in {enabled, eager_fallback}, use_kernels=false |
| VL merge → serve → base-vs-merged diff | BASE-04 | Requires GB10 GPU + Docker + (primary) Tinker account | Run tasks 20-04-01/02; confirm `output/base20/vl_merge_roundtrip.json` status=pass, merged_target_module_count==expected, (diff true \| confidence reduced) |

Each manual smoke's DETERMINISTIC proof is the automated receipt-status assertion in the row
above — no subjective "looks right" step. The manual part is the GB10/Docker/Tinker execution.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (8/8 tasks: 2 pytest, 6 receipt/static assertions)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (every task has one)
- [x] Wave 0 covers all MISSING references (both test files authored test-first in their tdd tasks)
- [x] No watch-mode flags (all pytest runs are `-x`, one-shot; no `--watch`)
- [x] Feedback latency < 15 s for unit tasks; smoke tasks bounded by vLLM boot timeout, proof is instant
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready

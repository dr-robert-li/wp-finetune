---
phase: 3
slug: model-prep-and-training
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-27
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, 46+ tests passing) |
| **Config file** | none (standard pytest discovery) |
| **Quick run command** | `python3 -m pytest tests/ -x -q` |
| **Full suite command** | `python3 -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds (unit tests only, no model loading) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/ -x -q`
- **After every plan wave:** Run `python3 -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | MODL-01 | smoke | `pytest tests/test_train_model.py::test_model_downloaded -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | MODL-02 | unit | `pytest tests/test_prepare_tokenizer.py::test_special_tokens_added -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | MODL-03 | unit | `pytest tests/test_prepare_tokenizer.py::test_embeddings_mean_init -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | MODL-04 | unit | `pytest tests/test_prepare_tokenizer.py::test_smoke_single_token_ids -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | EVAL-01 | integration | `pytest tests/test_eval_gen.py::test_phpcs_eval_runs -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | EVAL-02 | unit | `pytest tests/test_eval_judge.py::test_spearman_computation -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | EVAL-03 | unit | `pytest tests/test_eval_gen.py::test_security_rate_detection -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | EVAL-05 | unit | `pytest tests/test_eval_gate.py::test_gate_pass -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 2 | TRNG-01 | unit | `pytest tests/test_train_model.py::test_lora_config_params -x` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 2 | TRNG-02 | unit | `pytest tests/test_train_model.py::test_modules_to_save -x` | ❌ W0 | ⬜ pending |
| 03-03-03 | 03 | 2 | TRNG-03 | unit | `pytest tests/test_train_model.py::test_dataset_schema -x` | ❌ W0 | ⬜ pending |
| 03-03-04 | 03 | 2 | TRNG-04 | unit | `pytest tests/test_train_model.py::test_router_logits_enabled -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_prepare_tokenizer.py` — MODL-02, MODL-03, MODL-04
- [ ] `tests/test_train_model.py` — MODL-01, TRNG-01, TRNG-02, TRNG-03, TRNG-04
- [ ] `tests/test_eval_gen.py` — EVAL-01, EVAL-03 (mock phpcs subprocess)
- [ ] `tests/test_eval_judge.py` — EVAL-02 (pure Python Spearman)
- [ ] `tests/test_eval_gate.py` — EVAL-05 (exit code checks)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| W&B tracking shows loss curves | TRNG-05 | Requires live W&B account + training run | Check wandb.ai dashboard during/after training |
| Training completes without OOM | TRNG-06 | Requires actual DGX Spark GPU training | Monitor `nvidia-smi` during training run |
| Eval runs in DGX Toolbox container | EVAL-04 | Requires DGX Toolbox environment | Run `~/dgx-toolbox/eval/eval-toolbox.sh` and execute eval inside |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

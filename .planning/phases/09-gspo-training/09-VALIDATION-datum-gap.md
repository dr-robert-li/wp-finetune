---
phase: 09
slug: gspo-training
scope: corrective — GSPO Datum/logprob assembly gap
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
source: 09-RESEARCH-datum-gap.md § Validation Architecture
---

# Phase 9 (corrective) — GSPO Datum/Logprob Assembly · Validation Strategy

> Validation contract for the corrective plan that makes `forward_backward_custom` run with a
> REAL GSPO importance-sampling ratio (not the `seq_ratio=1.0` REINFORCE fallback).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (run via `.venv-tinker/bin/pytest` — the venv with tinker+openai+scipy) |
| **Config file** | `pyproject.toml` / `pytest.ini` (existing) |
| **Quick run command** | `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py -x -q` |
| **Full suite command** | `.venv-tinker/bin/pytest tests/ -q` |

---

## Sampling Rate

- **After every task commit:** `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py tests/test_rl_rollouts.py tests/test_rl_train.py -x -q`
- **After every wave:** `.venv-tinker/bin/pytest tests/ -q`
- **Phase gate:** full suite green **AND** the 1-step live re-smoke acceptance criteria met before `/gsd:verify-work`.

---

## Per-Task Verification Map

| Req | Behavior | Test Type | Automated Command | File | Status |
|-----|----------|-----------|-------------------|------|--------|
| DATUM-01 | `Datum.loss_fn_inputs` has target_tokens+logprobs+advantages | integration | `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py::test_trajectory_to_datum_schema -x` | ❌ W0 | ⬜ |
| DATUM-02 | `len(data)==len(advantages)` (single-turn) | integration | `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py::test_datum_assembly_len_matches_advantages -x` | ❌ W0 | ⬜ |
| GRPO-05 | Interleaved gen/judge rollouts → TrajectoryGroups | unit | `.venv-tinker/bin/pytest tests/test_rl_rollouts.py::TestBuildTrajectoryGroups -x` | ✅ update | ⬜ |
| GRPO-06 | Per-prompt group centering (CR-06) preserved | unit | `.venv-tinker/bin/pytest tests/test_rl_rollouts.py::TestComputeRolloutAdvantages -x` | ✅ update | ⬜ |
| GRPO-07 | Mixed-reward group → non-zero advantages | unit | `.venv-tinker/bin/pytest tests/test_rl_train.py::TestGSPOTrainingStep::test_grpo_advantages -x` | ✅ update | ⬜ |
| GRPO-08 | KL autohalt before optim_step (unchanged) | unit | `.venv-tinker/bin/pytest tests/test_rl_train.py::TestGSPOTrainingStep::test_kl_autohalt -x` | ✅ no change | ⬜ |
| DATUM-03 | real non-zero `sampling_sum` (the loss `except (AttributeError, KeyError)` fallback NOT taken) in live re-smoke | live smoke | re-smoke command below | manual | ⬜ |

---

## Wave 0 Requirements

- [ ] `tests/test_rl_datum_assembly.py` — DATUM-01/02 against a fixture `Trajectory` → `trajectory_to_data` (no Tinker client, no weights, <1s).
- [ ] `tests/test_rl_train_integration.py` — `_FakeSeq` + `_FakeSamplingClient` gain `.logprobs` (non-zero, so `seq_ratio != 1.0`); the fake prompt must be a valid `tinker.ModelInput`/stub `.chunks` that `trajectory_to_data` consumes.
- [ ] Update dict-asserting tests in `tests/test_rl_rollouts.py`, `tests/test_rl_train.py`, `tests/test_reward_pipeline*.py` for the new `(list[Datum], list[float], meta)` return contract.

---

## Manual-Only Verification — 1-step live re-smoke (DATUM-03)

Serve the judge first (`bash scripts/serve_v4_judge_vllm.sh`; wait for `/v1/models`), export `TINKER_API_KEY` from `.env`, then:

```bash
.venv-tinker/bin/python scripts/rl_train.py \
  --total-steps 1 --batch-size 2 --group-size 2 --max-pool 2 \
  --judge-base-url http://localhost:8000/v1 --judge-model wp_judge \
  --manifest-path output/_smoke/checkpoint_manifest.json \
  --metrics-path output/_smoke/metrics/rl_metrics.jsonl
```

Acceptance:
1. No `AttributeError: 'dict' object has no attribute 'loss_fn_inputs'` (the original blocker).
2. `optim_step` is reached (no halt) and `output/_smoke/metrics/rl_metrics.jsonl` has a `"step": 1` row with numeric `gspo/n_sequences`.
3. **Real logprobs flow** — the loss `gspo_loss_fn` reads a real non-zero `sampling_sum` from `datum.loss_fn_inputs["logprobs"]` and does NOT enter the `except (AttributeError, KeyError)` fallback. NOTE: on step 1 the sampling client is saved from the same weights, so `seq_ratio` legitimately CLAMPS to exactly 1.0 even with real logprobs — `seq_ratio != 1.0` is therefore NOT a valid signal. Verify via a temporary `logger.info("sampling_sum=%s fallback=%s", sampling_sum, used_fallback)` in `gspo_loss_fn`, then remove. Must NOT touch the canonical `output/rl_checkpoints/` manifest.

---

## Validation Sign-Off

- [ ] DATUM-01/02 offline tests green
- [ ] Updated unit tests green (new Datum return contract)
- [ ] Live re-smoke acceptance (1–3) met: real non-zero `sampling_sum`, loss `except` fallback NOT taken (NOT `seq_ratio != 1.0` — that legitimately clamps to 1.0 on step 1)
- [ ] No `output/rl_checkpoints/` clobber (smoke isolated)
- [ ] `nyquist_compliant: true` set

**Approval:** pending

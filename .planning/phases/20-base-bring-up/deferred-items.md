# Deferred Items — Phase 20 (Base Bring-Up)

Out-of-scope discoveries logged, not fixed (Scope Boundary rule — only auto-fix issues directly
caused by the current plan's task changes).

## 20-01

- **7 pre-existing test failures, unrelated to Phase 20 files** — surfaced only because
  `pytest tests/` could not even collect before the pytest 6.0.0rc2 → 9.1.1 fix (Rule 3, see
  SUMMARY). None touch `scripts/download_model.py`, `config/train_config_v4.yaml`, or
  `scripts/smoke_load_base20.py`:
  - `tests/test_reward_calibration.py::TestOracleValidAssertion::test_calibration_reward_impl_passes_gate`
    — `ci_lo=nan`, oracle-corpus/pipeline data issue (Phase 08.2 reward validity gate)
  - `tests/test_reward_form_sweep.py::test_weight_zero_matches_fix_correctness` — `assert None is not None`
  - `tests/test_reward_validity_gate.py::TestPairwiseRankAgreementValid::test_pairwise_rank_agreement_valid`
    — same NaN/ci_lo pattern as above
  - `tests/test_rl_judge_dispatch.py::TestScoreJudgeConsistencyBatch::test_timeout_imputes_from_group_mean`
  - `tests/test_rl_judge_dispatch.py::TestScoreJudgeConsistencyBatch::test_exception_imputes_from_group_mean`
  - `tests/test_rl_judge_dispatch.py::TestScoreJudgeConsistencyBatch::test_batch_preserves_input_order`
    — all three: score mismatch (0.5 vs expected 0.8), Phase 9 GSPO judge-dispatch logic
  - `tests/test_rl_train.py::TestRLTrainUnit::test_lora_config` — `ModuleNotFoundError: No module
    named 'tinker'` (Tinker SDK not installed in this host env; Phase 8/9 dependency, not a Phase
    20 concern)

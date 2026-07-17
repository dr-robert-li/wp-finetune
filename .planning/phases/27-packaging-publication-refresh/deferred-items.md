# Deferred Items — Phase 27

Out-of-scope discoveries logged per executor deviation-rules SCOPE BOUNDARY (not fixed, not this
plan's files).

## 27-01: pre-existing `pytest tests/` failures (unrelated to this plan)

`pytest tests/` (plan verification step 4, "cheap guard, not a gate") surfaces 8 pre-existing
issues, none touching files this plan modifies (ROADMAP.md, REQUIREMENTS.md,
`eval4_ext_gguf_convert.sh`, `pkg4_quant_type_check.py`, `pub4_validate_upload.py`). Last git touch
on all affected test files predates this plan (e.g. `tests/test_reward_form_sweep.py` last modified
at `e93f674`, Phase 08.2).

1. `tests/test_tinker_reasoning_data_v4.py` — collection error, `ModuleNotFoundError: No module
   named 'tinker_cookbook'` (env-specific, `.venv-tinker` dependency not on this `python3`'s path).
2. `tests/test_reward_calibration.py::TestOracleValidAssertion::test_calibration_reward_impl_passes_gate`
   — `ci_lo=nan`.
3. `tests/test_reward_form_sweep.py::test_weight_zero_matches_fix_correctness` — `assert None is not None`.
4. `tests/test_reward_validity_gate.py::TestPairwiseRankAgreementValid::test_pairwise_rank_agreement_valid`
   — `ci_lo>0` assertion.
5. `tests/test_rl_judge_dispatch.py::TestScoreJudgeConsistencyBatch` — 3 failures (timeout/exception
   imputation, order preservation).
6. `tests/test_rl_train.py::TestRLTrainUnit::test_lora_config` — `ModuleNotFoundError: No module
   named 'tinker'`.

None of these touch GGUF conversion, quant-type checks, or HF publication logic (this plan's
scope). Not fixed here.

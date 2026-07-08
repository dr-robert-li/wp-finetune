# Phase 11 — deferred items (out-of-scope discoveries during execution)

## From plan 11-03 (2026-07-09)

- `.venv-tinker` lacks `python-dotenv` → `tests/test_preflight.py` fails collection
  (`from dotenv import load_dotenv`). Pre-existing; unrelated to sieve work.
- 7 pre-existing failures in RL/reward test files (`test_reward_calibration.py`,
  `test_reward_form_sweep.py`, `test_reward_validity_gate.py`, `test_rl_judge_dispatch.py`,
  `test_rl_train.py`). RL is CLOSED (2026-07-05, 6/6 kills); these tests reference
  live-Tinker/served-judge fixtures. Not touched by plan 11-03 changes — candidates for
  skip-marking or removal in a cleanup pass.

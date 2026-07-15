# Deferred items — Phase 22

Out-of-scope discoveries found during 22-01 execution (SCOPE BOUNDARY: not caused by
this plan's changes, not fixed).

- `tests/test_preflight.py` fails collection: `ModuleNotFoundError: No module named
  'dotenv'`. Pre-existing (confirmed via `git stash` — reproduces on the unmodified
  tree). Unrelated to any file this plan touches.
- `tests/test_reward_calibration.py`, `tests/test_reward_form_sweep.py`,
  `tests/test_reward_validity_gate.py`, `tests/test_rl_judge_dispatch.py`,
  `tests/test_rl_train.py` (7 failures total): pre-existing, unrelated to Sieve/
  protected-mask tooling. Confirmed via `git stash` — reproduce identically on the
  unmodified tree (Tinker auth env / reward-gate env dependencies, not code bugs
  introduced by this plan).

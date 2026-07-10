# deprecated/

Frozen one-off scaffolding from the v3.0 campaign. None of it is part of the repeatable pipeline
(`PIPELINE.md`). It is kept, not deleted, because it is the archaeology of how the pipeline's gates were
established — the merge-equivalence probes, the RL smoke launchers, the format-stability bisects, the
pruning experiment drivers. If you are following the pipeline on a new base model, you do not need anything
in here.

Nothing in `deprecated/` is imported or called by any active script or `wp-finetune:*` skill. That was
verified by grep before the move: every file here returned zero references from the active tree.

**Correction (Phase 17-01):** the grep checked Python `import` statements only, which missed two helper
dirs referenced via runtime string-path construction (`PROJECT_ROOT / "scripts" / "_wpbench_pth"` in
`scripts/run_eval_reasoning.py`'s `_run_wpbench()`), not a static import. `_wpbench_pth/` and
`_wpbench_shim/` were restored to `scripts/` — see below.

## What's here (`deprecated/scripts/`, 95 files)

- **`_04.4_*`** — reasoning-merge fidelity/equivalence experiments (v3/v4 winner probes, byte-identity
  checks, post-merge validation, revl01a/revl04 harness confirmations).
- **`_p0_*`** — static-MoE merge and anchor forward-equivalence smoke probes (Unsloth-convention per-expert
  delta application, ckpt72 block-diagonal tests, extraction probes).
- **`_rlev01_*`** — RL-eval-01 checkpoint probe family (batch/score/wpbench step probes). Note: the active
  `scripts/_rlev01_wpbench_ckpt.py` was NOT moved — it is the single source of truth for the codegen bar
  constant `BASELINE_V12`, imported by the live `rl_codegen_tripwire.py`.
- **`_v3_*`, `_run_grid_*`, `_run_p0_*`** — grid liveness/dump probes and grid experiment drivers.
- **watchdogs / monitors / launchers** — `_*_watchdog.sh`, `_oom_guard.sh`, `_mem_watchdog.sh`,
  `_phase4.3_monitor.sh`, `_seedA_watch.sh`, `launch_*.sh`, `setup_container_phase1a.sh`,
  `monitor_phase1b_*.sh`, `w0_03_smoke_*.py`, `prune_run_13_04.sh`. Ephemeral run orchestration for jobs
  that have already produced their recorded results.
- **`reward_form_sweep.py`** + **`_probe_rl_reward.py`** — the phase-08.1 reward-form sweep experiment and
  its helper (moved as a pair; the sweep imports the helper).
- **`judge_batch_18.py`** — stray root-level one-off judge batch runner.

## Active dependencies deliberately kept in `scripts/`

Three underscore-prefixed files are real, imported dependencies and stayed in `scripts/`:
`_rlev01_wpbench_ckpt.py`, `_p0_vllm_smoke_serve.py` (vLLM boot/health/stop helper used by the eval, sieve,
and prune drivers), and `_reward_validity_oracle.py` (used by the active reward-validity gate).

Two more were restored to `scripts/` in Phase 17-01 after being wrongly archived here in Phase 16:
`_wpbench_pth/usercustomize.py` (PYTHONPATH-loaded monkeypatch that threads `enable_thinking=false` into
wp-bench's litellm calls and strips residual `<think>` blocks) and `_wpbench_shim/npx` (PATH shim so
wp-bench's `npx wp-env` calls resolve to the globally-installed `wp-env` bin). Both are required by every
full wp-bench run against this Qwen3 reasoning model — see `scripts/run_eval_reasoning.py::_run_wpbench`.

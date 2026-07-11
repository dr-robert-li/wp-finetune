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
- **`judge_batch_18.py`** — stray root-level one-off judge batch runner.

(`reward_form_sweep.py` + `_probe_rl_reward.py` were archived here as a pair in an earlier phase but were
restored to `scripts/` in Phase 18-01 — an active test still imports the sweep directly. See "Active
dependencies" below.)

## Active dependencies deliberately kept in `scripts/`

Three underscore-prefixed files are real, imported dependencies and stayed in `scripts/`:
`_rlev01_wpbench_ckpt.py`, `_p0_vllm_smoke_serve.py` (vLLM boot/health/stop helper used by the eval, sieve,
and prune drivers), and `_reward_validity_oracle.py` (used by the active reward-validity gate).

Two more were restored to `scripts/` in Phase 17-01 after being wrongly archived here in Phase 16:
`_wpbench_pth/usercustomize.py` (PYTHONPATH-loaded monkeypatch that threads `enable_thinking=false` into
wp-bench's litellm calls and strips residual `<think>` blocks) and `_wpbench_shim/npx` (PATH shim so
wp-bench's `npx wp-env` calls resolve to the globally-installed `wp-env` bin). Both are required by every
full wp-bench run against this Qwen3 reasoning model — see `scripts/run_eval_reasoning.py::_run_wpbench`.

Six more were found wrongly archived (Phase 16) and restored to `scripts/` in Phase 18-01, caught by
running the mandatory double-grep gate against *every* file already under `deprecated/`, not just new
candidates:

- `_p0_smoke_common.py` — imported directly by `tests/phase4_4/test_smoke_common.py`.
- `launch_validated_smoke.sh` — path-referenced by `tests/test_launch_validated_smoke.py` (`LAUNCHER =
  REPO_ROOT / "scripts" / "launch_validated_smoke.sh"`).
- `reward_form_sweep.py` + `_probe_rl_reward.py` — `tests/test_reward_form_sweep.py` imports the sweep
  directly (`import scripts.reward_form_sweep as s`); the sweep in turn imports `_probe_rl_reward` for
  `_compute_group_stats`, so both had to move together.
- `_p0_revl01_preflight.py` — invoked at runtime via `subprocess.run([sys.executable, "-m",
  "scripts._p0_revl01_preflight", ...])` in `scripts/run_eval_reasoning.py`.
- `_sieve_vllm_patch/sitecustomize.py` — bind-mounted into the vLLM container by
  `scripts/serve_30_70_vllm.sh` (`-v ".../scripts/_sieve_vllm_patch:/sieve_patch:ro"`) and documented as a
  live PYTHONPATH patch by `scripts/sieve_ksweep_run.py` and `scripts/sieve_expert_mask_inference.py`.

**Known false positives in the verify gate:** four archived files still turn up in a grep across the active
tree, but every hit is a comment or advisory string, not a runtime import/subprocess/path-construction call
— manually confirmed, left archived: `_04.4_revl04_v4.py` (mentioned in a `run_eval_reasoning.py` comment),
`_04.4_run_merge_v3.py` (mentioned in a human-facing `echo` message in `serve_reasoning_v3_vllm.sh`),
`_gen_judge_probe_corpus.py` (mentioned in a `test_reward_validity_gate.py` comment), `_rlev01_score.py`
(mentioned in a `_reward_validity_oracle.py` docstring).

## Phase 18-01 sweep — 10 files added

Double-grepped (import statements AND string literals, across `scripts/ eval/ tests/ wp-bench/ .claude/
telemetry/ config/ docs/` plus `PIPELINE.md`/`README.md`/`PROJECT.md`/`wp-moe.md`) before moving — zero hits
outside the moved set itself.

- **`bench17_swebench_consolidate_report.py`, `bench17_swebench_generate_predictions.py`,
  `bench17_swebench_throughput_probe.py`, `swebench_arm64_eval.py`, `bench17_wpbench_full_rerun.py`** — the
  Phase 17 SWE-bench / wp-bench-rerun driver family. Produced the receipts that MODEL_CARD.md's Benchmarks
  section cites (`output/bench17/*.json`); the receipts stay in `output/bench17/`, only the one-off run
  scripts move. Phase 17 numbers are final inputs for v3.1 — no rerun is planned, so these drivers have no
  future caller. `consolidate_report.py`'s one string mention of `swebench_arm64_eval.py` is report metadata
  (a methodology note baked into its JSON output), not a runtime import or subprocess call.
- **`judge_batch.py`, `generate_judge_batch.py`, `run_judge_batch5.py`, `run_judge_chunk.py`,
  `run_judge_batch_loop.sh`** — the manual judge-batch data-generation cluster from the original Phase
  1/2 dataset build (267K examples, exported and frozen since v1.0). `judge_batch.py` duplicates rubric
  logic wholesale with a hardcoded absolute path and was never wired into `phase1_judge.py` or
  `phase2_judge_dataset.py`, the pipeline's actual judge entrypoints. The other four are a self-contained
  manual-invocation cluster (`run_judge_batch5.py` and `run_judge_chunk.py` import
  `generate_judge_batch.py`; `run_judge_batch_loop.sh` shells out to it) with no external caller — ephemeral
  run orchestration for a dataset-build job that already produced its recorded results.

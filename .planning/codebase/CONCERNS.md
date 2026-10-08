---
last_mapped_commit: f5e93c309b551cf8744e50e6bc5119ce1aea071d
last_mapped_at: 2026-10-09
---
# Codebase Concerns

**Analysis Date:** 2026-10-09

## Tech Debt

**Repository size and tracked generated data:**
- Issue: About 127 MB of files are tracked in git, including 383 `.json`/`.jsonl` files. Several are multi-MB pipeline artifacts rather than source (for example `data/phase1b/rejudge_full_20k.pre_secn04.jsonl` at ~18 MB, `data/phase1b/rejudge_full_20k.jsonl` at ~18 MB, `data/phase4_reasoning/deep_judge_cot/deep_judge_cot_bulk.json` at ~4 MB). `.gitignore` excludes `data/phase1_extraction/`, `data/phase2_synthetic/`, `data/phase3_cot/`, `data/final_dataset/`, `models/`, `adapters/`, and `merged_model/`, but the `data/phase1b/`, `data/phase4_reasoning/`, `data/reasoning_dataset/`, and `output/` trees are committed.
- Files: `data/phase1b/`, `data/phase4_reasoning/`, `data/reasoning_dataset/`, `output/`, `deps/wp-finetune-data/`
- Impact: Clone and history growth, slow diffs, and it is unclear which artifacts are authoritative versus regenerable. Versions of the same dataset coexist (for example `openai_train.augmented.jsonl` and `openai_train.augmented.replay30.jsonl`/`replay50.jsonl`, and `rejudge_full_20k.jsonl` next to `.pre_secn04.jsonl`).
- Fix approach: Move regenerable datasets to git LFS or an external artifact store and record their hashes in a manifest. Keep only one canonical copy per dataset version. Extend `.gitignore` to cover regenerable outputs.

**Duplicated and superseded code kept in-tree:**
- Issue: `deprecated/` holds 114 files, including copies of live logic. `deprecated/scripts/judge_batch.py`, `judge_batch_18.py`, `_judge_batch.py`, and `_judge_repos.py` duplicate the `scripts/judge_*.py` family. `deprecated/scripts/_04.4_anchors_v3.py` and `_p0_anchor_forward_*.py` duplicate the merge-validation work in `scripts/`. Several `scripts/` names are near-duplicates: `generate_cot.py` (1287 lines), `generate_cot_real.py`, `generate_cot_vllm.py`, and `generate_deep_judge_cot.py`; `phase1_judge.py` and `phase2_judge.py`; `prune_gate_v4.py` and `prune_gated_eval.py`.
- Files: `deprecated/`, `scripts/generate_cot*.py`, `scripts/judge_*.py`, `scripts/prune_gate*.py`, `scripts/prune_gated_eval.py`
- Impact: Readers cannot tell which version produced a published number. Fixes applied to one copy do not reach the other.
- Fix approach: Confirm which variant produced each receipt in `output/`, then delete or move the rest into `deprecated/` with a pointer. Add a one-line header to each live script naming the phase it serves.

**Oversized orchestration and training scripts:**
- Issue: Several scripts are very large and mix concerns. `scripts/rl_train.py` (1767 lines), `scripts/run_eval_triage.py` (1415), `scripts/rl_rollouts.py` (1394), `scripts/generate_cot.py` (1287), and `scripts/merge_adapter.py` (974) each contain configuration, I/O, retry logic, and core math in one file.
- Files: `scripts/rl_train.py`, `scripts/run_eval_triage.py`, `scripts/rl_rollouts.py`, `scripts/generate_cot.py`, `scripts/merge_adapter.py`
- Impact: Hard to unit test in isolation and hard to review. Changes carry regression risk outside the function being edited.
- Fix approach: Split by responsibility (config loading, rollout generation, reward assembly, I/O) when next touched. Keep the public CLI stable.

**Hardcoded values in comparison outputs:**
- Issue: Non-gating v3.0 comparison figures are literal numbers without a `source_receipt` field. The phase-23 review (finding L-001) verified them as byte-accurate, but they are not derived at build time.
- Files: `scripts/build_eval4_comparison.py`, `output/eval4/`, `.planning/phases/23-final-evaluation/23-REVIEW.md`
- Impact: Low. Figures can drift from their sources without any check failing.
- Fix approach: Read the figures from the receipt files at build time and fail if a source is missing.

**Unimplemented LLM-assisted rubric checks:**
- Issue: `eval/rubric_scorer.py` lists four scoring tools. The fourth, LLM-assisted checks, is marked "TODO -- skipped for now" in the module docstring. Scoring currently uses PHPCS, PHPStan, and regex only.
- Files: `eval/rubric_scorer.py`, `eval/rubric_definitions.py`
- Impact: Dimensions that need semantic judgment (for example architecture and some i18n checks) are scored only by pattern. Rubric coverage is narrower than the documented 9-dimension rubric in `config/judge_system.md`.
- Fix approach: Either implement the LLM check behind `RUBRIC_USE_LLM_CHECKS` (already referenced in `scripts/reward_pipeline.py` and `scripts/phase0_score_seeds.py`) or update the docstring and rubric doc to state the deterministic-only scope.

**Stubbed and skipped test coverage:**
- Issue: 774 test functions exist under `tests/` and `scripts/test_*.py`, but several are skipped when fixtures or live-run outputs are absent. `tests/test_reward_pipeline.py` has stubbed contract tests. `tests/test_reward_pipeline_integration.py` skips when the fixture directory is missing. `tests/test_eval_integration.py` skips "requires Phase 9 live run". `tests/test_antihack.py` skips when a source JSONL is absent.
- Files: `tests/test_reward_pipeline.py`, `tests/test_reward_pipeline_integration.py`, `tests/test_eval_integration.py`, `tests/test_antihack.py`, `tests/test_calib_join.py`, `tests/test_generate_deep_judge_cot.py`
- Impact: A green run does not prove the pipeline works end to end. Skipped tests can hide regressions on machines without the data.
- Fix approach: Commit minimal fixtures for the integration tests so they run everywhere. Report skip counts in CI output.

## Known Bugs and Unresolved Blockers

**Reasoning-model eval harness incompatibility (recorded in planning state):**
- Symptoms: The eval parser `eval_judge.parse_judge_response` accepts only JSON, but the reasoning model emits prose. The model's 9-dimension taxonomy does not match the parser's D1-D9 taxonomy on two dimensions. Prose output has no `overall_score`.
- Files: `eval/eval_judge.py`, `eval/output_parsers.py`, `eval/dim_map.json`, `.planning/STATE.md`
- Trigger: Running the REVL-01 Spearman gate on reasoning-model output.
- Workaround: `eval/output_parsers.py` and `eval/dim_map.json` were added (Phase 4.4 resolution). Records in `.planning/STATE.md` say the gates were later run, but the derivation rule for prose `overall` (weighted mean via `DIMENSION_WEIGHTS`) must be kept in sync with the rubric.

**`<think>` contamination in generation evals:**
- Symptoms: With thinking enabled (the Qwen3 chat-template default), generation outputs can open `<think>` and never close it. The wp-bench harness then times out or scores the wrong text.
- Files: `scripts/_wpbench_pth/usercustomize.py`, `scripts/_wpbench_shim/npx`, `eval/eval_gen.py`
- Trigger: Calling the model through the default chat template without `enable_thinking: false`.
- Workaround: `usercustomize.py` replaces `ModelInterface.generate` to force `enable_thinking: false`. This only applies when that directory is on `PYTHONPATH`.

**Judge-capability recorded miss:**
- Symptoms: The v4 judge did not meet its primary target. Served s1 Spearman CI-lower is 0.7125 against a required 0.85 threshold. The 3-seed cheap-path ensemble CI-lower is 0.7563 against 0.87.
- Files: `.planning/STATE.md`, `output/eval4/`, `output/packaging/MODEL_CARD.md`
- Trigger: Any claim that the judge meets its target. The recorded disposition is `valid_recorded_miss`.
- Workaround: The miss is documented. The `README.md` headline rho of 0.8093 is a point estimate, and the CI spans roughly ±7-8 pp across quantization rungs.

## Security Considerations

**Plaintext API keys read from environment with no validation:**
- Risk: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `TINKER_API_KEY` are read directly from `os.environ` in several scripts. The `.env.example` file lists `ANTHROPIC_API_KEY`. `.env` and `*.env` are gitignored, which is correct, but nothing checks for a missing or malformed key before a costly run starts.
- Files: `scripts/rl_rollouts.py` (lines ~1032-1033), `scripts/preflight.py` (line ~37), `scripts/claude_agent.py`, `scripts/build_base20_probe_adapter.py` (line ~249), `scripts/build_base21_moe_probe_adapter.py`, `.env.example`
- Current mitigation: `.gitignore` excludes `.env` and `*.env`. `scripts/preflight.py` exists.
- Recommendations: Have `preflight.py` fail fast when a required key is absent. Never print or log key values in receipts. Confirm that `.env` has never been committed by running `git log --all -- .env`.

**Safety checks can be disabled by environment variable:**
- Risk: `REWARD_SKIP_PHPCS_ASSERT=1` disables the PHPCS presence assertion in the reward pipeline. It is set with `os.environ.setdefault` in at least 10 scripts, including `scripts/prune_gate_v4.py`, `scripts/eval4_ext_verdict.py`, `scripts/relabel/finalize_labels.py`, and `scripts/relabel/eval_relabel_ensemble.py`. Some of these are gate and verdict scripts. If PHPCS is missing, the reward signal silently degrades and the gate can still pass.
- Files: `scripts/reward_pipeline.py` (line ~466), `scripts/prune_gate_v4.py`, `scripts/eval4_ext_verdict.py`, `scripts/relabel/`, `scripts/_probe_rl_reward.py`
- Current mitigation: `reward_pipeline.py` logs a degraded-mode warning and requires explicit acknowledgment.
- Recommendations: Do not set this flag in gate or verdict scripts. Record the flag value in every receipt and make gate verdicts fail when it is `1`.

**Unsafe monkeypatching of the serving stack:**
- Risk: `scripts/_sieve_vllm_patch/sitecustomize.py` and `scripts/_sieve_profile_vllm_patch/sitecustomize.py` are auto-loaded via `PYTHONPATH` inside the vLLM container. They change router logits at inference. `scripts/_wpbench_pth/usercustomize.py` replaces wp-bench's `generate`. A stray `PYTHONPATH` makes these apply silently to other workloads.
- Files: `scripts/_sieve_vllm_patch/sitecustomize.py`, `scripts/_sieve_profile_vllm_patch/sitecustomize.py`, `scripts/_wpbench_pth/usercustomize.py`, `scripts/serve_v4_judge_vllm.sh`, `scripts/serve_v4_profile_vllm.sh`
- Current mitigation: `scripts/sieve_env_precheck.py` exists. Its coverage was not verified in this review.
- Recommendations: Have each patch refuse to run unless an explicit opt-in variable (for example `SIEVE_MASK_NPY`) is set. Log the patch name on load.

**Shell and remote-execution surface:**
- Risk: About 30 shell scripts (`scripts/*.sh`, `scripts/_pkg_*_run.sh`, `scripts/_pub*_upload.sh`) handle uploads and serving. Upload scripts take paths and tokens from the environment. The `scripts/_wpbench_shim/npx` shim intercepts `npx` on `PATH` when that directory is prepended.
- Files: `scripts/_pub03_upload.sh`, `scripts/_pub4_upload.sh`, `scripts/_wpbench_shim/npx`, `scripts/dgx_toolbox.py`
- Current mitigation: No `shell=True` subprocess calls were found in `scripts/` or `eval/`. Subprocess calls use list arguments.
- Recommendations: Quote all variables in the shell scripts and validate them with `set -euo pipefail`. Document that `_wpbench_shim` must not be on a global `PATH`.

**Third-party content and training data:**
- Risk: The training corpus is built from 236 cloned WordPress repositories, including deliberately poor-quality plugins (`deps/wp-finetune-data/wp_poor_plugins_raw.json`, ~40 MB). Code from these repos is executed or parsed by PHP tooling during scoring.
- Files: `config/repos.yaml`, `deps/wp-finetune-data/`, `scripts/phase1_clone.py`, `scripts/php_extract_functions.php`
- Current mitigation: Scoring is static (PHPCS, PHPStan, regex). Extraction uses the PHP tokenizer.
- Recommendations: Confirm that no training or scoring path executes extracted code. Record the license of each cloned repository in `config/repos.yaml`.

## Performance Bottlenecks

**Large JSON loaded whole into memory:**
- Problem: Several datasets are stored as single JSON arrays rather than JSONL. `data/phase4_reasoning/deep_judge_cot/deep_judge_cot_bulk.json` (~50k lines), `data/phase4_reasoning/deep_judge_cot/vllm_new_cot.json` (~37k lines), and `deps/wp-finetune-data/wp_plugins_raw.json` (~40k lines) are read in full.
- Files: `scripts/generate_deep_judge_cot.py`, `scripts/merge_reasoning_batches.py`, `scripts/build_reasoning_negatives.py`
- Cause: Whole-file `json.load` on each run.
- Improvement path: Convert to JSONL and stream. Keep a per-batch index so a single batch can be reprocessed.

**Model merging and quantization are memory-bound:**
- Problem: Merging LoRA adapters into a 35B MoE base and converting to GGUF needs most of the machine's memory. The planning record shows an Unsloth GPU merge exhausted GB10 unified memory (about 110 GiB pinned against a 121 GiB total).
- Files: `scripts/merge_adapter.py`, `scripts/_p0_unsloth_merge_v3.py` (referenced in `.planning/STATE.md`), `scripts/eval4_ext_gguf_convert.sh`, `scripts/_pkg_gguf_eval_run.sh`
- Cause: Full fp32 accumulation of MoE expert deltas. `max_memory` in accelerate is a hint, not a cap.
- Improvement path: Keep the CPU-only merge path (`merge_adapter.py`, fp32 accumulation tests in `tests/test_merge_adapter_fp32_accumulation.py`) as the default. Stream per-layer.

**Serial LLM-driven pipeline steps:**
- Problem: Data generation and judging run through Claude Code agents, one batch at a time (`scripts/claude_agent.py`, `scripts/rl_judge_dispatch.py`). `data/phase4_reasoning/deep_judge_cot/batches/` holds 5+ batch files of ~2.7k lines each.
- Files: `scripts/claude_agent.py`, `scripts/generate_deep_judge_cot.py`, `scripts/generate_critique_then_fix.py`
- Cause: Sequential dispatch with no shared rate limiting.
- Improvement path: Add bounded concurrency and checkpoint after each batch. `data/checkpoints/*.json` is already gitignored for this purpose.

## Fragile Areas

**Sieve expert mask and protected-expert logic:**
- Files: `scripts/sieve_expert_mask_inference.py`, `scripts/extract_protected_mask.py`, `scripts/prune_apply_physical.py`, `scripts/prune_apply_physical_v4.py`, `scripts/aimer_prune.py`, `scripts/aimer_prune_v4.py`, `output/sieve-v4/protected_expert_mask.json`
- Why fragile: Expert indices must match across the mask file, the router, and the physical prune. A mismatch silently changes which experts are kept. The v4 and earlier variants differ.
- Safe modification: Re-run `tests/test_protected_mask.py`, `tests/test_prune_physical.py`, and `tests/test_sieve_arch.py` after any change. Compare the mask hash to the value in the receipt.
- Test coverage: Unit tests exist. No end-to-end check that a pruned model's per-expert routing matches the mask.

**Unsloth versus PEFT MoE LoRA layout:**
- Files: `scripts/merge_adapter.py`, `scripts/tinker_export_checkpoint.py`, `scripts/build_exp23_fp32_merge_rho.py`, `tests/test_merge_adapter_moe_routing.py`
- Why fragile: Unsloth uses contiguous expert blocks (`B[:, e*R:(e+1)*R]`) and PEFT uses strided indexing (`B[:, e::E]`). The planning record says a naive `merge_and_unload()` would silently corrupt per-expert deltas.
- Safe modification: Never merge Unsloth-trained adapters with raw PEFT. Require the tensor-level and forward-pass equivalence anchors (`scripts/extract_pass_anchors.py`) to pass before promotion.
- Test coverage: `tests/test_merge_adapter_moe_routing.py` covers routing. Anchor checks are scripted but not enforced in CI.

**Reward pipeline with multiple modes:**
- Files: `scripts/reward_pipeline.py`, `scripts/reward_v2.py`, `scripts/reward_validity_gate.py`, `scripts/rl_rollouts.py`, `scripts/rl_codegen_tripwire.py`
- Why fragile: Behavior depends on `RUBRIC_USE_LLM_CHECKS`, `REWARD_SKIP_PHPCS_ASSERT`, and `SIEVE_MASK_NPY`. The reward used by GRPO must ground in the rubric scorer, not in assistant targets (per `.planning/STATE.md`).
- Safe modification: Run `tests/test_reward_pipeline.py` and `tests/test_reward_validity_gate.py` with the environment variables unset. Confirm the degraded-mode path is not taken.
- Test coverage: Contract tests are stubbed. Integration tests skip without fixtures.

**Hardcoded local tool locations:**
- Files: `scripts/dgx_toolbox.py` (reads `DGX_TOOLBOX_PATH`), `scripts/dgx_toolbox_watch.py`, `deps/dgx-toolbox` (git submodule, not initialised in this checkout), `.gitmodules`
- Why fragile: The pipeline is written for one DGX Spark host. `dgx_toolbox.py` depends on an external checkout at `DGX_TOOLBOX_PATH`. The submodule directory is empty here.
- Safe modification: Run `git submodule update --init` before any run that calls `dgx_toolbox`. Set `DGX_TOOLBOX_PATH` explicitly in the environment rather than relying on a default.
- Test coverage: `tests/test_gb10_load_safety.py` covers memory limits, not path resolution.

**Dependency manifest is missing:**
- Files: repository root (no `pyproject.toml`, `requirements*.txt` at root, `setup.cfg`, `tox.ini`, or `pytest.ini`), `config/requirements-profiling.txt`
- Why fragile: Python dependencies are documented in `README.md` ("Python 3.10+, `pyyaml`, `python-dotenv`") and in `config/requirements-profiling.txt`, but the tree has no single lock or manifest. Several scripts require `.venv-tinker/` and `.venv` paths (37 references in `scripts/*.py`).
- Safe modification: Add a root `pyproject.toml` or `requirements.txt` with pinned versions for the Tinker, vLLM, and Unsloth environments separately.
- Test coverage: Not applicable.

**No CI configuration:**
- Files: no `.github/workflows/` directory exists in the repository.
- Why fragile: Test and lint status depends on each developer running `pytest` by hand. The 774 tests have no enforced gate.
- Safe modification: Add a workflow that runs `pytest tests/` with fixtures that do not need the GPU host.
- Test coverage: Unknown without running the suite.

## Scaling Limits

**Single-host execution:**
- Current capacity: All training, serving, and eval run on one DGX Spark (GB10) with about 121 GiB unified memory (`README.md`, `.planning/STATE.md`).
- Limit: Any model or context larger than about 60-70 GiB of working set cannot be loaded alongside a second job. Concurrent eval and serving runs conflict.
- Scaling path: Use the queueing and `gb10_load_safety` logic in `tests/test_gb10_load_safety.py` as the gate. Move serving to a separate host if a second model is needed in parallel.

**Context length in judge output:**
- Current capacity: The server is started with `-c 16384`, and the README warns that shorter contexts truncate long critiques.
- Limit: `max_tokens: 8192` in the README example uses most of the window for long reasoning outputs.
- Scaling path: Measure the truncation rate per dimension and raise `-c` only if parse failures appear.

## Dependencies at Risk

**Pre-release and out-of-tree inference stacks:**
- Risk: The pipeline depends on llama.cpp `b9180+` (`README.md`), vLLM with custom patches, Unsloth (with fused-MoE LoRA conventions), and the Tinker SDK (`scripts/tinker_*.py`). These move fast and their internal layouts (for example vLLM router logits, Unsloth LoRA packing) are patched directly.
- Impact: A version bump in any of these can silently change tensor layout or routing and break the sieve, merge, or eval results without a test failure.
- Migration plan: Pin exact versions in a lock file. Re-run the anchor checks (`scripts/extract_pass_anchors.py`) after every upgrade.

**Git submodule `deps/dgx-toolbox`:**
- Risk: The submodule points at an external GitHub repository that is not vendored. The checkout here is empty.
- Impact: Scripts that import or shell out to the toolbox fail on a fresh clone.
- Migration plan: Vendor the required subset or document the required commit hash in `.gitmodules`.

## Missing Critical Features

**No deterministic LLM-check path in the scorer:**
- Problem: The fourth scoring tool (LLM-assisted checks) is absent from `eval/rubric_scorer.py`.
- Blocks: Scoring of rubric dimensions that need semantic judgment, and therefore a full 9-dimension ground truth for the judge.

**Generator not shipped:**
- Problem: `README.md` and `JOURNAL.md` state that the project ships a judge only. Generation relies on a recommended current base model.
- Blocks: Any request to publish a WordPress code generator from this repository will need new training data, because the existing generator targets are judged structurally weaker than the base (`JOURNAL.md`).

## Test Coverage Gaps

**End-to-end data pipeline:**
- What's not tested: Full run from `scripts/phase1_clone.py` through `scripts/phase1_extract.py`, `scripts/phase1_judge.py`, and `scripts/assemble_reasoning_dataset.py` to a JSONL that a training run accepts. Unit tests cover individual steps.
- Files: `scripts/phase1_clone.py`, `scripts/phase1_extract.py`, `scripts/phase1_judge.py`, `scripts/assemble_reasoning_dataset.py`, `tests/test_pipeline_integration.py`
- Risk: Schema drift between stages goes unnoticed. `scripts/export_dataset.py` and `scripts/validate_reasoning_consistency.py` are the only cross-stage checks.
- Priority: High

**Reward gating under degraded mode:**
- What's not tested: Gate verdicts with `REWARD_SKIP_PHPCS_ASSERT=1` set versus unset.
- Files: `scripts/reward_pipeline.py`, `scripts/prune_gate_v4.py`, `scripts/eval4_ext_verdict.py`, `tests/test_reward_pipeline.py`
- Risk: A gate can pass on degraded scoring with no failure reported.
- Priority: High

**Prose-output eval parsing:**
- What's not tested: Overall-score derivation from prose output (`DIMENSION_WEIGHTS` weighted mean) against the JSON path, on real model output.
- Files: `eval/output_parsers.py`, `eval/eval_judge.py`, `eval/dim_map.json`, `tests/test_eval_judge.py`
- Risk: The 2-of-9 dimension mismatch (Code Quality and Dependency Integrity versus D8 and D9) could shift Spearman without detection.
- Priority: Medium

**Live-run integration tests:**
- What's not tested: `tests/test_eval_integration.py` runs its live subtests only when Phase 9 output exists. `tests/test_antihack.py` and `tests/test_calib_join.py` need source files that are not always present.
- Files: `tests/test_eval_integration.py`, `tests/test_antihack.py`, `tests/test_calib_join.py`
- Risk: The live path is effectively untested in any environment other than the original run host.
- Priority: Medium

**Shell scripts:**
- What's not tested: The 30+ `.sh` scripts in `scripts/` have no tests (`tests/test_launch_validated_smoke.py` covers one).
- Files: `scripts/serve_*.sh`, `scripts/_pkg_*_run.sh`, `scripts/_pub*_upload.sh`, `scripts/eval4_ext_*.sh`
- Risk: A malformed path or missing variable only fails at run time on the host.
- Priority: Low

---

*Concerns audit: 2026-10-09*

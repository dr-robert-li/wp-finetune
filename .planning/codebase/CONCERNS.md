# Codebase Concerns

**Analysis Date:** 2026-03-31

## Tech Debt

**yaml.dump reorders config keys (cosmetic but causes diff noise):**
- Issue: The `/run-training` skill uses `yaml.dump()` without `sort_keys=False` when writing config files between training runs. This alphabetically reorders all YAML keys, making git diffs noisy and configs harder to read. The skill file at `.claude/skills/wp-finetune:run-training/SKILL.md` lines 279 and 1029 both use bare `yaml.dump(base_config, open(...))`. By contrast, `scripts/csv_to_repos.py` line 270 correctly uses `yaml.safe_dump(..., sort_keys=False)`.
- Files: `.claude/skills/wp-finetune:run-training/SKILL.md` (lines 279, 1029)
- Impact: Every adaptive planning cycle reorders the entire config YAML. Comparing configs across runs requires ignoring key order. Makes manual review of config changes between runs unnecessarily difficult.
- Fix approach: Replace `yaml.dump(base_config, open(run_config_path, "w"))` with `yaml.safe_dump(base_config, open(run_config_path, "w"), default_flow_style=False, sort_keys=False)` in both locations in the skill file. All config files would then preserve insertion order.

**Hardcoded 40/60 ratio in export_dataset.py:**
- Issue: `scripts/export_dataset.py` lines 27-28 hardcode `GEN_TARGET_RATIO = 0.40` and `JUDGE_TARGET_RATIO = 0.60`, but the project now exports 5 different ratio variants (30/70 through 70/30). The per-ratio config YAMLs (`config/train_config_40_60.yaml`, etc.) point to pre-split ratio directories, so the hardcoded ratio in the export script is only used for the default export path, not the ratio-specific ones. However, this creates confusion about whether the ratio enforcement in `enforce_ratio()` is active for ratio-specific exports.
- Files: `scripts/export_dataset.py` (lines 27-28, 121-138)
- Impact: Low immediate risk since ratio-specific datasets are pre-exported. But if someone runs the default export pipeline, it silently enforces 40/60 regardless of intent.
- Fix approach: Accept ratio as CLI argument or read from the active `train_config.yaml`. Remove hardcoded constants.

**Telemetry monitor scripts have hardcoded absolute paths:**
- Issue: `telemetry/training/monitor.sh`, `telemetry/training/_monitor.sh`, and `telemetry/training/monitor_30_70.sh` all contain hardcoded absolute paths to `/home/robert_li/Desktop/projects/wp-finetune/`. Container names are also hardcoded to `unsloth-headless`.
- Files: `telemetry/training/monitor.sh`, `telemetry/training/_monitor.sh`, `telemetry/training/monitor_30_70.sh`
- Impact: These scripts are not portable and would break on any other machine or if the project directory moves. The DGX toolbox config at `config/dgx_toolbox.yaml` already has these values centralized but the monitor scripts do not read from it.
- Fix approach: Source paths from environment variables or from `config/dgx_toolbox.yaml`. Use `PROJECT_ROOT` pattern matching the Python scripts.

**sample_weight metadata not consumed by trainer:**
- Issue: `scripts/export_dataset.py` line 141-153 adds `sample_weight` metadata to examples (1.5x for contrastive/low-score, 1.0x otherwise), but `scripts/train_model.py` does not read or use this field. SFTTrainer does not natively support per-example loss weighting from metadata. The weights are exported into the JSONL but have no effect on training.
- Files: `scripts/export_dataset.py` (lines 141-153), `scripts/train_model.py` (lines 309-359)
- Impact: The intended curriculum weighting for harder examples is not active. Contrastive and low-score examples are treated identically to high-quality generation examples during training loss computation.
- Fix approach: Either implement a custom data collator that reads `sample_weight` and scales loss per example, or remove the dead metadata to avoid confusion.

## Known Bugs

**Unsloth overrides batch_size from checkpoint trainer_state.json on resume:**
- Symptoms: When resuming from a checkpoint, Unsloth/HuggingFace Trainer reads `trainer_state.json` from the checkpoint directory, which contains the batch size from the original run. If the config has been changed between runs (e.g., adaptive planner reduced batch from 8 to 4), the trainer silently uses the checkpoint's batch size (8) instead of the config's batch size (4).
- Files: `scripts/train_model.py` (line 453 `trainer.train(resume_from_checkpoint=...)`), `.claude/skills/wp-finetune:run-training/SKILL.md`
- Trigger: Resume training after adaptive planner has changed `per_device_train_batch_size` in config.
- Workaround: Manually delete or edit `trainer_state.json` in the checkpoint directory before resuming. Or do not resume from checkpoints after batch size changes -- start fresh.

## Security Considerations

**No secrets in source code:**
- Risk: Low. The `.env` file exists but is gitignored. No API keys or credentials are embedded in Python source files. The project uses local MLflow (sqlite) with no cloud tracking.
- Files: `.env`, `.env.example`, `.gitignore`
- Current mitigation: `.gitignore` excludes `.env*`. MLflow uses local file store (`mlruns.db`). No cloud API calls from training code.
- Recommendations: None needed for current state. If cloud MLflow tracking is added later, ensure tokens are in `.env` only.

**Docker exec without resource limits:**
- Risk: The `scripts/dgx_toolbox.py` `execute()` method (line 464) runs `docker exec` with no memory or CPU limits. On unified memory (DGX Spark), a runaway process inside the container can consume all system RAM.
- Files: `scripts/dgx_toolbox.py` (lines 462-487)
- Current mitigation: Memory watchdog callback in `scripts/train_model.py` monitors `/proc/meminfo` during training. Pre-flight memory check runs before model loading.
- Recommendations: Consider adding `--memory` flag to Docker run commands (not exec). Note: NVIDIA forums report Docker memory limits do not fully contain the unified memory deadlock issue.

## Performance Bottlenecks

**DGX Spark unified memory zombie-OOM (driver-level deadlock):**
- Problem: On DGX Spark GB10, when system RAM approaches the ceiling (~120 GB of 128 GB), the NVIDIA driver's internal descriptor allocations fail below the CUDA runtime layer. Instead of a clean `RuntimeError: CUDA out of memory`, the driver enters an unrecoverable deadlock state. `nvidia-modeset` enters D-state, SSH dies, no logs flush. Hard reboot is the only recovery.
- Files: `scripts/train_model.py` (lines 254-302, MemoryWatchdogCallback), `config/train_config.yaml`
- Cause: Unified memory architecture (CPU + GPU + page cache in ~128 GB) has no discrete VRAM boundary. The CUDA runtime never gets a chance to throw a Python-level exception. PyTorch caching allocator fragmentation, dataloader worker accumulation, and checkpoint serialization spikes all contribute to gradual memory creep. See [pytorch/pytorch#174358](https://github.com/pytorch/pytorch/issues/174358).
- Improvement path: The memory watchdog checks `/proc/meminfo` every training step but **cannot prevent sudden allocation failures** -- only gradual creep. A single large tensor allocation (e.g., first forward pass at increased batch size) can jump from 10 GB headroom to deadlock instantaneously, before the watchdog's next check. Mitigations in place: 2 GB watchdog threshold, persistent workers to eliminate respawn spikes, peak-based headroom calculations, model-scale-aware batch ceilings. Still to investigate: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, Docker `--memory` limits, page cache dropping before training, swap disable.

**Long training times for larger datasets:**
- Problem: Training time scales linearly with dataset size. The 50/50 ratio (48,796 examples) at batch=4 with 2 epochs is estimated at ~53 hours. Larger ratios (60/40 at 60,996 examples, 70/30 at 81,328 examples) would take proportionally longer.
- Files: `config/train_config_50_50.yaml`, `config/train_config_60_40.yaml`, `config/train_config_70_30.yaml`
- Cause: Batch size is capped at 4 for 30B+ models on unified memory (XL ceiling in adaptive planner). Cannot safely increase batch size. GPU utilization is ~67% but the remaining idle time is the cost of the memory safety margin.
- Improvement path: The thermal exploitation ladder (prefetch_factor, save_steps, eval_steps) squeezes out pipeline efficiency without touching batch size. Further speedup requires either a smaller model (allows larger batches), more memory (different hardware), or fewer training epochs. The 5 GB optimization window (97 GB floor to 102 GB ceiling at 85% safety rule) leaves almost no room.

**Thermal monitoring has 10-minute sample interval:**
- Problem: The telemetry monitor scripts sample GPU metrics every 600 seconds (10 minutes). This interval misses sub-minute thermal spikes, OOM events, and training crashes that occur between samples.
- Files: `telemetry/training/monitor.sh` (line 57, `sleep 600`), `telemetry/training/_monitor.sh` (line 89, `sleep 600`)
- Cause: Conservative interval chosen to minimize monitoring overhead during multi-day training runs (84 checks * 10 min = 14 hours of coverage).
- Improvement path: Use a two-tier approach: lightweight metrics (GPU temp, RAM) at 60-second intervals via a background process writing to a ring buffer, with the full monitor script reading and summarizing at 10-minute intervals. The lightweight tier adds negligible overhead but catches transient events. Alternatively, use `nvidia-smi dmon` for continuous GPU monitoring.

## Fragile Areas

**Adaptive planner policy in skill file vs empirical evidence:**
- Files: `.claude/skills/wp-finetune:run-training/SKILL.md` (Step 8.5), `config/train_config.yaml`
- Why fragile: The adaptive resource planner lives in the skill file (a markdown document interpreted by Claude Code agents), not in executable Python code. The planner's model-scale batch ceilings (XL <= 4, Large <= 8, etc.) are derived from NVIDIA Developers Forum UGC, not from systematic benchmarking on this specific hardware. The 40/60 run completed successfully at batch=8 (before OOM from worker scaling), which contradicts the XL ceiling of 4. The planner would currently prevent batch=8 even though it worked for ~2200 steps.
- Safe modification: When adjusting the adaptive planner, always update both the skill file AND validate against the thermal history in `telemetry/training/thermal_history.json`. Do not change batch ceilings without running a warmup probe first.
- Test coverage: None. The adaptive planner is markdown pseudocode in a skill file, not testable Python. The entire training feedback loop (telemetry -> classify zone -> adjust config -> write YAML) is agent-interpreted, not unit-tested.

**Config file proliferation across training runs:**
- Files: `config/train_config.yaml`, `config/train_config_30_70.yaml`, `config/train_config_40_60.yaml`, `config/train_config_50_50.yaml`, `config/train_config_60_40.yaml`, `config/train_config_70_30.yaml`
- Why fragile: The base `train_config.yaml` is modified in-place by the adaptive planner between runs. Per-ratio configs are generated by the skill as copies of the base with data path and output_dir overrides. If the base config drifts (e.g., planner changes workers from 3 to 4), already-generated per-ratio configs still have the old value. There is no mechanism to regenerate per-ratio configs after adaptive changes.
- Safe modification: Always regenerate per-ratio configs from the base after any adaptive planning step. Compare configs with `diff` before starting a new run.
- Test coverage: `tests/test_config.py` and `tests/test_train_model.py` validate config schema but do not check consistency across ratio configs.

**Memory watchdog fail-open design:**
- Files: `scripts/train_model.py` (lines 277-287, `_available_mb()`)
- Why fragile: The watchdog returns `999_999` (never trigger) if it cannot read `/proc/meminfo`. This fail-open design means any filesystem issue, container isolation change, or /proc mount problem silently disables the only runtime OOM protection. The watchdog also only checks at step boundaries -- a single training step with a very long sequence could exhaust memory before the next check.
- Safe modification: Consider adding a fail-closed mode for production runs. Log a warning if `/proc/meminfo` is unreadable.
- Test coverage: No test for the watchdog callback. The `_available_mb()` method and the `on_step_end` trigger logic are untested.

## Scaling Limits

**DGX Spark 128 GB unified memory ceiling:**
- Current capacity: ~97 GB floor (Qwen3-30B-A3B BF16 + LoRA optimizer states + activations) with ~22 GB headroom at batch=4.
- Limit: Driver deadlock at ~120 GB. Effective ceiling is 85% of total (~109 GB). This gives a ~12 GB optimization window between the training floor and the safety ceiling.
- Scaling path: Cannot train larger models on this hardware without quantization (4-bit/8-bit), which is explicitly disabled for MoE models (`load_in_4bit=False` in `scripts/train_model.py` line 174, marked "LOCKED -- no QLoRA for MoE"). To scale up: use hardware with more memory or discrete VRAM, or switch to a smaller base model.

**Dataset size vs training time:**
- Current capacity: 5 ratio variants from ~35K to ~81K training examples. 2 epochs each.
- Limit: At batch=4 on DGX Spark, each epoch of the 70/30 dataset (81K examples) takes ~30-40 hours. 2 epochs = ~60-80 hours per run.
- Scaling path: Reduce to 1 epoch for larger datasets (diminishing returns on second epoch for LoRA). Or evaluate fewer ratio variants based on early results from 30/70 and 40/60.

## Dependencies at Risk

**Unsloth version coupling:**
- Risk: Unsloth is not pinned in `config/dgx_toolbox.yaml` `pinned_versions`. It is installed inside the Docker container via the dgx-toolbox unsloth-headless image. If the image is rebuilt or Unsloth updates, training behavior could change (different memory usage, different optimization paths, different checkpoint format).
- Impact: Unsloth controls the core training loop via `FastLanguageModel` wrapper. Version changes could affect: model loading (BF16 vs quantized paths), gradient checkpointing behavior (`use_gradient_checkpointing="unsloth"` in `scripts/train_model.py` line 219), and LoRA application.
- Migration plan: Pin Unsloth version in `config/dgx_toolbox.yaml` `pinned_versions` section. Record the currently-working version in the config.

**transformers/trl version sensitivity:**
- Risk: Pinned at `transformers==4.56.2` and `trl==0.24.0` in `config/dgx_toolbox.yaml`. The SFTTrainer API and SFTConfig parameters change across trl versions. The `trainer_state.json` batch size override bug may be version-specific.
- Impact: Upgrading either package could break the training pipeline silently (different default behaviors, renamed parameters).
- Migration plan: Keep versions pinned. Test any upgrade in isolation before a multi-day training run.

## Missing Critical Features

**No automated evaluation pipeline after training:**
- Problem: Eval scripts exist (`eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py`) but there is no automated flow from training completion to evaluation. The `/run-training` skill saves the adapter and stops. Evaluation requires manual intervention: merge the adapter, start a vLLM server, run eval scripts against the served model.
- Blocks: Cannot automatically compare ratio variants after training completes. Each comparison requires manual setup of the inference stack.

**No warmup probe implementation:**
- Problem: The adaptive planner's Rung 4 (batch size increase) requires a warmup probe -- run 1 real training step at the new batch size to verify it survives. This is documented in the JOURNAL but not implemented in code. The skill file describes writing a `_warmup_probe_required` flag but there is no code that reads this flag and executes the probe.
- Files: `.claude/skills/wp-finetune:run-training/SKILL.md` (Step 6 mentions warmup probe), `scripts/train_model.py` (no probe logic)
- Blocks: Batch size increases via the adaptive planner are either blocked (XL ceiling) or unguarded (if ceiling is raised).

## Test Coverage Gaps

**No tests for train_model.py runtime behavior:**
- What's not tested: Model loading, LoRA application, dataset loading, trainer construction, memory watchdog callback, resume-from-checkpoint logic. All tests in `tests/test_train_model.py` are static config validation (schema checks, string matching).
- Files: `tests/test_train_model.py` (129 lines, all config/static tests), `scripts/train_model.py` (499 lines)
- Risk: A regression in the training pipeline (e.g., wrong SFTConfig parameter, broken resume path, watchdog threshold too low) would not be caught until a multi-hour training run fails.
- Priority: Medium. The training script is relatively stable and changes infrequently, but each failure costs 6-12 hours of GPU time.

**No tests for dgx_toolbox.py:**
- What's not tested: Validation engine, container lifecycle management, execution engine, status reporting. All methods interact with Docker and the filesystem.
- Files: `scripts/dgx_toolbox.py` (639 lines), no corresponding test file
- Risk: Changes to container names, config structure, or Docker behavior could silently break the toolbox. The toolbox is the foundation layer for all training and evaluation operations.
- Priority: Medium. Could be tested with mocked subprocess calls.

**No tests for eval scoring correctness:**
- What's not tested: The 193-check rubric scoring in `eval/rubric_definitions.py` (1011 lines) and `eval/rubric_scorer.py` (657 lines) contain complex dimension weighting, floor rules (e.g., direct XSS vector caps D2 at 3/10), and multi-tool aggregation logic. Test files exist but coverage of edge cases in scoring is unknown.
- Files: `eval/rubric_definitions.py`, `eval/rubric_scorer.py`, `eval/eval_gen.py`, `eval/eval_judge.py`
- Risk: Incorrect evaluation scores could lead to selecting the wrong ratio variant. The scoring logic determines which model ships.
- Priority: High. Eval correctness determines which model ships.

**Adaptive planner is untestable:**
- What's not tested: The entire thermal zone classification, batch ceiling enforcement, memory backoff, and thermal ladder logic lives in `.claude/skills/wp-finetune:run-training/SKILL.md` as markdown pseudocode interpreted by agents. There is no Python implementation to unit test.
- Files: `.claude/skills/wp-finetune:run-training/SKILL.md` (Step 8.5, ~200 lines of logic)
- Risk: Agent misinterpretation of the adaptive planner pseudocode could cause OOM (batch too high), wasted compute (batch too low), or config corruption (YAML write errors).
- Priority: High. This logic directly controls resource allocation on expensive hardware with catastrophic failure modes.

---

*Concerns audit: 2026-03-31*

---
phase: 6
reviewers: [gemini, codex]
reviewed_at: 2026-03-31T23:30:00Z
plans_reviewed: [06-01-PLAN.md, 06-02-PLAN.md, 06-03-PLAN.md, 06-04-PLAN.md]
---

# Cross-AI Plan Review -- Phase 6

## Gemini Review

# Plan Review: Phase 06 - Adaptive Training Planner

This review covers the four implementation plans (06-01 through 06-04) for the **Adaptive Training Planner** (v1.1) milestone. The plans collectively implement a power-primary routing engine, thermal exploitation ladder, and batch/gradient-accumulation coupling.

## 1. Summary
The implementation strategy is exceptionally robust and follows a clear "Foundation - Implementation - Integration - Verification" lifecycle. It successfully shifts the project from simple temperature-based scaling to a sophisticated, power-aware adaptive engine (v4.0). The decision to centralize thresholds in `adaptive_planning.yaml` and delegate the decision logic to a dedicated Claude Code skill is architecturally sound, keeping the core training script focused on execution and telemetry production.

## 2. Strengths
*   **Power-Primary Logic:** Correctly identifies GPU wattage as the primary signal for resource headroom, treating temperature as a safety brake, which aligns with modern DGX GB10 thermal behavior.
*   **Thermal Exploitation Ladder (v4.0):** The reordered rung logic (Batch size first) ensures the highest-impact changes are attempted when headroom is detected.
*   **Effective Batch Maintenance:** The `new_accum = max(1, effective_batch // new_batch)` formula preserves training stability across batch size changes.
*   **Unsloth Silent Override Detection:** Capturing actual values from the startup banner via monkey-patching `builtins.print` is a clever solution to a common "silent failure" point in Unsloth training.
*   **Infrastructure Synergy:** Excellent use of the `dgx-toolbox` Phase 13 telemetry package (`GPUSampler`, `AnchorStore`, `FailureClassifier`), demonstrating strong cross-project integration.
*   **Safety Guards:** The inclusion of page cache dropping before memory checks and the multi-step warmup probe significantly reduces the risk of OOM-induced driver deadlocks on UMA systems.

## 3. Concerns
*   **Complexity of Skill Logic (MEDIUM):** The `adaptive-planner` skill (Plan 01) contains significant algorithmic logic (JSONL parsing, aggregates, ladder decisions). While Claude Code is capable of this, ensuring the agent correctly handles the "jitter_margin" and "p95_ram_gb" computations across long JSONL files is critical.
*   **Banner Parsing Fragility (LOW):** Reliance on regex-parsing the Unsloth banner is subject to upstream formatting changes. However, the plan includes a fallback and warning if parsing fails.
*   **Sudo Requirements (LOW):** The `drop_caches` command requires `sudo`. Ensure the environment is configured for passwordless sudo for these specific commands, or the training script may hang waiting for input.
*   **Memory Jitter Margin (LOW):** Plan 01 sets a 5GB safety margin. For Qwen3-30B-A3B on 128GB Spark, this is prudent, but we should monitor if this prevents scaling to Batch 8 in marginal cases.

## 4. Suggestions
*   **Telemetry Jitter:** In Plan 01, Step 3, suggest adding a check to skip the first 2-3 readings of a run to avoid "warm-up" noise (e.g., initial model loading spikes) from skewing the `avg_watts` and `avg_gpu_util`.
*   **Environment Variable Consistency:** Ensure `ADAPTIVE_THERMAL_LOG` is documented in `.env.example` as it is now a critical link between the container and the host-side planner.
*   **Failure Classifier Refinement:** In Plan 01, Step 5, consider adding a `MAX_RETRY_OOM` check in the `AnchorStore` logic to prevent the planner from repeatedly hitting a "soft OOM" cliff.

## 5. Risk Assessment
**Overall Risk Level: LOW**

The plan is well-defended. The inclusion of Plan 04 as a dedicated verification phase with automated threshold consistency checks (`82C` vs `85C` across all files) mitigates the most likely source of implementation error (threshold drift). The dependency on `dgx-toolbox` Phase 13 is clearly acknowledged and handled via imports and mounts.

**Recommendation:** Proceed with execution starting with **06-01-PLAN.md**.

---

## Codex Review

## 06-01-PLAN.md

### Summary
This is a strong foundation plan: it isolates the new planner skill and its config into a clean first wave, captures the core routing logic, and ties most requirements directly to artifacts and grepable acceptance criteria. The main weakness is that it pushes a large amount of execution logic into a markdown skill file, which makes correctness harder to validate than if the decision engine lived in tested Python code. As written, it likely achieves the documentation/instruction layer for the phase, but it leaves meaningful risk around determinism, parser robustness, and config/schema drift.

### Strengths
- Separates new files from existing-file edits, which reduces merge and sequencing risk.
- Maps requirements to explicit truths, artifacts, and key links.
- Correctly centralizes thresholds into `config/adaptive_planning.yaml`.
- Captures the intended routing model clearly: power-primary, thermal override, ladder ordering, batch coupling.
- Includes explicit handling for Unsloth override detection and anchor-store persistence.
- Acceptance criteria are concrete and mostly machine-checkable.

### Concerns
- **HIGH**: The plan puts substantial decision logic inside `.claude/.../SKILL.md` instead of a Python module. That makes behavior hard to unit test and easy to drift from the intended algorithm.
- **HIGH**: The coupling formula `max(1, effective_batch // new_batch)` does not actually preserve effective batch for non-divisible changes. The example later in Plan 04 confirms drift from 16 to 15, which contradicts BTCH-01.
- **MEDIUM**: The routing rules are partly inconsistent. The truth list names five zones, but the detailed dispatch logic mostly derives three watt bands plus thermal override and then conditionally refines TARGET into MODERATE.
- **MEDIUM**: The plan assumes the skill can reliably import and use `telemetry.*` modules, but there is no validation here that Phase 13 APIs match the expected signatures.
- **MEDIUM**: The telemetry parsing requirements assume stable JSONL fields and units, but there is no schema contract or fallback behavior if keys are missing or mixed by source.
- **LOW**: Verification overuses `grep` presence checks, which can pass even if logic is malformed or contradictory.

### Suggestions
- Move the actual routing/coupling logic into a Python helper module and keep the skill as orchestration instructions.
- Redefine BTCH-01 so it is mathematically consistent: either preserve effective batch exactly by constraining allowed batch sizes to divisors, or explicitly allow bounded drift with a stated tolerance.
- Add a formal schema section for canonical telemetry JSONL, including required/optional fields and units.
- Add a compatibility check against actual `dgx-toolbox` interfaces before depending on imports/signatures.
- Add test vectors for routing decisions, including no-power-data, thermal override, OOM, HANG, and anchor-hit cases.

### Risk Assessment
**MEDIUM-HIGH**. The phase boundary and sequencing are good, but the core algorithm is being encoded in a skill file with weak behavioral verification, and the batch-coupling rule currently conflicts with the stated requirement.

---

## 06-02-PLAN.md

### Summary
This plan is the highest-risk part of the set because it modifies the training script directly and introduces telemetry capture, print interception, page-cache dropping, and warmup probe behavior in one pass. It does address the needed plumbing for power-based planning, but several implementation details are brittle or unsafe for a long-running training process.

### Strengths
- Focuses on the minimum runtime changes needed for the planner to work.
- Correctly treats `train_model.py` as the source of truth for runtime telemetry.
- Preserves critical existing behavior in acceptance criteria.
- Calls out lazy imports for container-only dependencies.
- Adds specific verification for syntax and key feature presence.

### Concerns
- **HIGH**: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` inside `check_memory()` is operationally dangerous and unlikely to be reliable in containers.
- **HIGH**: Monkey-patching `builtins.print` around `trainer.train()` is brittle. It can capture unrelated output, break logging assumptions, miss non-`print` banner output.
- **HIGH**: The plan adds warmup probe execution inside the training script without clearly specifying how trainer/model resources are torn down between probe and full run.
- **MEDIUM**: The callback writes JSONL every 50 steps, but there is no durability strategy for partial writes or file contention.
- **MEDIUM**: The callback schema mentions `temp` while earlier docs use `temp_c`/`gpu_temps`; field-name consistency looks weak.
- **MEDIUM**: Falling back to a 1-step MemAvailable check when `telemetry.probe` import fails undercuts PROB-01.
- **LOW**: Running full `pytest tests/ -x -q` as acceptance may be expensive/noisy.

### Suggestions
- Remove the `sudo drop_caches` step; make it a pre-run operator action or optional wrapper.
- Replace `print` monkey-patching with parsing trainer startup logs or reading final trainer args after initialization.
- Define lifecycle rules for the warmup probe: model reuse vs rebuild, GPU cleanup.
- Normalize the canonical telemetry schema now and use the same field names in all plans.
- Add behavioral tests for the callback and banner parser.

### Risk Assessment
**HIGH**. Relies on fragile mechanisms that could fail during long GPU runs.

---

## 06-03-PLAN.md

### Summary
Directionally correct integration plan. Wires the new planner into the run flow, aligns thresholds, and adds container mount/PYTHONPATH.

### Strengths
- Clearly isolates integration work from core implementation.
- Replaces inline adaptive logic with delegated planner invocation.
- Enforces threshold consistency across files.
- Captures the container mount/PYTHONPATH dependency explicitly.

### Concerns
- **HIGH**: `config/dgx_toolbox.yaml` changes assume `container_pythonpath` is a real supported key.
- **MEDIUM**: Adding `telemetry.*` to `required_imports` can make validation fail before graceful degradation.
- **MEDIUM**: Sentinel files (`_thermal_cooldown_required`, `_warmup_probe_required`, `_probe_failed`) need clearer lifecycle/cleanup rules.
- **MEDIUM**: `$TELEMETRY` variable inconsistency (env var vs template var).
- **LOW**: Threshold updates via grep-like replacement risks altering explanatory text.

### Suggestions
- Confirm `container_pythonpath` is real; if not, define the launcher code.
- Split required imports into mandatory vs optional.
- Document sentinel-file ownership and cleanup rules.
- Add one end-to-end dry-run verification command.

### Risk Assessment
**MEDIUM**. Sound design but depends on configuration hooks that may not be fully real.

---

## 06-04-PLAN.md

### Summary
Sensible checkpoint plan with cross-file consistency validation and human review.

### Strengths
- Adds proper integration gate before expensive training.
- Checks cross-file threshold consistency.
- Includes both machine validation and human review.

### Concerns
- **HIGH**: The review example `batch=4 -> 5` producing effective batch `15` directly conflicts with BTCH-01.
- **MEDIUM**: Many checks are string-presence rather than behavioral validations.
- **MEDIUM**: Human review asks user to confirm watt ranges and regex behavior that should be automated first.

### Suggestions
- Resolve effective-batch invariance conflict first.
- Add a small decision-table test suite for the planner.
- Replace user review of mechanics with automated fixture tests; keep user review for policy choices.
- Add stale-sentinel-file check.

### Risk Assessment
**MEDIUM**. Useful checkpoint but validates a design with at least one unresolved correctness issue.

---

## Consensus Summary

### Agreed Strengths
- Power-primary routing is the correct architectural direction (both reviewers)
- Good dependency ordering across plans (both)
- Strong requirement traceability and centralized config (both)
- Appropriate inclusion of human checkpoint before costly training (both)

### Agreed Concerns
- **HIGH: Batch coupling formula (BTCH-01) is internally inconsistent** -- `max(1, effective_batch // new_batch)` doesn't preserve effective_batch for non-divisible batch sizes (e.g., batch 4->5 gives eff_batch 15, not 16). Both reviewers flag this.
- **HIGH: Decision logic in markdown skill vs testable Python** -- Codex rates this HIGH, Gemini rates as MEDIUM. Core routing algorithm should be in Python with unit tests.
- **HIGH: Plan 02 runtime fragility** -- `sudo drop_caches`, `builtins.print` monkey-patching, and unclear probe lifecycle. Both flag these as risky for long GPU runs.
- **MEDIUM: Telemetry schema consistency** -- Field names (`temp` vs `temp_c`) and required/optional fields are not rigorously defined across plans.
- **MEDIUM: Sentinel file lifecycle** -- Multiple sentinel files created but cleanup/ownership rules are incomplete.

### Divergent Views
- **Overall risk**: Gemini rates overall LOW, Codex rates HIGH. The difference is Gemini focuses on architectural soundness while Codex focuses on implementation correctness and runtime safety.
- **Skill-based logic**: Gemini sees this as architecturally sound for Claude Code. Codex sees it as untestable. Both have valid points -- the skill pattern is standard for this project but the algorithm complexity exceeds typical skill scope.

---
phase: 11-compression-packaging
plan: 04
subsystem: moe-inference-masking
tags: [moe-sieve, expert-masking, vllm-patch, judge-eval, wp-bench, gb10]

requires:
  - phase: 11-03
    provides: "sieve_profile_mode=shared (one masking profile for all 3 judge seeds), per-k protected-retention requirements (866/198/0 at-risk at k=13/32/64)"
provides:
  - "scripts/sieve_expert_mask_inference.py: build_ksweep_mask() + apply_mask(), Wave-0 test contract GREEN, no training/no gradients"
  - "scripts/_sieve_vllm_patch/sitecustomize.py: live vLLM router-logit masking patch, smoke-verified on real GB10 hardware"
  - "scripts/sieve_capture_judge_http.py: HTTP judge-response capture (4x concurrency), scoring cross-validated against existing canonical captures"
  - "scripts/sieve_ksweep_run.py: k-sweep driver (full/64/32/13, sequential GB10-safe serving)"
  - "output/sieve/k_sweep_results.json: k=full arm measured; k=64/32/13 NOT yet run (sweep halted by its own pre-registered sanity gate)"
affects:
  - "plan 11-05 (TOST equivalence gate) -- cannot proceed until the k=full baseline reproduces within the pre-registered tolerance of the canonical judge-ensemble number"
  - "Phase 13 AIMER prune-set -- masking mechanism (build_ksweep_mask/apply_mask) is directly reusable once the judge-axis harness gap below is resolved"

tech-stack:
  added: []
  patterns:
    - "vLLM inference-time masking via a __init__-patched forward_hook on the MoE gate (ReplicatedLinear), additive tensor precomputed once per layer at model-construction time (never inside the hot forward path) to satisfy CUDA-graph-capture's no-unpinned-H2D-copy constraint"
    - "SIEVE_MASK_NPY env-var threaded through serve_30_70_vllm.sh -> PYTHONPATH sitecustomize.py auto-load, additive/backward-compatible (unset = identical prior behavior)"
    - "ThreadPoolExecutor(max_workers=4) judge-response capture, matching wp-bench's own concurrency=4 convention against the same served endpoint"

key-files:
  created:
    - scripts/sieve_expert_mask_inference.py
    - scripts/_sieve_vllm_patch/sitecustomize.py
    - scripts/sieve_capture_judge_http.py
    - scripts/sieve_ksweep_run.py
  modified:
    - scripts/serve_30_70_vllm.sh

decisions:
  - "Inference-time masking implemented as a live vLLM router-logit patch (forward hook on Qwen3MoeSparseMoeBlock.gate, -inf on masked-out expert logits) rather than baking a static pruned checkpoint per k -- matches the plan's apply_mask semantics exactly (softmax renormalizes over the kept set automatically) and needs zero new on-disk model copies."
  - "k=full (unmasked) runs FIRST in the sweep order (full, 64, 32, 13) so the pre-registered sanity bounds gate the whole sweep before any masked arm burns GPU time -- exactly what happened."
  - "Two real, reproducible harness bugs were found and fixed during the mandatory real-hardware run (both committed): (1) judge capture max_tokens=1024 truncated 16-24/121 responses per seed vs the canonical 2048 -- fixed. (2) wp-bench's own wp-env-runtime-* grader containers are REUSED (not recreated) across invocations, letting WordPress-DB state from an earlier run silently degrade the next run's 'correctness' sub-score -- fixed by resetting them before every gen arm."
  - "After both fixes, wp_bench passes the sanity floor (0.4484 >= 0.4416) but judge_ensemble_rho does not (0.8075 vs floor 0.822, canonical 0.842) -- see Known Issues below. Per the plan's own explicit contract this HALTS the sweep; k=64/32/13 were never run."

metrics:
  duration: ~7h (dominated by 3 full real-hardware k=full attempts at ~90-120 min each, diagnosing 2 harness bugs in between)
  completed: 2026-07-09
status: blocked
requirements: [SIEVE-04]
requirements-completed: []
---

# Phase 11 Plan 04: Inference-Time Expert Masking + k-Sweep Summary

**Masking module built and unit-tested clean; the mandatory real-hardware run executed 3 times, found and fixed two genuine harness bugs (judge-capture truncation, wp-bench grader-container staleness), but the k=full baseline still falls short of the pre-registered judge-rho sanity floor -- the sweep correctly HALTED per its own gate before any masked arm ran.**

## What was done

**Task 1 — Inference-time expert-masking module.** `scripts/sieve_expert_mask_inference.py`:
`build_ksweep_mask(counts, protected, k)` (top-k hot UNION protected, per layer;
never drops a protected expert, masked-out experts are always the coldest
non-protected ones) and `apply_mask(router_logits, keep_mask_row)` (sets masked
logits to -inf; numpy + torch dual path). `tests/test_sieve_ksweep_mask.py`
(Wave-0 contract) GREEN — 3/3 passed. Assert-based `__main__` self-check
included. No training, no gradients. Committed `3bd27c8`.

**Task 2 — k-sweep infrastructure + the mandatory real-hardware run.**

- `scripts/_sieve_vllm_patch/sitecustomize.py`: auto-loaded via `PYTHONPATH`
  inside the vLLM serving container. Monkeypatches
  `Qwen3MoeSparseMoeBlock.__init__` to register a forward hook on each layer's
  `gate` (router) that adds -inf to masked-out experts' logits. Two device-
  placement bugs found and fixed via live smoke test before trusting it on
  real data: (1) vLLM's ambient `torch.utils._device` context silently pins
  bare `torch.tensor(...)` calls to `cuda:0`, which crashed when mixed with a
  genuinely-CPU `torch.from_numpy(...)` in the same op; (2) vLLM captures CUDA
  graphs for the served forward path, which forbids any unpinned CPU->CUDA
  copy inside the captured region -- the additive mask tensor must be built
  ONCE at layer-init time (explicit `device=self.gate.weight.device`), never
  inside the hot per-forward-call hook. **Smoke-verified real difference**: at
  k=13 (aggressive mask, ~25-40/128 experts kept per layer) the SAME prompt
  that produces a clean, correct completion on the unmasked model produces a
  qualitatively degenerate repeat-loop completion when masked -- confirms the
  patch does real, direction-correct work, not a no-op.
- `scripts/serve_30_70_vllm.sh`: additive `SIEVE_MASK_NPY` env var mounts the
  patch dir + mask file read-only into the container; unset behaves exactly
  as every prior caller (backward compatible).
- `scripts/sieve_capture_judge_http.py`: HTTP twin of
  `scripts/capture_judge_responses_tinker.py` (same `wp_judge`-filtered index
  discipline), reusing `eval.eval_judge._judge_create` (RC-A guard). Scoring
  logic (`score_judge_ensemble` in the driver) was validated end-to-end
  **before any new GPU run** against the EXISTING canonical Tinker captures
  (`output/relabel/eval_full_ep3`, `eval_s1_ep3`, `eval_s2_ep3`
  `judge_responses.jsonl`): reproduced `judge_ensemble_rho=0.8420` and
  `judge_single_s1_rho=0.8274`, an exact match to
  `output/relabel/eval_seed_curve.json`'s `N3=0.842` / `s1_ep3=0.8274`. The
  scoring/alignment code is proven correct; the discrepancy below is entirely
  in what data feeds it.
- `scripts/sieve_ksweep_run.py`: the k-sweep driver, order `full, 64, 32, 13`
  (full first so the sanity gate protects the whole sweep). Reuses
  `run_eval_reasoning._wpbench_with_boot` unmodified for the gen axis
  (`SIEVE_MASK_NPY` threaded via `os.environ`, zero code duplication).

**Real-hardware run, 3 attempts (this is VALIDATION.md's mandatory real-hardware
run for the phase):**

| Attempt | wp_bench (floor 0.4416) | judge_ensemble_rho (floor 0.822) | judge_single_s1_rho (canonical 0.8274) | Outcome |
|---|---|---|---|---|
| 1 (judge capture max_tokens=1024) | 0.4603 PASS | 0.8046 FAIL | 0.7902 | HALT — found judge truncation bug |
| 2 (max_tokens=2048 fix; wp-env still stale) | 0.4365 FAIL | 0.8243 PASS | 0.8018 | HALT — found wp-env staleness bug |
| 3 (both fixes applied) | 0.4484 PASS | 0.8075 FAIL | 0.8017 | HALT — judge axis still short, see below |

Attempts 1-2 each isolated and fixed one genuine, reproducible harness bug
(both committed, see Deviations). Attempt 3 applied both fixes and still HALTs
on the judge-rho sanity bound. `output/sieve/k_sweep_results.json` holds
attempt 3's k="full" arm (the last, most-correct measurement); k=64/32/13 were
never executed because the plan's own gate stops the sweep here:
*"If either sanity bound fails, HALT the sweep... harness misconfiguration,
not a masking result."*

## Diagnosis of the remaining judge-axis gap (unresolved -- needs a decision)

Ruled out during this session:
- **Truncation**: fixed (max_tokens 1024->2048), confirmed 0 parse failures.
- **Thinking-tag leakage**: 0/121 responses across all 3 seeds contain
  `<think>` -- `enable_thinking=False` was honored, no RC-A silent-rejection
  warning fired.
- **Dataset drift**: `data/reasoning_dataset/openai_val.jsonl` unchanged
  (git-tracked, last modified 2026-05-25, well before any historical capture).
- **Scoring/alignment bug**: the exact scoring function used here reproduces
  the canonical 0.8420/0.8274 numbers bit-for-bit against the EXISTING
  historical captures (see Task 2 above) -- the pairing/median/spearman logic
  is proven correct.
- **wp-bench-specific staleness**: fixed and gen wp_bench now passes reliably
  (0.4484, 0.4603 across repeats after the reset fix); this was a wp-bench-
  owned Docker fixture issue, unrelated to the judge axis.

Not yet ruled out (most likely candidate): **Tinker-vs-vLLM serving-path /
merge-fidelity gap.** All 3 canonical numbers (0.842 ensemble, 0.827 s1
single) were measured by sampling the **native, unmerged Tinker LoRA weights**
via Tinker's own `qwen3_disable_thinking` renderer. The k-sweep's judge axis
can only serve the **merged** checkpoints via vLLM (per 11-CONTEXT.md: *"Tinker
MoE-expert LoRA is NOT standard PEFT -- vLLM cannot load it as a runtime
adapter"* -- `merge_tinker_v3.py` does manual per-expert tensor arithmetic,
not a standard PEFT merge). s1's single-seed rho came back at 0.7902 / 0.8018
/ 0.8017 across 3 independent measurements of the SAME merged checkpoint --
consistently ~2.5-3.7pp below the canonical 0.8274, in the same direction
every time (not symmetric noise around 0.8274). This points at a systematic
merge-fidelity or renderer-formatting gap between the native Tinker LoRA and
its merged vLLM-served form, not masking, not a bug in this plan's new code.

**This is exactly the scenario the plan's `<critical_environment_facts>`
pre-registered a HALT for.** Resolving it requires a decision this executor
cannot make unilaterally: (a) recalibrate the sanity floor against a
freshly-measured "merged, unmasked" reference number instead of the
Tinker-native one, since vLLM-served masking can only ever be compared to the
merged baseline it's actually diffing against, or (b) invest in verifying/
improving `merge_tinker_v3.py`'s merge fidelity (or the chat-template/renderer
parity between Tinker's `qwen3_disable_thinking` and vLLM's
`chat_template_kwargs enable_thinking=false`) before trusting any k-sweep
result against the Tinker-native baselines.

## Task Commits

1. **Task 1: Inference-time expert-masking module** - `3bd27c8` (feat)
2. **Task 2: k-sweep infrastructure (patch + capture + driver)** - `f880195` (feat)
3. **Task 2 fix: parallelize judge capture** - `f09fdf7` (fix)
4. **Task 2 fix: judge max_tokens 1024->2048** - `cd36a5e` (fix)
5. **Task 2 fix: reset wp-bench grader containers** - `8c4b167` (fix)

## Files Created/Modified
- `scripts/sieve_expert_mask_inference.py` - build_ksweep_mask, apply_mask, CLI, self-check
- `scripts/_sieve_vllm_patch/sitecustomize.py` - live vLLM MoE-router masking patch
- `scripts/sieve_capture_judge_http.py` - HTTP judge-response capture, 4x concurrency
- `scripts/sieve_ksweep_run.py` - k-sweep driver (full/64/32/13), wp-env reset
- `scripts/serve_30_70_vllm.sh` - additive SIEVE_MASK_NPY mount support
- `output/sieve/k_sweep_results.json` (gitignored, on disk) - k="full" arm only, halted=true

## Decisions Made
- Live vLLM router-logit patch (not a pre-baked pruned checkpoint) for masking.
- Sweep order full-first so the sanity gate protects the whole run.
- 4x concurrency for judge capture (matches wp-bench's own convention on the
  same endpoint) after the initial sequential loop proved to be the actual
  throughput bottleneck.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Judge capture truncated at max_tokens=1024, biasing rho down**
- **Found during:** Task 2, real-hardware attempt 1
- **Issue:** `scripts/sieve_capture_judge_http.py` capped judge generations at
  1024 tokens; the canonical Tinker captures used 2048. 16-24/121 responses
  per seed were long enough to hit the cap and got cut off mid-rubric-
  dimension (1 outright unparseable truncation; partial-dimension-coverage
  bias on the rest).
- **Fix:** default `max_tokens` raised to 2048 (function default and CLI
  default), matching the project-wide convention (also used by wp-bench).
- **Files modified:** scripts/sieve_capture_judge_http.py
- **Verification:** re-ran the full arm; 0/121 parse failures across all
  3 seeds; response lengths now comfortably under the cap.
- **Committed in:** cd36a5e

**2. [Rule 3 - Blocking] Sequential judge capture was the actual throughput bottleneck**
- **Found during:** Task 2, real-hardware attempt 1 (before the truncation
  fix was even applied)
- **Issue:** one-request-at-a-time capture loop measured ~20-30s/item,
  projecting to hours for 121 items x 3 seeds x 4 k-arms.
- **Fix:** `ThreadPoolExecutor(max_workers=4)`, matching wp-bench's own
  concurrency=4 convention against the identical served endpoint. Output
  ordering unaffected (collected by index).
- **Files modified:** scripts/sieve_capture_judge_http.py
- **Verification:** subsequent captures completed in ~12-15 min/seed instead
  of an open-ended multi-hour projection.
- **Committed in:** f09fdf7

**3. [Rule 3 - Blocking] wp-bench's own grader containers accumulate stale WordPress-DB state**
- **Found during:** Task 2, real-hardware attempt 2
- **Issue:** the k=full gen wp-bench score reproduced 0.4603 (byte-identical
  to the historical REVL-04 rebench file) on the very first invocation, then
  0.4365 on two subsequent invocations (also byte-identical to each other,
  0/344 per-test diffs). "knowledge" (pure text-match) stayed bit-identical
  across all three; only "correctness" (docker-based PHP-execution grading)
  regressed -- pointing at the grader's own fixture, not vLLM/model
  nondeterminism. Root cause: `wp-env-runtime-*` (WordPress+MySQL) containers
  are REUSED across separate wp-bench invocations rather than recreated, so
  DB state from an earlier run degraded the next run's execution-based
  grading. Confirmed by manually removing the stale containers: score
  returned to 0.4603.
- **Fix:** `_reset_wpbench_grader()` (`docker rm -f` any `wp-env-runtime-*`
  container) added before every gen arm in the driver.
- **Files modified:** scripts/sieve_ksweep_run.py
- **Verification:** re-ran with fresh containers -> 0.4484 (attempt 3), well
  above the 0.4416 floor; without the reset it had failed at 0.4365.
- **Committed in:** 8c4b167

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking) + 1 unresolved finding
requiring a human decision (see Diagnosis section above and Known Issues).
**Impact on plan:** All 3 auto-fixes were necessary to get a trustworthy
real-hardware measurement at all; none were scope creep. The remaining
judge-axis gap is what stopped Task 2 from reaching k=64/32/13 -- exactly the
HALT-and-report path the plan's acceptance criteria pre-registered for this
failure mode.

## Known Issues

- **`output/sieve/k_sweep_results.json` has ONLY the k="full" arm.** k=64/32/13
  were never executed because the sanity gate correctly halted the sweep
  first. This is NOT a masking result and must not be read as "masking fails
  at every k" -- no masked arm has been measured yet.
- **judge_ensemble_rho for the unmasked baseline (0.8075-0.8243 across 3
  measurements) does not reliably clear the pre-registered 0.822 floor**, and
  `judge_single_s1_rho` (0.7902-0.8018) is consistently ~2.5-4pp below the
  canonical Tinker-native 0.8274. See Diagnosis above -- most likely a
  Tinker-vs-vLLM-merged-checkpoint fidelity gap, not resolved in this plan.
- **wp_bench for the unmasked baseline now passes reliably** (0.4484, 0.4603)
  after the wp-env reset fix, but showed one anomalous 0.4365 reading before
  the fix -- that specific value is explained (stale grader containers) and
  should not recur.

## Threat Flags

| Flag | File | Description |
|------|------|--------------|
| threat_flag: new-inference-surface | scripts/_sieve_vllm_patch/sitecustomize.py | Monkeypatches a vLLM internal class (`Qwen3MoeSparseMoeBlock.__init__`) via `PYTHONPATH`-based sitecustomize auto-load inside the serving container. Scoped to Sieve k-sweep use only (opt-in via `SIEVE_MASK_NPY`); raises loudly (does not silently no-op) if the mask file is missing or the patch fails to install, per T-11-08's inviolable-mask disposition. |

## User Setup Required
None - no external service configuration required. (The docker images, wp-bench
CLI, and merged checkpoints used were all already present in this environment.)

## Next Phase Readiness

**Not ready for plan 11-05 (TOST equivalence gate) yet.** Blocking question for
a human decision before any further Sieve k-sweep work:

1. Is the pre-registered judge-rho sanity floor (0.822, derived from the
   Tinker-native 0.842) the right reference for a harness that can only ever
   serve MERGED checkpoints? Or should the floor be recalibrated against a
   freshly-established "merged, unmasked, vLLM-served" reference number
   (which attempt 3 suggests sits around 0.80-0.82, not 0.842)?
2. Alternatively, is the merge process itself (`merge_tinker_v3.py`) worth
   auditing for fidelity loss vs the native Tinker LoRA before trusting any
   number measured through it?

Once that's resolved, `scripts/sieve_ksweep_run.py` needs no further changes
to actually run k=64/32/13 -- it already builds each k's masks, serves gen +
3 judge seeds sequentially (GB10-safe), and would append each arm to
`output/sieve/k_sweep_results.json` exactly like the k="full" arm already
recorded.

## Self-Check

- FOUND: scripts/sieve_expert_mask_inference.py
- FOUND: scripts/_sieve_vllm_patch/sitecustomize.py
- FOUND: scripts/sieve_capture_judge_http.py
- FOUND: scripts/sieve_ksweep_run.py
- FOUND: scripts/serve_30_70_vllm.sh (modified, SIEVE_MASK_NPY support)
- FOUND: output/sieve/k_sweep_results.json (gitignored, on disk, k="full" arm + halted=true)
- FOUND commits: 3bd27c8, f880195, f09fdf7, cd36a5e, 8c4b167
- `pytest tests/test_sieve_ksweep_mask.py -x -q` (via `.venv-tinker/bin/python -m pytest`): 3 passed
- Protected mask sha256 unchanged: 659af6eb... (.npy) / ade549e0... (.json) -- matches 11-03's recorded checksums

---
*Phase: 11-compression-packaging*
*Completed: 2026-07-09 (BLOCKED — see Next Phase Readiness)*

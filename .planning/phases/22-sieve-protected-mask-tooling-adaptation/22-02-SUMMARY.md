---
phase: 22-sieve-protected-mask-tooling-adaptation
plan: 02
subsystem: infra
tags: [moe-sieve, gate4-02, qwen3.6-35b-a3b, transformers, gb10, empirical-smoke]

# Dependency graph
requires:
  - phase: 22-sieve-protected-mask-tooling-adaptation
    provides: "22-01: scripts/sieve_arch.py (arch_dims, layer_strata, resolve_moe_layers) — the tooling this plan proves empirically"
provides:
  - "output/sieve-v4/tooling_smoke.json: signed receipt proving GATE4-02 SC1-SC4 empirically on the real models/Qwen3.6-35B-A3B-judge-v4-s1-merged checkpoint"
  - "scripts/sieve_v4_tooling_smoke.py: reusable bounded (N=32, max_seq_len=1024) single-GPU-load smoke harness for the v4 judge router"
  - "Empirical fact: the v4 judge VL-composite checkpoint must be loaded via AutoModelForImageTextToText (Qwen3_5MoeForConditionalGeneration), NOT AutoModelForCausalLM — the latter resolves a flat text-only class with 692/693 state_dict keys missing against this checkpoint's nested model.language_model.layers.* convention"
  - "Empirical fact: resolved_traversal_root == model.language_model.layers for this checkpoint (candidate #2 in sieve_arch's ordered list) — settles the ROADMAP-SC1-vs-20-04 question for the Phase 21 VL-composite merged judge, distinct from Phase 20's flat LoRA-merged text-only checkpoint"
affects: [25-k-sweep, 26-prune]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Meta-device state_dict key-set diff (torch.device('meta') + safetensors index.json) used to statically prove/disprove a checkpoint-to-model-class match BEFORE committing GPU time to a load — avoids wasting the plan's single bounded GB10 load on a doomed AutoModel class guess"

key-files:
  created:
    - scripts/sieve_v4_tooling_smoke.py
    - output/sieve-v4/tooling_smoke.json
  modified: []

key-decisions:
  - "Used AutoModelForImageTextToText instead of the plan's literal AutoModelForCausalLM (Rule 1 bug fix, see Deviations) — proven correct via a meta-device key diff: Qwen3_5MoeForConditionalGeneration has 0 missing keys against the checkpoint, Qwen3_5MoeForCausalLM has 692/693 missing"
  - "Ran the smoke directly on the host .venv-tinker environment (torch 2.12+cu130, transformers 5.5.3, native qwen3_5_moe support, CUDA already available) rather than the ngc-pytorch container the plan's read_first notes referenced — the container was needed in Phase 20 because the host lacked CUDA-enabled torch at that time; the host now has everything the load needs, so the container step would be pure overhead with no correctness benefit"
  - "resolved_traversal_root is computed at runtime from the live loaded model object (mirroring sieve_arch's private candidate-root walk), not hardcoded from static source-reading, so the receipt's SC4 claim is genuinely empirical"

requirements-completed: [GATE4-02]

coverage:
  - id: D1
    description: "Bounded GB10 smoke on the real v4 judge (models/Qwen3.6-35B-A3B-judge-v4-s1-merged) proves 40 router hooks register (SC1), strata 30 deltanet + 10 attention at the correct indices (SC2), router_logits last dim == 256 == config.num_experts with shared_expert confirmed as a separate module (SC3, empirical), and records the resolved traversal root (SC4) — output/sieve-v4/tooling_smoke.json status=pass"
    requirement: "GATE4-02"
    verification:
      - kind: other
        ref: ".venv-tinker/bin/python -c \"...\" reading output/sieve-v4/tooling_smoke.json and asserting every field (the plan's <verify> block, re-run post-hoc)"
        status: pass
    human_judgment: false

# Metrics
duration: ~35min
completed: 2026-07-15
status: complete
---

# Phase 22 Plan 02: Sieve v4 Tooling Empirical Smoke Summary

**One bounded GB10 load of the real 67 GiB v4 judge checkpoint proves the Plan 22-01 sieve_arch tooling correct end-to-end — 40 hooks, 256-wide router logits with the shared expert empirically absent, 30/10 strata split, and the actual resolved traversal root (`model.language_model.layers`) all captured in a hard-asserted JSON receipt.**

## Performance

- **Duration:** ~35 min (includes architecture investigation, ~4m12s weight load, forward pass, cleanup)
- **Completed:** 2026-07-15T06:23:38Z
- **Tasks:** 1
- **Files modified:** 2 (both new)

## Accomplishments

- `scripts/sieve_v4_tooling_smoke.py`: loads the v4 judge, registers hooks via `sieve_arch.resolve_moe_layers`, runs a bounded forward pass (N=32 examples from `data/reasoning_dataset/openai_val.jsonl`, max_seq_len=1024), and writes a hard-asserted receipt
- `output/sieve-v4/tooling_smoke.json`: `status: "pass"` with all four GATE4-02 sub-criteria proven on the real checkpoint (not a mock, not a config-only inference):
  - `hooks_registered: 40` == `expected_n_layers: 40` (SC1)
  - `strata_counts: {deltanet: 30, attention: 10}`, `attention_layer_indices: [3,7,11,15,19,23,27,31,35,39]` (SC2)
  - `router_logits_last_dim: 256` == `config_num_experts: 256`, `shared_expert_in_router_logits: false`, `shared_expert_module_present: true` (SC3)
  - `resolved_traversal_root: "model.language_model.layers"` (SC4)
- Empirically discovered and fixed a load-bearing bug in the plan's literal model-loading instruction before spending the bounded GPU load on it (see Deviations)
- GPU verified idle (3% util, `[N/A]` unified-memory query is normal for GB10) and no orphan container after the run

## Task Commits

Each task was committed atomically:

1. **Task 1: Bounded GB10 tooling smoke on the v4 judge → tooling_smoke.json receipt (SC1/SC2/SC3/SC4)** - `8e7fe00` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `scripts/sieve_v4_tooling_smoke.py` - New: bounded (N=32, max_seq_len=1024) single-load smoke harness; CUDA guard + `--allow-cpu` override matching the existing profiler scripts' convention; hooks `mlp.gate` via `sieve_arch.resolve_moe_layers`, captures `router_logits` last-dim on first fire, probes `mlp.shared_expert` presence, computes `layer_strata`, and derives `resolved_traversal_root` at runtime by mirroring `sieve_arch`'s private candidate-root walk
- `output/sieve-v4/tooling_smoke.json` - New: the GATE4-02 receipt (force-added past the `output/` gitignore, matching the existing repo convention for `output/base20/*.json` / `output/base21/*.json` receipts)

## Decisions Made

- **AutoModelForImageTextToText over AutoModelForCausalLM** — see Deviations below (Rule 1 auto-fix, not a discretionary choice)
- **No docker container** — the plan's `read_first` pointed at Phase 20's `bash deps/dgx-toolbox/containers/ngc-pytorch.sh` guidance, written when the host lacked CUDA-enabled torch. The host's `.venv-tinker` now has torch 2.12+cu130 (`torch.cuda.is_available() == True`) and transformers 5.5.3 with native `qwen3_5_moe` support (no `trust_remote_code` custom code needed — no `modeling_*.py` shipped with the checkpoint). Spinning up the container would add ~minutes of image/dependency setup with zero effect on correctness; ran directly on host per the CUDA guard already in the script.
- **`resolved_traversal_root` computed at runtime, not asserted from source reading** — even though a static meta-device/config investigation predicted `model.language_model.layers` before the load, the script itself walks the actual loaded model object at runtime (mirroring `sieve_arch._MOE_LAYER_ROOT_CANDIDATES`) so the receipt's SC4 claim is genuinely empirical, not hardcoded from the investigation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] AutoModelForCausalLM would silently produce a randomly-initialized forward pass**
- **Found during:** Task 1, pre-load investigation (before spending the plan's single bounded GB10 load)
- **Issue:** The plan's `<action>` literally specifies `AutoModelForCausalLM.from_pretrained(dtype=bfloat16, device_map="auto", trust_remote_code=True)`. Investigation of the checkpoint's `config.json` (`architectures: ["Qwen3_5MoeForConditionalGeneration"]`, composite `text_config`/`vision_config`) and `model.safetensors.index.json` (1045 keys including `model.visual.blocks.*` and `model.language_model.layers.*`) showed this is a VL-composite checkpoint with vision weights present, saved under a nested `model.language_model.layers.*` key convention — NOT the flat text-only convention some earlier merged checkpoints use. `AutoModelForCausalLM` resolves `qwen3_5_moe` to `Qwen3_5MoeForCausalLM`, whose `__init__` builds `self.model = Qwen3_5MoeTextModel(config)` (flat `model.layers.*`). A meta-device instantiation + `state_dict().keys()` diff against the checkpoint's key set confirmed 692 of 693 `Qwen3_5MoeForCausalLM` keys are MISSING from the checkpoint (and conversely all 1044 non-vision checkpoint keys would be reported "unexpected" and discarded) — loading this way would silently leave the entire text backbone randomly initialized while reporting success, directly undermining GATE4-02's empirical claim (the router hook/shape assertions are shape-based and would have passed even on garbage weights, making this a spoofing-adjacent failure mode not caught by the plan's hard asserts).
- **Fix:** Used `AutoModelForImageTextToText.from_pretrained(...)`, which resolves `qwen3_5_moe` to `Qwen3_5MoeForConditionalGeneration`. The same meta-device key diff showed 0 missing keys against the checkpoint (only 19 unrelated `mtp.*` multi-token-prediction keys are unexpected/dropped, a known non-load-bearing head). This is the class that actually matches the checkpoint's real weights.
- **Files modified:** scripts/sieve_v4_tooling_smoke.py (the loader call, with an inline comment documenting the finding and the key-count evidence)
- **Verification:** Model loaded successfully (4m12s, 1026/1026 shards), forward pass over 32 examples completed without shape/key errors, receipt `status: "pass"` on all four SC criteria, `resolved_traversal_root: "model.language_model.layers"` matches the empirically-confirmed nested convention
- **Committed in:** `8e7fe00` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug — caught before it could produce a false-positive receipt)
**Impact on plan:** Essential — the plan's literal model-loader class would have produced a structurally-valid but semantically-empty (randomly-initialized) receipt that still satisfied every shape-based assert. No scope creep: same script, same task, same receipt schema; only the `AutoModel*` class changed.

## Issues Encountered

- `rtk proxy nvidia-smi`/direct `nvidia-smi --query-gpu=memory.used` reports `[N/A]` for GB10's unified memory (expected — GB10 is a unified-memory Grace-Blackwell part, "Memory-Usage: Not Supported" in the full `nvidia-smi` table is the normal state, confirmed against `utilization.gpu` reporting correctly both idle-before (2%) and idle-after (3%) the run). Not a blocker; noted so the pattern isn't mistaken for a query failure in future plans.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 25's full k-sweep profiling pass now has an empirically-audited loader recipe for the v4 judge checkpoint family: `AutoModelForImageTextToText` (not `AutoModelForCausalLM`), `sieve_arch.resolve_moe_layers`/`arch_dims`/`layer_strata` all confirmed correct against the live model object, traversal root confirmed as `model.language_model.layers` for this VL-composite checkpoint shape.
- GATE4-02 is now closed independently of the RL gate (SC4) — the phase's empirical burden of proof is satisfied by `output/sieve-v4/tooling_smoke.json`.
- Open note for Phase 25: if a future v4 judge checkpoint is saved via a different pipeline (e.g. a flattened text-only merge, matching Phase 20's LoRA-merge convention rather than Phase 21's VL-composite SFT-merge convention), the AutoModel class choice should be re-verified with the same meta-device key-diff technique before committing a full profiling run — the two known v4-family checkpoint shapes in this repo (Phase 20 flat vs. Phase 21 nested) require different loader classes.
- No blockers for Phase 25.

---
*Phase: 22-sieve-protected-mask-tooling-adaptation*
*Completed: 2026-07-15*

## Self-Check: PASSED

All 3 claimed files found on disk; task commit hash (8e7fe00) found in git log.

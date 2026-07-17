---
phase: 27-packaging-publication-refresh
plan: 02
subsystem: packaging
tags: [gguf, llama.cpp, moe, quantization, mtp, spearman, judge-eval]

requires:
  - phase: 27-packaging-publication-refresh
    provides: "Plan 27-01 Wave-0 trust foundation: expert-count sanity check in eval4_ext_gguf_convert.sh, pkg4_quant_type_check.py, pub4_validate_upload.py"
provides:
  - "f16 GGUF master (--no-mtp) and Q8_0 GGUF of models/Qwen3.6-35B-A3B-judge-v4-pruned-k224, both proven 224-expert/40-block uniform and shared-expert-quant-type-uniform"
  - "REAL measured Q8 size: 30.37 GiB (32614463296 bytes) -- replaces the 33.6 GiB linear-scaling projection"
  - "Gate 1 baseline (f16-anchored, judge_rho=0.8002, n=121) -- the ±2pp floor Plan 03's Q6/Q5 descent gates against, frozen via floor_frozen_utc before any downstream quant byte exists"
  - "Q8 ladder rung: judge_rho=0.7851, delta_vs_gate1=-1.507pp, PASSES with only 0.493pp slack -- Q8 is NOT lossless on the pruned checkpoint (contradicts v3 precedent)"
  - "eval4_ext_gguf_convert.sh extended with an optional extra_convert_arg (--no-mtp) for checkpoints whose MTP layer wasn't pruned in lockstep with the trunk"
affects: [27-03-quantization-ladder, 27-04-model-card-and-manifest, 27-05-publish]

tech-stack:
  added: []
  patterns:
    - "--no-mtp GGUF conversion for checkpoints with heterogeneous per-layer expert counts (GGUF's expert_count metadata is a single GLOBAL field; llama.cpp's loader enforces it uniformly across every block)"
    - "Gate-1-by-path referencing: every downstream rung file (ladder_q8.json) reads gate1_f16_baseline_v4.json's judge_rho by path, never inlines it"

key-files:
  created:
    - output/pkg-v4/conversion_receipt_v4.json
    - output/pkg-v4/quant_type_q8.json
    - output/pkg-v4/gate1_f16_baseline_v4.json
    - output/pkg-v4/ladder_q8.json
    - output/pkg-v4/wp-judge-v4-pruned-k224.f16.gguf (61313087616 bytes, untracked binary)
    - output/pkg-v4/wp-judge-v4-pruned-k224.Q8_0.gguf (32614463296 bytes, untracked binary)
  modified:
    - scripts/eval4_ext_gguf_convert.sh

key-decisions:
  - "Gate 1 anchored to f16 (assumption A1), not Q8, per PKG4-02/ROADMAP wording -- this is the ONLY reason the Q8-not-lossless finding is visible at all; a Q8-anchored Gate 1 would report delta=0.000 by construction"
  - "--no-mtp GGUF conversion: Phase 26's prune surgery deliberately left the MTP/nextn layer at 256 experts while pruning the 40 trunk layers to 224 -- GGUF has no per-layer expert-count field, so the mixed-count GGUF failed to load in llama.cpp. Fixed with the officially-supported --no-mtp flag (drops mtp.* tensors, block_count=num_hidden_layers exactly). The shipped GGUF has no MTP/speculative-decoding head -- a real capability delta for 27-04's card to document."
  - "n=121, not 141: data/reasoning_dataset/openai_val.jsonl has 141 total lines but scripts.sieve_capture_judge_http filters to the wp_judge-tagged subset only (documented index discipline); the other 20 lines are wp_gen/other examples, irrelevant to a judge-only ship. Confirmed by direct count (121+20=141), not a truncation."
  - "Only small receipt JSON files (eval_summary.json, rho.txt, the four output/pkg-v4/*.json files) are force-added to git despite the blanket output/ gitignore, matching the output/packaging/{base,bf16}_eval precedent from prior phases. The large binaries (.gguf) and raw captures (judge_responses.jsonl, serve.log) stay on disk as evidence but are never committed."

requirements-completed: [PKG4-01]

coverage:
  - id: D1
    description: "f16 GGUF master + Q8_0 GGUF converted from the pruned v4 judge, both proven to carry exactly 224 experts and correct block count (40, --no-mtp), Q8's shared-expert tensors proven uniform with routed-expert tensors from the produced bytes"
    requirement: "PKG4-01"
    verification:
      - kind: other
        ref: "bash verify block in 27-02-PLAN.md Task 1 <verify>: conversion_receipt_v4.json size/sanity assertions + quant_type_q8.json shared_expert_uniform/routed_expert_types/deltanet_state_types assertions -- all pass"
        status: pass
    human_judgment: false
  - id: D2
    description: "Real Q8 byte size measured (30.37 GiB / 32614463296 bytes) and explicitly distinguished from the 33.6 GiB linear-scaling projection, which appears exactly once under size_vs_projection"
    requirement: "PKG4-01"
    verification:
      - kind: other
        ref: "python3 -c assertion: conversion_receipt_v4.json q8_0.size_bytes == stat -c%s of the produced file; grep -c '\"projected_q8_gib\": 33.6' == 1"
        status: pass
    human_judgment: false
  - id: D3
    description: "Gate 1 baseline measured on the shipped f16 GGUF/llama.cpp stack (anchor=f16_gguf_llamacpp, no OOM fallback needed), with stack/seeds/anchor named, v3 bands rejected, bf16-vLLM numbers quarantined as non-comparable"
    requirement: "PKG4-02"
    verification:
      - kind: other
        ref: "python3 -c assertion block in 27-02-PLAN.md Task 2 <verify>: judge_rho in (0,1), n>100, epsilon_pp==2, anchor==f16_gguf_llamacpp, judge_rho not in stale-number set, stack/seeds non-empty -- all pass"
        status: pass
    human_judgment: false
  - id: D4
    description: "Q8 ladder rung measured and gated against Gate 1 by path with an inclusive -2pp bar; concurrent-sequence CUDA-backend smoke evidence captured (4 parallel slots, 121/121 requests succeeded)"
    requirement: "PKG4-02"
    verification:
      - kind: other
        ref: "python3 -c assertion block in 27-02-PLAN.md Task 3 <verify>: gate.gate1_rho==gate1 file's judge_rho, delta recomputable to 1e-9, pass==(delta>=-0.02), size_bytes cited not re-derived, concurrent_sequence_smoke.requests_failed==0 -- all pass"
        status: pass
    human_judgment: false

duration: ~70min
completed: 2026-07-17
status: complete
---

# Phase 27 Plan 02: Convert & Measure — f16 Master, Q8 Rung, Gate 1 Baseline Summary

**Converted the pruned v4 judge to a `--no-mtp` f16 GGUF master and Q8_0 GGUF (30.37 GiB measured, not the 33.6 GiB projection), then discovered on the shipped stack that Q8 is NOT lossless on this surgically-pruned 224-expert MoE — it costs 1.507pp against the f16 Gate-1 floor, passing the ±2pp band with only 0.493pp of slack.**

## Performance

- **Duration:** ~70 min
- **Started:** 2026-07-17T07:35:00Z (approx)
- **Completed:** 2026-07-17T08:46:00Z
- **Tasks:** 3
- **Files modified:** 7 (1 extended script, 4 new receipt JSONs, 2 new GGUF binaries [untracked])

## Accomplishments
- f16 GGUF master and Q8_0 GGUF of `models/Qwen3.6-35B-A3B-judge-v4-pruned-k224` both exist, proven to carry exactly 224 experts and a correct, UNIFORM block count (40 blocks, `--no-mtp`) — the Q8's shared-expert tensors independently verified uniform with its routed-expert tensors (Q8_0) from the produced bytes, DeltaNet state tensors present (F32/Q8_0)
- REAL Q8 size measured: **30.37 GiB (32,614,463,296 bytes)** — smaller than both the 33.6 GiB linear-scaling projection AND v3's shipped 30.2 GiB; the projection appears exactly once in the receipt, explicitly labelled, never treated as measured
- **Gate 1 baseline established on the shipped f16 GGUF/llama.cpp stack**: judge_rho = 0.8001808600187146 (n=121, CI [0.7186, 0.8582], parse_fail=0). Assumption A1 (Gate 1 = f16, not Q8) held — the f16 serve did NOT OOM on the 121 GiB host, so `anchor=f16_gguf_llamacpp` with no fallback needed
- **Q8 ladder rung measured and gated**: judge_rho = 0.7851092593129675 (n=121, same wp_judge subset, parse_fail=0), delta_vs_gate1 = **-1.507pp**, PASSES the inclusive -2pp bar with only **0.493pp of slack**
- Concurrent-sequence CUDA-backend smoke (PKG4-01) satisfied: Q8 serve ran `-ngl 999 --parallel 4`, 121/121 requests succeeded, 0 failures — captured explicitly in `ladder_q8.json`'s `concurrent_sequence_smoke`

## HEADLINE FINDING: Q8 is NOT lossless on the pruned v4 judge

Q8 costs **1.507 percentage points** of Spearman rho against the same checkpoint's own f16 master (0.8002 → 0.7851). This **contradicts the v3 precedent**, where Q8 was confirmed lossless (v3 shipped ensemble rho 0.8056, no material degradation vs its bf16 baseline). The pruning of 32 experts/layer (256→224) appears to have made this checkpoint materially more quantization-sensitive than the unpruned v3 model was.

**This finding is visible ONLY because Gate 1 was anchored to f16 (assumption A1), not Q8.** A Q8-anchored Gate 1 would have produced `delta=0.000` by construction and reported "Q8 lossless" — a false conclusion that would have defined away the exact question this plan exists to answer ("is Q8 lossless on a surgically-pruned 224-expert MoE?"). That is the concrete payoff of the A1 override the planner and checker upheld.

**CI overlap caveat (not softened, not overstated):** the two 95% CIs (f16 [0.7186, 0.8582], Q8 [0.6984, 0.8492]) overlap substantially at n=121 — this does NOT establish the point-estimate drop is statistically insignificant, only that sampling noise at this n cannot rule out a smaller or larger true gap. The **1.507pp point-estimate drop is the honest headline** and is recorded as such in `ladder_q8.json`, not rounded to zero or called "within noise."

**Forward flag for Plan 27-03:** Q8 has already consumed 1.507pp of the 2pp budget, leaving 0.493pp. Q6_K and Q5_K_M are now **unlikely** to clear the f16 floor's ±2pp band — the ladder may well stop at Q8, making Q8 the ship tier. Plan 27-03 must still MEASURE Q6/Q5 rather than assume this; a Q6 failure should not be treated as an anomaly.

## Task Commits

Each task was committed atomically (plus one mid-plan fix commit):

1. **Task 1: Convert to f16 master + Q8_0, run sanity checks, measure REAL sizes** - `bf8ab1a` (feat)
2. **[Rule 1/3 fix] `--no-mtp` re-conversion after the first attempt failed to load** - `5e469a1` (fix)
3. **Task 2: Gate 1 — serve f16 master, measure ±2pp floor** - `d043890` (feat)
4. **Task 3: Q8 rung — serve, score, gate against Gate 1, concurrent-sequence smoke** - `130af17` (feat)

_No TDD split — this plan's tasks are measurement/conversion driver invocations, not new application logic._

## Files Created/Modified
- `output/pkg-v4/conversion_receipt_v4.json` - measured f16/Q8 sizes, sanity results, `--no-mtp` deviation note, projection-vs-measured closeout
- `output/pkg-v4/quant_type_q8.json` - per-tensor quant-type census (routed==shared==Q8_0, DeltaNet F32/Q8_0)
- `output/pkg-v4/gate1_f16_baseline_v4.json` - the frozen ±2pp floor (judge_rho=0.8002, n=121), `floor_frozen_utc` written before any Q6/Q5 byte exists
- `output/pkg-v4/ladder_q8.json` - Q8 rung result, delta_vs_gate1, headline finding, concurrent-sequence smoke evidence
- `output/pkg-v4/f16_eval/{eval_summary.json,rho.txt}` - Gate 1 receipt (tracked); `judge_responses.jsonl`/`serve.log` on disk, untracked
- `output/pkg-v4/q8_eval/{eval_summary.json,rho.txt}` - Q8 rung receipt (tracked); `judge_responses.jsonl`/`serve.log` on disk, untracked
- `scripts/eval4_ext_gguf_convert.sh` - added optional 4th `extra_convert_arg` positional (for `--no-mtp`), block-count sanity formula branches on it
- `output/pkg-v4/wp-judge-v4-pruned-k224.f16.gguf` (61.3 GB), `output/pkg-v4/wp-judge-v4-pruned-k224.Q8_0.gguf` (32.6 GB) - large binaries, on disk, deliberately NOT committed to git (matches project convention: `output/` is gitignored except explicitly force-added small receipt files)

## Decisions Made
- Gate 1 anchored to f16 (A1 override) held on the real hardware — no OOM fallback needed, `anchor=f16_gguf_llamacpp`
- `--no-mtp` conversion flag adopted as the fix for the mixed-expert-count MTP-layer load failure (see Deviations) — the shipped GGUF permanently lacks an MTP/speculative-decoding head; this is a real capability delta downstream plans (27-04's model card) must know about
- n=121 (not 141) confirmed correct by direct dataset count — the val-set file mixes wp_judge and wp_gen/other examples; the capture harness's documented `<wp_judge>`-prefix filter is working as designed, not truncating
- Only small receipt JSON files force-added to git (matching `output/packaging/{base,bf16}_eval` precedent); large GGUF binaries and raw judge-response captures/serve logs stay on disk, untracked

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1/3 - Blocking bug] `--no-mtp` re-conversion after llama.cpp refused to load the first GGUF**
- **Found during:** Task 2, first attempt to serve the f16 GGUF
- **Issue:** The first f16 conversion (block_count=41, MTP/nextn layer included per the plan's original block-count formula) produced a GGUF that FAILED TO LOAD in llama-server: `check_tensor_dims: tensor 'blk.40.ffn_gate_inp.weight' has wrong shape; expected 2048,224, got 2048,256`. Root cause: Phase 26's prune surgery deliberately left the MTP/nextn layer (and shared_expert.*) at the original 256 experts while pruning the 40 trunk layers to 224/256. GGUF's `expert_count` metadata is a single GLOBAL field and llama.cpp's loader enforces it uniformly against every block's tensor shapes — a checkpoint with heterogeneous per-layer expert counts cannot be represented in a single loadable GGUF via the default conversion path.
- **Fix:** Re-converted with the officially-supported `convert_hf_to_gguf.py --no-mtp` flag (confirmed in `~/llama.cpp/conversion/qwen.py`'s `_Qwen35MtpMixin`: drops all `mtp.*` tensors and sets `block_count=num_hidden_layers` exactly — no mixed-count block). Extended `scripts/eval4_ext_gguf_convert.sh` with an optional 4th `extra_convert_arg` positional passed through to the converter, and branched the block-count sanity formula on `--no-mtp` (expected=40, not 41). Re-derived tensor name patterns on the corrected f16 GGUF (all three non-empty), re-quantized to Q8_0, re-ran `pkg4_quant_type_check.py` (exit 0), re-measured the real Q8 size (30.37 GiB). Smoke-tested the corrected Q8_0 standalone (loads, listens, real generation succeeds) before proceeding to the full eval runs.
- **Files modified:** `scripts/eval4_ext_gguf_convert.sh`, `output/pkg-v4/conversion_receipt_v4.json`, `output/pkg-v4/quant_type_q8.json`
- **Verification:** All Task 1 acceptance criteria re-run and passed against the corrected artifacts; both f16 and Q8 GGUFs independently confirmed `block_count=40, expert_count=224` via `GGUFReader`
- **Committed in:** `5e469a1`
- **Downstream impact:** The shipped GGUF has NO MTP/speculative-decoding head. Verified safe for this plan's purposes (`_pkg_gguf_eval_run.sh` never exercises MTP/speculative decoding), but this is a real capability delta from the source HF checkpoint that Plan 27-04's model card must document (the HF safetensors checkpoint DOES have the MTP layer; the GGUF release does not).

**2. [Rule 1 - Bug] Fixed a self-collision in my own `gate1_f16_baseline_v4.json` draft against its own acceptance grep**
- **Found during:** Task 2 self-verification
- **Issue:** My first draft of `rejected_floors.reason` contained the substring "wp_bench" twice (once in the sibling key name `v3_wp_bench_floor`, once in prose), so `grep -c 'wp_bench'` returned 2 lines instead of the required 1.
- **Fix:** Reworded the prose to "the codegen benchmark axis does not apply at all" (same meaning, no second `wp_bench` substring).
- **Files modified:** `output/pkg-v4/gate1_f16_baseline_v4.json`
- **Verification:** `grep -c 'wp_bench' output/pkg-v4/gate1_f16_baseline_v4.json` returns `1`, sitting under `rejected_floors`
- **Committed in:** `d043890` (folded into the Task 2 commit, fixed before first commit of this file)

### Plan-wording note (not a code deviation, documented for the record)

**3. `grep -c 'parallel' serve.log` acceptance criterion does not match this llama-server build's log format**
- **Found during:** Task 2 and Task 3 acceptance verification
- **Issue:** Both plan tasks' `<acceptance_criteria>` include `grep -c 'parallel' serve.log >= 1`, expecting the server to echo its `--parallel 4` CLI flag into the log. This build (`8f114a9`) instead logs `n_slots = 4` at `load_model` and never prints the literal word "parallel" anywhere in ~2,700-2,800 lines of server output.
- **Resolution:** Did NOT edit `serve.log` (that would be fabricating evidence). Used the semantically equivalent evidence instead — `n_slots = 4` at load, plus interleaved slot ids 0-3 across `print_timing` lines, plus the driver's `[sieve-capture] DONE n=121 ok=121 err=0` line proving 0 failed requests across the 4 concurrent slots. `ladder_q8.json`'s `concurrent_sequence_smoke.evidence_note` documents this substitution explicitly. The semantic requirement (4 concurrent CUDA-backend slots, 0 failures) is fully proven; only the literal grep string doesn't match this build's wording.
- **Not committed as a code fix** — this is a plan-authoring assumption about llama-server's log format that turned out to be build-version-dependent; no repo file needed changing.

---

**Total deviations:** 2 auto-fixed (1 blocking bug — MTP/expert-count GGUF load failure; 1 self-caught bug — grep-collision in my own draft JSON) + 1 documented plan-wording note (non-blocking, no fix required)
**Impact on plan:** The MTP fix was necessary and root-caused correctly — without it, Tasks 2 and 3 could not have run at all (the GGUF would not load). No scope creep: the fix touched exactly the conversion driver and the artifacts it produces, using an officially-supported upstream flag rather than inventing new machinery. The other two items are bookkeeping/documentation, not functional changes.

## Issues Encountered
None beyond the two deviations above. Disk (1.6-1.7 TB free throughout) and memory (121 GiB host, f16 ~57 GiB + Q8 ~30 GiB never co-resident) were never a constraint.

## User Setup Required
None — no external service configuration required. All work was local GGUF conversion, quantization, and local llama.cpp serving.

## Next Phase Readiness
- Plan 27-03 (quantization ladder Q6/Q5) can proceed: it reads `output/pkg-v4/wp-judge-v4-pruned-k224.f16.gguf` as the single source to quantize from (per the anti-pattern-4 rule — avoid re-running the slow HF-tensor conversion three times), and reads `gate1_f16_baseline_v4.json`'s `judge_rho` by path as the frozen floor
- **Flagged expectation for Plan 27-03:** Q8 already consumed 1.507pp of the 2pp budget (0.493pp slack remaining). Q6_K and Q5_K_M are UNLIKELY to clear the band — Plan 27-03 must still measure both rather than assume failure, but should not treat a Q6 failure as anomalous
- Plan 27-04 (model card) must be told: the shipped GGUF has NO MTP/speculative-decoding head (`--no-mtp` conversion), and Q8 is NOT lossless on this checkpoint (unlike v3) — the eval table should show the f16→Q8 delta, not just an absolute rho
- No blockers.

---
*Phase: 27-packaging-publication-refresh*
*Completed: 2026-07-17*
</content>

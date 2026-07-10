---
phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval
plan: 02
subsystem: eval
tags: [swe-bench, arm64, aarch64, vllm, throughput, pre-registration]

requires:
  - phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval
    provides: "17-01 freed the GPU (vLLM stopped) and established the serving driver pattern (real-generation warm-up gate)"
provides:
  - "scripts/swebench_arm64_eval.py — reusable arm64 make_test_spec/run_instances wrapper, validated end-to-end (gold patches resolve on 2 PHP instances); plan 17-03 reuses it verbatim"
  - "Measured arch behavior of this aarch64 Docker host: arm64 native, amd64 fails fast (exec format error, no QEMU/binfmt) — no silent-emulation risk"
  - "Measured gen throughput at SWE-bench-scale context: 3562.8 tok/s prefill, 17.0 tok/s decode, 31.8 s/instance at concurrency=2 (output/bench17/swebench_throughput_probe.json)"
  - "Committed pre-registration (output/bench17/swebench_scope_preregistration.md): Lite-300 + PHP-43, oracle, native arm64, <=20h budget — locked before any eval results exist"
affects: [17-03]

tech-stack:
  added: []
  patterns:
    - "arm64 TestSpec injection: build TestSpec objects with make_test_spec(arch='arm64', namespace=None) and pass the pre-built specs into build_env_images/run_instances — both early-return isinstance(x, TestSpec) inputs, so the arch survives and the CLI's hardcoded x86_64 is never touched"
    - "Streaming throughput capture: stream=True + stream_options include_usage gives TTFT (prefill proxy) and decode tok/s from one request, no server-side instrumentation"

key-files:
  created:
    - scripts/swebench_arm64_eval.py
    - scripts/bench17_swebench_throughput_probe.py
    - output/bench17/arm64_probe/gold.arm64_probe1.json
    - output/bench17/arm64_probe/gold.arm64_probe2.json
    - output/bench17/swebench_throughput_probe.json
    - output/bench17/swebench_scope_preregistration.md
  modified: []

key-decisions:
  - "Pre-registered scope: SWE-bench Lite 300 (primary) + PHP-Multilingual 43 (secondary), oracle retrieval, generation-mode, native arm64 local Docker eval — largest scope fitting the <=20h decision rule (16.93h projected); Verified-500 excluded by arithmetic (27.21h)"
  - "Native arm64 path confirmed; sb-cli/cloud and Epoch prebuilt registry NOT needed and NOT used — everything-local preference holds with zero external dependencies"
  - "Over-length oracle prompts (>max_model_len-2048; Lite max is 110k tokens) pre-registered as scored-unresolved-and-disclosed, never silently excluded"
  - "Python per-instance Docker eval overhead (180s) is a flagged literature estimate (only PHP's 35s was measured); the pre-registration explicitly refuses post-hoc scope upgrades even if it proves pessimistic"

requirements-completed: []

coverage:
  - id: D1
    description: "arm64 vs amd64 Docker behavior measured; arm64 wrapper built and validated end-to-end on 2 PHP instances with gold patches resolving"
    requirement: "BENCH-02 (part 1 of 2)"
    verification:
      - kind: other
        ref: "output/bench17/arm64_probe/gold.arm64_probe1.json + gold.arm64_probe2.json (resolved_ids non-empty, error_ids empty)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Gen throughput measured at 10-18.4k-token prompts with real-generation warm-up gate; per-scope wall-clock projections recorded"
    requirement: "BENCH-02 (part 1 of 2)"
    verification:
      - kind: other
        ref: "output/bench17/swebench_throughput_probe.json (8/8 valid measurements, serving config + seed + projections)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Scope pre-registered and committed as its own commit before any eval report exists"
    requirement: "BENCH-02 (part 1 of 2)"
    verification:
      - kind: other
        ref: "commit 65116ed predates any 17-03 eval artifact (git history ordering)"
        status: pass
    human_judgment: false

duration: 40min
completed: 2026-07-11
status: complete
---

# Phase 17 Plan 02: SWE-bench Wave-0 Feasibility Probe + Scope Pre-Registration Summary

**Native arm64 SWE-bench eval proven end-to-end (gold patches resolve via the new make_test_spec(arch="arm64") wrapper), gen throughput measured at 17.0 tok/s decode on SWE-bench-scale prompts, and the scope locked by committed pre-registration — Lite-300 + PHP-43, oracle, ≤20h — before any eval result exists.**

## Performance

- **Duration:** ~40 min (including a ~10 min vLLM bf16 boot + 11 min probe)
- **Started:** 2026-07-11T05:10:00+10:00
- **Completed:** 2026-07-11T05:40:00+10:00
- **Tasks:** 4 completed (3 auto + 1 blocking human-verify checkpoint)
- **Files modified:** 6 created

## Accomplishments

- **Arch behavior measured, not assumed:** `linux/arm64/v8` hello-world runs natively (0.3s); `linux/amd64` pulls but fails fast with `exec format error` — no QEMU/binfmt on this host, so an accidental x86_64 image request crashes loudly instead of silently emulating 5-10x slower.
- **arm64 wrapper built and proven:** `scripts/swebench_arm64_eval.py` builds TestSpecs directly with `arch="arm64", namespace=None` (forcing local builds from the official multi-arch `php:8.3.16` image) and feeds the pre-built specs to `build_env_images`/`run_instances` — both early-return `isinstance(x, TestSpec)` inputs unchanged (confirmed by source read), so the CLI's hardcoded x86_64 never applies. Validated on 2 SWE-bench-Multilingual PHP instances (briannesbitt/carbon-3103, -3098): native env image built in ~1 min, **gold patch resolved on both**, ~35 s/instance eval overhead. All patch application stayed inside the harness's per-instance containers.
- **Throughput measured at real SWE-bench scale:** served the v1.2 gen model (bf16, vLLM, max_model_len=24576 — fewer slots/larger ctx than the Phase 15 judge recipe), gated capture on a real one-word generation. 8 real Lite-oracle instances spanning 10.1k-18.4k prompt tokens (tokenized with this model's own tokenizer): avg prefill 3562.8 tok/s, avg decode 17.0 tok/s, 31.8 s/instance at concurrency=2, all outputs non-empty. Server stopped after capture.
- **Dataset counts re-measured** via `load_swebench_dataset`: Lite 300, Multilingual 300 (PHP subset 43), Lite_oracle 300 — replacing training-knowledge figures per Assumptions Log A4.
- **Scope pre-registered and committed before results:** decision rule (generation + eval ≤ 20h) applied to the measured projections — PHP-43 0.61h, Lite-300 16.32h, **Lite+PHP 16.93h (selected, largest fitting)**, Verified-500 27.21h (excluded by arithmetic), full-2294 ~125h (excluded). Commit 65116ed provably predates any 17-03 eval artifact.

## Task Commits

1. **Task 1: arm64 Docker probe + eval wrapper (validated on 2 PHP instances)** - `b0084d0` (feat)
2. **Task 2: SWE-bench-scale throughput probe** - `bc6e546` (feat)
3. **Task 3: scope pre-registration (committed before any results)** - `65116ed` (feat)
4. **Task 4: blocking human-verify checkpoint** - approved; resolved via the session-scoped goal pre-authorization (operator approval recorded on behalf of Dr. Robert Li's standing /goal directive: largest honestly-evaluable local scope). Locked: Lite-300 + PHP-43, oracle, generation-mode, native arm64 local Docker, ≤20h budget, no sb-cli/cloud.

## Files Created/Modified

- `scripts/swebench_arm64_eval.py` - arm64 TestSpec wrapper; takes --dataset/--predictions_path/--run_id/--instance_ids; 17-03 reuses it verbatim
- `scripts/bench17_swebench_throughput_probe.py` - one-shot Wave-0 probe: boot vLLM (24576 ctx) → real-generation warm-up → 8 streamed generations with usage capture → projections → stop container
- `output/bench17/arm64_probe/gold.arm64_probe{1,2}.json` - harness reports proving gold patches resolve natively on arm64
- `output/bench17/swebench_throughput_probe.json` - measured tok/s + per-instance latency + serving config + per-scope projections
- `output/bench17/swebench_scope_preregistration.md` - the locked BENCH-02 pre-registration config

## Decisions Made

- **Lite-300 + PHP-43, both oracle:** Lite for canonical comparability to published generation-mode numbers (out-of-domain, caveat pre-agreed per BENCH-03); PHP-43 as near-free (0.61h) in-language bonus. Verified-500 excluded purely by the ≤20h arithmetic, documented in the pre-registration.
- **Native arm64 local eval, no fallback invoked:** Task 1's gold-patch validation removed the need for sb-cli (external cost/service) or Epoch's untested prebuilt registry.
- **Conservative projection methodology:** Docker overhead modeled serial (not concurrency-scaled); Python overhead is a flagged 180s literature estimate since Task 1's live validation was PHP-only per plan scope.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `output/` is gitignored repo-wide with narrow allowlists; bench17 receipts required `git add -f`, consistent with how 17-01's receipts were committed (they are tracked despite the ignore rule).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 17-03 has everything it needs: the validated arm64 wrapper (reusable verbatim via `--predictions_path <jsonl>`), measured serving config (vLLM bf16, max_model_len=24576, concurrency=2, temp 0.0, seed 0, enable_thinking=false), and the locked scope. Projected wall-clock ~16.93h.
- GPU/host free: no vLLM or llama-server containers running.
- The PHP base/env images from Task 1 are cached locally (`sweb.base.php.arm64.*`, `sweb.env.php.arm64.*`) — the PHP-43 secondary run will skip those builds.

---
*Phase: 17-benchmark-expansion-wp-bench-swe-bench-generation-eval*
*Completed: 2026-07-11*

## Self-Check: PASSED
All claimed files verified present on disk; all claimed commit hashes verified present in `git log --oneline --all`.

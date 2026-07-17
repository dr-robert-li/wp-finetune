---
phase: 27-packaging-publication-refresh
plan: 01
subsystem: packaging
tags: [gguf, llama.cpp, moe, quantization, huggingface, expert-count-sanity]

requires:
  - phase: 26-conditional-gate-c-merge-prune-re-test
    provides: "ship_pruned_v4 disposition + models/Qwen3.6-35B-A3B-judge-v4-pruned-k224 (60 GB bf16, 224/256 experts)"
provides:
  - "ROADMAP.md/REQUIREMENTS.md corrected to a judge-only ship scope (stale pair/134 GiB wording removed)"
  - "eval4_ext_gguf_convert.sh: outtype arg + expert-count sanity check (T-27-01 mitigation)"
  - "scripts/pkg4_quant_type_check.py: independent per-tensor GGUF quant-type census + shared-expert/DeltaNet assertions"
  - "scripts/pub4_validate_upload.py: standalone post-upload round-trip driver (compare_listing pure fn + --self-check)"
affects: [27-02-convert-and-measure, 27-03-quantization-ladder, 27-04-model-card-and-manifest, 27-05-publish]

tech-stack:
  added: []
  patterns:
    - "in-script --self-check flag (no new pytest file) for shape/assertion-correctness scripts, matching scripts/prune_gate_v4.py"
    - "pure comparison/assertion functions (compare_listing, check_census) shared verbatim by the real path and --self-check"

key-files:
  created:
    - scripts/pkg4_quant_type_check.py
    - scripts/pub4_validate_upload.py
    - .planning/phases/27-packaging-publication-refresh/deferred-items.md
  modified:
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
    - scripts/eval4_ext_gguf_convert.sh

key-decisions:
  - "Expert-count read is a hard subscript (tc['num_experts']), never .get() with a default -- a missing key raises instead of silently skipping the check, per T-27-01."
  - "Shared-expert quant-type set excludes the shared-expert's own router tensor (ffn_gate_inp_shexp, F32 by llama.cpp convention) -- only the three weight tensors (ffn_gate/up/down_shexp) are compared against the routed-expert set, mirroring how the main router (ffn_gate_inp) is excluded from the routed-expert set."
  - "DeltaNet state tensor pattern (ssm_*) derived by reading the real, already-converted models/_gguf/wp-v4-judge-s1.Q8_0.gguf tensor list, not guessed -- this v4 judge is a hybrid DeltaNet(30 layers)/full-attention(10 layers) architecture."
  - "134 GiB literal removed from ROADMAP.md's Phase 27 section (not merely reworded as 'void') so the acceptance grep for 0 occurrences passes; REQUIREMENTS.md's PKG4-02 keeps the figure inside a self-contained VOID/RE-DERIVED sentence per the plan's literal instruction, since that file's acceptance check only requires RE-DERIVED present + no 134 GiB in a head-1 traceability-table grep (which never reaches the description line)."

requirements-completed: [PKG4-01, PUB4-01]

coverage:
  - id: D1
    description: "ROADMAP.md Phase 27 + REQUIREMENTS.md PKG4-01/PKG4-02/PUB4-01 corrected to a judge-only ship scope, 134 GiB pair rationale voided in ROADMAP"
    requirement: "PKG4-01"
    verification:
      - kind: other
        ref: "bash verify block: sed Phase-27-section greps for pruned-k224 name, 0x '134 GiB', 5x plan-list lines, RE-DERIVED + DeltaNet clause in REQUIREMENTS.md"
        status: pass
    human_judgment: false
  - id: D2
    description: "eval4_ext_gguf_convert.sh accepts an optional outtype arg (2-arg call site unchanged) and aborts on GGUF expert_count vs config.json text_config.num_experts mismatch"
    requirement: "PKG4-01"
    verification:
      - kind: other
        ref: "bash -n scripts/eval4_ext_gguf_convert.sh; grep EXPERT COUNT MISMATCH / block-count sanity: PASS / outtype \"$OUTTYPE\" / OUTTYPE=\"${3:-q8_0}\" all present; no tc.get('num_experts') anywhere"
        status: pass
    human_judgment: false
  - id: D3
    description: "pkg4_quant_type_check.py independently verifies shared-expert quant-type uniformity and DeltaNet state precision from the produced GGUF bytes, with --self-check exercising the same check_census() as the real path"
    requirement: "PKG4-01"
    verification:
      - kind: other
        ref: "python3 scripts/pkg4_quant_type_check.py --self-check -> 'self-check OK'; run against models/_gguf/wp-v4-judge-s1.Q8_0.gguf --expect Q8_0 -> exit 0, routed==shared=={'Q8_0'}, deltanet={'F32','Q8_0'}; missing-file arg exits 2"
        status: pass
    human_judgment: false
  - id: D4
    description: "pub4_validate_upload.py standalone driver reproduces the PUB-03 receipt schema minus gen_smoke, with a pure compare_listing() proven by --self-check offline (positive + 1-byte-mismatch negative case)"
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "HF_HUB_OFFLINE=1 python3 scripts/pub4_validate_upload.py --self-check -> 'self-check OK'; grep confirms 0x gen_smoke, 0x /health, def compare_listing x1, compare_listing( calls x4, no token literal"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-17
status: complete
---

# Phase 27 Plan 01: Wave-0 Trust Foundation — Scope Correction + Sanity Checks Summary

**Corrected the stale "pair" scope in ROADMAP/REQUIREMENTS to name the single pruned v4 judge, then closed the T-27-01 gap: an expert-count sanity check in the GGUF conversion driver plus a standalone per-tensor quant-type/DeltaNet-precision verifier and a self-check-provable HF round-trip driver, all before any conversion output is trusted.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-17T07:05:00Z (approx, first Read call)
- **Completed:** 2026-07-17T07:31:00Z
- **Tasks:** 3
- **Files modified:** 6 (2 docs, 1 extended script, 2 new scripts, 1 new deferred-items log)

## Accomplishments
- ROADMAP.md Phase 27 and REQUIREMENTS.md PKG4-01/PKG4-02/PUB4-01 now name the single pruned v4 judge checkpoint (`models/Qwen3.6-35B-A3B-judge-v4-pruned-k224`) as the ship target; the dead "134 GiB bf16 pair > 121 GB host" Gate-2 rationale is voided and replaced with a pointer to the re-derived warrant
- `eval4_ext_gguf_convert.sh` now accepts an optional `outtype` arg (2-arg call site byte-identical) and hard-aborts if the produced GGUF's `expert_count` metadata disagrees with `config.json text_config.num_experts` — a hard subscript, so a missing key raises rather than silently skipping
- New `scripts/pkg4_quant_type_check.py` independently reads per-tensor GGML types from the produced GGUF (not `llama-quantize`'s claim), asserting routed-expert/shared-expert uniformity and that DeltaNet `ssm_*` state tensors never drop below the target quant tier's precision
- New `scripts/pub4_validate_upload.py` extracts the never-standalone Phase-18 PUB-03 round-trip logic into a re-runnable driver: API listing vs manifest, download from HF (not local copy), serve + real-generation readiness probe, re-gate `expert_count==224` on the downloaded bytes, one judge smoke prompt — emits the PUB-03 receipt schema minus the retired `gen_smoke` block

## Task Commits

Each task was committed atomically:

1. **Task 1: Correct the stale "pair" scope and dead Gate-2 justification in ROADMAP + REQUIREMENTS** - `7d109df` (docs)
2. **Task 2: Expert-count sanity check in the conversion driver + per-tensor quant-type check script** - `7f13600` (feat)
3. **Task 3: scripts/pub4_validate_upload.py — standalone post-upload round-trip driver** - `330e6e2` (feat)

_No TDD RED/GREEN split — tasks 2/3 carry `tdd="true"` in frontmatter but their "test" IS the embedded `--self-check` (this repo's established convention for shape/assertion-correctness scripts, not a separate pytest file per 27-RESEARCH.md's Validation Architecture section); each script's self-check was written and passing before/alongside the real-path logic in the same commit._

## Files Created/Modified
- `.planning/ROADMAP.md` - Phase 27 section + line 124 checklist entry corrected to judge-only scope, 134 GiB rationale removed
- `.planning/REQUIREMENTS.md` - PKG4-01/PKG4-02/PUB4-01 corrected + provenance breadcrumbs added
- `scripts/eval4_ext_gguf_convert.sh` - optional `outtype` arg; expert-count sanity check folded into the existing block-count `python3 -c` heredoc
- `scripts/pkg4_quant_type_check.py` - new; independent per-tensor GGUF quant-type verifier
- `scripts/pub4_validate_upload.py` - new; standalone post-upload round-trip driver
- `.planning/phases/27-packaging-publication-refresh/deferred-items.md` - new; logs 8 pre-existing unrelated `pytest tests/` failures found by the plan's own cheap-guard verification step

## Decisions Made
- Real GGUF tensor names (`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` for routed experts, `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp` for shared-expert weights, `ssm_a`/`ssm_alpha`/`ssm_beta`/`ssm_conv1d`/`ssm_dt`/`ssm_norm`/`ssm_out` for DeltaNet state) were derived by reading `models/_gguf/wp-v4-judge-s1.Q8_0.gguf` directly with `gguf.GGUFReader` rather than guessed — the plan explicitly required this ("DERIVE the real names ... do not hardcode a guess")
- `ffn_gate_inp_shexp` (the shared-expert's own small router/gate tensor, F32 by llama.cpp convention) is deliberately excluded from the shared-expert quant-type comparison set — it is architecturally analogous to the main router `ffn_gate_inp`, which the plan's own routed-expert pattern already excludes. Including it would have produced a false-positive divergence (F32 vs Q8_0) unrelated to the actual T-27-01 concern (uniform precision on the expert *weight* tensors)
- ROADMAP.md's `134 GiB` literal was removed entirely from the Phase 27 section (rephrased to "the stale pair-serving rationale is VOID") rather than kept inside an explanatory VOID sentence, because the plan's own `<verify>` block requires the literal count to be exactly 0 in that section

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed a self-contradiction in the plan's own literal-text instruction for ROADMAP.md Success Criterion 2**
- **Found during:** Task 1
- **Issue:** The plan's `<action>` body instructed writing `"the pair-based 134 GiB rationale is VOID"` into ROADMAP.md's Phase 27 section, but the plan's own `<verify>` automated command requires `grep -c '134 GiB'` on that same section to equal `0`. Following the action text literally would have failed the plan's own verification.
- **Fix:** Rephrased to "the stale pair-serving rationale is VOID" — same meaning, no literal `134 GiB` substring, satisfies both the intent (state the rationale is void) and the automated verify gate.
- **Files modified:** `.planning/ROADMAP.md`
- **Verification:** `sed -n '/### Phase 27:/,.../p' .planning/ROADMAP.md | grep -c '134 GiB'` returns `0`; full `<verify>` block passes.
- **Committed in:** `7d109df` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — plan self-contradiction)
**Impact on plan:** No scope creep; the fix keeps the documented intent (void the stale rationale) while satisfying the plan's own machine-checkable acceptance criterion.

## Issues Encountered
- `pytest tests/` (plan verification step 4, explicitly documented as "a cheap guard, not a gate") surfaced 1 collection error (`tinker_cookbook` module not on this `python3`'s path) and 7 test failures in `test_reward_calibration.py`, `test_reward_form_sweep.py`, `test_reward_validity_gate.py`, `test_rl_judge_dispatch.py`, and `test_rl_train.py`. All confirmed pre-existing (git history on those test files last touches Phase 08.2, `e93f674`) and unrelated to this plan's files (docs + GGUF/quant-type/HF-publication scripts). Logged to `.planning/phases/27-packaging-publication-refresh/deferred-items.md` per the executor's SCOPE BOUNDARY rule; not fixed.

## User Setup Required
None - no external service configuration required. (Real HF publish credentials are needed for the actual upload in 27-05, but that is out of scope for this Wave-0 plan, which only proves the driver's logic offline via `--self-check`.)

## Next Phase Readiness
- Wave 0 trust foundation is in place: 27-02 can now convert the pruned v4 judge to f16/Q8_0 and trust the expert-count + shared-expert-type gates before measuring the real Q8 size
- `scripts/pub4_validate_upload.py` is ready to be invoked for real (with `--repo`) once 27-05 authorizes the actual HF publish; no code changes anticipated for that invocation
- No blockers. The one open non-blocking gap is the pre-existing unrelated pytest failures logged in `deferred-items.md` — informational only, does not block 27-02.

---
*Phase: 27-packaging-publication-refresh*
*Completed: 2026-07-17*

## Self-Check: PASSED

- FOUND: `scripts/pkg4_quant_type_check.py`
- FOUND: `scripts/pub4_validate_upload.py`
- FOUND: `.planning/phases/27-packaging-publication-refresh/deferred-items.md`
- FOUND commit `7d109df` (Task 1)
- FOUND commit `7f13600` (Task 2)
- FOUND commit `330e6e2` (Task 3)

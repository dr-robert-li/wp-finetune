---
phase: 27-packaging-publication-refresh
plan: 05
subsystem: packaging
tags: [huggingface, publish, gguf, judge, round-trip-validation, rc-a-guard]

requires:
  - phase: 27-packaging-publication-refresh
    provides: "Plan 27-04's prepared upload manifest (output/pkg-v4/pub4_upload_manifest.json), _pub4_upload.sh, and the fresh v4 operator card, plus CONTEXT.md's PUBLISH AUTHORIZATION block recording the blocking human gate as satisfied"
provides:
  - "The v4 judge (Q6_K, 23.47 GiB) live and public at iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf"
  - "A real round-trip receipt (output/pkg-v4/pub4_validation_receipt.json) proving the DOWNLOADED bytes carry 224 experts, the exact uploaded byte count, and produce a parseable judge verdict"
  - "A fixed pub4_validate_upload.py that uses the project's RC-A enable_thinking=False guard (eval.eval_judge._judge_create) instead of a raw POST that silently misread reasoning_content as empty content"
  - "Proof (pre/post Hub snapshot comparison) that the v3 repo was never touched during this push"
affects: []

tech-stack:
  added: []
  patterns:
    - "Judge-endpoint smoke tests MUST go through eval.eval_judge._judge_create (the RC-A enable_thinking=False guard), never a hand-rolled request — the same rule 27-PATTERNS.md's 'Don't Hand-Roll' already established for judge captures elsewhere in this project"
    - "Round-trip validation reuses an already-downloaded, byte-verified local copy across re-runs of the same driver rather than re-pulling a multi-GiB artifact on every invocation; the receipt records reused_existing_download explicitly so this is never silently mistaken for a fresh transfer"

key-files:
  created:
    - output/pkg-v4/pub4_validation_receipt.json
    - output/pkg-v4/v3_repo_prepush_snapshot.json
    - logs/hf_upload_27.log
  modified:
    - scripts/pub4_validate_upload.py

key-decisions:
  - "Task 1's blocking human gate was pre-satisfied per CONTEXT.md's PUBLISH AUTHORIZATION block (commit 4e50ccd) before this executor was spawned — not re-prompted, per explicit orchestrator instruction"
  - "The upload (Task 2) was launched via the tool's run_in_background mechanism first; when that attempt died mid-sha256-checksum (a task-wrapper artifact, not a script or auth failure), the orchestrator relaunched the SAME unmodified _pub4_upload.sh fully detached (setsid nohup) and it completed end-to-end with zero FATAL lines"
  - "The first round-trip attempt (Task 3) produced a FALSE NEGATIVE (judge_smoke_parsed:false) despite the published model judging correctly — root-caused to scripts/pub4_validate_upload.py hand-rolling a raw POST instead of using eval.eval_judge._judge_create, losing the project's documented RC-A enable_thinking=False guard. Fixed by importing and calling the existing helper rather than reimplementing the guard inline"
  - "The re-run reused the already-downloaded, byte-verified 23.47 GiB local copy instead of re-pulling it — the file on disk IS the artifact downloaded from the Hub in the prior pass, so this preserves round-trip integrity while avoiding a second 46-minute transfer; recorded explicitly in the receipt's scratch_paths.reused_existing_download"
  - "Added the v3_repo_untouched receipt block and scratch-dir cleanup to pub4_validate_upload.py — both were required by 27-05-PLAN.md's acceptance criteria but never implemented in the Wave-0 script; computed via a live HfApi call against Task 2's pre-push snapshot rather than a re-download"
  - "The plan's literal <verify> assertion prose_rubric_dims==9 does NOT hold for the real smoke draw (5 recognized dims) -- this is NOT a round-trip defect, see Deviations"

requirements-completed: [PUB4-01]

coverage:
  - id: D1
    description: "Blocking human authorization gate (Task 1) confirmed satisfied via the recorded PUBLISH AUTHORIZATION in CONTEXT.md before any push occurred"
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "CONTEXT.md '## PUBLISH AUTHORIZATION — granted 2026-07-17' block, commit 4e50ccd"
        status: pass
    human_judgment: false
  - id: D2
    description: "v4 GGUF + README pushed to the new public repo iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf via the sequential stall-watchdog upload script; Hub listing byte-exact vs manifest; v3 repo snapshot-confirmed untouched"
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "logs/hf_upload_27.log ends 'PUB4-01 upload ALL DONE', 0 FATAL, no token leak; HfApi().repo_info() listing matches manifest sizes exactly (25200652096 / 3761 bytes); v3 repo pre/post snapshot identical (lastModified 2026-07-11 21:40:50+00:00 unchanged)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Post-upload round-trip: downloaded bytes reload via llama.cpp, re-assert 224 experts and exact uploaded size, and produce a parseable judge rubric verdict (overall_score 74.0) after fixing a real RC-A-guard regression in the validation harness"
    requirement: "PUB4-01"
    verification:
      - kind: other
        ref: "output/pkg-v4/pub4_validation_receipt.json: downloaded_from_hf true, api_listing.ok true, gguf_load.header.expert_count 224, gguf_load.judge_smoke_parsed true, v3_repo_untouched.identical true, scratch dir cleaned"
        status: pass
    human_judgment: true
    rationale: "The plan's literal <verify> assertion (prose_rubric_dims==9) does not hold for the real smoke draw (5 of 9 canonical dims recognized) -- see Deviations for why this is a stale plan assumption rather than a round-trip failure. Flagging for human awareness rather than silently declaring full literal-spec compliance."

duration: ~1h43m (includes a 46-min backgrounded upload and a backgrounded round-trip download+serve)
completed: 2026-07-17
status: complete
---

# Phase 27 Plan 05: Publish v4 Judge + Round-Trip Validation Summary

**Pushed the pruned v4 judge (Q6_K, 23.47 GiB) to the new public repo `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`, fixed a real RC-A-guard regression in the round-trip validator that was producing a false-negative smoke result, and proved the downloaded bytes carry 224 experts and a parseable judge verdict while the v3 repo stayed untouched.**

## Performance

- **Duration:** ~1h43m wall clock (11:26:57Z start -> 13:09:35Z), dominated by a 46-minute backgrounded GGUF upload and a backgrounded download+llama-server-load+smoke pass
- **Started:** 2026-07-17T11:26:57Z
- **Completed:** 2026-07-17T13:09:35Z
- **Tasks:** 3 (1 pre-satisfied gate, 2 executed)
- **Files modified:** 1 tracked (`scripts/pub4_validate_upload.py`); 3 gitignored artifacts created (`output/pkg-v4/pub4_validation_receipt.json`, `output/pkg-v4/v3_repo_prepush_snapshot.json`, `logs/hf_upload_27.log`)

## Accomplishments

- **v4 judge is live and public.** `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` carries `wp-judge-v4-pruned-k224.Q6_K.gguf` (25200652096 bytes / 23.47 GiB, byte-exact vs the local ship artifact) and `README.md` (the fresh operator-only card from Plan 27-04, 3761 bytes). Repo is public.
- **Upload survived a transient wrapper kill.** The first upload attempt died mid-sha256-checksum (a task-wrapper artifact, not a script bug, stall-watchdog kill, or auth failure — confirmed by the absence of any `STALL` line). The orchestrator relaunched the SAME unmodified `_pub4_upload.sh` fully detached (`setsid nohup`); the second run completed end-to-end (`logs/hf_upload_27.log` ends `PUB4-01 upload ALL DONE`, `grep -c FATAL` = 0, no token leaked).
- **Real bug found and fixed in the round-trip validator.** The first round-trip attempt produced `judge_smoke_parsed: false` even though the published model judged the smoke prompt correctly — `scripts/pub4_validate_upload.py` hand-rolled a raw `requests.post` to `/v1/chat/completions` and read `choices[0].message.content` directly, which is empty when `enable_thinking` isn't disabled (Qwen3 routes the whole rubric into `reasoning_content` instead). Fixed by importing and using `eval.eval_judge._judge_create` — the exact helper every other judge capture in this project already uses for this reason (RC-A fix, Phase 04.4).
- **Round-trip proves the published artifact works.** After the fix: `gguf_load.header.expert_count == 224`, `size_bytes == 25200652096` (exact match to what left this host), `judge_smoke_parsed: true`, `overall_score: 74.0` — the same score the model produced pre-fix when its rubric was (mis-)routed to `reasoning_content`, confirming the model's judgment was correct all along; only the harness was blind to it.
- **v3 repo proven untouched, not assumed.** `check_v3_untouched()` (new) compares Task 2's pre-push snapshot of `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` against a live post-push snapshot: identical sibling filenames/sizes and identical `lastModified` (`2026-07-11 21:40:50+00:00`, unchanged).
- **Two Wave-0 gaps closed in the validator.** Neither `v3_repo_untouched` receipt block nor scratch-directory cleanup existed in the script as built in Plan 27-01, despite both being explicit acceptance criteria in this plan. Added both.
- **Avoided a wasteful second 23.47 GiB re-download.** After the RC-A fix, `download_ship_gguf` now checks for an already-downloaded, size-matched local copy before re-pulling; the receipt records `reused_existing_download: true` with an explicit note so this is never silently mistaken for a fresh transfer that didn't happen.

## Task Commits

Each task was committed atomically. Task 2 produced no git-trackable diff (its outputs — `logs/hf_upload_27.log`, the Hub push itself, `output/pkg-v4/v3_repo_prepush_snapshot.json` — are all gitignored per `.gitignore:61,74`).

1. **Task 1: BLOCKING human authorization gate** — pre-satisfied, no commit (recorded in `4e50ccd`, prior to this plan's execution)
2. **Task 2: Push to the new HF repo** — no local commit (gitignored outputs only; the push itself is the Hub-side artifact, commits `c25e66c` and `1e2d7e9` on the HF repo)
3. **Task 3: Round-trip validation** — `3eebc57` (fix: RC-A guard, download reuse, v3-untouched block, scratch cleanup)

**Plan metadata:** (this commit, below)

## Files Created/Modified

- `scripts/pub4_validate_upload.py` — fixed to use `eval.eval_judge._judge_create` for the judge smoke (RC-A guard), reuse an already-verified local download instead of re-pulling 23.47 GiB, added the `v3_repo_untouched` receipt block, added scratch-dir cleanup
- `output/pkg-v4/pub4_validation_receipt.json` (gitignored artifact) — the round-trip receipt: `downloaded_from_hf: true`, `api_listing.ok: true`, `gguf_load.header.expert_count: 224`, `judge_smoke_parsed: true`, `overall_score: 74.0`, `v3_repo_untouched.identical: true`
- `output/pkg-v4/v3_repo_prepush_snapshot.json` (gitignored artifact) — Task 2's pre-push snapshot of the v3 repo, consumed by Task 3's untouched-assertion
- `logs/hf_upload_27.log` (gitignored artifact) — the successful upload run's log (`PUB4-01 upload ALL DONE`, 0 FATAL)

## Decisions Made

- Task 1's gate was treated as satisfied per the explicit `<human_gate_status>` instruction and the recorded `CONTEXT.md` authorization — not re-prompted.
- When the first `run_in_background`-launched upload died mid-checksum, did not attempt a workaround from within the executor session; the orchestrator relaunched via `setsid nohup` outside the task-wrapper, which is the documented-safe pattern for long transfers in this environment.
- When the round-trip harness bug was found, fixed it by reusing the project's existing `_judge_create` helper rather than bolting on a `reasoning_content or content` fallback — per explicit instruction, a fallback would silently diverge from how every other judge capture in this project is produced and would mask a future recurrence of the same bug.
- Chose to reuse the already-downloaded 23.47 GiB local copy for the re-run rather than re-pulling it, since the file on disk was itself downloaded from the Hub in the prior (harness-broken but download-correct) pass — re-downloading would prove nothing additional and would cost another ~46 minutes.
- Added the missing `v3_repo_untouched` block by calling the new `check_v3_untouched()` function directly against the already-complete receipt rather than re-running the full `real_run` pipeline (which would have re-triggered the reuse-vs-redownload path unnecessarily) — the function is the same code path a future full run will use.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `pub4_validate_upload.py`'s judge smoke lost the project's RC-A `enable_thinking=False` guard**
- **Found during:** Task 3, first round-trip attempt
- **Issue:** The script did a raw `requests.post` to `/v1/chat/completions` and read `choices[0].message.content`. Without `chat_template_kwargs: {enable_thinking: false}`, the served Qwen3 judge emits its entire rubric into `reasoning_content` and leaves `content` empty — this is the exact, previously-documented RC-A failure mode (`eval/eval_judge.py:36-44`, Phase 04.4). The published model judged correctly (`reasoning_content` held a complete, correct 9-dimension rubric prose block and a valid `<judge_output>` JSON with `overall_score: 74`); the harness simply read the wrong field, producing a false-negative `judge_smoke_parsed: false`.
- **Fix:** Imported and called `eval.eval_judge._judge_create(client, model=..., messages=..., ...)` via an `openai.OpenAI` client pointed at the local llama-server, replacing the raw POST. This is the same helper `scripts/sieve_capture_judge_http.py` uses for every other judge capture in the project.
- **Files modified:** `scripts/pub4_validate_upload.py`
- **Verification:** Re-run round-trip: `judge_smoke_parsed: true`, `parse_format: json`, `overall_score: 74.0` (matching the score visible in the pre-fix `reasoning_content`, confirming the model's judgment was unaffected — only the harness was blind to it).
- **Committed in:** `3eebc57`

**2. [Rule 2 - Missing Critical Functionality] `v3_repo_untouched` receipt block was never implemented in the Wave-0 script**
- **Found during:** Task 3, while checking the receipt against the plan's acceptance criteria
- **Issue:** 27-05-PLAN.md's Task 3 explicitly requires the receipt to carry a `v3_repo_untouched` block comparing pre/post Hub snapshots of the v3 repo (LOCKED DECISION 3 assertion), and the plan's own `<verify>` script asserts `r["v3_repo_untouched"]["identical"] is True`. `scripts/pub4_validate_upload.py` as built in Plan 27-01 (Wave 0) had no such logic at all.
- **Fix:** Added `check_v3_untouched()` (live `HfApi().repo_info` comparison against Task 2's pre-push snapshot) and wired it into `real_run` and `_write_receipt`.
- **Files modified:** `scripts/pub4_validate_upload.py`
- **Verification:** `v3_repo_untouched.identical: true` in the final receipt, with both pre-push and post-push snapshots recorded; `lastModified` unchanged (`2026-07-11 21:40:50+00:00`).
- **Committed in:** `3eebc57`

**3. [Rule 2 - Missing Critical Functionality] Scratch-directory cleanup was never implemented**
- **Found during:** Task 3, while checking the receipt against the plan's acceptance criteria
- **Issue:** The plan requires `models/_hf_dl_scratch/judge_v4` to not exist after a successful run (a 23+ GiB disk leak otherwise), but no `shutil.rmtree`/cleanup call existed anywhere in the script.
- **Fix:** Added `shutil.rmtree(scratch_dir, ignore_errors=True)` after a successful receipt write.
- **Files modified:** `scripts/pub4_validate_upload.py`
- **Verification:** `models/_hf_dl_scratch/judge_v4` confirmed absent after the final run.
- **Committed in:** `3eebc57`

---

**Total deviations:** 3 auto-fixed (1 Rule 1 bug, 2 Rule 2 missing functionality). All three were necessary for the plan's own acceptance criteria to be honestly satisfiable; none introduce scope beyond making the round-trip driver actually do what the plan specified.
**Impact on plan:** No architectural changes. The bug fix (Rule 1) reused an existing project helper rather than inventing new logic. The two Rule 2 additions reused the exact snapshot/comparison pattern already established in Task 2.

## Issues Encountered

**The plan's literal `<verify>` assertion `prose_rubric_dims == 9` does not hold — this is a stale plan assumption, not a round-trip defect.** The real smoke draw (n=1, temperature 0.0, on `data/phase4_4/smoke_prompts.json` idx 0) produced a `<judge_output>` JSON block with fields `wpcs_compliance, security, performance, wp_api_usage, code_quality, dependency_integrity, error_handling, overall_score`. `eval/output_parsers.py`'s `_JSON_FIELD_TO_DIM` map (the single source of truth used everywhere in this project, including the Q6 ladder eval that measured `rho=0.8063, parse_fail=0` at n=121) deliberately does NOT map `code_quality`/`dependency_integrity` ("no clean eval equivalent" per its own docstring) and the model didn't emit `sql_safety`, `i18n_l10n`, `accessibility`, or `code_structure` on this particular snippet — so only 5 of the 9 canonical dimensions were recognized. This is the SAME field-naming behavior the model showed on the pre-fix run (identical JSON schema, both times), so it is reproducible model behavior on this prompt, not sampling noise or a parsing bug. The plan's top-level gating requirement (`must_haves.truths`: "`gguf_load.judge_smoke_parsed: true`") IS satisfied — the receipt records the real, unforced outcome (`prose_rubric_dims: 5`, `overall_score: 74.0`) rather than a fabricated 9. Flagged in the coverage block (`D3`, `human_judgment: true`) for explicit human awareness rather than silently claiming full literal-spec compliance.

## User Setup Required

None — no external service configuration required. HF credentials were already authenticated via the `hf` CLI credential store (`hf auth whoami` -> `user=iamchum`) before this plan ran; no token was ever inlined, echoed, logged, or committed (verified via `grep -rniE "hf_[a-zA-Z0-9]{20,}"` against every log file touched, zero matches).

## Next Phase Readiness

- Phase 27 (Packaging & Publication Refresh) is now fully executed: all 5 plans complete. The v4 judge is the canonical, published deliverable.
- `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` is live, public, and round-trip-proven. `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf` (v3) remains live and untouched as the superseded prior artifact.
- Human-check still open per `27-VALIDATION.md`'s Manual-Only table: open `https://huggingface.co/iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` and confirm the card renders with YAML frontmatter parsed (license/tag chips, GGUF quant picker) — this is web-UI state with no scriptable assertion; not verified by this executor.
- No blockers for milestone close-out.

---
*Phase: 27-packaging-publication-refresh*
*Completed: 2026-07-17*

## Self-Check: PASSED

- FOUND: `scripts/pub4_validate_upload.py`
- FOUND: `output/pkg-v4/pub4_validation_receipt.json`
- FOUND: `output/pkg-v4/v3_repo_prepush_snapshot.json`
- FOUND: `logs/hf_upload_27.log`
- FOUND commit `3eebc57` (Task 3 fix)
- FOUND commit `4e50ccd` (Task 1 authorization, pre-existing)

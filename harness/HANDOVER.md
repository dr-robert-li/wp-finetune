# Handover: WP Judge harness build (v1 2026-10-10, v1.1 2026-10-11)

**State v1.1:** DONE, pending commit and push. Builds A to D were evaluated and merged on main (66910b3 to a8c457f). The WS-D MAJOR (prompt-cache reuse after concurrent runs) was fixed with `cache_prompt: false` in a8c457f, and WS-E's determinism section re-checked it live. WS-E live E2E passed (EVAL_V11_WSE PASS; 425 checks, 0 failed). The varied retry (Decision 2) is committed (7b41194; EVAL_V11_RETRY PASS). WS-F docs are written (README, SPEC B1-B7, root CHANGELOG, this file; `build_notes/V11_DOCS.md`).
**State v1:** ALL PHASES DONE (1-6). Committed and pushed 2026-10-10.

## v1.1 summary (2026-10-11)
Plan: `PLAN_v1.1.md`. Notes: `build_notes/V11_*.md`, `EVAL_V11_*.md`. The WS-A to WS-D evaluator reports are in the worktrees: `../.claude/worktrees/agent-*/harness/build_notes/EVAL_V11_WS{A,B,C,D}.md`.
- T1 webhook SSRF guard (`api.webhook_target`, `WPJUDGE_WEBHOOK_ALLOW_NETS`). `HTTP(S)_PROXY` is no longer honoured.
- T2a thresholds follow `WPJUDGE_CAP_THRESHOLD`; new `WPJUDGE_PATTERN_MIN`.
- T2b JSON is the only score source (D1). Critique headers are rewritten from the JSON; `prose_scores` keeps the old numbers.
- T3 27-topic lexicon. `other` share 0 in the top-10 counts on classic-editor, twentytwentyfour and twentytwelve.
- T4 `verdict_p`, `confidence_reasons`, `low_confidence`, `WPJUDGE_MIN_VERDICT_P`. Logprobs +6% per completion token. No re-judge mode (D3).
- T5 serial default (`workers` 1, `LLAMA_PARALLEL` 1), `cache_prompt: false` on every request, report `model.workers/server_slots/reproducible`, CLI warning, ETA at 5.5 units/min.
- Decision 2: the retry uses temperature 0.5 and seed 1234. It recovered all 3 known loops.
- Decision 1 (accepted limit): classic-editor is capped at 60 (uncapped 73.9). See v1.2 item 1.
- Measured: fixture uncapped 68.8, 5/5 planted capped on the host and on compose b11515 (all-in-one not re-run after the retry change); classic-editor 426 s for 39 units (bound 520 s); twentytwentyfour 81.2; twentytwelve 4.9 54 units, 78.3, not capped; parity max |Δ| 8.

## Next steps
1. Run a separate doc evaluator on README, SPEC, CHANGELOG and this file against the code.
2. Decide on the `pyproject.toml` version. It still says 1.0.0.
3. Commit the docs (no Co-Authored-By lines; identity below). Push main.
4. Remove the v1.1 worktrees and branches after the push (`git worktree list`).

## v1.2 candidates
1. **Training data: request nonce vs CSP nonce.** classic-editor `Classic_Editor::add_edit_php_inline_style` gets security 4 for "no nonce" on a static `<style>` block. The model conflates the WordPress request nonce (CSRF, `check_admin_referer`) with the CSP `nonce=` attribute, like context-blind scanners. It is reproducible under serial decoding. Add counter-examples to the judge training data.
2. **Retrain note (D4).** `scripts/relabel/rebuild_targets.py` has a regex gap: it leaves the severity-format prose numbers stale. 29% of judge training examples use the severity format, and 67% of their prose numbers differ from the JSON. Fix the script before the next retrain. The fix is outside the harness.
3. False negatives are systematic model leniency. Only training can fix them.

## Known minors (v1.1, not fixed)
- Webhook: the 400 detail echoes the resolved internal IP (a DNS oracle). `getaddrinfo` has no timeout, so slow DNS can hold threadpool threads.
- `_sync_headers`: a header without a `— reason` tail is not rewritten (0 of 1789 saved headers lack it).
- `prose_disagrees` is imprecise: it fires on about 25% of units on a clean plugin.
- Lexicon: `docblock` matches `\bcomments?\b`, so block-markup and `phpcs:ignore` comments are mis-tagged. The NA filter sometimes drops a real defect sentence from an anti line.
- `logprobs: true` is always sent. A server that rejects it outright makes every unit unjudged.
- Tests: the `parity-cmp` failure message prints "max 8 <= 8" when it fails on the delta count. `sec_real` does not assert the classic-editor cap or the 520 s wall bound.

## Carry-over v1 open items
- Auth on `/docs` and `/openapi.json`: they stay open when `WPJUDGE_API_KEY` is set.
- Float knobs take whole numbers only: `retention_days`, `cap_value` and `coverage_threshold` have int defaults, so `_cast` rejects `1.5`. The README says "whole numbers".

## Read first (in order)
1. `SPEC.md`: decisions R1.1–R3.3, amendments A1–A6, report schema v1 (accepted).
2. `PLAN.md`: contracts, workstreams W1–W7, phases 1–6, **Locked plan**, and the **binding changes 1–16** from the plan review. The binding changes override the locked-plan text where they conflict.
3. `research/r1_extract.md`, `r2_judge.md`, `r3_api.md`, `r4_deploy.md`, `plan_review.md`.
4. `research/proto/`: working prototypes to build from.
   - `ts_extract.py` and `intake.py`: the W1 basis.
   - `edge.php`: the W1 edge-case fixture.
   - `rank_r2.py`: the W3 ranking basis. The fuller version is in r2 §5.
   - `api_r3.py` and `api_r3_test.sh`: the W4 basis.
5. `INVENTORY.md`: the repo assets to copy, e.g. `parse_judge_response` = `eval/eval_judge.py:310-383`, copied verbatim.

## Done so far
- Phase 1: research (4 agents). Phase 2: lock, with an independent evaluator verdict of APPROVE WITH CHANGES; all 16 changes are applied in PLAN.md.
- Pre-build CUDA check passed:
  - `ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515` is pulled locally.
  - On driver 580 / GB10 it loads IQ4_XS in about 10 s, using 16,078 MiB GPU, at 66.4 tok/s.
  - Its output on val prompt 0 matches the study's host build: same verdict and the same 6 dimension scores, overall 78 vs 79.
  - The 16-prompt parity check (binding change 13) is still to be run in phase 5.

## Next steps (phase 3 onward, per PLAN.md)
1. **Orchestrator pre-creates the shared files** (binding change 10). The workstreams only read them.
   - `harness/pyproject.toml`: package `wpjudge`. Deps: openai, fastapi, uvicorn, python-multipart, tree-sitter>=0.25,<0.27, tree-sitter-php>=0.24,<0.26. Extra `test`: httpx, pytest. Console script `wpjudge = wpjudge.cli:main`.
   - `harness/wpjudge/__init__.py`: a stub exporting `scan`. W4 fills it in.
   - `harness/wpjudge/config.py`: one dataclass with env overrides `WPJUDGE_*`. Every knob:
     - cap: dims security and sql_safety, threshold ≤ 4, cap value 60;
     - coverage threshold: 90;
     - extraction: `MAX_CHARS` 15000, templates flag on;
     - archive limits: 20000 files, 200 MB total, 100× ratio;
     - upload limit: 50 MB;
     - workers: 4;
     - `base_url` and model file (`wp-judge-v4-pruned-k224.IQ4_XS.gguf`);
     - jobs dir and retention (7 days);
     - API key;
     - `max_tokens` 2048.
2. **Spawn build agents in parallel**, one per workstream: W1 extract, W2 judge, W3 aggregate, W4 api+cli (against the contracts), W5 deploy, W6 fixtures plus `tests/e2e.sh`. Tell each one which files it owns, the contracts, its binding changes, and "do not commit; do not self-validate".
3. **Phase 4:** a separate evaluator agent per workstream. Defects go back to the builder; loop until PASS.
4. **Phase 5:** integration and functional tests. Run them through a test agent, then have a separate evaluator verify the results. Containers need `sg docker -c '...'` (see below).
5. **Phase 6:**
   - Write `harness/README.md` (operator guide).
   - Add a section to the root `README.md`.
   - Add a root `CHANGELOG.md` entry.
   - The evaluator checks the docs against actual behaviour.
   - Commit and push.

## Environment facts
- The host is aarch64 GB10, driver 580.178.04 (CUDA 13.0). No php (not needed).
- Python venv for repo tools: `/home/rli/work/venv` (it has openai). Do **not** pip-install harness deps into it. Use a fresh venv, e.g. `harness/.venv` (add it to .gitignore), or a scratch venv.
- Host llama-server: `~/work/llama.cpp/build/bin/llama-server`. Model files are in `models/_gguf/`:
  - `shrink-iq4xs-im_f16.gguf`: the default, the same bytes as the HF IQ4_XS file;
  - `shrink-mix-eiq3xxs_f16.gguf`: the small file;
  - the Q6_K and f16 files.
- Docker: `rli` is in the `docker` group, but older shells lack it, so run docker as `sg docker -c 'docker ...'` until re-login. GPU passthrough works (`--gpus all`) with no config changes.
- The GPU is shared. Run one model server at a time; kill servers and containers after use; check `nvidia-smi`.
- The session scratchpad is wiped between sessions. Anything worth keeping goes in `harness/`.

## Working rules (user)
- Ponytail mode (full): the laziest correct solution, reuse first, a tight MVP, no enterprise sprawl.
- Validation is always done by a **separate evaluator sub-agent**, never by the builder.
- Testing must go beyond unit tests: real E2E against the live model and container (phase 5).
- Commits: **no** `Co-Authored-By` or "Generated with" lines. Identity: `git -c user.name="Robert Li" -c user.email=dr.robert.li.au@gmail.com`. `gh` is authenticated, and `git push` works.
- Leave alone: `.planning/config.json` (the user's graphify setting) and `.planning/graphs/`. Both are uncommitted on purpose.
- Docs style: STE (short active sentences).

## Known quirk
Sub-agents' final reply text has arrived **empty** several times this session. Always tell agents to write their output to a file and reply "done". Then read the file.

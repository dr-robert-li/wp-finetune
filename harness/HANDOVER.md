# Handover: WP Judge harness build (2026-10-10)

**State:** ALL PHASES DONE (1-6). Phase 5: EVAL_PHASE5 PASS WITH NOTES (parity limit widened to ≤ 8 by the user). Phase 6: EVAL_W7 PASS (round 2). Committed and pushed 2026-10-10. Follow-ups (optional): webhook host check (SSRF), auth on /docs, low_rate tied to cap threshold, warning when prose and score disagree, TOPICS lexicon gaps ("other").
Commit: see git log (feat(harness)).

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

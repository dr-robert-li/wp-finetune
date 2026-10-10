# WP Judge Harness — Build Plan

Source of truth: `SPEC.md` (decisions R1.1–R3.3, report schema v1 accepted). Reuse map: `INVENTORY.md`.
Principle: tightly defined MVP. Reuse before writing. One elegant option per component. No enterprise sprawl.

## Rules for every agent
- Stay inside `harness/` (the only exceptions are the root README and CHANGELOG, phase 6).
- Copy repo assets in. Do not import from `scripts/` or `eval/`; the package must be self-contained.
- Build agents never validate their own work. A **separate evaluator sub-agent** validates each workstream.
- Each non-trivial piece of logic leaves ONE runnable check. Real verification comes from the integration and functional tests (phase 5), not from mocks.
- Do not commit. The orchestrator commits.

## Contracts (fixed now, so workstreams build in parallel)

`Unit` (extract → judge), a plain dict:
```
{"id": "<rel_file>:<start_line>:<name>", "file": "<path relative to target root>", "line": int, "end_line": int,
 "name": "Class::method | function | <template>", "kind": "function|method|template",
 "code": "<source, after truncation>", "loc": int, "truncated": bool}
```
`Skip` record: `{"file", "line", "name", "reason": "test|vendor|node_modules|build|minified|tiny|parse_error|too_large"}` (per-file skips: `line: 0, name: ""`).

`Judgment` (judge → aggregate) = the Unit fields +
```
{"status": "judged|unjudged", "attempts": int, "verdict": "PASS|FAIL|null", "overall_score": int|null,
 "scores": {"<native dim key>": int}, "critique": "<prose before [/REASONING]>", "raw": "<full response>", "error": str|null}
```
Native dimension keys: those emitted inside `<judge_output>`, e.g. `wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility` (`error_handling` is kept if present). The keys are not renamed.

Report: `SPEC.md` schema v1. Public entry points:
```
wpjudge.extract.load_target(path) -> (root_dir, target_meta)          # dir | .zip | .tar.gz, guarded
wpjudge.extract.collect_units(root_dir) -> (units, skips)
wpjudge.judge.judge_units(units, base_url, model, workers=4, on_result=None) -> judgments
wpjudge.aggregate.build_report(target_meta, model_meta, judgments, skips, cfg) -> report dict
wpjudge.scan(path, **cfg) -> report dict                                 # library one-shot
```
Config: one `wpjudge/config.py` dataclass, overridable from env vars (`WPJUDGE_*`). It holds the thresholds (cap ≤ 4, cap value 60, coverage 90%), limits, workers, base_url and model file.

## Workstreams

| WS | Module / files | Reuses | New |
|---|---|---|---|
| W1 extract | `wpjudge/extract.py` | skip list from `phase1_extract.py`, `has_test_code` (with 2 false-positive fixes), the r1 prototype | archive intake with traversal, symlink, size and ratio guards; minified detection; template units; truncation |
| W2 judge | `wpjudge/judge.py` | `_judge_create`, `parse_judge_response`, the threaded pattern of `sieve_capture_judge_http.py` | prompt builder; retry-once; critique split; status |
| W3 aggregate | `wpjudge/aggregate.py` | — | LOC-weighted mean, cap, dimension stats, worst functions, pattern and anti-pattern ranking, report assembly |
| W4 api+cli | `wpjudge/cli.py`, `wpjudge/api.py`, `pyproject.toml` | — | FastAPI async jobs on a disk store; auth; limits; retention; webhook; partial results |
| W5 deploy | `Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh` | llama-server flags from `shrink_eval.sh` and README | first-run `hf download`; all-in-one target |
| W6 tests | `tests/` | — | fixture plugin with planted bugs; malicious archives; E2E scripts |
| W7 docs | `harness/README.md`, root `README.md` + `CHANGELOG.md` | — | — |

## Phases

1. **Research (parallel, 4 agents; read and experiment only).** Each agent writes `research/<topic>.md`: options, evidence (commands run, results), one recommendation, risks.
   - R1 extract: fix the PHP tokenizer script vs tree-sitter-php (pip, no PHP runtime) vs nikic/php-parser. Run each candidate on real plugins (a few wordpress.org plugins and a theme). Cover the known bugs, template units, minified detection and archive-guard approach (stdlib zipfile and tarfile `filter="data"`).
   - R2 judge + patterns: confirm the prompt against the training data. Run real calls on IQ4_XS through llama-server with `--parallel 4` and measure throughput against 1 slot. Measure whether greedy output differs across slots. Study the critique text and pick the simplest robust way to group findings into patterns and anti-patterns (for example dimension + low score + a normalized key phrase). No embeddings unless that is shown necessary.
   - R3 api: the minimal FastAPI async-job design (BackgroundTasks or one worker thread plus a disk store), upload streaming plus size limits, retention sweep, webhook. Choose the smallest design that satisfies R2.4.
   - R4 deploy: aarch64 + CUDA reality. Does an official llama.cpp server image exist for linux/arm64 with CUDA? If not, find the smallest path (build stage from the llama.cpp source, or a host-run server). Cover nvidia-container-toolkit on this host, the hf CLI in the image, and the all-in-one feasibility.
2. **Lock.** The orchestrator folds the research into a "Locked plan" section per workstream (file list, approach, check). A **separate evaluator agent** reviews the locked plan against SPEC for gaps and over-engineering. **The user signs off** before the build.
3. **Build (parallel).** W1, W2, W3 and W5 run at once (disjoint files, fixed contracts). W4 starts at the same time against the contracts. W6 fixtures are built in parallel by their own agent.
4. **Validate each workstream.** A separate evaluator agent per workstream reads the code and runs its check against the contract and SPEC. Defects go back to the builder. The loop repeats until PASS.
5. **Integration and functional testing** (run by a test agent; a separate evaluator verifies the results):
   - **CLI E2E, host:** `wpjudge scan` on the fixture plugin as a folder, .zip and .tar.gz, against a real llama-server (IQ4_XS). The 3 reports must agree on units and skips. Golden expectations: planted SQL injection → `capped: true` with that function in `cap_reasons`; anti_patterns include security/sql; skip counts exact; clean functions not in worst_functions.
   - **Real targets:** one real wordpress.org plugin and one theme. Check report schema validity, coverage ≥ 90%, and wall time recorded. The evaluator recomputes every summary number from `functions[]` independently.
   - **Malicious input:** zip-slip, symlink, zip bomb, oversize, non-archive. All must be rejected cleanly with no files written outside the job directory.
   - **API functional:** `docker compose up`. Submit, poll, fetch the report and the partial results; auth on and off; webhook received by a local listener; retention sweep deletes expired jobs; a job survives an API restart only if its result is on disk (document the behaviour).
   - **All-in-one image:** one scan smoke run.
   - **Library in-session:** `from wpjudge import scan` in a Python session against the host server.
6. **Docs + commit.** `harness/README.md` (operator guide: install, run, configure, API reference, report fields, limits, the known small-model caveats). Root README section + CHANGELOG entry. The evaluator checks the docs against the actual behaviour. Then commit and push (no Claude signature lines).

## Known blockers (user action)
- Docker socket: `sudo usermod -aG docker rli` (then re-login) or run the container tests with sudo. Needed in phase 5.

## Locked plan (after research; pending evaluator review + user sign-off)

**W1 extract**: `wpjudge/extract.py`, about 180 lines. Build it from the r1 prototype (`research/proto/{ts_extract,intake}.py`, copied in).
- `load_target`: a dir passes through. Zip/tar intake uses the r1 guards: unsafe paths, symlinks and hardlinks, 20000 files, 200 MB total, 100× ratio, tar `filter="data"`. A single top-level dir becomes the root. Return `target_meta` (name, type plugin|theme by `style.css` Theme Name header else plugin, files_scanned, sha256 of the archive or "" for a dir).
- `collect_units`: tree-sitter walk using the r1 rules: skip rules, minified, template units, `too_large`, truncation at 15000 chars with a marker, and skipping symlinked files. Template units are on by default; `WPJUDGE_TEMPLATES=0` turns them off.
- Check: `tests/test_extract.py`, covering the edge.php 6 units with lines, plus the malicious archives being rejected.

**W2 judge**: `wpjudge/judge.py`, about 150 lines. The r2 core, plus `parse_judge_response` copied verbatim from `eval/eval_judge.py:310-383`.
- No system prompt. Temperature 0, max_tokens 2048, thinking off through extra_body.
- Retry once. (The overflow-halving retry is dropped. Add it only if phase 5 sees a context overflow.)
- It does not own `DIM`/`TOPICS`/`rank`; those live in W3.
- `scores` = integer keys from `<judge_output>` except verdict/overall_score.
- Check: `tests/test_judge.py` parses saved real responses (both prose formats and a broken one), with no server needed.

**W3 aggregate**: `wpjudge/aggregate.py`, about 150 lines.
- Overall = LOC-weighted mean of overall_score over judged units (rounded to 1 decimal place).
- Cap (R2.1): any judged unit with `security` or `sql_safety` ≤ 4 sets `capped`, `overall_score = min(uncapped, 60)` and `cap_reasons = [{id, dimension, score}]`.
- Dimensions: mean, std (population), min, n_rated, and low_rate (the share ≤ 4).
- Patterns and anti-patterns use the r2 `rank` (TOPICS lexicon) with top 10.
- worst_functions: the 10 lowest-scoring judged units.
- coverage: judged / (judged + unjudged), with `low_coverage` below 90%. `skipped` counts the skip records.
- verdict_counts are taken from the model verdicts as given.
- `functions[]` holds every judgment (minus `raw`) plus the skip records with status "skipped".
- Check: `tests/test_aggregate.py` runs on hand-built judgments with known answers (cap triggered and not triggered, weighting, coverage).

**W4 api+cli**: `wpjudge/api.py` (about 170 lines, r3 design B: one daemon worker thread + `queue.Queue`, disk store, endpoints and restart behaviour exactly as r3), `wpjudge/cli.py` (`scan`, `serve` via argparse), `wpjudge/__init__.py` (`scan()`), `wpjudge/config.py`, and `pyproject.toml`.
- Dependencies: openai, fastapi, uvicorn, python-multipart, httpx, tree-sitter, tree-sitter-php.
- `serve` runs a single uvicorn worker.
- Check: `tests/test_api.py` drives the real app with a stub scan through TestClient: auth, 413/411/415, the queue, partial results, restart and retention.

**W5 deploy**: `Dockerfile` (targets `harness` and `allinone`), `docker-compose.yml`, `docker/llama-entrypoint.sh`, `docker/allinone.sh`.
- These are the r4 drafts with php-cli removed.
- The model download uses curl. The pinned tag is `server-cuda13-b11515`.
- The healthcheck has a start_period of 40 minutes.
- The small file is never run with q8_0 KV.

**W6 tests and fixtures**: `tests/fixtures/vuln-plugin/` holds about 10 functions.
- Planted: 2 SQL injections, 1 AJAX handler with no nonce or capability check, 1 unescaped echo, 1 unsanitized `$_POST`.
- Also 4 clean functions, 1 tiny function, `vendor/x.php`, `tests/test-x.php`, `build/index.asset.php`, `x.min.php`, and a template file.
- Built as .zip and .tar.gz at test time.
- `tests/e2e.sh` runs the phase 5 integration and functional suite against a live server.

**W7 docs**: `harness/README.md`, the root README section, and the CHANGELOG entry.

Build agents: W1, W2, W3, W5 and W6 run in parallel. W4 runs in parallel against the contracts. Each workstream is then validated by its own evaluator agent.

### Binding changes from the plan review (research/plan_review.md; all accepted)
1. **worst_functions golden:** assert that every planted unit scores below every clean unit; do not assert list membership.
2. **Cap golden:**
   - W6 pre-runs the planted functions once through the live judge and records their scores.
   - The golden asserts the invariant `capped == any judged unit has security|sql_safety ≤ 4`, with exactly those units in `cap_reasons`.
   - The golden asserts the planted SQL injection caps only if the pre-run showed it.
   - If nothing scores ≤ 4, strengthen the fixture, not the harness.
3. **Determinism:**
   - The golden scans (dir, .zip, .tar.gz) run with `workers=1`.
   - One `workers=4` run asserts only units, skips and verdicts.
   - Score assertions use ±6 tolerances or order relations.
4. **`functions[]` rows:**
   - A skip row is `{file, line, name, loc: 0, truncated: false, status: "skipped", reason, verdict: null, overall_score: null, scores: {}, critique: ""}`.
   - A judged or unjudged row is the schema fields + `id, kind, end_line, attempts, error`.
   - `code` and `raw` are dropped from the report.
5. **rank:** examples are `"file:line name"`, only for anti_patterns (patterns have no examples). It ranks judged rows only.
6. `DIM`, `TOPICS` and `rank` live in `aggregate.py`.
7. **Owners:**
   - `model.file` and `model.base_url` are filled by `scan()` (W4) from config.
   - Compose passes `WPJUDGE_MODEL_FILE` to both services.
   - `collect_units` returns `files_scanned`, the count of `.php` files walked.
   - `coverage.units` = judged + unjudged; `pct` = judged / units × 100.
   - `cap_reasons` = `[{id, dimension, score}]`.
8. **Zero judged units:** the overall scores are null and `low_coverage` is true. Phase 5 adds a server-down E2E: exit 0, all units unjudged, a valid report.
9. `OpenAI(..., max_retries=0)`.
10. **The orchestrator pre-creates** `pyproject.toml` (all deps), a `wpjudge/__init__.py` stub and `wpjudge/config.py` (every knob) before the build. Workstreams only read them. Fixture ownership: W1 owns `tests/fixtures/extract/`, W2 owns `tests/fixtures/responses/`, W6 owns `tests/fixtures/vuln-plugin/`.
11. **Docker:** `docker-compose.yml` and `Dockerfile` live in `harness/`, with build context `.` and `COPY . /app/`.
12. **Guards:**
    - Compose publishes `127.0.0.1:8000:8000`.
    - `webhook_url` must use the http or https scheme.
    - The job id must match `re.fullmatch(r"[0-9a-f]{32}")`.
    - A non-numeric Content-Length returns 400.
    - The CLI extracts into a `TemporaryDirectory`.
    - Phase 5 tests `/jobs/..%2f../functions` and auth on `/functions`.
13. **Parity check (phase 5):**
    - First, a driver-580 smoke test of the image.
    - Then r2's 16 prompts at `workers=1` on the host build and on the container b11515.
    - Expect 16/16 verdicts equal and overall |Δ| ≤ 8. (Was ≤ 6. That limit was calibrated on slot noise within one build. Phase 5 measured cross-build drift of max 8 on 1/16, mean 1.5, host ≥ container on all 16, with all verdicts equal. The user widened it to 8 on 2026-10-10.)
14. **Real targets:**
    - Use classic-editor (39 units) and twentytwentyfour (60 units), not cf7 or woocommerce.
    - The CLI prints the unit count and an ETA (units ÷ 11/min) to stderr.
15. **Templates:** phase 5 records the theme's template vs function score split. The README documents the out-of-distribution caveat and `WPJUDGE_TEMPLATES=0` (default on).
16. **CLI flags:** `-o`, `--workers`, `--base-url`, `--model-file`.
- **Optional items adopted:**
  - The webhook uses stdlib `urllib.request` (httpx only for tests).
  - `allinone.sh` reuses the download block in `llama-entrypoint.sh`.
  - `judge_units` uses `submit` + `as_completed`, with the output sorted by `id`.
  - Ties in `worst_functions` are broken by `id`.
  - The README notes that one large job blocks the queue and v1 has no cancel.

## Out of scope (v1)
PHPCS/WPCS, static cross-function rules, multi-GPU, queue backends, a UI, report diffing, and model-quality re-evaluation (already done in the shrink study).

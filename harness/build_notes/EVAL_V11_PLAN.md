# Evaluation: PLAN_v1.1.md "Phase 2 lock"

Evaluator: an independent sub-agent. Read only: no code edited, no GPU server started. The only command run was one CPU-only `docker run` of the b11515 image, to check its Python version.
Inputs: `PLAN_v1.1.md`, `research/v11_r1..r5`, `wpjudge/*.py`, `docker/*`, `docker-compose.yml`, `Dockerfile`, `tests/*.py`, `tests/e2e.sh`.

## What checks out
- **T1 (WS-A)** follows R1 option C exactly: the guard function, re-checking at send time, no redirects, the allowlist, and the E2E phases. I checked the allinone image by running `python3` in b11515: it is 3.12.3, the same as `python:3.12-slim`. So R1's `is_global`, multicast and NAT64 findings hold in both images. `_create_connection` is an instance attribute that `HTTPConnection.__init__` sets, so the hook works.
- **T2a (WS-B)** matches R2 §a. The literal `4` sits at `aggregate.py:27` (the `rank` default) and `aggregate.py:86` (`low_rate`), the literal `8` at `aggregate.py:27`, and the call site at `aggregate.py:88`.
- **T2b / D1** follow R2. `aggregate.DIM` (`aggregate.py:7-9`) parses both header formats, `score N/10` and `severity w (N/10)`. `rank()` takes polarity from the JSON (`aggregate.py:35`). Rewriting the headers to `score X/10` therefore leaves the `rank` output unchanged, and T3's numbers, which R3 measured on unsynced critiques, still apply. The reason text after the dash is kept.
- **T3 (WS-B)** is R3 option F, including the measured E2E bounds. Leaving pattern `examples` out is justified by the schema freeze.
- **T4 / D3** follow R4's recommendation and its "not built" list.
- **T5 (WS-D)** follows R5 option A. The plan asks to grep every PARALLEL/WORKERS default. The real list is short:
  - `docker/llama-entrypoint.sh:13` and `docker-compose.yml:8,33` both default to 4;
  - `allinone.sh` has no default of its own: it runs `llama-entrypoint.sh`, and the `Dockerfile` sets no `WPJUDGE_WORKERS`, so it inherits the config default;
  - `e2e.sh:22` sets `HOST_PARALLEL` to 4.
  The plan covers all of these. Requiring exact w1 = w4 at `HOST_PARALLEL=1` does not contradict the |Δ| ≤ 8 parity limit. Parity compares two different builds (host 9954 vs container b11515). The exact check compares runs on one server, as R5 requires.
- **Cross-task interactions:**
  - D2 computes `prose_disagrees` before the header rewrite, which is the right order.
  - `confidence_reasons` uses `cap_threshold`, as T2a does.
  - Check the parenthesised allowlist condition (R1's bug note) in a unit test.

## Findings

### 1. MAJOR: `cap_dims` and `cap_threshold` cannot reach `to_judgment` as planned, and `__init__.py` belongs to no workstream
- **Evidence.**
  - `to_judgment` (`judge.py:116`) has one production caller, `judge_one` (`judge.py:138`).
  - `judge_one` is called only by `judge_units` (`judge.py:148`), whose signature is `(units, base_url, model, workers=4, on_result, max_tokens)`. It has no cfg.
  - `judge_units` is called from `wpjudge/__init__.py:26`.
  - "Pass them from the caller" (PLAN line 87) therefore means changing the signatures of `judge_units` and `judge_one` and editing `__init__.py`.
- **The ownership gap.**
  - `__init__.py` is not in the ownership table, yet WS-D also needs it: `model_meta` is built at `__init__.py:28`, and the `/props` probe has to run in `scan()`.
  - C and D would both edit `__init__.py` and the `judge_units` signature at the same time.
  - The flag would also be frozen at the threshold in force when the unit was judged. WS-B's E2E re-aggregates with `WPJUDGE_CAP_THRESHOLD=5` (PLAN line 79), and `prose_disagrees` would go stale there.
- **Fix: keep `to_judgment` cfg-free.**
  - Before rewriting the headers, `to_judgment` stores the original prose numbers in the judgment, for example `prose_scores: {"security": 4, "sql_safety": 6}` taken from `DIM` (prose numbers only, or every dimension).
  - `aggregate._row` computes `prose_disagrees` from `prose_scores`, `scores`, `cfg.cap_dims` and `cfg.cap_threshold`.
- **What this fixes at once.**
  - No signatures need threading.
  - Re-aggregating at a new threshold stays correct.
  - The report reader can see why the flag fired, which the synced critique no longer shows (see finding 7).
- **Ownership.** Give `__init__.py` to WS-D only, for the `server_slots` probe and passing `workers`/`server_slots` into `model_meta`.

### 2. MAJOR: running builders B, C and D in one tree is not safe
- **Evidence: several workstreams edit the same code.**
  - `build_report`, `aggregate.py:61-112`, is edited by three workstreams:
    - B: the `low_rate` line at 86 and the `rank` call at 88;
    - C: the `_row` and summary dict at 98-111;
    - D: the `model` dict at 97.
  - `judge.py` is edited by C (logprobs, `judge_one`, `to_judgment`) and by D (`server_slots`, and possibly the `judge_units` default).
  - `tests/test_aggregate.py` is edited by B and C.
  - `tests/e2e.sh` is edited by A (`sec_api`, `compose_override` at 413), by D (`HOST_PARALLEL` at 22, `cmd_golden` at 153, a new section, and the case list at 534), and by E (chk checks for every workstream).
  - `config.py` is edited by all four.
- **Why "B lands first, or C re-reads" (PLAN lines 50-51) is a race, not an ordering:**
  - all builders start in parallel;
  - a per-workstream evaluator running `pytest` or e2e in the shared tree tests other builders' half-finished edits, so one failure cannot be traced to one workstream;
  - `docker build` (`Dockerfile` `COPY . /app/`) for `api` or `allinone` bakes in whatever the other builders have left mid-edit.
- **Fix.** Pick one of two:
  - (a) Run each workstream in its own git worktree (Agent `isolation: "worktree"`). Each evaluator runs in that worktree. The orchestrator merges in the order A, B, C, D and re-runs the unit tests after each merge.
  - (b) Run A in parallel with a serial B → C → D chain, so each builder starts only after the previous one passes its evaluator.
  Either way, assign each `e2e.sh` function to exactly one owner.

### 3. MAJOR: the wall-time check "+20% of v1 (221 s)" will always fail
- **Evidence.**
  - The 221 s for classic-editor (`build_notes/PHASE5_RUN.md:61`, `EVAL_PHASE5.md:95`) was measured with v1 defaults: `HOST_PARALLEL=4` and `cfg.workers=4`. `sec_real` passes no `--workers`.
  - After T5 the run is serial. R5 measured 10.0 → 5.4 units/min, so 39 units take about 433 s before any logprobs cost.
  - R4's +10% logprobs overhead was also measured at w4, with `top_logprobs=5`.
  - The check (PLAN line 102, copied from R4) mixes the T5 slowdown with the T4 cost.
- **Fix.** Bound each cost separately:
  - **T5:** classic-editor wall time ≤ 1.2 × (39 / 5.4 min) ≈ 520 s. Better: measure a serial v1 baseline once, with git-stashed code at `-np 1`/w1.
  - **T4:** logprobs cost ≤ +15% of the per-completion-token time against that serial baseline. Record the number in `V11_E2E.md`. R4 never measured `logprobs=true` without `top_logprobs`.
  - Also state the expected total E2E wall time at serial (cli runs five fixture scans, about 9 to 10 min).

### 4. MAJOR: the WS-E section list does not run and leaves out sections that matter
- **Evidence.** PLAN line 131 runs `unit cli real api allinone sec_api parity`. The valid sections are those at `e2e.sh:534`: `cli malicious serverdown api allinone library parity real prerun`.
  - `unit` and `sec_api` are not sections, so each one counts as `fail "unknown section"`. `sec_api` is the function behind `api`. `golden` is a chk subcommand inside `sec_cli`, not a section.
  - The new WS-D `determinism` section is not in the list, so T5's fast-mode E2E never runs.
  - `serverdown` is left out, yet it is the only live check of unjudged rows. It matters for C's new fields: `low_confidence=false`, `[]` and `null`, and the summary count.
  - `library` is left out, yet `scan()` gains `server_slots` and `workers` in WS-D.
  - `malicious` costs nothing and guards the extraction code during the merges.
- **Fix.** Use `pytest tests/` (run separately) plus `tests/e2e.sh malicious serverdown cli library real determinism parity api allinone`. Add `determinism` to the `case` list and to the `all` list (`e2e.sh:530`).

### 5. MINOR: the `determinism` section will silently reuse the serial server
- **Evidence.** `host_up` (`e2e.sh:58`) returns at once if anything answers on `PORT`. After `cli` has run, the `-np 1` server is still up, so "against a `--parallel 4` server" would actually run at `-np 1`. `server_slots` would then be 1, `reproducible` true, and the assertion would fail. `host_down` kills only a server it started itself (`HOST_PID`).
- **Fix.**
  - `sec_determinism` runs `host_down`, then starts its own server with `HOST_PARALLEL=4` on a separate port (for example 8099). It asserts `model.server_slots == 4` before testing `reproducible == false` and the warning.
  - Golden reports should also assert `model.server_slots == 1`, so a leftover `-np 4` server is caught for the right reason.

### 6. MINOR: the T2a E2E assertions depend on the data or the truncation
- **Evidence.**
  - "`low_rate` changes at T=5" (PLAN line 79) holds only if some canonical fixture score is exactly 5. Nothing shows that the serial canonical run has one. R5 lists `sql_safety` 4, 5 and 7 only under concurrent runs.
  - "anti counts ≥ those at 4" can fail for a key that drops out of the top 10 at T=5.
  - `chk.py` hard-codes `CAP_T = 4` (`e2e.sh:90`) and `x <= 4` in `recompute` (`e2e.sh:132`). Any recompute on a T=5 report fails.
- **Fix.**
  - Assert that `low_rate[d]` equals the share of scores ≤ 5, recomputed from `functions[]`.
  - Compare anti counts per key present in both lists, or compare the sum.
  - Make `recompute` take the threshold from `WPJUDGE_CAP_THRESHOLD`, or from an argument.

### 7. MINOR: D2 overstates what `prose_disagrees` does, and the reader cannot check it
- **Evidence.**
  - R4 finding 4 says the classic-editor CSRF "FP" never reaches the cap: JSON security for `add_redirect_helper` was 6 or 7 in 8 of 8 runs. After D1 there is no FP left in the cap or the statistics.
  - So the flag fires on a unit whose JSON is right. Its value is a warning that the prose body still argues CSRF.
  - R2 says the cap-dimension version of the flag is noise. R4 says it fires on about 25% of classic-editor units, which is presumed clean.
  - After the header rewrite, the report no longer shows the original prose number (`raw` is dropped). A reader sees `Security: score 6/10` next to `prose_disagrees` with no explanation.
  - R4 itself recommends dropping this reason once the prose and the JSON agree. The plan's deviation is defensible, because the flag catches 3 of 7 val FNs at no cost, but D2 should give that reason, not "catches the FP".
- **Fix.**
  - Reword D2.
  - Expose `prose_scores` (finding 1).
  - Say in the README that about 25% of units on a clean plugin are flagged.
  - Record `summary.low_confidence` / `units` for classic-editor in `V11_E2E.md` as a precision reference.

### 8. MINOR: the "`cap_reasons` ids are exactly the 5 planted ids" check has no evidence on b11515 at serial
- **Evidence.** On the host the check is backed by R5's `ref`: uncapped 68.8 at serial, the same as `r_dir`. The api and allinone reports (b11515) were measured only at v1's `-np 4`/w4, which is nondeterministic (`aio`: R2 §a). After T5, compose and allinone run at `-np 1`, and their canonical output has never been seen. PLAN line 97 does not say which reports this hard check covers.
- **Fix.** Make it a hard check on the host cli and golden reports. On api and aio, make it soft (warn) until one serial b11515 run is recorded. `cmd_fixture` is shared, so add a flag argument.

### 9. MINOR: the plan misses stale "fast-mode" defaults in the code
- **Evidence.**
  - The CLI ETA uses `total / 11` (`cli.py:30`), which assumes about 11 units/min. At serial it underestimates by about 2×.
  - `judge_units(..., workers=4)` (`judge.py:148`) keeps a default of 4 for direct library callers.
- **Fix (WS-D).**
  - ETA: `total / (5.5 if workers == 1 else 10)`, or a single constant of 5.5.
  - Change the `judge_units` default to 1.

### 10. MINOR: the live-E2E rule contradicts the workstream table, and docs have no evaluator
- **Evidence.**
  - Rule line 9 says each workstream adds live E2E evidence.
  - The table (line 52) gives live runs to E, after A to D pass their evaluators. B and C own no `e2e.sh` code.
  - WS-F (docs) has no evaluator, even though the README defaults (lines 51, 85, 100, 130, 150) and the `e2e.sh` header comment (line 4, `HOST_PARALLEL (4)`) must all change.
- **Fix.**
  - Each workstream evaluator runs its own live section under `flock`, inside its worktree: A `api`, B `real` plus re-aggregate, C `cli`, D `cli determinism`. E then runs the merged full suite.
  - Add a doc evaluator that checks every default, env var and schema field in the README and SPEC against the code.

### 11. MINOR: the third real target is not named, and serial halves throughput
- **Evidence.** PLAN line 134 says "a WooCommerce-style plugin or a classic PHP theme". WooCommerce itself has thousands of functions, which is hours at 5.4 units/min.
- **Fix.** Name one pinned download URL with ≤ 150 units, and add it to the `sec_real` spec list (`e2e.sh:362`) with its expected unit count and type.

### 12. MINOR: the allinone SSRF target is not covered by E2E
- **Evidence.** In allinone, the harness and llama-server share a container, so `http://127.0.0.1:8080/` is the live SSRF target there. `sec_allinone` (`e2e.sh:514`) submits no webhook.
- **Fix.** Add one check: `webhook_url=http://127.0.0.1:8080/health` returns 400 in `sec_allinone`. It costs nothing, since the stack is already up.

### 13. MINOR: nobody has checked determinism with logprobs on
- **Evidence.**
  - R5's byte-identical serial runs were made without logprobs.
  - Golden compares w1 with w4, and both have logprobs on. It would not notice if logprobs changed the canonical output.
  - The functional fixture checks (5 planted bugs capped, every planted unit below every clean unit) are the only guard.
- **Fix.** In `V11_E2E.md`, compare golden `r_dir` per-unit scores with R5's `ref`, which is the v1 `r_dir`: uncapped 68.8. Record any difference. A difference is not a failure, but it must be explained.

### 14. MINOR: unit-test details that will break or prove little
- **Evidence.**
  - The `Stub` (`tests/test_judge.py:46-56`) returns choices with no `logprobs` attribute. `verdict_p` extraction must use `getattr(choice, "logprobs", None)`, or `test_request_shape` and the retry tests crash.
  - Existing tests call `to_judgment(UNIT, raw, 1)` (lines 28, 36, 40), so new parameters need defaults.
  - The "fake client with jitter" test (PLAN line 121) only re-proves the client-side sort at `judge.py:148-159`. It cannot detect server-side nondeterminism.
  - `_skip_row` (`aggregate.py:55`) is not listed for C, yet skip rows also need `verdict_p=null`, `low_confidence=false` and `confidence_reasons=[]`.
- **Fix.**
  - Use `getattr`.
  - Give the new parameters defaults.
  - Label the jitter test as a client-side guard.
  - Add `_skip_row` to C's scope.

## Requirement coverage
| Req | Covered | Notes |
|---|---|---|
| T1 SSRF | yes | findings 12 (allinone E2E) and 2 (e2e ownership) |
| T2a thresholds | yes | finding 6 |
| T2b root cause and fix | yes (R2 root cause in `rebuild_targets.py`; JSON authoritative; headers synced) | findings 1 and 7 |
| T3 "other" | yes | finding 11 |
| T4 FP/FN evaluation and surfacing | yes (R4 measured all candidates with cost) | findings 3, 7 and 13 |
| T5 worker-count independence | yes (serial default, `reproducible` stamp) | findings 4, 5 and 9 |
| Parallel workstreams | planned, not safe as written | finding 2 |
| Separate evaluator per workstream | A to E yes; F no | finding 10 |
| Live E2E plus functional | yes in E; not per workstream; section list broken; wall-time check wrong | findings 3, 4 and 10 |

VERDICT: FAIL

## Round 2

Evaluator: a second independent sub-agent. Read only. No GPU server and no docker run. I re-checked against `wpjudge/{judge,aggregate,__init__,cli,config,api,extract}.py`, `tests/e2e.sh`, `tests/test_{judge,aggregate}.py`, `docker/*`, `docker-compose.yml`, `research/v11_r2/r4/r5`, `build_notes/e2e_work/r_dir.json`, the git state and `.venv`.

### Round-1 findings
| # | Status | Reason |
|---|---|---|
| 1 | resolved | `to_judgment` stays cfg-free and only gains `verdict_p=None`. `prose_scores` is stored in the judgment. `_row` already copies every judgment key except code/raw (`aggregate.py:51`), so `prose_scores` reaches the row with no plumbing. `prose_disagrees` is computed in `_row` from cfg, so re-aggregating at T=5 stays correct. `__init__.py` belongs to D only. `api.py:91` goes through `scan()`, so D's `/props` probe covers the API too. |
| 2 | resolved in design | Per-WS worktrees, a merge order and one owner per `e2e.sh` function. But the worktree environment as specified does not work: see N1. |
| 3 | resolved | 520 s = 1.2 × 39/5.4 min. Logprobs are bounded separately at +15% per completion token. The baseline has no stated source: see N3. |
| 4 | resolved | `malicious serverdown cli library real determinism parity api allinone` are all real sections once D adds `determinism` to `case` and `all` (`e2e.sh:530,534`). pytest runs separately. |
| 5 | resolved | `host_down`, then its own server on 8099 (free: PORT is 8097 and PARITY_PORT is 8098, `e2e.sh:20,25`). It asserts `server_slots == 4`, and golden asserts `server_slots == 1`. R5 table rows 25 and 28 support "w1 on `-np 4` == `-np 1` golden" (with no `-kvu`; the entrypoint sets none). |
| 6 | resolved | `low_rate` is recomputed from `functions[]`. `recompute` takes the threshold from env. The summed top-10 anti counts are monotone in T: each (dim, topic) count can only grow and patterns are untouched, so ≥ is sound. |
| 7 | resolved | D2 is reworded (3/7 FNs, CSRF only a warning), with `prose_scores`, a README precision note and the classic-editor ratio recorded. |
| 8 | resolved | The check is hard on host cli and golden, soft on api and aio. I verified the v1 `r_dir` `cap_reasons` ids are exactly the 5 planted ids (sqli×2, ajax, echo, option), at uncapped 68.8. The mechanism (a flag argument to the shared `cmd_fixture`, and which mode `library` uses) is left to the builder. Acceptable. |
| 9 | resolved | ETA becomes 5.5 (`cli.py:30`), and the `judge_units` default becomes 1 (D). |
| 10 | resolved | Each WS evaluator runs its own live section, and docs get their own evaluator. |
| 11 | resolved | `twentytwelve` is named (classic theme; `extract.py:99-101` detects `style.css` "Theme Name:"), with a pinned version and a ≤ 150 gate. The fallback plugin is unnamed. Minor, accepted. |
| 12 | resolved | `sec_allinone` checks that a `127.0.0.1:8080/health` webhook gives 400. That matches `allinone.sh` (llama is on 127.0.0.1:8080 in the same container). |
| 13 | resolved | E compares golden `r_dir` against v1 `r_dir` (68.8) and explains any difference. |
| 14 | resolved | `getattr` on the Stub with no `logprobs` (`test_judge.py:54`). The parameters get defaults. The jitter test is labelled client-side. `_skip_row` is in C's scope. |

### New findings

#### N1. MAJOR: as specified, the worktrees lack the plan, the venv and the model, and a reused venv silently tests main's code
- **Evidence.**
  - `EnterWorktree`/`isolation: "worktree"` branches from `origin/main` by default (`baseRef: fresh`), which is b819e78. `PLAN_v1.1.md`, `research/v11_r*.md` and `build_notes/EVAL_V11_PLAN.md` are untracked, and `HANDOVER.md` is modified (git status). So builders get no plan and no research.
  - `.venv/` is gitignored (root `.gitignore`), and `e2e.sh:10-11` uses `$H/.venv/bin/{python,wpjudge}`, so a fresh worktree has none.
  - The existing venv is an editable install pointing at the main tree: `.venv/.../wpjudge-1.0.0.dist-info/direct_url.json` gives `file:///home/rli/Desktop/projects/wp-finetune/harness`. If the main venv is symlinked or reused, the `wpjudge` CLI and `from wpjudge import scan` in the cli, library and real sections import main's unmodified code. The evaluator then passes without testing the workstream at all.
  - `GGUF` defaults to `$H/../models/_gguf/...` (`e2e.sh:16`) and `VAL` to `$H/../data/...` (`e2e.sh:28`). `models/` is gitignored, so in `<repo>/.claude/worktrees/<n>/harness` both paths are missing.
- **Fix.** Add a "worktree setup" step to the plan:
  - commit (or copy in) the plan, the research and the EVAL file before spawning, or set `worktree.baseRef: head` after committing them;
  - in each worktree run `python -m venv .venv && .venv/bin/pip install -e '.[dev]'` (or the equivalent), then assert that `.venv/bin/python -c 'import wpjudge; print(wpjudge.__file__)'` is under that worktree;
  - every live run exports `GGUF=/home/rli/Desktop/projects/wp-finetune/models/_gguf/shrink-iq4xs-im_f16.gguf` and points `VAL` (or the repo-root data path) at the main checkout.

#### N2. MINOR: the B and C worktrees run live sections on the old fast-mode defaults
- **Evidence.** B and C branch before D's change, so `HOST_PARALLEL` is still 4 (`e2e.sh:22`) and `cfg.workers` is still 4. `sec_real` passes no `--workers` (`e2e.sh:367`). B's `real` other-share checks and any C timing would run at `-np 4`/w4. That is nondeterministic (R5), so the results are not comparable with E's merged run, and "no other in top 3" can flake.
- **Fix.** The B and C evaluators export `HOST_PARALLEL=1 WPJUDGE_WORKERS=1`.

#### N3. MINOR: the logprobs-cost baseline has no source
- **Evidence.** WS-C/E: "+15% … against a serial baseline run without logprobs". No knob turns logprobs off. R4's per-token numbers (28.9 vs 32.4 ms) are at w4 with `top_logprobs=5`, and R5 records only units/min.
- **Fix.** State the baseline: the C evaluator runs classic-editor once on the worktree base commit (pre-C code) at `HOST_PARALLEL=1`/w1, then once with C, and records the ms per completion token for both in `V11_E2E.md`.

#### N4. MINOR: the wall-time estimate for `real` is low
- **Evidence.** classic-editor (39) plus twentytwentyfour (60) at 5.4 units/min is already about 18 min. twentytwelve adds an estimated 50 to 100 units, about 9 to 18 min. "real about 20 min including the third target" is therefore about 30 to 40 min.
- **Fix.** Correct the estimate, so nobody kills a slow but healthy run.

#### N5. MINOR: the ownership of `e2e.sh` contradicts itself
- **Evidence.** "Each `e2e.sh` function has exactly one owner", yet WS-E owns "`tests/e2e.sh` checks for every WS".
- **Fix.** E owns only the twentytwelve spec line and fixes for post-merge breakage. Every other check stays with its WS.

#### N6. MINOR: details that would surprise a builder
- `test_worst_ties_and_rows` (`test_aggregate.py:95`) asserts the skip row as an exact dict, so C's new `_skip_row` fields break it. Name it in C's scope.
- The `prose_disagrees` rule only says what a JSON None means. `DIM` also captures a prose `None` (`score None/10`). State that a prose None or N/A counts as above (R4 line 116).
- `DIM` only matches header lines that carry a `— reason` tail, and only those are captured or rewritten. The E2E "every header number equals `scores[d]`" must therefore use the same `DIM` regex, or it will flag lines that were never in scope.
- `sec_determinism` needs `$WORK/r_dir.json` from `cli` in the same `WORK`. It must fail with a clear message when the file is absent, and it must use its own PORT/BASE, because `host_up` reads the globals (`e2e.sh:20-21,58-62`).

VERDICT: FAIL

## Round 3

Evaluator: a third independent sub-agent. Read only. No GPU server. I checked the plan against `pyproject.toml`, `tests/e2e.sh`, `tests/test_aggregate.py`, `wpjudge/{config,aggregate,cli}.py`, both `.gitignore` files, the git state, the main `.venv`, and the model, data and llama-server paths on disk.

### Round-2 findings
| # | Status | Reason |
|---|---|---|
| N1 | partly resolved | Steps 1 and 3 (GGUF) and the import-path check are correct. The paths exist: `models/_gguf/shrink-iq4xs-im_f16.gguf` (16.6 GB), `data/reasoning_dataset/openai_val.jsonl` and `~/work/llama.cpp/build/bin/llama-server`. `.claude/worktrees/` and `.venv/` are gitignored, so a fresh venv per worktree is needed, as the plan says. But the step-2 pip command names an extra that does not exist (R3-1). The VAL export has no effect (R3-2). |
| N2 | resolved | Step 4 exports `HOST_PARALLEL=1 WPJUDGE_WORKERS=1`. `e2e.sh:22` reads `HOST_PARALLEL` from env, and `config.load` maps `WPJUDGE_WORKERS` onto `workers` (`config.py:1,40-45`). |
| N3 | resolved | The baseline is defined: classic-editor at the worktree base commit (which is pre-C, because the base is HEAD) and with C, both at w1, recording ms per completion token. |
| N4 | resolved | `real` is now estimated at about 30 to 40 min. |
| N5 | resolved | E owns only the twentytwelve spec line and post-merge fixes. |
| N6 | resolved | `test_worst_ties_and_rows` (`test_aggregate.py:95`, exact skip-row dict) is named in C. A prose or JSON None, N/A or absent counts as above. The header check uses `aggregate.DIM` (`aggregate.py:7`). `sec_determinism` fails clearly without `$WORK/r_dir.json` and uses its own PORT/BASE. |

### New findings

#### R3-1. MAJOR: `pip install -e '.[dev]'` installs no test tools, so the worktree pytest falls back to main's code
- **Evidence.**
  - `pyproject.toml` has only `[project.optional-dependencies] test = ["httpx", "pytest"]`. There is no `dev` extra. pip only warns ("does not provide the extra 'dev'") and exits 0, so the worktree `.venv` gets no pytest and no httpx.
  - Step 2 checks only `import wpjudge`, which passes.
  - An evaluator that then runs `pytest tests/` gets either the system pytest or main's `.venv/bin/pytest`, which does have pytest and httpx. Main's venv is an editable install of the main tree, and pytest's default `prepend` import mode puts only `tests/` on `sys.path`. So `import wpjudge` resolves to main's unmodified code, and the unit tests pass without testing the workstream at all. This is the same failure mode as N1. `test_api.py` also needs httpx.
- **Fix.**
  - Use `.venv/bin/pip install -e '.[test]'`, which matches README:17 and HANDOVER:27.
  - Extend the step-2 check: `.venv/bin/python -m pytest --version` must succeed.
  - Every unit run uses `.venv/bin/python -m pytest tests/`, never a bare `pytest`.

#### R3-2. MINOR: exporting `VAL` has no effect
- **Evidence.** `e2e.sh:28` sets `VAL=$(cd "$H/.." && pwd)/data/...` unconditionally (it is not `${VAL:-...}`). In a worktree, that path is missing. Only `sec_parity` reads it (`e2e.sh:391`), and only E runs parity, on main, where the path exists. So nothing breaks today, but step 3 is wrong.
- **Fix.** Make it `VAL=${VAL:-...}` (D owns the e2e.sh header), or drop VAL from step 3.

VERDICT: FAIL

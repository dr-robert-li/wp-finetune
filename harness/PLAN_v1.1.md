# WP Judge harness v1.1 plan

Status: phase 2 lock (draft), awaiting evaluator review and user sign-off. Do not lock the build until all research files exist and the user signs off.
The process is the same as v1 (PLAN.md): research → lock → evaluator review → user sign-off → parallel build → evaluator per workstream → E2E → docs → commit.

## Rules for every agent
- Ponytail: the smallest correct change. Reuse first.
- Builders never validate their own work. A separate evaluator sub-agent does.
- Testing goes beyond unit tests. Each workstream evaluator runs that workstream's live `e2e.sh` section: a real model server, the real CLI and API, and docker where relevant. WS-E then runs the full suite on merged main.
- GPU lock: wrap every model-server run in `flock /tmp/wpjudge-gpu.lock <cmd>`. Run one server at a time. Kill it after use.
- Write output to a file and reply "done". Sub-agent reply text sometimes arrives empty.
- Do not commit. The orchestrator commits.

## Tasks (from the user, 2026-10-10)
- **T1 Webhook SSRF.** `webhook_url` checks only the scheme. Block or restrict internal targets (loopback, link-local, private ranges, metadata IPs, DNS rebinding, redirects).
- **T2 Score consistency.**
  - (a) `low_rate` and the pattern/anti-pattern thresholds in `aggregate.py` are hard-coded at ≤ 4 and ≥ 8. They ignore `WPJUDGE_CAP_THRESHOLD`.
  - (b) Root-cause the critique-prose vs stored-score mismatch. Example: classic-editor `add_redirect_helper` prose says "Security 4/10" while `<judge_output>` gives 5; the prose also discusses dims that have no score. Find where the divergence comes from (training data? generation order? parsing?). Decide the fix so the cap and the statistics reflect the judge's actual assessment.
- **T3 "other" pattern labels.** Find why many patterns get `finding: "other"` (TOPICS lexicon coverage vs critique phrasing vs the rank logic) and fix it.
- **T4 FP/FN risk.** Test candidate ways to reduce false positives and false negatives, or to surface low-confidence units whose verdict or cap could flip. Candidates to evaluate with data: score near the threshold, prose/score disagreement, multi-sample agreement (n samples at temp>0), logprobs of the score tokens, re-judge with workers=1, a second-pass check of flagged units. Measure the cost (time per unit) and the benefit against the fixture planted bugs plus the known FP (classic-editor CSRF).
- **T5 Worker-count stability.** With 4 workers, verdicts stayed stable, but unit scores and `cap_reasons` changed. One fixture unit moved 14→42. The report (scores, cap and verdicts) must not depend on the worker count. Root-cause the nondeterminism across slots: llama.cpp batched decoding, KV/slot layout, or sampling. Fix it, or bound and surface it.

## Phase 1 research (parallel)
| ID | Topic | Output | GPU |
|---|---|---|---|
| R1 | T1 SSRF guard design | `research/v11_r1_ssrf.md` | no |
| R2 | T2 root cause + threshold alignment | `research/v11_r2_scores.md` | yes, with the lock |
| R3 | T3 "other" labels | `research/v11_r3_topics.md` | no (uses saved reports) |
| R4 | T4 confidence / FP-FN | `research/v11_r4_confidence.md` | yes, with the lock |
| R5 | T5 worker-count stability | `research/v11_r5_determinism.md` | yes, with the lock |

Each research file: the options, evidence (the commands run and their results), one recommendation, the files to change, the proposed E2E test, and the risks.

## Phase 2 lock (after research)
Status: revised after evaluator round 1 (`build_notes/EVAL_V11_PLAN.md`, FAIL: 4 major, 10 minor; all addressed). Round 2 FAIL (N1 major: worktree setup; N2 to N6 minor), all addressed. Round 3 FAIL (R3-1 major: the `[dev]` extra does not exist, so use `[test]`; R3-2 minor: VAL), both fixed. Next: user sign-off.

### Cross-task decisions
- **D1. `<judge_output>` stays the only score source** (R2). The cap, the statistics and the ranking use the JSON scores. The prose numbers are stale severity buckets left by a regex gap in `scripts/relabel/rebuild_targets.py`. 29% of the judge training examples use the severity format, and 67% of their prose numbers differ from the JSON. Trusting the prose, or taking min(prose, JSON), adds the fixture clean-unit FP and caps classic-editor and twentytwentyfour.
- **D2. R2 vs R4 conflict.** R2 rewrites the critique headers from the JSON. R4's `prose_disagrees` flag reads the prose numbers. Resolution:
  - Before the rewrite, `judge.to_judgment` (still cfg-free) stores the original prose header numbers as `prose_scores` in each judgment and report row.
  - `aggregate._row` computes `prose_disagrees` from `prose_scores`, `scores`, `cfg.cap_dims` and `cfg.cap_threshold`. This stays correct when the report is re-aggregated at another threshold, and the reader can see why the flag fired.
  - Why keep the flag: it costs nothing and catches 3 of 7 val FNs, where the prose said 4 and the JSON said 6. It also warns when the prose body argues a defect that the score does not reflect (the classic-editor CSRF case; after D1 that case never reaches the cap).
  - The flag is imprecise: it fires on about 25% of units on a presumed-clean plugin. The README says so.
- **D3. No re-judge mode.** No n-sample, no second pass, no workers=1 re-judge (R4: 0 errors changed at +14 to 400% calls). The FN problem is systematic model leniency. The README says so.
- **D4. Retrain note.** Record the `rebuild_targets.py` gap in HANDOVER, with R2's counts. The fix is outside the harness.

### Workstreams and file ownership
Each builder A to D runs in its **own git worktree** (Agent `isolation: "worktree"`), in parallel. Its evaluator runs in the same worktree, including that workstream's live E2E section under the GPU flock. The orchestrator merges A → B → C → D into main, resolves the small `build_report`/`config.py`/`e2e.sh` overlaps, and re-runs `pytest tests/` after each merge. Each `e2e.sh` function has exactly one owner. WS-E owns only the twentytwelve spec line and fixes for breakage after the merge.

**Worktree setup** (the orchestrator does this before spawning; round-2 N1):
1. Commit the plan, the research and the eval files on main, then spawn the worktrees from HEAD (`worktree.baseRef: head`).
2. In each worktree: `cd harness && python3 -m venv .venv && .venv/bin/pip install -e '.[test]'`. Then check that `.venv/bin/python -c 'import wpjudge;print(wpjudge.__file__)'` prints a path inside that worktree, and that `.venv/bin/python -m pytest --version` succeeds. Run every unit test as `.venv/bin/python -m pytest tests/`, never a bare `pytest`. Never reuse main's venv: it is an editable install of main.
3. Every live run exports `GGUF=/home/rli/Desktop/projects/wp-finetune/models/_gguf/shrink-iq4xs-im_f16.gguf` because `models/` is not in the worktree. (`VAL` is only read by `parity`, which E runs on main.)
4. The B and C evaluators export `HOST_PARALLEL=1 WPJUDGE_WORKERS=1`, because their branches predate D's defaults.

| WS | Tasks | Owns | Depends on |
|---|---|---|---|
| A | T1 | `api.py` (webhook, create_job), `config.webhook_allow_nets`, `tests/test_api.py`, `e2e.sh` `sec_api`/`compose_override`/the webhook check in `sec_allinone` | none |
| B | T2a, T3 | `aggregate.py` TOPICS/RX/DIM_TOPICS/NA/`rank`, the `low_rate` line, `config.pattern_min`, `tests/test_aggregate.py` (topic and threshold tests), `e2e.sh` chk `cmd_real` other-share checks and `recompute` threshold | none |
| C | T2b, T4 | `judge.py` `_judge_create`/`judge_one`/`to_judgment` (logprobs, `verdict_p`, `prose_scores`, header sync), `aggregate.py` `_row`/`_skip_row`/summary (confidence fields), `config.min_verdict_p`, `tests/test_judge.py`, confidence tests in `test_aggregate.py`, `e2e.sh` chk `cmd_fixture` checks | worktree, merged after B |
| D | T5 | `config.workers`, `docker/llama-entrypoint.sh`, `docker-compose.yml`, `judge.server_slots` and the `judge_units` default, `__init__.py` (`scan` probe and `model_meta`), the `aggregate.build_report` model dict, `cli.py` (warning and ETA), `e2e.sh` `HOST_PARALLEL`/header comment/`cmd_golden`/new `sec_determinism`/case and all lists, `tests/test_config.py` | worktree, merged after C |
| E | E2E | the twentytwelve spec line, fixes for breakage after the merge, live runs, `build_notes/V11_E2E.md` | A to D pass their evaluators and are merged |
| F | Docs | `README.md`, `SPEC.md`, `CHANGELOG.md`, `HANDOVER.md`; then a separate doc evaluator checks every default, env var and schema field against the code | E passes |

### WS-A: T1 webhook SSRF (R1, option C)
- `config.py`: `webhook_allow_nets: tuple = ()`, a CIDR list (`WPJUDGE_WEBHOOK_ALLOW_NETS`).
- `api.py`:
  - add `webhook_target(url)`: `urlsplit` once; `getaddrinfo`; unwrap `ipv4_mapped` and NAT64; block when `not is_global or is_multicast`, unless the IP is in the allowlist. Returns `(parts, port, ip)` or raises ValueError.
  - `webhook()`: `http.client` connects to the checked IP (Host, SNI and cert by name). It re-checks at send time and does not follow redirects. A block is not retried.
  - `create_job`: `await run_in_threadpool(webhook_target, url)`, and a ValueError becomes a 400.
  - drop `urllib.request`.
- Unit tests: R1's BLOCKED list (400 and no job dir), allowlist (127/8 allowed, metadata still blocked), rebinding pinned, no redirect, and `test_webhook_delivered` sets the allowlist.
- E2E (`sec_api`, real compose):
  - phase 1, allowlist `127.0.0.0/8`: loopback delivery works, and metadata gives 400;
  - phase 2, no allowlist: 400 for `127.0.0.1`, `2130706433`, `[::ffff:127.0.0.1]`, `localhost` and `http://llama:8080/health`.
  - `sec_allinone`: `webhook_url=http://127.0.0.1:8080/health` gives 400. In that image the harness and the llama server share one container.
  - The unit tests include the parenthesised allowlist condition (R1's bug note).
- Accepted risks: `HTTP(S)_PROXY` is no longer honoured (documented); the private `_create_connection` hook is pinned by a unit test.

### WS-B: T2a thresholds + T3 "other" labels (R2 §a, R3 option F)
- `config.py`: `pattern_min: int = 8` (`WPJUDGE_PATTERN_MIN`).
- `aggregate.py`:
  - `low_rate` uses `x <= cfg.cap_threshold`;
  - `rank(judged, lo=cfg.cap_threshold, hi=cfg.pattern_min)`;
  - replace `TOPICS` with R3's 27-topic lexicon, and add `DIM_TOPICS`, `NA`, `SENT` and `_topics`. Lines are tagged with their own dimension's topics. Only anti lines fall back to the whole lexicon. N/A sentences are dropped, and a line left empty is skipped.
- Not built: pattern `examples` (R3 item 4). It would change the frozen schema. Add it if the user asks.
- Unit tests: `test_thresholds_follow_cfg` (R2), `test_topics_v11` and `test_anti_fallback_and_code_quality_lexicon` (R3). The existing `test_ranking` stays green.
- E2E, on the real and fixture reports:
  - the "other" share of the top-10 counts is ≤ 15% for both lists;
  - no "other" in the top 3;
  - re-aggregate the saved fixture judgments with `WPJUDGE_CAP_THRESHOLD=5`: each `low_rate[d]` equals the share of scores ≤ 5, recomputed from `functions[]`; the sum of the anti counts is ≥ the sum at 4;
  - chk `recompute` takes the threshold from `WPJUDGE_CAP_THRESHOLD` instead of the hard-coded 4 (`e2e.sh:90,132`).
  - Evaluator live run: `e2e.sh real`.
- Document next to the knob: the default of 4 matters. 7 of 8 planted-bug cap reasons sit at exactly 4, so T=3 misses most of them.

### WS-C: T2b critique sync + T4 confidence (R2 §b, R4)
- `judge.py`:
  - both `_judge_create` calls send `logprobs=True` (no `top_logprobs`);
  - `judge_one` takes `verdict_p = exp(logprob)` of the first non-blank, non-quote token after `"verdict":` inside `<judge_output>`. Read the logprobs with `getattr(choice, "logprobs", None)`, because the test Stub has none. `verdict_p` is `None` when there are no logprobs or no verdict;
  - `to_judgment(unit, raw, attempts, verdict_p=None)` stays cfg-free, and the new parameter has a default. It stores `prose_scores` (dim → the original prose number, from the `DIM` regex), then rewrites each dimension header to `Label: score <json or None>/10`. `raw` is unchanged.
- `aggregate.py` `_row` and the summary:
  - each row gets `verdict_p` and `prose_scores`;
  - `confidence_reasons` holds `"verdict_p"` when the value is below `cfg.min_verdict_p`, and `"prose_disagrees"` when the rule below holds;
  - the `prose_disagrees` rule: for some cap dimension, the prose score and the JSON score sit on different sides of `cfg.cap_threshold`. A JSON or prose score that is None, N/A or absent counts as above the threshold;
  - each row gets `low_confidence`;
  - `summary.low_confidence` is the count of rows with it. Skipped rows (`_skip_row`) and unjudged rows get false, `[]` and null.
- `config.py`: `min_verdict_p: float = 0.9` (`WPJUDGE_MIN_VERDICT_P`; 0 disables it).
- Unit tests: R2's `test_critique_synced_to_json` and `test_cap_ignores_prose`, and R4's `verdict_p` extraction cases (split tokens, no logprobs, template path) and confidence cases.
- E2E:
  - fixture: every critique header number equals `scores[d]` or `None`. The check matches headers with the same `aggregate.DIM` regex, which covers only lines that have a `— reason` tail;
  - the `cap_reasons` ids are exactly the 5 planted-bug ids, with no clean unit. This is a hard check on the host cli and golden reports, and a soft check (warning only) on api and aio until one serial b11515 run is recorded;
  - every judged row has `0 ≤ verdict_p ≤ 1`, and the summary count matches;
  - with `WPJUDGE_MIN_VERDICT_P=0`, no row has the `verdict_p` reason;
  - api and allinone (build b11515): `verdict_p` is not null;
  - classic-editor `add_redirect_helper` has `prose_disagrees` (a soft check that warns);
  - record `summary.low_confidence`/units for classic-editor in `V11_E2E.md`, as a precision reference;
  - the logprobs cost is at most +15% of the per-completion-token time, against a serial baseline run without logprobs. Record the number.
  - Unit tests also cover skip rows, by updating the exact-dict assertion in `test_worst_ties_and_rows` (`test_aggregate.py:95`), and existing `to_judgment(UNIT, raw, 1)` calls.
  - Logprobs-cost baseline (round-2 N3): the C evaluator runs classic-editor once at the worktree base commit and once with C, both at `HOST_PARALLEL=1`/w1. It records the ms per completion token for both.
  - Evaluator live run: `e2e.sh cli serverdown`.

### WS-D: T5 worker-count stability (R5, option A)
Root cause: llama.cpp CUDA kernels are not batch-invariant. Concurrent decoding changes the reduction order and so the argmax token. No server flag or sampling option fixes it (cache_prompt, seed/top_k, `-kvu` and `-nocb` were all tested). Serial decoding is byte-identical across runs, configs and processes. With serial decoding, the client worker count is irrelevant (`-np 1` with w4 gives the same result as `-np 4` with w1).
- Default to serial:
  - `config.workers` 4 → 1;
  - `docker/llama-entrypoint.sh` and `docker-compose.yml`: `LLAMA_PARALLEL` default 4 → 1, for both the server and `WPJUDGE_WORKERS`;
  - the allinone entrypoint too, if it sets parallel (grep every `PARALLEL`/`WORKERS` default);
  - `e2e.sh` `HOST_PARALLEL` default 1.
- Surface the fast mode:
  - `judge.server_slots(base_url)` reads `GET <root>/props` → `total_slots`, or `None` on any error;
  - the report `model` gets `workers`, `server_slots` and `reproducible = workers == 1 or server_slots == 1`;
  - the CLI warns once on stderr when `reproducible` is false;
  - the CLI ETA uses 5.5 units/min instead of 11 (`cli.py:30`), and the `judge_units` default changes to `workers=1`;
  - `schema_version` stays 1.0 (additive).
- Cost: −46% throughput (10.0 → 5.4 units/min). Fast mode stays opt-in (`LLAMA_PARALLEL=4 --workers 4`).
- Unit tests:
  - the defaults are 1 in config, the entrypoint and compose;
  - the `reproducible` truth table;
  - `/props` probe success and failure;
  - a fake client with jitter: `judge_units` gives the same result at workers 1 and 4. This guards the client side only; it cannot catch server-side nondeterminism.
- E2E:
  - `golden` at `HOST_PARALLEL=1`: w1 vs w4 must be **exactly** equal on the per-unit scores and verdicts, `summary` (overall, uncapped, `cap_reasons`) and `dimensions`. dir vs zip vs tgz must also be exactly equal. All reports must have `model.server_slots == 1` and `reproducible == true`.
  - new `sec_determinism`, added to the `case` list and to `all`:
    - fail with a clear message if `$WORK/r_dir.json` (from `cli`) is absent;
    - run `host_down`, then start its own `--parallel 4` server on port 8099, with its own PORT/BASE, because `host_up` reads the globals;
    - w4 must report `model.server_slots == 4`, `reproducible == false`, and the stderr warning;
    - w1 on the same server must exactly equal the golden `r_dir`;
    - kill the server afterwards.
  - Evaluator live run: `e2e.sh cli determinism`.
- Accepted limits:
  - Reproducibility is per build and per server config. Host vs container keeps the parity limit of |Δ| ≤ 8.
  - A shared multi-slot server with other traffic is not covered, so the README says to use a dedicated `--parallel 1` server.
  - Deterministic is not the same as accurate: `ser_render_search_box` canonically gives 14.

### WS-E: E2E (separate test agent, then its own evaluator)
- Runs `pytest tests/` and then `tests/e2e.sh malicious serverdown cli library real determinism parity api allinone` live, on merged main, with the GPU lock.
- Records the commands, times and numbers in `build_notes/V11_E2E.md`. Expected serial wall time: cli about 10 min, real about 30 to 40 min including the third target.
- Wall-time bound (T5): classic-editor ≤ 520 s (1.2 × 39 units at 5.4 units/min), replacing the v1 221 s figure, which was measured at 4 workers.
- Determinism with logprobs on: compare the golden `r_dir` per-unit scores against the v1 `r_dir` (uncapped 68.8). Explain any difference; a difference is not a failure.
- Functional checks beyond pass/fail:
  - the fixture still catches all 5 planted bugs, with no clean unit capped;
  - classic-editor is not capped;
  - a third real target: `twentytwelve`, a classic PHP theme, as a pinned versioned zip from downloads.wordpress.org. E counts its units with `extract` before the run, and it must be ≤ 150, otherwise pick a pinned small plugin. Add it to the `sec_real` spec list with its unit count and type. It checks R3's lexicon on unseen code;
  - the parity limit stays |Δ| ≤ 8.

## Evidence on hand
- v1 reports: `build_notes/e2e_work/` (gitignored, local only): `real/*.json`, `r_*.json`, `api_report.json`, `aio_report.json`. Reports drop `raw`. The critique is kept.
- Host server: `~/work/llama.cpp/build/bin/llama-server`. Models: `../models/_gguf/shrink-iq4xs-im_f16.gguf`. Training data: `../data/reasoning_dataset/`.
- Tests: `tests/e2e.sh <section>` and `harness/.venv`.

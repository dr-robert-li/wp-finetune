VERDICT: PASS WITH NOTES

Independent evaluation of phase 5 (tests/e2e.sh run, reported in build_notes/PHASE5_RUN.md).
Evaluator did not run the tests, start GPU servers or containers, edit harness files, or commit.

## Summary

- Every PASS/FAIL count, fixture number, real-target number, theme-split number and parity delta in PHASE5_RUN.md matches e2e.log, the run_<section>.out files and the report JSONs. Only cosmetic claims about the logs are wrong (finding 6).
- An independent recompute of every summary number from functions[] matches exactly for all 12 reports. Nothing in the check relies on chk.py or wpjudge.aggregate.
- Phase-5 coverage is good. A few items are tested only by unit tests, or not at all (finding 4).
- The parity FAIL is real, and the build-drift diagnosis is sound. Binding change 13 is still **not met as written**. The orchestrator or user must decide before phase 6 (finding 1).

## Findings

1. **HIGH: binding change 13 is not met. A decision is needed.**
   - Evidence: parity_host.json vs parity_container.json, recomputed:
     - verdicts 16/16 equal;
     - |Δ| = 1,2,0,0,0,2,2,0,1,4,**8**,2,0,2,0,0, so max 8 and mean 1.50;
     - host ≥ container on all 16 prompts: 9 higher, 7 tied.
   - Both runs are byte-identical: `cmp` against the `*.run1.json` copies is equal.
   - The diagnosis holds:
     - Same GGUF, same 16 prompts, temperature 0.
     - The host rerun at `--parallel 1 -c 8192` (n_slots = 1 per llama-host-p1.log, the same as the container) reproduces the host scores on all 15 parsed prompts, including 82 on prompt 11.
     - So the remaining variable is the build: host `version: 9954 (8f114a9b5)`, GNU 13.3 (I checked with `llama-server --version`; no e2e artifact records it), against container b11515, GNU 14.2, CUDA 13.
     - The one-sided sign pattern (9 of 9 non-zero deltas favour the host) points to a systematic offset, not noise.
   - Caveats the report understates:
     - (a) The ±6 limit was calibrated in r2 on within-build slot nondeterminism. Cross-build drift is a different quantity, so the limit may simply be the wrong yardstick.
     - (b) All 16 parity prompts are PASS on both sides, so "16/16 verdicts equal" says nothing about FAIL-side stability. That gap is covered elsewhere: the fixture runs on b11515 (api, aio) give the same 5 PASS / 5 FAIL per unit as the host runs.
     - (c) The diagnostic does not separate llama.cpp code from CUDA toolkit or compiler. For this decision that does not matter.
   - Suggested action: (b) rebuild the host at b11515 and re-run `tests/e2e.sh parity`, which closes the item cleanly. Or (a) accept and record it as known cross-build drift (max 8 on 1/16). Do not silently widen the limit.

2. **MEDIUM: the golden, library and real-target numbers come from host build 9954, not the shipped build.**
   - Evidence: cli, library and real ran against `127.0.0.1:8097` (the host 9954 server, per model.base_url in the reports). Only api and aio ran on b11515: fixture uncapped 67.7 / 69.7 vs 68.8 on the host.
   - classic-editor 74.9 and twentytwentyfour 81.7 (including the theme split) have not been measured on the container build.
   - Suggested action: if the README quotes these numbers, label them as host-build numbers, or re-run `real` once the host is at b11515.

3. **LOW: on the fixture, the cap hangs on scores exactly at the threshold.**
   - Evidence: of the 8 cap_reasons in r_dir, 6 are score == 4, exactly at the threshold. Only `ser_search_registrations sql_safety 2` and `ser_render_search_box security 1` sit clearly below it.
   - r_w4 and aio have only 1 entry below 4 (`sql_safety 2`).
   - The golden "planted SQLi caps" assertion uses `any(...)`, so it survives. The fixture would stay capped through `ser_search_registrations` even after a +1 drift on the others.
   - Suggested action: none for the harness. Keep binding change 2(b) in mind if the build changes.

4. **LOW: these phase-5 and binding-change items have no live E2E evidence.** They have unit-test or static evidence only, or none.
   - Malicious input goes through the API only for zip-slip. Symlink, bomb and oversize-content archives are not submitted through the API. The CLI path covers all of them.
   - No zip with forged (understated) header sizes. extract.py relies on zipfile stopping at the declared size plus CRC, and that is not exercised.
   - No tar.gz bomb.
   - "No files written outside the job directory" is checked by searching for the planted escape filenames: `find WORK /tmp -maxdepth 4`, the `outside/` dir, and `find /` in the container. There is no full before/after filesystem diff. That is adequate given the fixed names.
   - Restart:
     - a done job survives (tested);
     - a running job becomes failed "interrupted" (tested);
     - a **queued** job is not tested (api.py:129 re-enqueues it);
     - the hourly idle retention sweep is not tested (only the sweep at startup).
   - Binding change 12, job-id `fullmatch`: the `..%2f` probes return 404 from Starlette routing, because a decoded `/` cannot match `{id}`. The regex guard itself is not reached in E2E. tests/test_api.py::test_bad_job_ids covers it.
   - Binding change 12, compose `127.0.0.1:8000:8000`: e2e replaces ports with `!override`. The base file is correct statically (docker-compose.yml:37-38) but is not exercised.
   - The webhook was delivered to a listener inside the harness container (loopback). Delivery across the network to the host is not tested.
   - Binding change 15 README part and "document the restart behaviour" belong to phase 6 (docs). They are not yet evidenced.
   - Suggested action: optional. Add a queued-job restart case and one API symlink or bomb submit. Make sure phase 6 docs cover restart, retention and the template caveat.

5. **LOW: the diagnostic `--parallel 1` pass shows the host slot config changes output in at least one case.**
   - Evidence: in llama-host-p1.log the 14th request decoded 2048 tokens, a runaway that hit max_tokens. That is why prompt 14 has no `<judge_output>`. At `--parallel 4` the same prompt finished normally.
   - "Slot config ruled out" is true for the Δ 8 on prompt 11, not for byte-identity in general.
   - It also shows that runaway generations to max_tokens occur on this model, which the retry-once / unjudged path has to absorb.
   - Suggested action: note it in the docs' small-model caveats. No harness change needed now.

6. **COSMETIC: small inaccuracies in PHASE5_RUN.md.**
   - "Per-section output: run_<section>.out": there is no run_malicious.out or run_serverdown.out.
   - Their walls of 4 s and 1 s are only implied by e2e.log timestamps: malicious 12:45:41 to 12:45:46, serverdown to 12:45:50. That is consistent but not recorded.
   - "The `post:` line at the end of each run_<section>.out": run_cli.out has no `post:` line, although its in-suite "nvidia-smi compute apps after stop:" is empty.
   - The log's real-section INFO line says "function n=39" for classic-editor, because chk.py lumps non-template units together. The report correctly says kind=method, which I confirmed: all 39 rows are `method`.
   - Suggested action: none.

## Verified claims (all match)

- **Counts:**
  - malicious 17/0, serverdown 18/0, cli 87/0, library 33/0, real 28/0, parity 2/1 (twice), api 41/0, allinone 17/0;
  - first-run total 243/1;
  - walls from the run files: cli 573, library 257, real 570, parity 351/352, api 114, allinone 86 s;
  - session 12:45:41 to about 13:29:31 (44 min). The diagnostic gap is 13:22:06 to 13:25:55.
- **Fixture:** overall/uncapped/capped/n_cap/verdicts:
  - dir/zip/tgz 60/68.8/T/8/5-5;
  - w4 69.1;
  - lib_dir/lib_zip 68.8;
  - api 67.7;
  - aio 69.7.
- **Fixture, cross-checks:**
  - dir vs zip vs tgz: identical per-unit scores (max Δ 0);
  - w4 verdicts equal;
  - planted [36,46,46,14,46] < clean [80,80,80,84];
  - the planted < clean ordering also holds in w4, api and aio (max planted 46, min clean 78);
  - the pre-run capped both SQLi units (sec 4/sql 2 and 4/4);
  - skips {build, tiny, test, vendor, minified} = 1 each;
  - files_scanned 7;
  - sha256 empty for dir, set for archives.
- **Real targets:**
  - classic-editor: 39 units, 1 skip (tiny), 100%, 74.9 uncapped = capped, 39 PASS, files_scanned 1, type plugin, ETA "~3.5 min", wall 221 s, all kind=method, mean 75.3, LOC-weighted 74.9, LOC 892.
  - twentytwentyfour: 60 units, 0 skips, 100%, 81.7, 60 PASS, type theme, ETA "~5.5 min" (60/11 = 5.45), wall 337 s.
  - Theme split: template n=57 mean 81.6, LOC-weighted 81.6, LOC 3329; function n=3, 84.0/84.0, LOC 152; difference 2.4.
- **API:**
  - hook.json is byte-for-byte the same object as api_report.json (15 rows);
  - j2.json = failed "interrupted by server restart; resubmit";
  - st.json shows running with done=1 (partial results);
  - all guard status codes appear in e2e.log as reported.
- **Parity:** smoke `build 11515, commit 3d65c90d0`; the deltas are as listed in finding 1.
- **serverdown (r_down.json):** 10 unjudged, overall null, low_coverage true, not capped, a valid report.

## Independent recompute (my own script, not chk.py)

Script: `<scratchpad>/recompute.py`, run with `python3 -I`. It recomputes from functions[] only, using the PLAN/SPEC definitions:

- coverage (units, judged, unjudged, skipped, pct rounded to 1 dp, low_coverage);
- LOC-weighted uncapped mean rounded to 1 dp, and capped = min(uncapped, 60) when any judged security|sql_safety ≤ 4;
- cap_reasons compared as an exact set of (id, dim, score);
- verdict_counts including zero keys;
- dimensions (mean and population std rounded to 2 dp, min, n_rated, low_rate rounded to 3 dp, and an exact key set);
- worst_functions as the 10 lowest by (score, id), with all 5 fields compared;
- skip-row shape per binding change 4 (exact key set, loc 0, nulls);
- unit-row required keys, and id == file:line:name;
- anti_pattern examples refer to judged rows; patterns have no examples; at most 10 of each.

Results, with zero discrepancies in each:

| Report | Overall | Uncapped (raw) | Capped | n cap_reasons | Dimensions / worst / coverage / verdicts |
|---|---|---|---|---|---|
| real/classic-editor.json | 74.9 | 74.88 | no | 0 | exact |
| real/twentytwentyfour.json | 81.7 | 81.664 | no | 0 | exact |
| r_dir.json | 60 | 68.784 | yes | 8 (= my set) | exact |
| api_report.json | 60 | 67.712 | yes | 8 | exact |
| aio_report.json | 60 | 69.696 | yes | 8 | exact |

I also ran the same check on r_zip, r_tgz, r_w4, lib_dir, lib_zip, r_down and r_stdout: zero discrepancies. No judged row has loc < 1, so the `max(loc,1)` weighting in aggregate.py makes no difference here.

## What I ran

- Read PLAN.md (phase 5, binding changes 1-16), SPEC.md (schema v1), PHASE5_RUN.md, tests/e2e.sh (including the embedded chk.py), wpjudge/aggregate.py, the intake part of wpjudge/extract.py, and the api.py grep for restart, sweep and job id.
- e2e.log and run_*.out: PASS/FAIL tallies, `== wpjudge` / `== total` timestamps, INFO lines.
- `cmp` of the parity_*.json files against their *.run1.json copies; a per-prompt diff of host vs container, host_p1 vs host, and host_p1 vs container.
- llama-host*.log: n_slots, and the decoded-token counts per request (p1 request 14 = 2048).
- `~/work/llama.cpp/build/bin/llama-server --version` and `git rev-list --count HEAD` → 9954 (8f114a9b5). This is a read-only version query; no server was started.
- The independent recompute script above, plus a per-unit table of verdict, score, security and sql_safety across dir/zip/tgz/w4/api/aio, against vuln-plugin.expected.json and vuln-plugin.prerun.json.
- A comparison of hook.json with api_report.json, and inspection of j2.json, st.json, real/*.err, r_*.err and mal/tmp (empty).

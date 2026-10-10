# EVAL v1.1 WS-E: live E2E on merged main (independent evaluator)

Tree: HEAD a8c457f plus the uncommitted `tests/e2e.sh` diff (2 hunks). Inputs: `V11_EVAL_BRIEF.md`, `PLAN_v1.1.md` WS-E, `V11_E2E.md`, and the artifacts in `build_notes/e2e_work/v11/`. I edited nothing except this file and made no commits. The `wpjudge` import resolves to `harness/wpjudge/judge.py` (main checkout).

## 1. Re-derived numbers (from the artifact JSONs, not the report)
| check | result |
|---|---|
| golden `v11/r_dir.json` vs v1 `e2e_work/r_dir.json`, per unit (overall_score, verdict, scores) | **identical on all 10 judged units** (differing units: []). v1 summary 60 / uncapped 68.8, v1.1 the same |
| r_zip, r_tgz, r_w4, lib_dir, lib_zip vs golden | units, summary (overall, uncapped, cap_reasons) and dimensions all exactly equal. All have `server_slots 1, reproducible true` (r_w4 workers 4) |
| det_w1 (`-np 4`, w1) vs golden | exactly equal. `server_slots 4, reproducible true` |
| det_w4 | `server_slots 4, workers 4, reproducible false`. Uncapped 69.9, differs from golden. The warning is in `det_w4.err`, and `det_w1.err` has no warning |
| cap_reasons in host reports (r_dir/zip/tgz/w4, lib_dir/zip, det_w1) | function set == the 5 planted bugs {search_registrations, delete_registration, ajax_export_registrations, render_search_box, save_settings} |
| planted vs clean (golden) | planted 36/46/46/14/46, all FAIL. Clean 80/80/80/84/84, all PASS, none capped. (The report lists 4 clean values; there are 5. The `<template>` scores 84. Cosmetic.) |
| api/aio (container b11515, serial) | 4/5 planted capped. ser_search_registrations is `unjudged` (parse_failure, attempts 2), coverage 90.0, overall 60, uncapped 72.0. The check is soft, so it logs WARN |
| classic-editor | 39 units, 39 judged, wall 426 s (limit 520 s). **capped True**, overall 60, uncapped 73.9. Single cap_reason `classic-editor.php:925:Classic_Editor::add_edit_php_inline_style` security 4 (v1 had security 7, overall 68). 26/39 units changed some score vs v1 (20/39 changed overall_score) |
| twentytwelve | zip sha256 a151ec03…5b9e. I re-extracted it: **54 units (28 function, 26 template), 0 skips, 28 php, type theme**. Report: 53 judged / 1 unjudged (`search.php` parse_failure), coverage 98.1, overall 78.3, not capped, 53 PASS. 'other' share 0/306 in patterns, none in the top 3, anti_patterns empty |
| twentytwentyfour | 60/60 judged, 81.2, not capped |
| parity run 1 | host prompt 13 = null, so 15/16 verdicts and only 15 deltas. Both checks FAIL |
| parity re-run | 16/16 verdicts, max abs delta 8 (prompt 10: 82 vs 74), mean 1.50 |
| sec_*.out | malicious 17/0, serverdown 18/0, cli 154/0, library 43/0, real 110/0, determinism 7/0, parity run1 1/2, parity 3/0, api 51/0 (1 WARN), allinone 22/0 (1 WARN). Matches `times.txt` and the report |

## 2. Review of the e2e.sh edits
- twentytwelve spec line: pinned versioned URL (`twentytwelve.4.9.zip`, not latest-stable), unit count 54 matches my extract, type theme. It uses the existing spec format. Correct and minimal.
- `_send` gets `"cache_prompt": False`. This is the same extra_body field that `judge.py::_judge_create` already sends. It affects `parity-run` (host and container) and `prerun`. One token added on one line. Correct and minimal. Note that the change also moves the container's own prompt 11 from 76 to 78, which is expected because both sides now decode without slot-KV reuse.

## 3. Independent check of the parity root cause
I started my own host server (port 8191, `--parallel 1 -c 8192`, same flags as `host_up`) under `flock /tmp/wpjudge-gpu.lock` and sent the 16 parity prompts in sequence twice:
- `cache_prompt` default (true): prompt 13 is **null, finish_reason=length** (repetition loop). Prompt 11 = 78.
- `cache_prompt: false`: prompt 13 = **PASS 66, stop**. Prompt 11 = 76.

This reproduces both parity run 1 (13 null, 11 = 78) and the re-run (13 = 66, 11 = 76) exactly. I then ran `tests/e2e.sh parity` on the patched tree with a fresh WORK (`scratchpad/evw/pwork`). Result: **3 passed, 0 failed, 16/16 verdicts, max 8, mean 1.50.** The per-prompt host/container pairs are identical to the report's re-run. The root-cause claim holds, and the fix is in the right place: the test harness's private client was missing a field the product client already sends.

After my runs, `nvidia-smi` shows no compute apps, there is no `llama-server` process, and `docker ps` is empty.

## 4. Can the retry after a parse_failure recover?
From the code: `judge_one` loops `for attempt in (1, 2)` and sends the **identical** messages through `_judge_create` with temperature 0.0 and `cache_prompt: False`. Nothing changes between attempts: no seed, no temperature bump, no prompt change. On a single-slot server each attempt computes the full prompt from scratch with the same batch shape, so greedy decode is deterministic.

Live evidence from the same probe server:
- twentytwelve `search.php` via `_judge_create` twice: both finish_reason=length, len 8005, **md5 87ac1436… identical**. `judge_one` then returns `unjudged, attempts 2, parse_failure`.
- Parity prompt 13 via `_judge_create` twice: both 66, md5 identical.

Conclusion: **at temperature 0 with cache_prompt false on a serial server, the retry is byte-identical, so it can never recover a repetition loop.** It only helps with transport exceptions, and in theory with output from concurrent multi-slot decoding, which is now flagged non-reproducible. Any planned fix has to change the second request. Examples: a repetition penalty or DRY sampler, a small temperature or seed on attempt 2, or a larger `max_tokens`. The last would not help a true loop.

The api `ser_search_registrations` row has attempts 2 and parse_failure. The report keeps only one raw (`ssr.py.raw`, which loops until length), so the container artifact alone cannot show that the two attempts were identical. The host probe above settles the mechanism.

## 5. Plan WS-E items
| item | status |
|---|---|
| pytest then the 9 sections, live, GPU lock, recorded | met (pytest 70 passed per report; all sections pass after the fix) |
| classic-editor wall ≤ 520 s | met (426 s). INFO only, no e2e assert. MINOR: a check for this would be cheap |
| golden r_dir vs v1, explain differences | met (no difference) |
| fixture catches 5 planted bugs, no clean unit capped | met on host. Container 4/5 (soft WARN, see decisions) |
| **classic-editor not capped** | **NOT met.** Capped by a security-4 false positive on a static inline `<style>` echo. This is reproducible under serial decoding. The harness is wired correctly (the cap invariant recomputes, and the model gives the score). This is a model-accuracy outcome, covered by the accepted limit "Deterministic is not the same as accurate". It does not block WS-E as a harness or test defect, but it needs a user decision |
| twentytwelve third target, pinned, ≤ 150 units, in spec list | met (54, theme, pinned 4.9). R3 lexicon 'other' share 0 |
| parity ≤ 8 | met after the `_send` fix (max exactly 8) |

Findings:
1. MAJOR (fixed in the uncommitted diff, verified): the parity/prerun `_send` lacked `cache_prompt: false`, so parity failed under the new `HOST_PARALLEL=1` default. The fix is correct and reproduced. It needs committing.
2. MINOR: `parity-cmp`'s second message reads "max 8 <= 8" while failing, because it actually failed on `len(d) != 16`. The message could state the delta count.
3. MINOR: neither "classic-editor wall ≤ 520 s" nor "classic-editor not capped" is asserted in `sec_real`. Both are only visible as INFO.
4. MINOR (report accuracy): the report lists 4 clean scores instead of 5, and says 26/39 "changed score" where 20/39 changed overall_score. Neither changes a conclusion.

## Decisions for the user
1. **classic-editor capped** (add_edit_php_inline_style security 4, a false positive, reproducible). Options: accept it as a model limit under "deterministic ≠ accurate", or add it as v1.2 training data. A harness change cannot fix it, short of changing the cap rule.
2. **Container serial b11515 run**: ser_search_registrations loops until length on both attempts, so 4/5 planted bugs are capped and coverage is 90%. This is the serial b11515 run the soft check (`e2e.sh:200`) was waiting for. Decide whether api/aio stays soft (known model degeneration on that build) or becomes hard.
3. **Repetition loops** (container ser_search_registrations, host twentytwelve search.php, host parity 13 when the cache is reused): the retry is byte-identical (section 4), so the only fix is a change to the judge's second request, such as a repetition penalty, DRY sampler or temperature on attempt 2. That is a judge-side change outside WS-E. It would change reproducible scores only for units that currently fail to parse, if it applies on attempt 2 only.

## Commands run
- Re-derivation scripts over `e2e_work/v11/*.json`, `v11/real/*.json`, `e2e_work/r_dir.json`, `e2e_work/real/classic-editor.json`.
- `wpjudge.extract.load_target` + `collect_units` on `twentytwelve.zip` (54 units).
- `flock /tmp/wpjudge-gpu.lock bash scratchpad/evw/probe.sh` (host server on 8191: seq-16 with and without cache_prompt, search.php x2, judge_one, p13 x2). Output in `scratchpad/evw/probe.out`.
- `flock /tmp/wpjudge-gpu.lock sg docker -c "GGUF=… WORK=scratchpad/evw/pwork tests/e2e.sh parity"`: 3 passed, 0 failed, 5 min 19 s.
- After the runs: nvidia-smi empty, no llama-server, no docker containers.

VERDICT: PASS

# V1.1 WS-B build notes: T2a thresholds + T3 "other" labels

Branch: `worktree-agent-a19a4a11a3d7ac300`. Worktree: `/home/rli/Desktop/projects/wp-finetune/.claude/worktrees/agent-a19a4a11a3d7ac300`.
Code commit: `907fffe` (on top of main `5e5ea1a`). This notes file is in a second commit.

## What changed
- `wpjudge/config.py` `Config`: new `pattern_min: int = 8` (`WPJUDGE_PATTERN_MIN`). The comment on `cap_threshold` now says it also sets the `low_rate`/anti-pattern bound, and that 4 is load-bearing (7 of 8 planted-bug cap reasons sit at exactly 4; v11_r2 §a).
- `wpjudge/aggregate.py`:
  - `TOPICS` is replaced with R3's 27-topic lexicon, copied verbatim. New: `DIM_TOPICS`, `NA`, `SENT` and `_topics(d, reason, kind)`. A line is tagged with its own dimension's topics. Only anti lines fall back to the whole lexicon.
  - `rank`: N/A sentences are stripped from the reason (split by `SENT`). A line left empty is skipped. Tags come from `_topics`. The seen/count/examples logic is unchanged.
  - `build_report`: `low_rate` uses `x <= cfg.cap_threshold`. The call is now `rank(judged, lo=cfg.cap_threshold, hi=cfg.pattern_min)`.
- `tests/test_aggregate.py`: new `test_thresholds_follow_cfg`, `test_topics_v11` and `test_anti_fallback_and_code_quality_lexicon`. The last two are R3's sketches, copied verbatim. `test_ranking` is unchanged and still passes.
- `tests/e2e.sh` (chk.py only, plus 2 lines in `sec_real`):
  - `CAP_T` now comes from `WPJUDGE_CAP_THRESHOLD` (default 4). `recompute(r, tag, t=CAP_T)` uses `t` for the cap invariant and for `low_rate`.
  - new `topics(r, tag)`: for both lists, the "other" share of the top-10 counts is ≤ 15%, and there is no "other" in the top 3. It is called from `cmd_real`.
  - new `chk rethreshold <report>...`: it rebuilds each saved report from `functions[]` with the current `aggregate.build_report`, at `WPJUDGE_CAP_THRESHOLD=4` and then `=5`. On the T=4 rebuild it runs `topics`. On the T=5 rebuild it runs `recompute(..., 5)`, and it checks that the anti count sum at T=5 is ≥ the sum at T=4.
  - `sec_real` runs `rethreshold` on each real report, and on `$WORK/r_dir.json` (the fixture) when that file exists. If it is absent, it notes that the fixture re-threshold check was skipped.

## Tests
- `.venv/bin/python -m pytest tests/`: **38 passed**.
- Offline, against the saved v1 reports (`harness/build_notes/e2e_work`):
  - `chk rethreshold` on classic-editor, twentytwentyfour, r_dir, api_report and aio_report: all PASS. On the old critiques, "other" in the top 10 is 0 everywhere.
  - At T=5, classic-editor becomes capped (`add_redirect_helper` security 5). This is expected, and it is why the default stays 4.
  - Control: `chk real` on the v1 classic-editor report **FAILS** the new check (patterns "other" 35/105, and "other" in the top 3). So the check catches the v1 behaviour.

## Live run (`e2e.sh real`, under the GPU flock)
`GGUF=…shrink-iq4xs-im_f16.gguf HOST_PARALLEL=1 WPJUDGE_WORKERS=1 PORT=8197 WORK=<scratch, with v1 r_dir.json copied in> flock /tmp/wpjudge-gpu.lock bash tests/e2e.sh real`
- Result: **69 passed, 0 failed**. The server was stopped afterwards (the GPU compute app list was empty).
- classic-editor: 39 units, wall 398 s (serial), overall 73.9, not capped.
  - Patterns "other": 0/131.
  - Top 5: WP API choice 25, dependencies 18, readability 17, efficiency 14, formatting 11.
  - No anti rows.
- twentytwentyfour: 60 units, wall 632 s, overall 81.1, not capped.
  - Patterns "other": 0/419.
  - Top 3: block markup 49, dependencies 48, WP API choice 47.
  - Anti: 1 row (accessibility markup), with "other" 0/1.
- Re-threshold at T=5, anti count sums (T=5 vs T=4):
  - classic-editor 0 vs 0;
  - twentytwentyfour 3 vs 1;
  - fixture r_dir 23 vs 22.
  - At T=5, `low_rate` and the cap invariant recompute from `functions[]` with PASS.

## Deviations
- The other-share check is applied to the fixture via `rethreshold` on `r_dir.json`, a rebuild with the current code. `cmd_fixture`/`cmd_golden` belong to WS-C/WS-D, so I did not touch them. To run the check against r_dir, run `cli` before `real` in the same `WORK`.
- The `rethreshold` call lines in `sec_real` are the only edits outside chk.py. WS-E owns the spec list in that function, which I left untouched.
- Pattern `examples` (R3 item 4) are not built, as the plan says.
- Docs (SPEC A5 "27-topic lexicon", README knob note) are left to WS-F.

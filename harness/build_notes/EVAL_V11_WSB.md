# EVAL v1.1 WS-B: T2a thresholds + T3 "other" labels

Evaluator: independent (did not build). Worktree `agent-a19a4a11a3d7ac300`, HEAD `8421656` (code `907fffe`), diff `5e5ea1a..HEAD`.
`wpjudge.__file__` = `<wt>/harness/wpjudge/__init__.py` (worktree venv). No code edited.

## Spec compliance (PLAN_v1.1.md WS-B)
| Item | Status |
|---|---|
| `config.pattern_min = 8` (`WPJUDGE_PATTERN_MIN`, via generic `load()` env cast) | done |
| `low_rate` uses `<= cfg.cap_threshold` | done (aggregate.py build_report) |
| `rank(judged, lo=cfg.cap_threshold, hi=cfg.pattern_min)` | done |
| 27-topic `TOPICS`, `DIM_TOPICS`, `NA`, `SENT`, `_topics`; own-dim tags, anti-only fallback, N/A sentences dropped, empty line skipped | done; byte-for-byte R3 proposal |
| Pattern `examples` not built | correctly not built (schema frozen) |
| Unit tests `test_thresholds_follow_cfg`, `test_topics_v11`, `test_anti_fallback_and_code_quality_lexicon`; `test_ranking` green | done |
| E2E: "other" share <= 15% of top-10 and none in top 3, on real + fixture | done (`topics()` in `cmd_real`; fixture via `rethreshold` rebuild of `r_dir.json`) |
| E2E: re-aggregate fixture at `WPJUDGE_CAP_THRESHOLD=5`, `low_rate` recomputed from `functions[]`, anti sum T5 >= T4 | done (`cmd_rethreshold`, uses independent `recompute(r5, tag, 5)`) |
| chk `recompute` threshold from env, not hard-coded 4 | done (`CAP_T` from env; both cap and `low_rate` use `t`) |
| Knob documentation (keep 4; 7/8 planted caps at 4) | done (config.py comment) |

Deviations (fixture check runs as a rebuild of saved `r_dir.json` rather than inside `cmd_fixture`; SPEC/README to WS-F) are justified by workstream ownership and are documented in V11_WSB.md.

## Findings

1. **MINOR: `docblock` regex `\bcomments?\b` mis-tags block-markup comments and `phpcs:ignore` comments as docblock.**
   Evidence (live twentytwentyfour report): 26 code_quality lines match `docblock`, and 17 of them match only on "comment(s)". Examples: "The use of `<!-- wp: ... -->` comments is correct for block parsing" and "block comments are used correctly". classic-editor: the wpcs line "suppresses InputNotSanitized ... with a phpcs:ignore comment" is tagged `docblock`. This accounts for 3 of the 4 wrong labels in the hand-check below. It inflates `code_quality/docblock` on block themes.
   Fix: drop the bare `\bcomments?\b` alternative, or guard it with `(?<!block )(?<!wp: )comments?` and `(?<!ignore )`. The `inline (comment|documentation)` and `well-documented` alternatives already cover the real cases.

2. **MINOR: the `NA` sentence filter occasionally drops real defect sentences on anti lines.**
   This was measured on the training/val prose (`data/reasoning_dataset/openai_*.jsonl`), with 6395 lines at score <= 4.
   - Whole-line drops outside accessibility/i18n: 0. All the whole-line drops are genuine "No HTML output; not applicable" lines, which is the intended behaviour.
   - Sentence-level: 409 sentences matched NA outside a11y/i18n. In a 30-sample, about 4 were real defects. Examples:
     - "format_error returns a string containing __() output concatenated directly into HTML without escaping": dropped by `returns (only )?(a |an )?(...string...)`;
     - "calling get_option() on every save, even if no changes are made, is wasteful": dropped by `\bno (\w+[ -]){0,2}changes?`;
     - "the function returns an array keyed by ...".
   - Synthetic phrasings also drop whole lines: "Does not perform a nonce check before saving.", "Does not handle errors from wp_remote_get.", "No user input is sanitized before use." (`(does not) (perform|handle)`, `no (user )?input`).
   - Impact: anti-pattern counts only. The cap and `low_rate` come from the JSON scores and are unaffected. The estimated loss is under 1% of anti sentences, and the real-report anti lines lose 0.
   Fix (optional, later): apply NA to anti lines only when the sentence also names the line's own dimension or contains "applicab", or drop the `returns ...` and `perform|handle` verbs from the anti-side NA. R3 already lists this as a known risk.

3. **INFO: top-10 "other" metric vs line level.** The top-10 counts are 0% "other" on all reports. Each (dim, "other") pair is its own row, so fragmentation can hide "other" in that metric. At line level on the live reports, 22 of 466 tagged lines (4.7%) are "other". That is still well under 15%, and down from about 14% in v1 (R3), so there is no action needed.

4. **INFO: polarity follows the JSON score by design.** One wpcs line scored 8 describes a defect ("paper-shielding an unsanitized superglobal read") and is counted as a *pattern*. This is model behaviour, outside WS-B's scope. WS-C's prose/JSON sync touches the related cap case.

No BLOCKER or MAJOR.

## Code review
- The diff is small and in the house style: regex dict, `ponytail:` comment, no new deps. The `rank` loop logic is unchanged apart from the NA strip and `_topics`.
- `_topics`: own-dimension pool first, then the full lexicon for anti only, then `['other']`. An unknown dim falls back to the full lexicon. This is correct.
- An edge case that is harmless: `pattern_min <= cap_threshold` gives no "ignored" band (scores above lo become patterns). It is not validated, and it is a nonsense setting anyway.
- `cmd_rethreshold` sets `os.environ` in its own chk process only. `recompute` at T=5 is independent of `aggregate`, so it is a real check and not circular.

## Unit tests
`.venv/bin/python -m pytest tests/ -q`: **38 passed**.
The new tests assert concrete behaviour:
- the cap/anti/low_rate flip at T=5;
- the pattern drop at pattern_min=9;
- N/A drop vs "No nonce" kept;
- `esc_html_e` in i18n not counted as escaping;
- anti fallback.

All of them fail on the v1 code (`Config` has no `pattern_min`, and the v1 lexicon has no `efficiency`/`readability`/`debug code`). They are meaningful.

## Live e2e (run by evaluator)
```
cd <wt>/harness
GGUF=/home/rli/Desktop/projects/wp-finetune/models/_gguf/shrink-iq4xs-im_f16.gguf HOST_PARALLEL=1 WPJUDGE_WORKERS=1 \
  PORT=8297 WORK=<scratchpad>/wsb flock /tmp/wpjudge-gpu.lock bash tests/e2e.sh real
```
(`WORK` was pre-seeded with the v1 fixture `r_dir.json` from `harness/build_notes/e2e_work/`, so the fixture re-threshold check ran. The coordinator's `vuln-plugin/build` note does not apply, because the `real` section does no fixture scan.)
- **Result: 69 passed, 0 failed.** The server was stopped afterwards (nvidia-smi compute list empty after stop).
- classic-editor: 39 units, wall 391 s, overall 73.9, not capped.
  - Patterns "other" 0/131.
  - Top: WP API choice 25, dependencies 18, readability 17, efficiency 14, formatting 11.
  - 0 anti rows.
- twentytwentyfour: 60 units, coverage 98.3%, wall 640 s, overall 81.1.
  - Patterns "other" 0/419.
  - Top 3: block markup 49, dependencies 48, WP API choice 47.
  - Anti: accessibility markup 1.
- Re-threshold, anti sums at T5 vs T4:
  - classic-editor 0 vs 0;
  - twentytwentyfour 3 vs 1;
  - r_dir 23 vs 22.
  - The T5 cap invariant, `low_rate` and dimensions all recompute (r_dir at T5: 8 cap reasons, overall 60).
- These numbers match the builder's report.

## Label precision hand-check (live reports)
I regenerated the per-line tags with `aggregate._topics` from both live reports, and took a random sample of 30 pattern lines (seed 7) plus the only anti line. That gives 38 non-"other" labels, of which **34 are correct (89%)**. This matches R3's 90% target, against 71% for v1.
- The 4 wrong labels: `docblock` x3 (finding 1); `block markup` for "standard block editor classes" in a dependency_integrity line (borderline).
- The 6 "other" lines in the sample are recall misses, not wrong tags. Examples: "calls update_network_option() twice conditionally"; "var(--wp--preset--spacing--50) ... preset".
- The anti line (alt="" without aria/role) → `accessibility markup`: correct.
- NA-dropped pattern lines on the live reports: 6. All are genuine N/A, for example "no external dependencies; dependency integrity does not apply" and "no performance concerns".

VERDICT: PASS

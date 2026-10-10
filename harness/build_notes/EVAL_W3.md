VERDICT: PASS

# W3 aggregate — independent evaluation

Judged against SPEC.md report schema v1, PLAN.md Contracts + W3 locked entry + binding changes 4/5/6/7/8,
research/r2_judge.md §3/§5, research/proto/rank_r2.py, and build_notes/W3.md. No blockers or majors.

## Defects
1. minor — wpjudge/aggregate.py:79,104. `low_coverage` compares the *rounded* pct to the threshold, so true
   coverage just under 90% is not flagged: 899/999 judged = 89.99% -> pct 90.0 -> low_coverage false
   (probed). R2.3 says flag under 90%. Fix: compute `ratio = len(judged)/n_units*100` and use
   `ratio < cfg.coverage_threshold`; round only for the `pct` field.
2. minor — wpjudge/aggregate.py:101. A judged row whose JSON has no `verdict` (judge.py:125 allows
   `j.get("verdict")` = None) adds a `None` key to verdict_counts, serialised as `"null": n`. Plan says
   verdicts "as given", so tolerable; if unwanted, count only string verdicts or map None to "UNKNOWN".
3. minor — wpjudge/aggregate.py:89. worst_functions ties broken by lexical id: ties at lines 1,2,10,11 in
   one file come out 10,11,1,2 (probed). Deterministic and literally matches the adopted optional item
   ("ties broken by id"), so acceptable; reads oddly in a report. Cheap improvement:
   `key=lambda j: (j["overall_score"], j["file"], j["line"], j["name"])`. Not required.
4. minor (outside W3, config.py:16) — `cap_value: float = 60` holds int 60, so a capped report emits
   `overall_score: 60` (int) while uncapped is a float (80.0); also `WPJUDGE_CAP_VALUE=59.5` would fail
   `int("59.5")` in `_cast`. Fix in config: `cap_value: float = 60.0`. JSON-valid either way.

## Conformance confirmed (recomputed independently)
- Top-level keys, target/model/summary/coverage/dimension/anti_pattern/pattern/worst/functions shapes match
  schema v1 exactly; `json.dumps(allow_nan=False)` succeeds.
- Overall = LOC-weighted mean (weight max(loc,1)), 1 dp; cap: any judged unit with security|sql_safety <=
  cap_threshold -> capped, min(uncapped, 60), cap_reasons `[{id, dimension, score}]`, one per (unit, dim);
  prose-only scores ignored. Cap never raises a lower score.
- Zero judged: overall/uncapped null, capped false, low_coverage true, dimensions {}, worst [];
  zero units: pct 0.0 (binding 8).
- Coverage: units = judged+unjudged, skipped = len(skips), pct = judged/units*100 (binding 7).
- Dimensions: fmean, pstdev (population, verified vs [2,8] -> 3.0), min, n_rated, low_rate (<=4), judged
  rows only, numeric JSON scores only.
- rank: matches r2 §5 logic; anti <=4 / pattern >=8, 5-7 dropped (boundary probed 4/5/7/8); counted once
  per (kind, dim, topic) per function even across duplicate lines; anti examples "file:line name",
  patterns have no examples; judged rows only (binding 5); DIM/TOPICS/rank live here (binding 6).
  DIM matched 9 lines in each of the 4 parseable real fixture responses (both prose formats).
- functions[]: judgments minus code/raw (keeps id, kind, end_line, attempts, error); skip rows exactly the
  binding-4 shape. files_scanned arg wired from scan() in __init__.py:29.
- No over-engineering found; 112 lines, stdlib only. The test file is a real known-answer check.

## What I ran
- `harness/.venv/bin/python -m pytest tests/test_aggregate.py -q` -> 6 passed.
- Scratchpad probe (probe_w3.py): 40 random judged + 4 unjudged + 2 skips with my own recomputation of
  overall, cap set, coverage, verdict_counts, per-dim mean/pop-std/min/n/low_rate, worst order, key sets
  -> all equal; real response fixtures through judge.to_judgment -> rank; edge probes for empty input,
  coverage rounding, None verdict, loc 0, id ties, cap type, duplicate-topic lines, score boundaries.

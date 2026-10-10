# v1.1 R2: score consistency (T2)

Scripts and outputs: session scratchpad `train_cmp.py`, `train_fmt.py`, `rep_cmp.py`. No GPU run was needed (see §3).

## TL;DR
- **(a)** `aggregate.py` hard-codes `4` twice (`low_rate`, the `rank` anti-pattern bound) and `8` once (the `rank` pattern bound). None of them reads `cfg`. Fix: both `4`s become `cfg.cap_threshold`, and `8` becomes a new knob `cfg.pattern_min` (default 8).
- **(b) Root cause: the training data.** `scripts/relabel/rebuild_targets.py` relabeled the judge targets (v1.3 relabel, reused for v4) and rewrote the prose numbers with a regex that only matches `Header: score N/10`. Prose written as `Header: severity <word> (N/10)` (29% of judge examples) kept the old teacher's numbers. Those numbers are a fixed lookup from the severity word: low→8, medium→6, high→4, critical→2. Dimensions the relabel dropped as not applicable also kept their old prose numbers, and `error_handling` was added to the JSON with no prose line. The model reproduces this exactly. Generation order and parsing are fine.
- **Recommendation:** keep `<judge_output>` as the only score source (status quo for the cap and the statistics). Then rewrite the dimension headers in the stored critique from the JSON, so the report agrees with itself. Do not trust the prose, and do not take min(prose, JSON): either one adds false positives (fixture clean unit, classic-editor). Do not add a disagreement flag either: it is systemic noise. Add a retrain note to fix `rebuild_targets.py`.

## (a) Hard-coded thresholds in aggregate.py

| Where | Literal | Meaning | Derive from |
|---|---|---|---|
| `build_report` dims: `low_rate = sum(x <= 4 ...)` | `4` | share of "bad" scores per dimension | `cfg.cap_threshold` |
| `rank(judged, lo=4, ...)`, `kind = 'anti' if s <= lo` | `4` | anti-pattern bound | `cfg.cap_threshold` (same rule: "≤ T is bad", which is the cap's meaning) |
| `rank(..., hi=8)`, `lo < s < hi` skipped, `s >= hi` = pattern | `8` | good-practice bound | new `Config.pattern_min: int = 8` (env `WPJUDGE_PATTERN_MIN`). It is independent of the cap, and 8 is the model's modal "fine" score. |
| `rank(..., top=10)`, `worst[:10]` | `10` | list lengths | leave as is (presentation, not a threshold) |
| call site `rank(judged)` | n/a | passes no cfg | `rank(judged, lo=cfg.cap_threshold, hi=cfg.pattern_min)` |

The cap (`cfg.cap_dims`, `cfg.cap_threshold`, `cfg.cap_value`) and `low_coverage` (`cfg.coverage_threshold`) already read cfg. If `pattern_min <= cap_threshold`, the `anti` branch is checked first, so nothing breaks. No validation is needed.

Calibration warning (relevant to anyone setting `WPJUDGE_CAP_THRESHOLD`): on the fixture, 7 of the 8 v1 cap reasons sit at exactly 4 (`r_dir.json`). In `aio_report.json` all are 4 except sql 2. With T=3 the cap catches only 2 of 5 planted bugs (r_dir) or 1 of 5 (aio). The default of 4 is load-bearing. Document this next to the knob.

## (b) Prose vs stored score

### 1. Mismatch in the saved reports
`python3 rep_cmp.py 4 real/classic-editor.json real/twentytwentyfour.json r_dir.json api_report.json aio_report.json`. The script parses each dimension header in `critique` and compares it with `scores`.

| | count |
|---|---|
| judged units | 129 (52 severity format, 79 score format; one format per unit) |
| prose numeric and JSON present | 857 pairs |
| score-format pairs equal | **534 / 534 (100%)** |
| severity-format pairs differing | **184 / 323 (57%)** |
| units with at least one mismatch | 46 / 129 (36%) |
| prose numeric, JSON absent (N/A in JSON) | 280 |
| JSON dimension with no prose line (all `error_handling`) | 37 |
| prose header with the word "low" but N=4 (incoherent) | 9 (one twentytwentyfour `<template>` unit; JSON gives 7 to 8) |

JSON minus prose: −6:1, −5:1, −4:8, −3:9, −2:34, −1:73, 0:673, +1:4, +2:41, +3:3, +4:10. Severity word to N in the reports: low→8 (342) or 4 (9), medium→6, high→4, critical→2. The prose number is the severity word, not a score.

**Cap flips at T=4:** 13 (dimension, unit) cases. **All of them go the same way: prose says `high (4/10)` (≤ T) and JSON says 5 to 8 (> T).** Trusting the prose would cap where JSON does not:
- classic-editor: 8 units (`init_actions` ×2, `settings_1`, `settings_2`, `user_settings`, `network_settings`, `save_network_settings`, `add_redirect_helper`). The plugin goes from uncapped 74.9 to **capped 60**. This is the known classic-editor CSRF false positive.
- twentytwentyfour `<template>`: the theme goes from 81.7 to capped 60.
- vuln-plugin `ser_handle_cancel_registration` (in r_dir, api and aio). The fixture labels this unit **clean**.

### 2. Training data (root cause)
The deployed judge's behavior matches only `openai_train_relabel_v1.jsonl` (v1.3 relabel SFT: `output/tinker/PROMOTED_v1.3.json` `train_data`, and v4 reuses the v1.3 labels per CHANGELOG). `openai_train.jsonl`, `openai_train_v4_rebuilt.jsonl` and `openai_val.jsonl` all have prose = JSON on 100% of pairs.

`python3 train_fmt.py data/reasoning_dataset/openai_train_relabel_v1.jsonl` (478 judge examples):

| format | examples | prose = JSON | prose ≠ JSON | prose numeric, JSON absent |
|---|---|---|---|---|
| `score N/10` | 338 | 1869 | **0** | 891 (mostly sql_safety 211, a11y 223, i18n 208, security 143) |
| `severity w (N/10)` | 140 | 284 | **570 (67%)** | 406 |

Across all files the training severity map is fixed: low=8 (848), medium=6 (212), high=4 (141), critical=2 (59). Cap-dimension disagreements in training are mostly prose 2 or 4 with JSON 5 to 8: the same direction as the reports. `error_handling` is in the JSON of 278 examples and in the prose of none.

The cause is in `scripts/relabel/rebuild_targets.py`:
```python
content = re.sub(r"(?m)^([A-Za-z0-9 /_]+?): score (\S+)/10", _patch, content)   # never matches "severity high (4/10)"
...
return mo.group(0)  # unknown/omitted dim -> leave as-is   # N/A-by-relabel dims keep the old number
```
Its self-check uses the same `score` regex, so it passed even though the severity rows were never patched. The model learned exactly what it was trained on. Score-format prose copies the JSON. Severity-format prose is the old teacher's bucket, unrelated to the JSON. Dimensions the relabel rubric treats as not applicable (no SQL, no output, no strings) still get prose numbers but are absent from the JSON. That is the "prose covers SQL Safety, i18n, Accessibility with no stored score" symptom: those dimensions are N/A by the trained rubric. `judge_output` is the target that the judge rho (0.80 to 0.83) was measured on. The prose numbers were never validated.

### 3. Generation order and parse path
- Order (fixtures `tests/fixtures/responses/severity_format_*.txt` and the training targets): prose → `[/REASONING]` → `<judge_output>{...}</judge_output>`. The prose comes first, so the JSON is not conditioned on numbers the model emits later. It still ignores its own severity numbers, as trained.
- `judge.py::to_judgment`: the scores come only from `parse_judge_response` Strategy 0 (the tag block). The critique is `raw` split at the first `[/REASONING]` or `<judge_output>`. The stored critique is therefore the exact prose prefix, and `scores` is the exact JSON. **No parsing bug.** `aggregate.DIM` parses both header formats correctly, and `rank()` already takes polarity from the JSON, not the prose.
- GPU step skipped: the saved reports already hold the full prose and the parsed JSON, and the training data explains the cause completely. A live call would only reproduce the same pair.

### 4. Which source matches true severity (fixture, `vuln-plugin.expected.json`)
Cap dimensions at T=4, `r_dir.json` (`api` and `aio` agree):

| unit (group) | JSON sec/sql | prose sec/sql | JSON correct | prose correct |
|---|---|---|---|---|
| ser_search_registrations (sqli) | 4 / 2 | high4 / crit2 | yes | yes |
| ser_delete_registration (sqli) | 4 / 4 | crit2 / crit2 | yes | yes |
| ser_ajax_export_registrations (no nonce/cap) | 4 / 4 | crit2 / high4 | yes | yes |
| ser_render_search_box (unescaped echo) | 1 / N/A | crit2 / low8 | yes | yes |
| ser_save_settings (unsanitized option) | 4 / N/A | crit2 / low8 | yes | yes |
| ser_get_registrations_for_event (clean) | 8 / 8 | 8 / 8 | yes | yes |
| ser_render_registration_count (clean) | 8 / N/A | 8 / 10 | yes | yes |
| **ser_handle_cancel_registration (clean)** | 8 / 8 | **high4** / med6 | yes | **no: false positive, would cap** |
| ser_sanitize_settings (clean) | 8 / N/A | 8 / 10 | yes | yes |
| `<template>` | 8 / N/A | 8 / 10 | yes | yes |

JSON is 10/10 and prose is 9/10. On real code the prose adds 9 more cap triggers (classic-editor and twentytwentyfour), and none of them catches anything the JSON misses: the JSON flags every planted bug. The prose is more severe in degree (critical 2 where JSON gives 4), which does not matter at T=4 but makes the JSON margin thin (see §a).

## Options
| # | Option | Verdict |
|---|---|---|
| 1 | Trust the prose numbers | **Reject.** They are stale buckets (even "low (4/10)"). Adds 1 fixture false positive and caps classic-editor and twentytwentyfour. The rho was never measured on them. |
| 2 | min(prose, JSON) on the cap dimensions | **Reject.** Same 13 flips as option 1, because every disagreement has prose lower. |
| 3 | Flag disagreement as low confidence | **Weak; hand to R4.** It fires on 36% of units (all severity-format units), so it measures format, not doubt. A cap-dimension-only version (prose ≤ T and JSON > T) fires 13 times, and every case it fires on is a prose false positive. No evidence it finds false negatives. |
| 4 | **Keep the JSON authoritative and rewrite the critique headers from the JSON** | **Recommend.** The cap and the statistics are already right. The visible bug is a report that contradicts itself. |
| 5 | Retrain with fixed targets | **Record as a retrain note** (outside the harness). Patch the `severity w (N/10)` headers in `rebuild_targets.py`, write `None`/N/A for dimensions the relabel dropped, and add `Error Handling:` prose when the JSON has it. Re-run the self-check with both regexes. |

## Recommendation (files to change)
1. `wpjudge/config.py`: add `pattern_min: int = 8  # rank(): a dim scored >= this is a pattern`.
2. `wpjudge/aggregate.py`:
   - `low_rate`: `x <= cfg.cap_threshold`.
   - `anti, pat = rank(judged, lo=cfg.cap_threshold, hi=cfg.pattern_min)`.
3. `wpjudge/judge.py::to_judgment`: after the split, normalize the dimension headers in the critique to the JSON. Keep `raw` unchanged. Reuse `aggregate.DIM`'s label list (move the label regex to one shared place or import it):
   ```python
   HDR = re.compile(r'^(WPCS Compliance|SQL Safety|Security|Performance|WP API Usage|Code Quality|'
                    r'Dependency Integrity|i18n|Accessibility|Error Handling):\s*'
                    r'(?:score (?:\d+|None)/10|severity \w+ \((?:\d+|None)/10\))', re.M)
   def _sync(critique, scores):  # A5: numbers in the report come only from <judge_output>
       return HDR.sub(lambda m: f'{m[1]}: score {scores.get(m[1].lower().replace(" ", "_"), "None")}/10', critique)
   ```
   The output stays in the `score N/10` form, which `aggregate.DIM` already parses, so `rank()` is unchanged. Dropping the stale severity word removes "severity high" next to a 5.
4. SPEC/README one-liner: "Scores and dimension numbers in `critique` come from `<judge_output>`. `None/10` means the judge rated the dimension not applicable. `error_handling` may be scored without a prose line."
5. Retrain note (HANDOVER or `.planning`): the `rebuild_targets.py` regex gap, with the counts above.

## Tests
Unit (no GPU):
- `test_aggregate.py::test_thresholds_follow_cfg`: one judgment with security 5. With `cap_threshold=5`, `low_rate` is 1.0, the unit appears in `anti_patterns`, and it is in `cap_reasons`. With the default 4, none of that holds. `pattern_min=9` drops an 8 from `patterns`.
- `test_aggregate.py::test_cap_ignores_prose`: critique `Security: severity critical (2/10) — ...` with `scores.security=8` → `capped` is False.
- `test_judge.py::test_critique_synced_to_json`: new fixture `responses/severity_mismatch_add_redirect_helper.txt` (the stored classic-editor critique + `[/REASONING]` + its JSON). After `to_judgment`, the critique contains `Security: score 5/10`, `SQL Safety: score None/10` and `WPCS Compliance: score 6/10`, with no `severity` header left. `scores` is unchanged, and `aggregate.DIM.findall` still returns 9 lines.
- The existing `test_real_responses_parse` keeps passing (scores untouched).

E2E (`tests/e2e.sh`, the fixture section that already runs vuln-plugin):
- Consistency: for every judged unit, each critique header number equals `scores[d]`, or is `None` when `d` is absent.
- Cap truth: the set of `cap_reasons` ids equals the 5 planted-bug ids (groups sqli, ajax_no_nonce_no_cap, unescaped_echo, unsanitized_option). No `clean` unit appears in it.
- Threshold knob without a second model run: re-aggregate the saved judgments with `WPJUDGE_CAP_THRESHOLD=5`. The `low_rate` values change and `anti_patterns` counts are ≥ those at T=4.

## Risks
- The rewrite hides the model's original severity word from report readers. Anyone debugging the model needs `raw` (not in reports today). If the original prose is ever needed, keep it in the job dir, not in the report.
- The JSON margin is thin (most planted bugs at exactly 4). Run-to-run variance across workers (render_search_box got 1 in r_dir and 4 in aio) could push a real bug to 5. That belongs to R4/T4, not this fix.
- The reason text of a dimension that is N/A in the JSON may still describe a real issue that the JSON chose not to score. That is rare in the fixture, but it stays visible in the prose.
- A different model build (retrained with fixed targets) makes the rewrite a no-op. It is harmless to keep.

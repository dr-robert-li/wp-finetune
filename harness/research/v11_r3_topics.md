# v1.1 R3: why patterns get `finding: "other"` (T3)

Research only. No harness files changed. Scratch scripts were in the session scratchpad (`load.py`, `m1.py`..`m14.py`, `proto3.py`). The final prototype code is copied in full under "Proposed code" below, so this file does not depend on the scratchpad.

## 1. Answer in one paragraph
The cause is **lexicon gaps**. It is not the per-dimension split. "No concern applies" lines and the rank logic are only small factors. Almost every "other" line is generic praise that the 16-topic lexicon has no word for: "correctly uses the canonical WP API" (wp_api_usage), "clean, readable, well-structured" (code_quality), "lightweight, negligible overhead" (performance), "follows WordPress Coding Standards, proper indentation" (wpcs), and block-theme markup (`wp:template-part`, `wp:pattern`). One more defect shows up in the measurements. Even when a line is tagged, v1 credits every topic word in it, wherever it appears. "Relies only on `esc_html_x()`" in a dependency line counts as **output escaping**. "Uses `esc_html_e`" in an i18n line counts as **output escaping**. Hand-checked precision is 71%. The recommended fix has three parts: (a) a larger lexicon of about 27 topics, still neutral and regex-only; (b) a per-dimension topic allowlist, falling back to the whole lexicon for anti-patterns only; (c) dropping only the "not applicable" *sentences* of a reason, never the whole line. On the saved reports, the pattern "other" share falls from 14% to 3% (classic-editor 19% → 5%, twentytwentyfour 14% → 2%). "other" rows in the top 10 fall from 3 to 0 on classic-editor. Tag precision rises from 71% to 90%. Embeddings are not needed.

## 2. Evidence

### 2.1 Corpus
- Reports: `real/classic-editor.json` (39 judged), `real/twentytwentyfour.json` (60), `r_dir/r_stdout/r_tgz/r_w4/r_zip.json`, `api_report.json`, `aio_report.json` (10 judged each, the same fixture plugin). `r_down.json` has 0 judged. Deduplicated by critique text: 140 units, 1258 dimension lines.
- Training prose (held out, never used to tune): `data/reasoning_dataset/openai_train_v4_rebuilt.jsonl`, `openai_val.jsonl` and `openai_train_relabel_v1.jsonl`, `<wp_judge>` rows only, deduplicated: 9684 dimension lines.

### 2.2 The split is not the cause (`m2.py`)
- Every line starting with a dimension label matched `DIM`: 1256/1256, 0 unmatched.
- 0 captured reasons still contain `Fix:`. The `Issue:` prefix is already stripped. The regex already isolates the Issue clause, so "use the Issue: clause" changes nothing.
- 46 `error_handling` JSON scores have no prose line, so they can never produce a pattern. This is minor (see Risks).

### 2.3 v1 "other" share per report (current `aggregate.rank`, `m1.py`/`m14.py`)
| report | pattern other | "other" rows in pattern top 10 | anti other |
|---|---|---|---|
| real/classic-editor | 39/206 = 19% | 3 (wp_api_usage 15, performance 12, code_quality 8) | no anti rows |
| real/twentytwentyfour | 77/551 = 14% | 1 (code_quality 32) | no anti rows |
| r_* fixture (each) | 4-11% | 0 | 0-4% |
| api / aio | 7% / 8% | 1 / 0 | 0% / 10% |
| all, deduplicated | 131/963 = 14% | 2 | 4/120 = 3% |

Per dimension, deduplicated (pattern lines tagged / "other"): wp_api_usage 75/36, code_quality 55/46, performance 25/23, wpcs 85/12, security 68/7, dependency 93/4. Anti-patterns are almost clean (3%). **The problem is on the pattern side.**

Held-out training prose with v1: pattern other is **33%** and anti other is **21%**. Anti-patterns look clean on the reports only because the fixture's bugs are the classic nonce/SQL/escaping ones.

### 2.4 Hand clustering of the 131 "other" pattern lines (`other_dump.json`)
| cluster | n (approx.) | typical sentence | missing lexicon word |
|---|---|---|---|
| correct core API use | 33 of 36 wp_api_usage | "get_post_meta() and update_post_meta() are the canonical WP meta APIs" | correctly uses, canonical, idiomatic, standard, `apply_filters`, hooks, `wp_*(` |
| block-theme markup | ~25 across dimensions | "`wp:template-part` is the correct block-level inclusion mechanism" | `wp:…`, template part, block markup |
| readability praise | ~40 of 46 code_quality | "The code is clean, readable, and well-structured" | clean, readable, well-structured, maintainable, clear |
| efficiency praise | ~20 of 23 performance | "lightweight … no DB or network calls" | lightweight, negligible, overhead, efficient, once |
| WPCS style praise | ~10 of 12 wpcs | "follows WordPress Coding Standards … proper indentation" | indentation, coding standards, line endings |
| N/A remarks | ~10 | "no performance concern", "No external dependencies" | (not a pattern at all) |
| real but rare | ~5 | truthiness check, offset as a string, print_r debug | defensive checks, debug code |

Training prose adds anti-side gaps that the reports do not have yet: code_quality "syntactically invalid / truncated / closing brace / debug statement" (94 lines), sql_safety raw `$wpdb` use, performance "query inside foreach", wpcs brace/indent/tabs, wp_api "file_get_contents instead of WP_Filesystem", accessibility/i18n N/A lines scored low.

### 2.5 "No concern applies" lines are a small factor on real reports
- Only **7 of 618** pattern lines (score ≥ 8) are *purely* N/A. The model leaves most N/A dimensions out of the JSON, and the rank function then skips them. v1 tagged all 6 checked: "No external dependencies" → `dependencies`, "no SQL … prepare() not applicable" → `SQL preparation`. These credit a non-pattern as a strength.
- Dropping **whole lines** that contain a "no …" phrase is wrong. My first attempt dropped 120 pattern lines and 3 anti lines. Lost content included "uses `$wpdb->prepare()` with `%d` … no user input" (a real pattern) and the anti line "has **no input validation**". **Sentence-level** stripping is required, split on `.`/`;`/` — `.
- Training-data artefact: the teacher's "max-2-N/A rule" scored extra N/A dimensions **2-3** ("No HTML output; scored 3 as entirely inapplicable"): 45 of 1333 training anti lines (3.4%), mostly accessibility and i18n. The fine-tuned model can reproduce this. If it does, v1 counts them as `accessibility markup`/`translation` **anti-patterns**. It cannot trip the cap, because `cap_dims` = security and sql_safety. None of the 18 low accessibility/i18n/sql lines in the real reports was N/A (`m11.py`), so this is a precaution. The same sentence-level strip covers it.

### 2.6 Rank logic: a precision problem, not an "other" problem
- Thresholds and the JSON-score gate are fine: no N/A dimensions leak through, and the existing `test_ranking` semantics still hold.
- Multi-topic crediting tags any topic word anywhere in the line, so cross-dimension words mis-tag. Examples: an i18n line naming `esc_html_x` → output escaping; a dependency line listing `current_user_can` → capability check; a performance line → `accessibility markup` via "heading"; a security line → `SQL preparation` via "injection" (already noted in r2 §5).
- Hand check: 50 random gated lines (seed 42, `m12.py`), each tag judged right or wrong by reading the sentence:

| | tags | correct | precision | lines with ≥ 1 correct label | "other" |
|---|---|---|---|---|---|
| v1 | 89 | 63 | **71%** | 41/50 | 6 |
| proposed | 98 | 88 | **90%** | 48/50 | 0 |

Remaining errors in the proposal are N/A fragments not caught ("no direct user input"), and "static"/"self::" counted as coupling.

## 3. Options tried (all measured on the same reports, `m3.py`/`m6.py`/`m14.py`)
| option | pattern other (dedup) | top-10 other (classic) | tags/line | note |
|---|---|---|---|---|
| A. v1 | 14% | 3 | ~1.6 | baseline |
| B. v1 + whole-line N/A filter | 14% | 2 | | does not fix the gaps; drops real content |
| C. expanded lexicon, all topics | 1% | 0 | 2.7 | hides "other" but adds mis-tags (inflation 963 → 1715 hits) |
| D. C + "first match only" | 1% | 0 | 1.0 | loses real second topics; earliest word is often the wrong one ("correctly uses esc_attr" → WP API choice) |
| E. C + per-dimension allowlist + sentence N/A strip, fallback to whole lexicon for every line | 0.5% | 0 | 1.8 | pattern fallback mostly mis-tags (28 lines, ~80% wrong) |
| **F. E, fallback only for anti** (recommended) | **3%** | **0** | 1.8 | 7 anti fallbacks, all correct |

Final numbers for F, per report (`m14.py`):
| report | pattern other v1 → F | top-10 other v1 → F | anti other v1 → F |
|---|---|---|---|
| classic-editor | 19% → 5% | 3 → 0 | – |
| twentytwentyfour | 14% → 2% | 1 → 0 | – |
| r_* fixture | 4-11% → 0% | 0 → 0 | 0-4% → 0% |
| api / aio | 7%/8% → 0%/2% | 1/0 → 0/0 | 0%/10% → 0%/0% |
| all, deduplicated | 14% → 3% | 2 → 0 | 3% → 0% |
| held-out training prose | 33% → 21% (of lines kept) | | 21% → 3% |

The held-out pattern residual is mostly N/A remainder sentences ("It is a backend data transformation function.", "SQL safety does not apply") in accessibility (204) and sql_safety. Training prose scores N/A dimensions 10, but the deployed model mostly leaves them out of the JSON, so few of these reach `rank` in real scans.

New top 10 on classic-editor (was: dependencies 26, **other 15**, **other 12**, translation 10, **other 8**, …):
```
30 wp_api_usage WP API choice | 22 dependency_integrity dependencies | 19 code_quality readability
16 performance efficiency     | 11 i18n translation                  | 10 code_quality docblock
10 wpcs formatting            |  9 wpcs docblock                     |  8 code_quality globals / coupling | 8 wp_api_usage translation
```
twentytwentyfour: code_quality readability 49, dependencies 48, WP API choice 45, block markup 43, wpcs docblock 41, formatting 37, security output escaping 37, i18n translation 37, … The "i18n output escaping 33" mis-tag is gone. Anti on aio is unchanged in substance (nonce, SQL preparation, input sanitization, capability check), plus correct new `code_quality` labels in place of "other".

## 4. Recommendation
Option F. It is lexicon-only and adds no dependency. In `aggregate.py`:
1. Replace `TOPICS` with the 27-topic lexicon below. Keep the names neutral; polarity still comes from the JSON score.
2. Add `DIM_TOPICS` (allowlist per dimension). Tag a reason with the topics of its own dimension. For **anti** lines only, if nothing matches, fall back to the whole lexicon. Anything left over is `other`.
3. Add `NA` + `strip_na()`: split the reason on `.`, `;` or ` — ` and drop sentences that say the dimension does not apply. Apply it to both kinds. If nothing is left, skip the line (no topic, no "other").
4. (Cheap, optional, recommended) Give pattern rows `examples` like anti rows. Most remaining doubt about a label goes away when the reader sees the sentence. This changes `SPEC.md` A5/schema line 49, `PLAN.md` binding change 5, and the e2e check at `tests/e2e.sh:103`, which asserts that patterns have no examples. Skip it if the schema freeze matters more.

### Proposed code (drop-in for `TOPICS`/`RX`/`rank` in `harness/wpjudge/aggregate.py`)
```python
# Neutral topics; polarity comes from the JSON score. ponytail: fixed lexicon, extend when 'other' leads a dimension.
TOPICS = {
 'nonce': r'nonce|check_admin_referer|check_ajax_referer|csrf',
 'capability check': r'capabilit|current_user_can|access control|permission|authori[sz]|ownership|privilege',
 'output escaping': r'escap|esc_(html|attr|url|js|textarea)|wp_kses|xss',
 'input sanitization': r'sanitiz|unslash|validat|absint|intval|\$_(get|post|request|server|cookie|files)|user input',
 'SQL preparation': r'prepare|interpolat|concatenat|injection|placeholder|\$wpdb|raw sql|parameteri[sz]',
 'query scope': r'no limit|limit clause|select \*|unbounded|all rows|per_?page|posts_per_page|all columns',
 'caching': r'cach|transient',
 'queries in loops': r'n\+1|in a loop|per iteration|inside (a |the )?(foreach|loop|for\b)|foreach loop|for every|each (post|row|item)',
 'efficiency': r'lightweight|negligible|overhead|efficient|o\(1\)|early (return|exit)|heavy|expensive|redundant|once\b'
               r'|single (\w+ )?(call|query)|no loops|minimal|small|acceptable',
 'naming': r'camel ?case|snake_case|naming|(function|unique|theme|plugin) prefix|prefixed|namespac',
 'docblock': r'docblock|phpdoc|@param|@return|@since|@package|inline (comment|documentation)|well[- ]documented|\bcomments?\b',
 'formatting': r'indent|spacing|brace|\btabs?\b|whitespace|line ending|crlf|yoda|strict comparison|formatt'
               r'|coding (standards|conventions)|opening tag|closing tag',
 'translation': r'translat|text ?domain|\b_[_enx]\(|\b_n\b|_ex\(|esc_(html|attr)_[_ex]|i18n|internationali|plural',
 'accessibility markup': r'aria|\blabel|alt\b|alt text|screen reader|assistive|semantic|heading'
                         r'|<(main|header|footer|nav|aside|figure)|keyboard|focus|contrast',
 'error handling': r'error|exception|wp_error|return value|try|catch|fallback|wp_die',
 'readability': r'readab|clean|well[- ]structured|well[- ]organi[sz]ed|maintainab|concise|\bclear(ly)?\b|straightforward|logical|descriptive',
 'size / responsibilities': r'responsibilit|too long|complex|nested|single[- ]purpose|separation of concerns|mixes|self[- ]contained|minimal',
 'defensive checks': r'edge case|null coalescing|\?\?|isset|empty\(|is_array|cast|guard|defensive|undefined index',
 'debug code': r'print_r|var_dump|error_log|\bdebug',
 'syntax / truncation': r'syntax error|syntactically|truncat|invalid php|parse error|incomplete',
 'dead code': r'dead code|\bunused|unreachable',
 'globals / coupling': r'\bglobal|coupl|singleton|\bstatic\b|self::',
 'WP API choice': r'wordpress api|wp api|core (function|api)|helper|correct(ly)? (use|utiliz|call)|canonical|idiomatic|standard'
                  r'|(wp|wordpress|core|woocommerce|rest|options|meta) (\w+ )?apis?\b|apply_filters|add_action|add_filter'
                  r'|remove_action|\bhooks?\b|instead of|wp_\w+\(|redirect',
 'block markup': r'wp:[a-z-]+|block (markup|template|pattern|theme|comment|editor|attribute)|template[- ]part|pattern slug|core blocks?',
 'file handling': r'file_get_contents|wp_filesystem|filesystem|\blfi\b|path traversal|file inclusion',
 'privacy / PII': r'\bpii\b|personal data|gdpr',
 'dependencies': r'dependenc|depends|relies|rely|class_exists|function_exists|guaranteed|available|core (wordpress|wp)'
                 r'|wordpress core|wp core|jquery|enqueue|external'}
RX = {k: re.compile(v, re.I) for k, v in TOPICS.items()}
_SEC = ['nonce', 'capability check', 'output escaping', 'input sanitization', 'SQL preparation', 'file handling', 'privacy / PII']
DIM_TOPICS = {   # a dimension's line is tagged with its own topics first (cuts cross-dimension mis-tags)
 'security': _SEC,
 'sql_safety': ['SQL preparation', 'input sanitization', 'query scope', 'caching'],
 'wpcs_compliance': ['naming', 'docblock', 'formatting'] + _SEC[:5] + ['translation'],
 'performance': ['caching', 'queries in loops', 'query scope', 'efficiency'],
 'wp_api_usage': ['block markup', 'file handling'] + _SEC[:4] + ['translation', 'WP API choice'],
 'code_quality': ['debug code', 'syntax / truncation', 'dead code', 'error handling', 'defensive checks', 'naming',
                  'docblock', 'globals / coupling', 'size / responsibilities', 'readability'],
 'dependency_integrity': ['globals / coupling', 'block markup', 'dependencies'],
 'i18n': ['translation'], 'accessibility': ['accessibility markup'], 'error_handling': ['error handling']}
# "this dimension does not apply" sentences: not a strength, not a defect (also the teacher's max-2-N/A low scores)
NA = re.compile(
    r"not (applicable|relevant|required|needed|a concern)|\binapplicable\b|\bn/a\b|\bnot? (\w+ ){0,3}appl(y|ies|icable)\b"
    r"|do(es)? not apply|\bno (\w+[ -]){0,2}(concerns?|issues?|changes?|violations?)\b"
    r"|\bno (direct |raw )?(sql|database|db)( \w+)? (quer|interaction|access|call|operation)"
    r"|\bno (translatable|user[- ]facing|user[- ]visible|ui|visible) (strings?|text)"
    r"|\bno (html|markup|interactive html|dom) (output|generat|emit)|\bno (external )?dependenc|security surface"
    r"|\bno (user[- ]supplied |user |external |direct |dynamic )?(input|data)\b(?! validation)"
    r"|(does not|doesn.t|do not) (output|generate|emit|render|produce|interact|accept|handle|process|perform|make|touch|execute|query|contain)"
    r"|\bpure (php|function|utility|backend)|backend (logic|data|utility)|only returns"
    r"|returns (only )?(a |an )?(boolean|string|array|class name)", re.I)
SENT = re.compile(r'(?<=[.;!?])\s+|\s+[—–]\s+')


def _topics(d, reason, kind):
    """Own-dimension topics; anti lines fall back to the whole lexicon (patterns would only mis-tag)."""
    for pool in ([DIM_TOPICS.get(d, TOPICS)] + ([TOPICS] if kind == 'anti' else [])):
        ts = [t for t in pool if RX[t].search(reason)]
        if ts:
            return ts
    return ['other']


def rank(judged, lo=4, hi=8, top=10):
    cnt = {'anti': Counter(), 'pattern': Counter()}
    ex = {}
    for j in judged:
        seen = set()
        for lab, _, _, reason in DIM.findall(j.get("critique") or ""):
            d = lab.lower().replace(' ', '_')
            s = j["scores"].get(d)
            if not isinstance(s, (int, float)) or lo < s < hi:
                continue
            kind = 'anti' if s <= lo else 'pattern'
            reason = ' '.join(x for x in SENT.split(reason) if not NA.search(x))
            if not reason:
                continue                               # N/A line: neither a pattern nor an anti-pattern
            for t in _topics(d, reason, kind):
                ...                                    # unchanged from v1 (seen / cnt / ex)
```
(The prototype keeps match order by position; plain allowlist order works equally well for counting.)

## 5. Files to change
- `harness/wpjudge/aggregate.py`: `TOPICS`, new `DIM_TOPICS`, `NA`, `SENT`, `_topics`, and two lines in `rank`.
- `harness/tests/test_aggregate.py`: new unit tests (below). The existing `test_ranking` passes unchanged against the prototype, and both sketches below give the asserted results against it (checked).
- `harness/tests/e2e.sh`: add the other-share assertion to `cmd_real` (and to `chk`, which covers fixture reports).
- `harness/SPEC.md` A5 ("16-topic lexicon" → "27-topic lexicon, per-dimension, N/A sentences dropped"). If pattern examples are adopted, also the schema line and `PLAN.md` change 5.

## 6. Tests
Unit (`test_aggregate.py`, no GPU):
```python
def test_topics_v11():
    J2 = lambda f, sc, c: J(f, 60, scores=sc, critique=c)
    js = [J2("a.php:1:f", {"wp_api_usage": 9, "code_quality": 9, "performance": 9, "i18n": 9, "dependency_integrity": 9},
             "WP API Usage: score 9/10 — get_post_meta() is the canonical WP meta API; usage is idiomatic.\n"
             "Code Quality: score 9/10 — The code is clean, readable, and well-structured.\n"
             "Performance: score 9/10 — Lightweight; negligible overhead.\n"
             "i18n: score 9/10 — Strings use esc_html_e() with the text domain.\n"            # not output escaping
             "Dependency Integrity: score 9/10 — No external dependencies.\n"),               # N/A -> dropped
          J2("b.php:1:g", {"accessibility": 3, "security": 2, "sql_safety": 9},
             "Accessibility: score 3/10 — No HTML output; scored 3 as entirely inapplicable.\n"  # N/A low score -> dropped
             "Security: score 2/10 — No nonce verification before update_option().\n"          # 'No nonce' is NOT N/A
             "SQL Safety: score 9/10 — No user input reaches it; uses $wpdb->prepare() with %d.\n")]  # keep 2nd sentence
    r = build_report(META, MODEL, js, [], CFG)
    pat = {(p["dimension"], p["finding"]) for p in r["patterns"]}
    assert pat == {("wp_api_usage", "WP API choice"), ("code_quality", "readability"), ("performance", "efficiency"),
                   ("i18n", "translation"), ("sql_safety", "SQL preparation")}
    assert [(a["dimension"], a["finding"]) for a in r["anti_patterns"]] == [("security", "nonce")]

def test_anti_fallback_and_code_quality_lexicon():
    js = [J("a.php:1:f", 30, scores={"code_quality": 3},
            critique="Code Quality: score 3/10 — ser_delete has no capability check before the DELETE.\n"),
          J("b.php:1:g", 30, scores={"code_quality": 2},
            critique="Code Quality: score 2/10 — Leftover print_r() debug statement.\n")]
    anti = {(a["dimension"], a["finding"]) for a in build_report(META, MODEL, js, [], CFG)["anti_patterns"]}
    assert anti == {("code_quality", "capability check"), ("code_quality", "debug code")}
```
E2E (`tests/e2e.sh`, in `cmd_real` and the fixture `chk`; it runs on the reports each section already writes):
```python
for k in ("patterns", "anti_patterns"):
    rows = r[k]; tot = sum(x["count"] for x in rows); oth = sum(x["count"] for x in rows if x["finding"] == "other")
    ok(tot == 0 or oth / tot <= 0.15, f"{tag}: {k} 'other' share {oth}/{tot} <= 15% of top-10 counts")
    ok(not any(x["finding"] == "other" for x in rows[:3]), f"{tag}: no 'other' in {k} top 3")
```
Thresholds and margins. On the top-10 rows (what the report holds), "other" in F is 0 everywhere measured. v1 is 35/105 = 33% on classic-editor (twentytwentyfour 32/322 = 10%, api 2/23 = 9%), so the 15% bound **fails on v1 for classic-editor and passes on F** with margin. The top-3 check also catches v1 (classic-editor has `wp_api_usage/other` at rank 2). Over all hits, the worst real report under F is 5%. The bound is loose enough to survive the ±few-point run-to-run wobble from 4-worker nondeterminism.

## 7. Risks
- **Tuned on the data it is measured on.** The held-out training prose still improves (pattern other 33% → 21%, anti 21% → 3%), but the 90% precision figure comes from the tuning reports. Re-check on a third real target (a WooCommerce-style plugin, a classic PHP theme) in the v1.1 real section.
- **Broad words.** "standard", "error", "minimal", "comments", "static" are broad. The per-dimension allowlist limits them to the dimension where they are meaningful. Do not add them to more dimensions without re-measuring.
- **More labels per line.** Tags per line rise from about 1.6 to 1.8, and pattern counts grow about 15%. Counts mean "functions whose line mentions the topic in that dimension", not "functions with exactly this finding". This is the same semantics as v1.
- **Score/prose mismatch flows into patterns.** Some score-8 lines describe a defect ("retrieves registration data without any access control checks" → pattern `capability check`). This is a T2/T4 problem (JSON score vs prose), not a lexicon one. Pattern examples (item 4) would make it visible.
- **N/A strip removes an occasional useful clause** ("no user input, so …"). Anti lines are protected by the `(?! validation)` guard and by "no nonce" / "no capability check" not being N/A phrases (unit-tested above). Watch for new N/A phrasings in future model versions. The held-out residual shows the remaining phrasing gaps ("It is a backend … function").
- **`error_handling`** has JSON scores (46) but no prose lines, so it never ranks. Out of T3 scope. Note it for the prompt/format owner.

# R2: judge client, serving and pattern ranking

## 1. Prompt (confirmed)
In `openai_val.jsonl` and all 10 `openai_train*.jsonl` files, every `<wp_judge>` row (121 val, 478–512 train) has **no system message** and matches exactly `"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"`. Answers: dimension prose, `[/REASONING]`, `<judge_output>{json}</judge_output>`.

## 2. Serving (IQ4_XS im_f16, port 8101, `-ngl 999 --jinja --chat-template-kwargs '{"enable_thinking": false}' --parallel 4 -c 32768`)
`n_slots = 4, n_ctx_slot = 8192`. First 16 val judge prompts, temperature 0, `max_tokens` 2048, largest prompt 874 tokens.

| Run | Wall | Functions/min | Generated tok/s | Mean completion tokens |
|---|---|---|---|---|
| 4 workers (a) | 92.1 s | 10.4 | 123 | 708 |
| 4 workers (b) | 83.6 s | 11.5 | 132 | 688 |
| 1 worker (a) | 170.7 s | 5.6 | 63 | 677 |
| 1 worker (b) | 169.7 s | 5.7 | 64 | 677 |

- **4 slots ≈ 1.9× throughput.** A 300-function plugin ≈ 27 min.
- **Parse:** 64/64 parsed (copied `parse_judge_response`), 0 `finish_reason=length`, longest completion 1286 tokens. 0 repetition loops in 64 live + 242 captured responses. 1/242 captures fails to parse (~0.4%); retry covers it.
- **Determinism:**
  - Sequential is deterministic: two 1-worker runs byte-identical 16/16.
  - Parallel is not: 0/16 identical between the two 4-worker runs and 0/16 vs sequential (divergence after 23–818 chars; batching changes numerics).
  - JSON effect: verdict 16/16 equal in every pairing, `security` never differed, 26/106 fields differed, `overall_score` |Δ| mean 0.9–1.3, max 6.
  - Keep 4 workers default; document the wobble; `workers=1` for reproducible reports.
- **Cleanup:** server killed; `nvidia-smi` shows only desktop graphics; port 8101 down.

## 3. Prose dimension lines: reliability
242 captures (iq4xs_im_f16 + baseline_q6k, 121 functions × 2) use **two formats**: 143 `Label: score N/10 — reason`, 99 (41%) `Label: severity low|medium|high (N/10) — Issue: … . Fix: …`. Prose score = JSON score in 1017/1417 lines; **all 400 mismatches (±1–4) are severity-format**. The model also writes prose scores for dimensions it omits from the JSON ("SQL Safety 10/10 — no queries").

**Decision:** scores **only from `<judge_output>`**; absent key = not applicable. Prose lines supply **reason text only**. `DIM` below matches both formats in 242/242 responses.

## 4. Recommended `judge.py` (core)
```python
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
PROMPT = "<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"
DIM = re.compile(r'^(WPCS Compliance|SQL Safety|Security|Performance|WP API Usage|Code Quality|'
                 r'Dependency Integrity|i18n|Accessibility|Error Handling):\s*(?:score (\d+|None)/10|'
                 r'severity \w+ \((\d+|None)/10\))\s*[—–-]+\s*(?:Issue:\s*)?(.+?)(?:\.?\s*Fix:.*)?$', re.M)

def _call(client, model, code):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=2048,
        messages=[{"role": "user", "content": PROMPT.format(code=code)}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content or ""

def judge_one(client, model, unit):
    err, raw = None, ""
    for attempt in (1, 2):                                  # R2.3: retry once
        try:
            raw = _call(client, model, unit["code"])
            j = parse_judge_response(raw)                   # copied verbatim from eval_judge.py:310-383
            if j and isinstance(j.get("overall_score"), (int, float)):
                scores = {k: v for k, v in j.items() if k not in ("verdict", "overall_score") and isinstance(v, int)}
                return {**unit, "status": "judged", "attempts": attempt, "verdict": j.get("verdict"),
                        "overall_score": int(j["overall_score"]), "scores": scores,
                        "critique": raw.partition("[/REASONING]")[0].strip(), "raw": raw, "error": None}
            err = "parse_failure"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return {**unit, "status": "unjudged", "attempts": 2, "verdict": None, "overall_score": None,
            "scores": {}, "critique": "", "raw": raw, "error": err}

def judge_units(units, base_url, model, workers=4, on_result=None):
    client = OpenAI(base_url=base_url, api_key="none", timeout=600)
    with ThreadPoolExecutor(workers) as ex:
        out = []
        for j in ex.map(lambda u: judge_one(client, model, u), units):
            if on_result: on_result(j)
            out.append(j)
    return out
```
Skipped: `_judge_create`'s kwarg-rejection fallback (server flags are pinned) and overall derivation from dimensions (all captures emit `overall_score`). Slot budget 8192 − 2048 ≈ 6000 prompt tokens: W1 truncation must stay under (~20 KB code).

## 5. Pattern and anti-pattern ranking (R1.4)
**Normalized-clause counting fails:** 155 anti keys (139 singletons), 1015 pattern keys (871 singletons); top entries are near-duplicates ("camelcase method name violates wpcs snake case" vs "…convention") or empty ("calls", "wp api").

**Recommended (no embeddings):** JSON score ≤ 4 → anti-pattern, ≥ 8 → pattern; tag the reason with a fixed lexicon of 16 **neutral** topics (polarity comes from the score); count (kind, dimension, topic) once per function; keep one example.

```python
TOPICS = {'nonce': r'nonce', 'capability check': r'capabilit|current_user_can',
 'output escaping': r'escap|esc_(html|attr|url)|xss', 'input sanitization': r'sanitiz|unslash|\$_(get|post|request|server|cookie)',
 'SQL preparation': r'prepare|interpolat|injection', 'naming': r'camel ?case|snake_case|naming',
 'docblock': r'docblock|phpdoc|@param|@return|@since', 'translation': r'translat|text ?domain|\b_[_enx]\(|esc_html__',
 'caching': r'cach|transient', 'queries in loops': r'n\+1|in a loop|per iteration',
 'error handling': r'error handling|wp_error|exception|return value', 'accessibility markup': r'aria|\blabel|alt text|screen reader',
 'globals / coupling': r'\bglobal|coupl|singleton', 'size / responsibilities': r'responsibilit|too long|complex|nested',
 'WP API choice': r'wordpress api|wp api|core function|helper', 'dependencies': r'dependenc'}
RX = {k: re.compile(v, re.I) for k, v in TOPICS.items()}

from collections import Counter
def rank(judgments, lo=4, hi=8, top=10):
    cnt = {'anti': Counter(), 'pattern': Counter()}; ex = {}
    for j in judgments:
        seen = set()
        for lab, _, _, reason in DIM.findall(j["critique"]):
            d = lab.lower().replace(' ', '_'); s = j["scores"].get(d)   # JSON score; absent = N/A
            if s is None or lo < s < hi: continue
            kind = 'anti' if s <= lo else 'pattern'
            for t in [t for t, rx in RX.items() if rx.search(reason)] or ['other']:
                if (kind, d, t) in seen: continue
                seen.add((kind, d, t)); cnt[kind][(d, t)] += 1
                ex.setdefault((kind, d, t), f'{j["id"]}: {reason[:80]}')
    return {k: [{"dimension": d, "finding": t, "count": n, "examples": [ex[(k, d, t)]]}
                for (d, t), n in c.most_common(top)] for k, c in cnt.items()}
```

**Output on the 242 captures** (counts doubled: each function appears twice). Anti: 146 hits, 15 "other". Pattern: 864 hits, 148 "other".
```
ANTI  29 wpcs_compliance docblock         | camelCase method name violates WPCS snake_case; ... no PHPDoc
      24 wpcs_compliance naming           | camelCase method name violates WPCS snake_case
       9 security        output escaping  | $wcfmNonce ... echoed directly into a script tag without escaping
       8 wpcs_compliance output escaping  | addToHead echoes $wcfmNonce directly ... without esc_html_attr()
       8 security        input sanitization | dqfwc_save_taxonomy_custom_meta processes $_POST['term_meta'] without nonce
       7 security        nonce            | Critical security vulnerability: Missing nonce verification ...
       7 code_quality    other            | column_default() contains a print_r($item, true) debug statement
       5 i18n            translation      | exception message is hardcoded English string
       4 security        SQL preparation  | $wcfmNonce ... unescaped ... "could allow injection"  (false hit)
       4 sql_safety      SQL preparation  | SQL query uses $wpdb->prefix inside a double-quoted string
PATTERN 85 dependency_integrity dependencies | 68 wpcs_compliance docblock | 56 performance other |
        36 code_quality other | 35 wp_api_usage other | 34 i18n translation | 29 wp_api_usage WP API choice | ...
```
**Verdict:** anti-patterns are **meaningful** (real recurring WP defects; 10% "other"). Patterns are **weak but usable** as "strengths" (generic praise; 17% "other"). Noise: multi-topic lines credit every topic; stray words ("injection") mis-tag. Always show the example so readers can judge.

## 6. Risks
- Parallel nondeterminism: scores wobble a few points run to run at 4 workers.
- Prose scores are wrong in the severity format (41%): never read them.
- Fixed lexicon: extend when "other" leads a dimension.
- Small samples (16 throughput, 121 unique functions for patterns); template/HTML-heavy plugins untested.
- `sql_safety` appears in only 10/120 JSONs, so SQL issues surface mostly via `security`.

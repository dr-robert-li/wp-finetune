"""Judgments + skips -> report dict."""
import re
import statistics
from collections import Counter

# Prose dimension lines, both formats. Reason text only; scores come from <judge_output> JSON.
DIM = re.compile(r'^(WPCS Compliance|SQL Safety|Security|Performance|WP API Usage|Code Quality|'
                 r'Dependency Integrity|i18n|Accessibility|Error Handling):\s*(?:score (\d+|None)/10|'
                 r'severity \w+ \((\d+|None)/10\))\s*[—–-]+\s*(?:Issue:\s*)?(.+?)(?:\.?\s*Fix:.*)?$', re.M)

# Neutral topics; polarity comes from the JSON score.
# ponytail: fixed lexicon, extend when 'other' leads a dimension.
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
    for pool in [DIM_TOPICS.get(d, list(TOPICS))] + ([list(TOPICS)] if kind == 'anti' else []):
        ts = [t for t in pool if RX[t].search(reason)]
        if ts:
            return ts
    return ['other']


def rank(judged, lo=4, hi=8, top=10):
    """Tag each prose reason with topics, count (kind, dim, topic) once per function."""
    cnt = {'anti': Counter(), 'pattern': Counter()}
    ex = {}
    for j in judged:
        seen = set()
        for lab, _, _, reason in DIM.findall(j.get("critique") or ""):
            d = lab.lower().replace(' ', '_')
            s = j["scores"].get(d)                     # JSON score; absent = N/A
            if not isinstance(s, (int, float)) or lo < s < hi:
                continue
            kind = 'anti' if s <= lo else 'pattern'
            reason = ' '.join(x for x in SENT.split(reason) if not NA.search(x))
            if not reason:
                continue                               # N/A line: neither a pattern nor an anti-pattern
            for t in _topics(d, reason, kind):
                if (kind, d, t) in seen:
                    continue
                seen.add((kind, d, t))
                cnt[kind][(d, t)] += 1
                ex.setdefault((kind, d, t), f'{j["file"]}:{j["line"]} {j["name"]}')
    anti = [{"dimension": d, "finding": t, "count": n, "examples": [ex[('anti', d, t)]]}
            for (d, t), n in cnt['anti'].most_common(top)]
    pat = [{"dimension": d, "finding": t, "count": n} for (d, t), n in cnt['pattern'].most_common(top)]
    return anti, pat


def _row(j, cfg):
    """Review leads only; they never change the score or the cap."""
    r = {k: v for k, v in j.items() if k not in ("code", "raw")}
    prose, why = r.setdefault("prose_scores", {}), []
    if j["status"] == "judged":
        p = r.setdefault("verdict_p", None)
        if p is not None and p < cfg.min_verdict_p:
            why.append("verdict_p")
        low = lambda x: isinstance(x, (int, float)) and x <= cfg.cap_threshold  # None / N/A / absent = above
        if any(low(prose.get(d)) != low(j["scores"].get(d)) for d in cfg.cap_dims):
            why.append("prose_disagrees")
        if j.get("attempts") == 2:                     # judged on the varied retry (judge.RETRY_SAMPLING), not attempt 1
            why.append("retry")
    else:
        r["verdict_p"] = None
    return {**r, "confidence_reasons": why, "low_confidence": bool(why)}


def _skip_row(s):
    return {"file": s["file"], "line": s["line"], "name": s["name"], "loc": 0, "truncated": False,
            "status": "skipped", "reason": s["reason"], "verdict": None, "overall_score": None,
            "scores": {}, "critique": "", "verdict_p": None, "prose_scores": {},
            "confidence_reasons": [], "low_confidence": False}


def build_report(target_meta, model_meta, judgments, skips, cfg, files_scanned=None):
    """files_scanned: the explicit arg wins; else target_meta["files_scanned"]; else 0."""
    judged = [j for j in judgments if j["status"] == "judged"]
    n_units = len(judgments)

    # overall: LOC-weighted mean, cap, null when nothing judged
    if judged:
        w = [max(j["loc"], 1) for j in judged]
        uncapped = round(sum(j["overall_score"] * x for j, x in zip(judged, w)) / sum(w), 1)
        cap_reasons = [{"id": j["id"], "dimension": d, "score": j["scores"][d]}
                       for j in judged for d in cfg.cap_dims
                       if isinstance(j["scores"].get(d), (int, float)) and j["scores"][d] <= cfg.cap_threshold]
        capped = bool(cap_reasons)
        overall = min(uncapped, cfg.cap_value) if capped else uncapped
    else:
        uncapped = overall = None
        cap_reasons, capped = [], False

    pct = round(len(judged) / n_units * 100, 1) if n_units else 0.0

    dims = {}
    for d in sorted({d for j in judged for d in j["scores"]}):
        v = [j["scores"][d] for j in judged if isinstance(j["scores"].get(d), (int, float))]
        if v:
            dims[d] = {"mean": round(statistics.fmean(v), 2), "std": round(statistics.pstdev(v), 2),
                       "min": min(v), "n_rated": len(v), "low_rate": round(sum(x <= cfg.cap_threshold for x in v) / len(v), 3)}

    anti, pat = rank(judged, lo=cfg.cap_threshold, hi=cfg.pattern_min)
    worst = sorted(judged, key=lambda j: (j["overall_score"], j["id"]))[:10]

    rows = [_row(j, cfg) for j in judgments] + [_skip_row(s) for s in skips]
    if files_scanned is None:
        files_scanned = target_meta.get("files_scanned", 0)
    workers, slots = model_meta.get("workers", cfg.workers), model_meta.get("server_slots")
    return {
        "schema_version": "1.0",
        "target": {"name": target_meta.get("name", ""), "type": target_meta.get("type", "plugin"),
                   "files_scanned": files_scanned, "sha256": target_meta.get("sha256", "")},
        "model": {"file": model_meta.get("file", ""), "base_url": model_meta.get("base_url", ""),
                  "workers": workers, "server_slots": slots,
                  "reproducible": workers == 1 or slots == 1},  # only serial decoding is deterministic
        "summary": {
            "overall_score": overall, "overall_score_uncapped": uncapped, "capped": capped,
            "cap_reasons": cap_reasons,
            "verdict_counts": {"PASS": 0, "FAIL": 0, **Counter(j["verdict"] for j in judged)},
            "coverage": {"units": n_units, "judged": len(judged), "unjudged": n_units - len(judged),
                         "skipped": len(skips), "pct": pct,
                         "low_coverage": not judged or pct < cfg.coverage_threshold},
            "low_confidence": sum(r["low_confidence"] for r in rows),
        },
        "dimensions": dims,
        "anti_patterns": anti,
        "patterns": pat,
        "worst_functions": [{"file": j["file"], "line": j["line"], "name": j["name"],
                             "overall_score": j["overall_score"], "verdict": j["verdict"]} for j in worst],
        "functions": rows,
    }

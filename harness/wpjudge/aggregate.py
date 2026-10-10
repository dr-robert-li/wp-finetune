"""Judgments + skips -> report dict (SPEC.md schema v1, PLAN.md binding changes 1-8)."""
import re
import statistics
from collections import Counter

# Prose dimension lines, both formats (r2 §3). Reason text only; scores come from <judge_output> JSON.
DIM = re.compile(r'^(WPCS Compliance|SQL Safety|Security|Performance|WP API Usage|Code Quality|'
                 r'Dependency Integrity|i18n|Accessibility|Error Handling):\s*(?:score (\d+|None)/10|'
                 r'severity \w+ \((\d+|None)/10\))\s*[—–-]+\s*(?:Issue:\s*)?(.+?)(?:\.?\s*Fix:.*)?$', re.M)

# Neutral topics; polarity comes from the JSON score. ponytail: fixed lexicon, extend when 'other' dominates.
TOPICS = {'nonce': r'nonce', 'capability check': r'capabilit|current_user_can',
          'output escaping': r'escap|esc_(html|attr|url)|xss',
          'input sanitization': r'sanitiz|unslash|\$_(get|post|request|server|cookie)',
          'SQL preparation': r'prepare|interpolat|injection', 'naming': r'camel ?case|snake_case|naming',
          'docblock': r'docblock|phpdoc|@param|@return|@since',
          'translation': r'translat|text ?domain|\b_[_enx]\(|esc_html__',
          'caching': r'cach|transient', 'queries in loops': r'n\+1|in a loop|per iteration',
          'error handling': r'error handling|wp_error|exception|return value',
          'accessibility markup': r'aria|\blabel|alt text|screen reader',
          'globals / coupling': r'\bglobal|coupl|singleton',
          'size / responsibilities': r'responsibilit|too long|complex|nested',
          'WP API choice': r'wordpress api|wp api|core function|helper', 'dependencies': r'dependenc'}
RX = {k: re.compile(v, re.I) for k, v in TOPICS.items()}


def rank(judged, lo=4, hi=8, top=10):
    """r2 §5: tag each prose reason with topics, count (kind, dim, topic) once per function."""
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
            for t in [t for t, rx in RX.items() if rx.search(reason)] or ['other']:
                if (kind, d, t) in seen:
                    continue
                seen.add((kind, d, t))
                cnt[kind][(d, t)] += 1
                ex.setdefault((kind, d, t), f'{j["file"]}:{j["line"]} {j["name"]}')
    anti = [{"dimension": d, "finding": t, "count": n, "examples": [ex[('anti', d, t)]]}
            for (d, t), n in cnt['anti'].most_common(top)]
    pat = [{"dimension": d, "finding": t, "count": n} for (d, t), n in cnt['pattern'].most_common(top)]
    return anti, pat


def _row(j):
    return {k: v for k, v in j.items() if k not in ("code", "raw")}


def _skip_row(s):
    return {"file": s["file"], "line": s["line"], "name": s["name"], "loc": 0, "truncated": False,
            "status": "skipped", "reason": s["reason"], "verdict": None, "overall_score": None,
            "scores": {}, "critique": ""}


def build_report(target_meta, model_meta, judgments, skips, cfg, files_scanned=None):
    """files_scanned: the explicit arg wins; else target_meta["files_scanned"]; else 0."""
    judged = [j for j in judgments if j["status"] == "judged"]
    n_units = len(judgments)

    # overall: LOC-weighted mean (R1.3), cap (R2.1), null when nothing judged (change 8)
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
                       "min": min(v), "n_rated": len(v), "low_rate": round(sum(x <= 4 for x in v) / len(v), 3)}

    anti, pat = rank(judged)
    worst = sorted(judged, key=lambda j: (j["overall_score"], j["id"]))[:10]

    if files_scanned is None:
        files_scanned = target_meta.get("files_scanned", 0)
    workers, slots = model_meta.get("workers", cfg.workers), model_meta.get("server_slots")
    return {
        "schema_version": "1.0",
        "target": {"name": target_meta.get("name", ""), "type": target_meta.get("type", "plugin"),
                   "files_scanned": files_scanned, "sha256": target_meta.get("sha256", "")},
        "model": {"file": model_meta.get("file", ""), "base_url": model_meta.get("base_url", ""),
                  "workers": workers, "server_slots": slots,
                  "reproducible": workers == 1 or slots == 1},  # T5: only serial decoding is deterministic
        "summary": {
            "overall_score": overall, "overall_score_uncapped": uncapped, "capped": capped,
            "cap_reasons": cap_reasons,
            "verdict_counts": {"PASS": 0, "FAIL": 0, **Counter(j["verdict"] for j in judged)},
            "coverage": {"units": n_units, "judged": len(judged), "unjudged": n_units - len(judged),
                         "skipped": len(skips), "pct": pct,
                         "low_coverage": not judged or pct < cfg.coverage_threshold},
        },
        "dimensions": dims,
        "anti_patterns": anti,
        "patterns": pat,
        "worst_functions": [{"file": j["file"], "line": j["line"], "name": j["name"],
                             "overall_score": j["overall_score"], "verdict": j["verdict"]} for j in worst],
        "functions": [_row(j) for j in judgments] + [_skip_row(s) for s in skips],
    }

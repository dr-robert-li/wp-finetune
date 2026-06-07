"""Three-layer merge-fidelity gate for wp-reasoning-v3 merged-served (plan 04.4-02).

Proves the merged-vLLM-served v3 reproduces Tinker-sampled JUDGE behavior so the Phase-4.3
Tinker REVL-01 judge Spearman (0.263) can carry. REVL-02 generation PHPCS is OUT of scope
(judge fidelity does not transfer to generation; plan 04 measures REVL-02 fresh — SC2).

Task 1 (this commit): serve preconditions only — served-model identity, tokenizer parity
(corrected), thinking disabled. Task 2 adds L1 (forward anchor, corroboration), L2 (24-prompt
invalid-PHP sentinel verdict agreement, BLOCKING), L3 (Spearman >= 0.95 on 121 val rows, BLOCKING)
and the carry_judge_evidence decision.

Fidelity logic helpers come from plan 01 (unit-tested in test_fidelity_protocol.py):
    from scripts.merge_tinker_v3 import sentinel_agreement, spearman_agree
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.merge_tinker_v3 import sentinel_agreement, spearman_agree  # noqa: E402,F401  (F401: used in Task 2)

STOCK_VOCAB = 151936       # served MODEL embedding / output vocab (config.vocab_size)
STOCK_TOK_LEN = 151669     # stock tokenizer real token count (see plan 01: padded model vs tokenizer)
SERVED_MODEL_NAME = "wp-reasoning-v3"
STAGING_DIR = "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3"
MERGE_REPORT = "output/merge_v3/merge_report.json"
EXPECTED_SHARDS = 13


class PreconditionError(RuntimeError):
    pass


def _http_get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def assert_served_identity(endpoint: str, merge_report_path: str = MERGE_REPORT,
                           staging_dir: str = STAGING_DIR) -> dict:
    """Confirm the vLLM endpoint serves the v3 staging weights, NOT the stale ckpt-72 canonical.

    Fingerprint = served-model-name == wp-reasoning-v3 AND the staging it was launched from is
    the anchor-certified v3 (merge_report status + 13 shards on disk). /v1/models alone cannot
    prove weight identity, so we bind the served name to the certified staging fingerprint.
    """
    base = endpoint.rstrip("/")
    models = _http_get_json(f"{base}/models")
    served = [m.get("id") for m in models.get("data", [])]
    name_ok = SERVED_MODEL_NAME in served
    rep = json.load(open(merge_report_path)) if os.path.exists(merge_report_path) else {}
    report_ok = (rep.get("status") == "staging_anchor_certified"
                 and rep.get("shard_count") == EXPECTED_SHARDS
                 and rep.get("anchors_all_pass") is True)
    shards_on_disk = len([f for f in os.listdir(staging_dir)
                          if f.startswith("model-") and f.endswith(".safetensors")]) \
        if os.path.isdir(staging_dir) else 0
    disk_ok = shards_on_disk == EXPECTED_SHARDS
    ok = bool(name_ok and report_ok and disk_ok)
    detail = {"served_models": served, "name_ok": name_ok, "report_certified": report_ok,
              "shards_on_disk": shards_on_disk, "fingerprint_shards": EXPECTED_SHARDS, "ok": ok}
    if not ok:
        raise PreconditionError(f"served identity check failed (stale ckpt-72?): {detail}")
    return detail


def assert_tokenizer_vocab(staging_dir: str = STAGING_DIR) -> dict:
    """Tokenizer parity. CORRECTED from the plan's `len==151936` spec error (see plan 01):
    151936 is the padded MODEL embedding; the stock tokenizer has 151669 tokens. The real
    invariant is (a) served model vocab_size == 151936, (b) the tokenizer is STOCK — task
    markers <wp_gen>/<wp_judge> tokenize as PLAIN TEXT (>1 BPE pieces), matching v3 training,
    NOT the extended tokenizer's single special ids. A 151671 extended tokenizer would invalidate
    the comparison.
    """
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(staging_dir)
    tok = AutoTokenizer.from_pretrained(staging_dir)
    model_vocab = int(cfg.vocab_size)
    wp_pieces = len(tok.encode("<wp_judge>", add_special_tokens=False))
    tok_len = len(tok)
    max_id = max(tok.get_vocab().values())
    stock_text_routing = wp_pieces > 1 and tok_len == STOCK_TOK_LEN and max_id < model_vocab
    ok = (model_vocab == STOCK_VOCAB) and stock_text_routing
    detail = {"tokenizer_vocab": model_vocab, "tokenizer_len": tok_len,
              "wp_judge_pieces": wp_pieces, "max_id": max_id,
              "stock_text_routing": stock_text_routing, "ok": ok}
    if not ok:
        raise PreconditionError(f"tokenizer parity failed (extended tokenizer?): {detail}")
    return detail


def assert_think_disabled(endpoint: str, served_model: str = SERVED_MODEL_NAME) -> dict:
    """Probe one short generation; v3 must match Tinker qwen3_disable_thinking. We send
    chat_template_kwargs enable_thinking=false AND apply strip_think_blocks before scoring.
    Precondition passes if the probe does not lead with <think> OR strip_think_blocks neutralizes
    any think scaffold (robust double-guard).
    """
    from eval.output_parsers import strip_think_blocks

    base = endpoint.rstrip("/")
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 16, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{base}/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    text = resp["choices"][0]["message"]["content"]
    leads_think = text.lstrip().startswith("<think>")
    stripped = strip_think_blocks(text)
    strip_effective = (stripped != text) or (not leads_think)
    ok = (not leads_think) or strip_effective
    detail = {"probe_leads_with_think": leads_think, "strip_effective": strip_effective,
              "think_handling": "enable_thinking_false+strip_think_blocks", "ok": ok}
    if not ok:
        raise PreconditionError(f"think-disable precondition failed: {detail}")
    return detail


def run_preconditions(endpoint: str) -> dict:
    """Run all three serve preconditions; raise on any failure. Returns a report dict."""
    return {
        "served_identity": assert_served_identity(endpoint),
        "tokenizer": assert_tokenizer_vocab(),
        "think": assert_think_disabled(endpoint),
    }


# --------------------------------------------------------------------------- #
# Three-layer fidelity gate (L1 corroboration, L2 sentinel BLOCKING, L3 Spearman BLOCKING).
# --------------------------------------------------------------------------- #

POLICY_THRESHOLD = 70.0          # VERDICT-POLICY: effective FAIL if model FAIL OR overall < 70
SENTINEL_PROMPTS = "data/reasoning_dataset/invalid_php_sentinel.jsonl"
SENTINEL_TINKER_SUMMARY = "output/format_stability/invalid_php_sentinel_v3_summary.json"
VAL_PROMPTS = "data/reasoning_dataset/openai_val.jsonl"
TINKER_PAIRS = "output/eval_reasoning/reasoning_v3_tinker/eval_judge_results.pairs.jsonl"
L2_MIN_AGREE = 22                # >= 22/24 sentinel verdict agreement
L3_MIN_SPEARMAN = 0.95
L3_MIN_PARSE_COVERAGE = 0.90


def _generate_chat(endpoint: str, model: str, user_contents: list, max_tokens: int = 1024) -> list:
    """Generate at temp 0 with thinking disabled; return strip_think_blocks(text) per prompt."""
    import openai

    from eval.output_parsers import strip_think_blocks
    client = openai.OpenAI(base_url=endpoint, api_key="none")
    outs = []
    for c in user_contents:
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": c}],
                max_tokens=max_tokens, temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            outs.append(strip_think_blocks(resp.choices[0].message.content or ""))
        except Exception as e:  # noqa: BLE001
            print(f"[gen] error: {e}", file=sys.stderr)
            outs.append("")
    return outs


def _policy_verdict(text: str) -> tuple:
    """(effective_verdict, overall, false_pass) under VERDICT-POLICY (PASS iff overall>=70 & model not FAIL)."""
    from scripts.check_invalid_php_sentinel import _verdict
    from eval.output_parsers import parse_judge_scores

    model_v = _verdict(text)
    parsed = parse_judge_scores(text, "auto") or {}
    overall = parsed.get("overall")
    if model_v == "FAIL":
        eff, fp = "FAIL", False
    elif overall is not None:
        fp = overall >= POLICY_THRESHOLD
        eff = "PASS" if fp else "FAIL"
    else:
        fp = model_v == "PASS"
        eff = "PASS" if fp else "FAIL"
    return eff, overall, fp


def run_l2_sentinel(endpoint: str, out_jsonl: str) -> dict:
    """L2 (BLOCKING): 24 invalid-PHP sentinel prompts on merged-served must agree with the
    Tinker-sampled policy verdicts (all FAIL) >= 22/24, with 0/24 merged false-pass."""
    prompts = [json.loads(l) for l in open(SENTINEL_PROMPTS)]
    user = [p["messages"][0]["content"] for p in prompts]
    responses = _generate_chat(endpoint, SERVED_MODEL_NAME, user)
    tinker = json.load(open(SENTINEL_TINKER_SUMMARY))
    # Tinker effective policy verdicts: rows recorded policy_false_pass; effective FAIL unless false-pass.
    tinker_eff = ["PASS" if r.get("policy_false_pass") else "FAIL" for r in tinker["rows"]]
    merged_eff, false_pass, rows = [], 0, []
    for i, (p, txt) in enumerate(zip(prompts, responses)):
        eff, overall, fp = _policy_verdict(txt)
        merged_eff.append(eff)
        false_pass += int(fp)
        rows.append({"index": i, "defect_category": p["metadata"].get("defect_category"),
                     "verdict": eff, "overall_score": overall, "policy_false_pass": fp,
                     "response_head": txt[:160]})
    with open(out_jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    agree = sentinel_agreement(tinker_eff, merged_eff)
    n = len(tinker_eff)
    passed = (agree >= L2_MIN_AGREE) and (false_pass == 0)
    return {"layer": "L2_sentinel", "n": n, "agreement": agree, "min_agree": L2_MIN_AGREE,
            "merged_false_pass": false_pass, "tinker_false_pass": sum(r.get("policy_false_pass", 0) for r in tinker["rows"]),
            "pass": passed}


def run_l3_spearman(endpoint: str, out_jsonl: str) -> dict:
    """L3 (BLOCKING): Spearman(tinker_overall, merged_overall) >= 0.95 on the 121 REVL-01 val rows."""
    from eval.output_parsers import parse_judge_scores

    pairs = [json.loads(l) for l in open(TINKER_PAIRS)]
    val = [json.loads(l) for l in open(VAL_PROMPTS)]
    judge = [r for r in val if any("<wp_judge>" in m.get("content", "")
                                    for m in r["messages"] if m["role"] == "user")]
    assert len(judge) == len(pairs), f"val judge {len(judge)} != tinker pairs {len(pairs)}"
    user = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in judge]
    responses = _generate_chat(endpoint, SERVED_MODEL_NAME, user)
    tinker_used, merged_used, rows = [], [], []
    for i, (pr, txt) in enumerate(zip(pairs, responses)):
        parsed = parse_judge_scores(txt, "auto") or {}
        m_overall = parsed.get("overall")
        t_overall = pr.get("model_overall")
        ok = (m_overall is not None and t_overall is not None)
        if ok:
            tinker_used.append(t_overall); merged_used.append(m_overall)
        rows.append({"index": pr.get("index", i), "tinker_overall": t_overall,
                     "merged_overall": m_overall, "parse_ok": ok})
    with open(out_jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from scripts.merge_tinker_v3 import spearman_rho
    rho = spearman_rho(tinker_used, merged_used) if len(tinker_used) >= 2 else 0.0
    coverage = len(tinker_used) / len(pairs)
    passed = (rho >= L3_MIN_SPEARMAN) and (coverage >= L3_MIN_PARSE_COVERAGE)
    return {"layer": "L3_spearman", "n": len(pairs), "n_used": len(tinker_used),
            "parse_coverage": round(coverage, 4), "spearman": round(rho, 4),
            "min_spearman": L3_MIN_SPEARMAN, "pass": passed}


def l2_failure_diagnosis() -> list:
    """Encoded diagnostic order for a failed L2 (formatting is likeliest, not merge arithmetic)."""
    return ["tokenizer_parity", "think_disable", "prompt_template", "merge_math"]


if __name__ == "__main__":  # pragma: no cover
    # Task 2 wires the full L1/L2/L3 gate. Standalone preconditions probe:
    ep = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8021/v1"
    print(json.dumps(run_preconditions(ep), indent=2))

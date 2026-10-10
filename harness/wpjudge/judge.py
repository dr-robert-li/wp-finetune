"""Units -> judgments via the wp-judge model behind an OpenAI-compatible server (llama-server)."""
import json
import math
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI

from .aggregate import DIM

PROMPT = "<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"  # training format, no system prompt

# Copied from eval/eval_judge.py _judge_create: if the server rejects the llama.cpp
# extra_body fields, drop them for the rest of the process and warn.
# cache_prompt False: a reused slot's prompt KV from a concurrent (batched) run moves w1 scores.
_thinking_kwarg_supported = True


def _judge_create(client, *, model, messages, max_tokens=2048, temperature=0.0):
    global _thinking_kwarg_supported
    if _thinking_kwarg_supported:
        try:
            return client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, logprobs=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False},
            )
        except Exception as e:  # noqa: BLE001
            emsg = str(e).lower()
            if "template" in emsg or "enable_thinking" in emsg or "cache_prompt" in emsg:
                _thinking_kwarg_supported = False
                print("WARNING [wpjudge]: server rejected chat_template_kwargs/cache_prompt; "
                      "dropping them for this process. Unclosed <think> "
                      "blocks will show up as parse failures.", file=sys.stderr, flush=True)
            else:
                raise
    return client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, logprobs=True)


# Copied verbatim from eval/eval_judge.py:310-383.
def parse_judge_response(response: str) -> Optional[dict]:
    """Parse model judge response and extract score fields.

    PURE parser — returns exactly what the model emitted (no derived fields), so
    it is safe for BOTH the served-judge reward path AND teacher-GT extraction
    (_extract_gt_from_assistant), which requires overall_score to be ABSENT when
    the target omits it (canonical GT = rubric_scorer, never a derived proxy).
    overall_score derivation for the reward path lives in judge_score_single.

    Handles:
      - <judge_output>...</judge_output> tag block (v1.2 reasoning judge)
      - Raw JSON string
      - JSON in markdown code fences (```json ... ``` or ``` ... ```)
      - Embedded {...} object
      - Missing keys -> returns dict without those keys

    Args:
        response: Raw model response string.

    Returns:
        Parsed dict with judge fields (may include overall_score), or None
        if the response cannot be parsed as JSON.
    """
    text = response.strip()

    # Strip <think>...</think> blocks (Qwen3 reasoning mode).
    # Must happen before JSON extraction — thinking content may contain
    # curly braces that confuse the greedy {.*} regex below.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strategy 0: <judge_output>...</judge_output> tag block. MUST precede the
    # greedy {.*} scan: the v1.2 judge prefixes a [REASONING] prose block that
    # quotes the code under review (e.g. `{$wpdb->prefix}`), so a greedy first-{
    # to last-} match spans prose+JSON and fails. The tags bound the JSON exactly.
    # (Teacher-GT targets carry no such tags, so this strategy never alters them.)
    match = re.search(r"<judge_output>\s*(\{.*\})\s*</judge_output>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 1: raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: ```json fenced block
    match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: generic ``` fenced block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 4: embedded JSON object (first { ... } match)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Cannot parse
    return None


def _sync_headers(critique, scores):
    """T2b: prose header numbers are stale severity buckets (R2); keep them as prose_scores, rewrite from JSON."""
    prose = {}

    def fix(m):
        d = m.group(1).lower().replace(' ', '_')
        n = m.group(2) or m.group(3)
        prose.setdefault(d, int(n) if n.isdigit() else None)
        end = m.end(2) + 3 if m.group(2) else m.end(3) + 4      # past "/10" or "/10)"
        return f"{m.group(1)}: score {scores.get(d)}/10" + m.string[end:m.end()]
    return DIM.sub(fix, critique), prose


def to_judgment(unit, raw, attempts, verdict_p=None):
    """Raw model text -> judged Judgment, or None if unusable (no JSON or no numeric overall_score)."""
    j = parse_judge_response(raw)
    if not isinstance(j, dict) or not isinstance(j.get("overall_score"), (int, float)) \
            or isinstance(j["overall_score"], bool):
        return None
    scores = {k: int(v) for k, v in j.items()  # A5: scores only from the JSON, never the prose
              if k not in ("verdict", "overall_score") and isinstance(v, (int, float))
              and not isinstance(v, bool) and float(v).is_integer()}
    critique, prose = _sync_headers(re.split(r"\[/REASONING\]|<judge_output>", raw, maxsplit=1)[0].strip(), scores)
    return {**unit, "status": "judged", "attempts": attempts, "verdict": j.get("verdict"),
            "overall_score": int(round(j["overall_score"])), "scores": scores, "verdict_p": verdict_p,
            "prose_scores": prose, "critique": critique, "raw": raw, "error": None}


def verdict_prob(choice):
    """T4: p of the first non-blank, non-quote token after "verdict": inside <judge_output>; None if absent."""
    toks = getattr(getattr(choice, "logprobs", None), "content", None) or []
    starts, text = [], ""
    for t in toks:
        starts.append(len(text))
        text += t.token
    i = text.find("<judge_output>")
    m = re.compile(r'"verdict"\s*:').search(text, i) if i >= 0 else None
    if not m:
        return None
    for p, t in zip(starts, toks):
        if p + len(t.token) > m.end() and t.token[max(0, m.end() - p):].strip(' \t\r\n"'):
            return round(math.exp(t.logprob), 3)
    return None


def judge_one(client, model, unit, max_tokens=2048):
    raw, err = "", None
    for attempt in (1, 2):  # retry once
        try:
            r = _judge_create(client, model=model, max_tokens=max_tokens,
                              messages=[{"role": "user", "content": PROMPT.format(code=unit["code"])}])
            raw = r.choices[0].message.content or ""
            j = to_judgment(unit, raw, attempt, verdict_prob(r.choices[0]))
            if j:
                return j
            err = "parse_failure"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return {**unit, "status": "unjudged", "attempts": 2, "verdict": None, "overall_score": None,
            "scores": {}, "critique": "", "raw": raw, "error": err}


def server_slots(base_url):
    """llama-server slot count from GET <root>/props (total_slots), or None on any error."""
    try:
        with urllib.request.urlopen(re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/props", timeout=5) as r:
            n = json.load(r).get("total_slots")
        return n if type(n) is int else None
    except Exception:  # noqa: BLE001
        return None


def judge_units(units, base_url, model, workers=1, on_result=None, max_tokens=2048):
    """Judge every unit concurrently; on_result(judgment) fires per finished unit. Returns judgments sorted by id."""
    client = OpenAI(base_url=base_url, api_key="none", timeout=600, max_retries=0)
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(judge_one, client, model, u, max_tokens) for u in units]
        for fut in as_completed(futures):
            j = fut.result()
            if on_result:
                on_result(j)
            out.append(j)
    return sorted(out, key=lambda j: j["id"])

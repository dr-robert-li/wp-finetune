"""Per-dimension Spearman correlation evaluation for judge mode.

Compares model's <wp_judge> dimension scores against ground truth scores
extracted from the test dataset's assistant response JSON.

Ground truth source: The test set's assistant response already contains
scored judge output (overall_score, wpcs_compliance, security_score, etc.)
with real variance (min=10, max=100, stdev≈14). Using rubric_scorer as GT
produces near-zero variance (stdev≈0.4), making Spearman meaningless.

rubric_scorer is retained as a fallback for dimensions not covered by the
test set's GT fields, and for examples whose assistant response cannot be
parsed.

Usage:
    python -m eval.eval_judge [--limit N] [--output PATH]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import openai
from scipy.stats import spearmanr

from eval.rubric_definitions import DIM_NAME_MAP, DIMENSION_WEIGHTS
from eval.rubric_scorer import score_code
from scripts.dgx_toolbox import get_toolbox

# Module-level toolbox singleton (lazy)
_dgx = None

# RC-A fix (Phase 04.4 / D-IT-02): Qwen3 reasoning adapters Tinker-trained under the
# qwen3_disable_thinking renderer never learned to CLOSE a <think> block. Served via vLLM
# WITHOUT chat_template_kwargs enable_thinking=False, the merged model emits an UNCLOSED
# <think> block -> judge JSON is unparseable -> 19-25% parse failures + Spearman collapse.
# strip_think() CANNOT rescue this: its regex `<think>.*?</think>` requires the closing tag,
# which is exactly what's missing. So the kwarg is the ONLY guard. If a served template
# rejects the kwarg we drop it for the rest of the run but WARN LOUDLY — a silently dropped
# kwarg would reproduce the bug with no backstop and masquerade as a green run.
_thinking_kwarg_supported = True


def _judge_create(client, *, model, messages, max_tokens=1024, temperature=0.0):
    """Query a vLLM judge endpoint with enable_thinking=False (RC-A fix).

    Mirrors the graceful-fallback pattern in scripts/fidelity_gate_v3.py. Non-template
    exceptions propagate to the caller's existing api_error handling.
    """
    global _thinking_kwarg_supported
    if _thinking_kwarg_supported:
        try:
            return client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:  # noqa: BLE001
            emsg = str(e).lower()
            if "enable_thinking" in emsg or "chat_template" in emsg or "template" in emsg:
                _thinking_kwarg_supported = False
                print(
                    "WARNING [eval_judge RC-A]: served template REJECTED "
                    "chat_template_kwargs enable_thinking=False; dropping it for the rest "
                    "of this run. strip_think CANNOT remove UNCLOSED <think> blocks, so "
                    "judge parse failures from unterminated reasoning are NOT guarded this "
                    "run. Treat any parse_failure_rate from this run as SUSPECT.",
                    file=sys.stderr, flush=True,
                )
            else:
                raise
    return client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)


def _get_dgx():
    global _dgx
    if _dgx is None:
        _dgx = get_toolbox()
    return _dgx


DEFAULT_MODEL = "openai/qwen3-wp"


def _detect_model(client: openai.OpenAI) -> str:
    """Auto-detect the served model name from /v1/models endpoint."""
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Model field -> dimension key mapping
# ---------------------------------------------------------------------------

# Model outputs these fields; map each to the rubric dimension key.
# Fields not in this map (like overall_score, documentation_score) are
# handled separately.
#
# The trained 30/70 judge emits the same schema as the test set's assistant
# response — `_score`-suffixed for perf/i18n/accessibility/security, plain for
# the rest. Earlier the map was derived from DIM_NAME_MAP which used a different
# (older) field naming, causing model-side dim scores to never be recorded for
# D4/D6/D7 (n=0 paired data in Phase 0.3). Keep this in sync with
# _GT_FIELD_TO_DIM below.
_MODEL_FIELD_TO_DIM: dict[str, str] = {
    "wpcs_compliance": "D1_wpcs",
    "security_score": "D2_security",
    "sql_safety": "D3_sql",
    "performance_score": "D4_perf",
    "wp_api_usage": "D5_wp_api",
    "i18n_score": "D6_i18n",
    "accessibility_score": "D7_a11y",
    "error_handling": "D8_errors",
    "code_structure": "D9_structure",
}

# ---------------------------------------------------------------------------
# GT field -> dimension key mapping
# ---------------------------------------------------------------------------

# The test dataset's assistant response uses a different (simpler) set of
# field names than the model output fields in DIM_NAME_MAP. All 9 rubric
# dimensions are supported — older test sets may omit some fields, in which
# case those dims are not scored from the GT and fall back to rubric_scorer.
#
# NOTE: documentation_score exists in some GT records but has no corresponding
# rubric dimension — it is intentionally omitted.
_GT_FIELD_TO_DIM: dict[str, str] = {
    "wpcs_compliance": "D1_wpcs",
    "security_score": "D2_security",
    "sql_safety": "D3_sql",
    "performance_score": "D4_perf",
    "wp_api_usage": "D5_wp_api",
    "i18n_score": "D6_i18n",
    "accessibility_score": "D7_a11y",
    "error_handling": "D8_errors",
    "code_structure": "D9_structure",
}


# ---------------------------------------------------------------------------
# Parse helpers (kept from original)
# ---------------------------------------------------------------------------


# v1.2 reasoning-judge <judge_output> field name -> internal rubric dim key.
# The served judge emits SHORT field names ("security", "performance", "i18n",
# "accessibility") AND the [/REASONING]-block sometimes uses the *_score variants
# the eval harness historically expected — accept both so derivation never
# silently drops a present dimension. "code_quality" / "dependency_integrity" are
# intentionally UNMAPPED: dim_map.json rules them "no clean eval equivalent", so
# they are excluded from the weighted overall (not silently folded into D9).
_JUDGE_FIELD_TO_DIM = {
    "wpcs_compliance": "D1_wpcs",
    "security": "D2_security", "security_score": "D2_security",
    "sql_safety": "D3_sql",
    "performance": "D4_perf", "performance_score": "D4_perf",
    "wp_api_usage": "D5_wp_api",
    "i18n": "D6_i18n", "i18n_score": "D6_i18n", "i18n_l10n": "D6_i18n",
    "accessibility": "D7_a11y", "accessibility_score": "D7_a11y",
    "error_handling": "D8_errors",
    "code_structure": "D9_structure",
}

# PASS threshold (verdict POLICY, 04.3 VERDICT-POLICY): PASS iff overall >= 70.
# A FAIL verdict must therefore never derive an overall >= 70.
_PASS_THRESHOLD = 70.0


def _dump_judge_failure(php_code: str, raw_text: str, resp: object) -> None:
    """Diagnostic: when judge_score_single is about to return None, append the
    code-under-judgement + the raw judge output + finish_reason to the JSONL at
    $WP_JUDGE_DEBUG_DUMP. No-op unless that env var is set (zero prod overhead).
    Lets us SEE the real live parse-failure population instead of theorizing it."""
    path = os.environ.get("WP_JUDGE_DEBUG_DUMP")
    if not path:
        return
    try:
        finish = None
        usage_completion = None
        try:
            finish = resp.choices[0].finish_reason  # type: ignore[attr-defined]
            usage_completion = getattr(getattr(resp, "usage", None), "completion_tokens", None)
        except Exception:  # noqa: BLE001
            pass
        rec = {
            "finish_reason": finish,
            "completion_tokens": usage_completion,
            "raw_len": len(raw_text),
            "code_len": len(php_code),
            "has_judge_output_tag": "<judge_output>" in raw_text,
            "has_prose_score": bool(re.search(r"score\s+\d+\s*/\s*10", raw_text)),
            # Residual triage: was the judged code already extracted clean PHP, or
            # did a fence survive (extract miss) / is there no code at all (not-code
            # generation -> correct reward is low, not impute)?
            "code_has_fence": "```" in php_code,
            "code_starts_phpish": php_code.lstrip()[:6].lower().startswith(("<?php", "functi", "class ", "add_ac", "regist")),
            "code": php_code[:6000],
            "raw_text": raw_text[:6000],
        }
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 — diagnostics must never break scoring
        pass


def _derive_overall_from_dims(obj: dict) -> "Optional[float]":
    """Derive overall_score (0-100) from per-dimension 0-10 scores.

    The v1.2 reasoning judge is bimodal: it frequently emits the per-dimension
    block + verdict but OMITS overall_score. Imputing those to the group mean
    (D-08-07) erases the (usually low) score the output earned — a directional
    reward bias. Instead derive a proxy overall:

      - weighted mean over the dims the judge actually emitted+mapped, weights =
        canonical DIMENSION_WEIGHTS renormalized over present dims (symmetric
        with rubric aggregation; dim_map.json is the single source so Phase 7/10
        reward shaping does not drift);
      - fallback to plain mean x10 if NO emitted field maps to a weighted dim;
      - cap below the PASS threshold when verdict == FAIL (a FAIL must not
        derive a passing overall).

    Returns None if no numeric dimension scores are present at all.
    """
    dim_scores: dict[str, float] = {}   # internal dim -> 0-10
    numeric_fallback: list[float] = []   # any numeric dim-like field, 0-10
    for field, val in obj.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if field == "overall_score":
            continue
        numeric_fallback.append(float(val))
        dim = _JUDGE_FIELD_TO_DIM.get(field)
        if dim is not None:
            dim_scores[dim] = float(val)

    if dim_scores:
        total_w = sum(DIMENSION_WEIGHTS[d] for d in dim_scores)
        overall10 = sum(dim_scores[d] * DIMENSION_WEIGHTS[d] for d in dim_scores) / total_w
        overall = overall10 * 10.0
    elif numeric_fallback:
        overall = (sum(numeric_fallback) / len(numeric_fallback)) * 10.0
    else:
        return None

    verdict = obj.get("verdict")
    if isinstance(verdict, str) and verdict.strip().upper() == "FAIL":
        overall = min(overall, _PASS_THRESHOLD - 1.0)
    return max(0.0, min(100.0, overall))


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


def judge_score_single(
    php_code: str,
    client: "openai.OpenAI",
    model: str,
    max_tokens: int = 1024,
) -> "Optional[float]":
    """Invoke wp_judge on a single PHP code string.

    Returns raw overall_score (0-100) as a float, or None on parse failure.

    CRITICAL: MUST use _judge_create (not client.chat.completions.create directly)
    to preserve the RC-A enable_thinking=False guard.  Without the guard, the
    merged Qwen3 model emits unclosed <think> blocks that parse_judge_response
    cannot rescue — causing 19-25% parse failures (see RC-A fix, Phase 04.4).

    The v1.2 reasoning judge emits a [REASONING] prose block followed by a
    <judge_output> JSON block (~750+ tokens total). The default MUST cover that —
    512 truncates before the JSON, yielding a silent None (then group-mean
    imputation in the RL reward path). 1024 matches _judge_create and the eval
    path (run_eval) and is the validated budget; raise it for very long inputs.

    Args:
        php_code:   The PHP source string to evaluate.
        client:     An openai.OpenAI instance pointed at the vLLM judge endpoint.
        model:      The served model name (e.g. "openai/qwen3-wp").
        max_tokens: Max tokens for the judge response (default 1024).

    Returns:
        float: raw overall_score from the parsed judge response.
        None:  if the response cannot be parsed or overall_score is missing/non-numeric.
    """
    messages = [
        {
            "role": "user",
            "content": f"<wp_judge> Evaluate this WordPress code:\n\n{php_code}",
        }
    ]
    resp = _judge_create(
        client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    raw_text = resp.choices[0].message.content or ""
    parsed = parse_judge_response(raw_text)
    score: Optional[float] = None
    if parsed is not None:
        overall = parsed.get("overall_score")
        if isinstance(overall, (int, float)) and not isinstance(overall, bool):
            score = float(overall)
        else:
            # Bimodal judge: per-dim block + verdict present but overall_score
            # OMITTED. Derive it (served-judge dims are 0-10) rather than None — a
            # None here triggers group-mean imputation (D-08-07) that erases the
            # low score the output earned, biasing reward toward verbose/rough
            # generations. Derivation lives HERE (reward boundary), NOT in
            # parse_judge_response, so teacher-GT extraction stays pure.
            score = _derive_overall_from_dims(parsed)
    if score is None:
        _dump_judge_failure(php_code, raw_text, resp)
    return score


def _extract_code_from_judge_prompt(user_message: str) -> str:
    """Extract the PHP code being judged from a <wp_judge> user message.

    The wp_judge format is:
        <wp_judge> Evaluate this WordPress code:\n\n<?php\n...code...

    Returns the code portion, or empty string if not found.
    """
    # Remove the <wp_judge> tag and "Evaluate this WordPress code:" prefix
    text = re.sub(r"<wp_judge>\s*", "", user_message, count=1)
    text = re.sub(
        r"Evaluate this WordPress code:\s*", "", text, count=1, flags=re.IGNORECASE
    )
    return text.strip()


def _extract_gt_from_assistant(messages: list[dict]) -> Optional[dict]:
    """Extract ground truth scores from the test example's assistant response.

    The test dataset's assistant response contains a JSON object with scored
    judge output, e.g.::

        {
            "overall_score": 45,
            "wpcs_compliance": 55,
            "security_score": 10,
            "performance_score": 80,
            "i18n_score": 55,
            "accessibility_score": 65,
            "documentation_score": 55,
            ...
        }

    Returns a dict with keys ``overall`` (float) and ``dimension_scores``
    (dict[str, float]) using internal dimension keys (D1_wpcs, etc.), or
    ``None`` if the assistant response cannot be parsed or does not contain
    ``overall_score``.

    Only dimensions present in _GT_FIELD_TO_DIM are included; dimensions not
    covered by the GT (D3_sql, D5_wp_api, D8_errors, D9_structure) are absent
    from the returned dict.
    """
    assistant_content = next(
        (m["content"] for m in messages if m["role"] == "assistant"), None
    )
    if not assistant_content:
        return None

    parsed = parse_judge_response(assistant_content)
    if parsed is None or "overall_score" not in parsed:
        return None

    overall = parsed.get("overall_score")
    if not isinstance(overall, (int, float)):
        return None

    dim_scores: dict[str, float] = {}
    for gt_field, dim_key in _GT_FIELD_TO_DIM.items():
        val = parsed.get(gt_field)
        if isinstance(val, (int, float)):
            dim_scores[dim_key] = float(val)

    return {"overall": float(overall), "dimension_scores": dim_scores}


# ---------------------------------------------------------------------------
# Spearman helper
# ---------------------------------------------------------------------------


def _safe_spearman(xs: list[float], ys: list[float]) -> dict:
    """Compute Spearman correlation, returning a standard dict.

    Returns {"corr": float, "p_value": float, "n_pairs": int}.
    If fewer than 2 pairs or all values identical, corr=0.0, p_value=1.0.
    """
    n = len(xs)
    if n < 2:
        return {"corr": 0.0, "p_value": 1.0, "n_pairs": n}
    # scipy warns / returns nan when all values are identical
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"corr": 0.0, "p_value": 1.0, "n_pairs": n}
    result = spearmanr(xs, ys)
    return {
        "corr": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_eval(
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    limit: Optional[int] = None,
    output_path: str = "output/eval_judge_results.json",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    output_format: str = "auto",
    gt_mode: str = "dataset",
    responses_jsonl: Optional[str] = None,
) -> dict:
    """...

    output_format: 'json'|'prose'|'auto' — how to parse MODEL judge output
        (council Option B). 'auto' = JSON-first, prose fallback. Reasoning v1.2
        emits prose; Phase-4 JSON callers use 'auto' (unaffected).
    gt_mode: 'dataset' (legacy: dataset assistant-target GT, Phase-4 behavior) or
        'calibrated_canonical' (Phase-4.4 REVL-01: canonical GT = rubric
        calibrated_overall HARD + dataset teacher as SOFT diagnostic; per-dim
        Spearman restricted to dim_map clean set; rows missing calibrated_overall
        EXCLUDED + counted, no raw-rubric fallback; GT-variance preflight).
    """
    """Compute per-dimension Spearman correlation between model judge scores
    and ground truth scores from the test dataset.

    Loads wp_judge examples from the test dataset, queries the served model via
    vLLM endpoint (base_url arg, or EVAL_JUDGE_BASE_URL env, or DGX Toolbox
    default), parses dimension scores from judge responses, extracts GT scores
    from the test example's assistant response JSON, and computes per-dimension
    + overall Spearman correlations.

    GT source priority:
    1. Test dataset's assistant response JSON (real variance, preferred).
    2. rubric_scorer.score_code() fallback — used only when the assistant
       response cannot be parsed, or for rubric dimensions not covered by the
       GT fields (D3_sql, D5_wp_api, D8_errors, D9_structure).

    Args:
        dataset_path: Path to OpenAI-format JSONL test dataset.
        limit: Maximum number of examples to evaluate (None = all).
        output_path: Path to save JSON results.
        model: Override served model name (else auto-detect from /v1/models).
        base_url: Override OpenAI-compatible endpoint base URL. Falls back to
            EVAL_JUDGE_BASE_URL env then to DGX Toolbox's vllm_endpoint().

    Returns:
        dict with overall_spearman, per_dimension, and backward-compat fields.
    """
    if output_format not in ("json", "prose", "auto"):
        raise ValueError(f"output_format must be json|prose|auto, got {output_format!r}")
    if gt_mode not in ("dataset", "calibrated_canonical"):
        raise ValueError(f"gt_mode must be dataset|calibrated_canonical, got {gt_mode!r}")
    if gt_mode == "calibrated_canonical":
        return _run_eval_reasoning(
            dataset_path=dataset_path, limit=limit, output_path=output_path,
            model=model, base_url=base_url, output_format=output_format,
            responses_jsonl=responses_jsonl,
        )

    import os
    resolved_base_url = base_url or os.environ.get("EVAL_JUDGE_BASE_URL")
    if not resolved_base_url:
        dgx = _get_dgx()
        resolved_base_url = dgx.vllm_endpoint()
    client = openai.OpenAI(base_url=resolved_base_url, api_key="none")
    resolved_model = model or _detect_model(client)

    # Load and filter wp_judge examples
    examples = []
    dataset_file = Path(dataset_path)
    with dataset_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            messages = example.get("messages", [])
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            if "<wp_judge>" in user_msg:
                examples.append(example)

    if limit is not None:
        examples = examples[:limit]

    print(
        f"Evaluating {len(examples)} wp_judge examples via {resolved_base_url} (model={resolved_model})",
        file=sys.stderr,
    )

    # Collectors: per-dimension pairs and overall pairs
    dim_model_scores: dict[str, list[float]] = {
        dim: [] for dim in DIMENSION_WEIGHTS
    }
    dim_gt_scores: dict[str, list[float]] = {dim: [] for dim in DIMENSION_WEIGHTS}
    overall_model: list[float] = []
    overall_gt: list[float] = []
    skipped = 0

    # Per-example debug records
    pair_records: list[dict] = []
    skipped_records: list[dict] = []

    # JSONL output paths (siblings of main output)
    pairs_path = Path(output_path).with_suffix(".pairs.jsonl")
    skipped_path = Path(output_path).with_name(
        Path(output_path).stem + "_skipped.jsonl"
    )

    for i, example in enumerate(examples):
        messages = example["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        user_msg_text = user_messages[0]["content"] if user_messages else ""

        # Extract code being judged
        code = _extract_code_from_judge_prompt(user_msg_text)

        # Query model in wp_judge mode
        try:
            response = _judge_create(
                client,
                model=resolved_model,
                messages=user_messages,
                max_tokens=1024,
                temperature=0.0,
            )
            generated = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i}] Model error: {e}", file=sys.stderr)
            skipped += 1
            skipped_records.append({
                "index": i, "reason": "api_error",
                "error": str(e)[:500], "response": "",
            })
            continue

        # Parse dimension scores from response
        parsed = parse_judge_response(generated)
        if parsed is None or "overall_score" not in parsed:
            skipped += 1
            skipped_records.append({
                "index": i, "reason": "parse_fail",
                "response": generated[:2000],
                "parsed_keys": list(parsed.keys()) if parsed else None,
            })
            continue

        model_overall = parsed.get("overall_score")
        if not isinstance(model_overall, (int, float)):
            skipped += 1
            skipped_records.append({
                "index": i, "reason": "type_error",
                "response": generated[:500],
                "overall_score_value": repr(model_overall),
            })
            continue

        # Ground truth: extract from test example's assistant response.
        # The test dataset already contains scored judge output with real
        # variance (min=10, max=100, stdev≈14).  Using rubric_scorer as GT
        # produces near-zero variance (stdev≈0.4) and meaningless Spearman.
        gt_from_dataset = _extract_gt_from_assistant(messages)

        if gt_from_dataset is None:
            # Fallback: re-score via rubric_scorer (logs a warning so the
            # operator knows GT provenance for this example).
            print(
                f"  [{i}] WARNING: GT not in assistant response; "
                "falling back to rubric_scorer",
                file=sys.stderr,
            )
            gt_rubric = score_code(code)
            # Prefer XGBoost-calibrated overall when available (Phase 1a);
            # fall back to raw deterministic rubric overall otherwise.
            gt_overall_val = (
                gt_rubric.calibrated_overall
                if gt_rubric.calibrated_overall is not None
                else gt_rubric.overall
            )
            gt_dim_scores = {
                k: v
                for k, v in gt_rubric.dimension_scores.items()
                if v is not None
            }
            gt_source = (
                "rubric_scorer_calibrated"
                if gt_rubric.calibrated_overall is not None
                else "rubric_scorer"
            )
        else:
            gt_overall_val = gt_from_dataset["overall"]
            gt_dim_scores = gt_from_dataset["dimension_scores"]
            gt_source = "dataset"

        # Record overall pair
        overall_model.append(float(model_overall))
        overall_gt.append(float(gt_overall_val))

        # Record per-dimension pairs.
        # For dimensions covered by the GT dataset fields, use dataset GT.
        # For dimensions not in the GT (D3_sql, D5_wp_api, D8_errors,
        # D9_structure), fall back to rubric_scorer if we already have it,
        # otherwise skip (to avoid running rubric_scorer just for fallback dims).
        pair_record: dict = {
            "index": i,
            "prompt": user_msg_text,
            "response": generated,
            "code": code,
            "model_overall": float(model_overall),
            "gt_overall": float(gt_overall_val),
            "gt_source": gt_source,
            "dimensions": {},
        }

        # Lazy rubric fallback — only run score_code() once if needed for
        # dimensions not covered by the GT dataset.  When gt_from_dataset is
        # None we already ran score_code() above; reuse that result.  When
        # gt_from_dataset is present we start as None and score lazily only if
        # a per-dimension lookup misses.
        _rubric_fallback: Optional[object] = (
            gt_rubric if gt_from_dataset is None else None  # type: ignore[possibly-undefined]
        )

        for model_field, dim_key in _MODEL_FIELD_TO_DIM.items():
            model_val = parsed.get(model_field)
            if model_val is None or not isinstance(model_val, (int, float)):
                continue

            # Try dataset GT first
            gt_val = gt_dim_scores.get(dim_key)

            if gt_val is None:
                # Dimension not covered by dataset GT; try rubric_scorer fallback
                if _rubric_fallback is None:
                    _rubric_fallback = score_code(code)
                rb_val = _rubric_fallback.dimension_scores.get(dim_key)  # type: ignore[union-attr]
                if rb_val is not None:
                    gt_val = float(rb_val)

            if gt_val is not None:
                dim_model_scores[dim_key].append(float(model_val))
                dim_gt_scores[dim_key].append(float(gt_val))
                pair_record["dimensions"][dim_key] = {
                    "model": float(model_val),
                    "gt": float(gt_val),
                }

        pair_records.append(pair_record)

        if (i + 1) % 50 == 0:
            print(
                f"  [{i+1}/{len(examples)}] pairs collected={len(overall_model)}",
                file=sys.stderr,
            )

    # --- Compute correlations ---

    overall_result = _safe_spearman(overall_model, overall_gt)

    per_dimension: dict[str, dict] = {}
    for dim_key in DIMENSION_WEIGHTS:
        per_dimension[dim_key] = _safe_spearman(
            dim_model_scores[dim_key], dim_gt_scores[dim_key]
        )

    summary = {
        "overall_spearman": overall_result,
        "per_dimension": per_dimension,
        "skipped": skipped,
        "total": len(examples),
        # Backward compat
        "spearman_corr": overall_result["corr"],
        "p_value": overall_result["p_value"],
        "total_pairs": overall_result["n_pairs"],
    }

    # Save results JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {output_path}", file=sys.stderr)

    # Save per-example pairs JSONL for debugging
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w") as pf:
        for rec in pair_records:
            pf.write(json.dumps(rec) + "\n")
    print(f"Per-example pairs saved to {pairs_path}", file=sys.stderr)

    # Save skipped examples JSONL for diagnostic analysis
    if skipped_records:
        with skipped_path.open("w") as sf:
            for rec in skipped_records:
                sf.write(json.dumps(rec) + "\n")
        # Print skip reason summary
        reasons = {}
        for rec in skipped_records:
            r = rec["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        print(f"Skipped {len(skipped_records)} examples: {reasons}", file=sys.stderr)
        print(f"Skipped details saved to {skipped_path}", file=sys.stderr)

    # Print human-readable correlation table to stderr
    _print_correlation_table(summary)

    return summary


def _derive_prose_overall(dim_scores_0_100: dict, weights: dict) -> Optional[float]:
    """Weighted mean over emitted+mapped dims, weights renormalized over present dims.

    Symmetric with rubric aggregation (council Option 1). dim_scores are 0-100.
    Returns None if no weighted dims present.
    """
    num = 0.0
    den = 0.0
    for dim, score in dim_scores_0_100.items():
        w = weights.get(dim)
        if w is not None:
            num += w * score
            den += w
    return (num / den) if den > 0 else None


def _gt_variance_ok(values: list[float], min_stdev: float, min_unique: int) -> tuple[bool, dict]:
    """REVL-01A guard: refuse degenerate (rank-collapsed) canonical GT."""
    import statistics
    if len(values) < 2:
        return False, {"reason": "n<2", "n": len(values)}
    stdev = statistics.pstdev(values)
    uniq = len(set(round(v, 3) for v in values))
    ok = stdev >= min_stdev and uniq >= min_unique
    return ok, {"stdev": round(stdev, 3), "unique": uniq,
                "min_stdev": min_stdev, "min_unique": min_unique, "ok": ok}


def _run_eval_reasoning(
    dataset_path: str,
    limit: Optional[int],
    output_path: str,
    model: Optional[str],
    base_url: Optional[str],
    output_format: str,
    responses_jsonl: Optional[str] = None,
) -> dict:
    """REVL-01 calibrated-canonical eval path (Phase 4.4, council two-GT/Option 3).

    Canonical GT = rubric calibrated_overall (HARD, REVL-01A overall Spearman).
    Teacher GT = dataset assistant-target (SOFT diagnostic, REVL-01B).
    Per-dim Spearman = 6 clean dims only, RAW rubric per-dim basis (calibration is
    overall-only). Rows missing calibrated_overall EXCLUDED + counted (no raw
    fallback). GT-variance preflight on canonical GT before computing REVL-01A.

    OFFLINE mode (responses_jsonl): instead of querying a served endpoint, model
    responses are read from a pre-captured `{index, response}` JSONL (e.g. produced
    by Tinker sampling — the model has no HTTP endpoint). `index` MUST be the
    position in the filtered wp_judge example list, matching the enumeration below.
    GT extraction + Spearman + preflight are byte-identical to the online path, so
    the resulting REVL-01A is comparable to the served-vLLM baseline.
    """
    import os
    from eval.output_parsers import parse_judge_scores, load_dim_map

    dm = load_dim_map()
    clean_dims = set(v for k, v in dm["clean_mapped_dims"].items() if not k.startswith("_"))
    weights = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
    pf = dm["gt_variance_preflight"]

    offline_responses: Optional[dict[int, str]] = None
    client = None
    resolved_model = model or "offline-captured"
    if responses_jsonl:
        offline_responses = {}
        with open(responses_jsonl) as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "__provenance__" in rec:
                    continue
                offline_responses[int(rec["index"])] = rec.get("response", "")
        print(f"[REVL-01] OFFLINE: {len(offline_responses)} captured responses "
              f"from {responses_jsonl}", file=sys.stderr)
    else:
        resolved_base_url = base_url or os.environ.get("EVAL_JUDGE_BASE_URL")
        if not resolved_base_url:
            resolved_base_url = _get_dgx().vllm_endpoint()
        client = openai.OpenAI(base_url=resolved_base_url, api_key="none")
        resolved_model = model or _detect_model(client)

    examples = []
    with Path(dataset_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            um = next((m["content"] for m in ex.get("messages", []) if m["role"] == "user"), "")
            if um.startswith("<wp_judge>"):
                examples.append(ex)
    if limit is not None:
        examples = examples[:limit]
    _src = f"OFFLINE:{responses_jsonl}" if offline_responses is not None else resolved_base_url
    print(f"[REVL-01] {len(examples)} judge examples via {_src} "
          f"(model={resolved_model}, fmt={output_format}, gt=calibrated_canonical)",
          file=sys.stderr)

    overall_model: list[float] = []
    overall_gt_canon: list[float] = []     # calibrated rubric (HARD)
    overall_model_b: list[float] = []
    overall_gt_teacher: list[float] = []   # dataset teacher (SOFT)
    dim_model: dict[str, list[float]] = {d: [] for d in clean_dims}
    dim_gt: dict[str, list[float]] = {d: [] for d in clean_dims}
    pair_records: list[dict] = []
    excluded = {"parse_fail": 0, "no_calibrated_gt": 0, "api_error": 0}

    for i, ex in enumerate(examples):
        msgs = ex["messages"]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        user_text = user_msgs[0]["content"] if user_msgs else ""
        code = _extract_code_from_judge_prompt(user_text)
        if offline_responses is not None:
            if i not in offline_responses:
                excluded["api_error"] += 1
                continue
            generated = offline_responses[i] or ""
        else:
            try:
                resp = _judge_create(
                    client, model=resolved_model, messages=user_msgs, max_tokens=1024, temperature=0.0)
                generated = resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                excluded["api_error"] += 1
                continue

        parsed = parse_judge_scores(generated, output_format)
        if parsed is None or not parsed.get("dimension_scores"):
            excluded["parse_fail"] += 1
            continue
        parse_mode = parsed["_format"]
        model_dims = parsed["dimension_scores"]  # internal keys, 0-100

        # model_overall: json emits overall; prose -> weighted-mean derivation
        if "overall" in parsed:
            model_overall = float(parsed["overall"])
            derived = False
        else:
            mo = _derive_prose_overall(model_dims, weights)
            if mo is None:
                excluded["parse_fail"] += 1
                continue
            model_overall = mo
            derived = True

        # Canonical GT = rubric calibrated_overall. NO raw fallback.
        rub = score_code(code)
        if rub.calibrated_overall is None:
            excluded["no_calibrated_gt"] += 1
            continue
        gt_canon = float(rub.calibrated_overall)
        # raw rubric per-dim (0-10 -> 0-100) for SOFT per-dim
        rub_dims_100 = {k: float(v) * 10.0 for k, v in rub.dimension_scores.items()
                        if v is not None}

        # Teacher GT (SOFT diagnostic) from dataset assistant-target
        teacher = _extract_gt_from_assistant(msgs)

        overall_model.append(model_overall)
        overall_gt_canon.append(gt_canon)
        if teacher is not None:
            overall_model_b.append(model_overall)
            overall_gt_teacher.append(float(teacher["overall"]))

        rec = {"index": i, "parse_mode": parse_mode, "derived_overall": derived,
               "model_overall": model_overall, "gt_canonical": gt_canon,
               "gt_canonical_source": "rubric_calibrated_overall",
               "gt_teacher": (float(teacher["overall"]) if teacher else None),
               "gt_teacher_source": ("dataset_assistant_target" if teacher else "missing"),
               "dimensions": {}}
        for d in clean_dims:
            if d in model_dims and d in rub_dims_100:
                dim_model[d].append(model_dims[d])
                dim_gt[d].append(rub_dims_100[d])
                rec["dimensions"][d] = {"model": model_dims[d], "gt_raw_rubric": rub_dims_100[d]}
        pair_records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(examples)}] paired={len(overall_model)}", file=sys.stderr)

    # GT-variance preflight (HARD guard) on canonical GT
    var_ok, var_detail = _gt_variance_ok(overall_gt_canon, pf["min_stdev"], pf["min_unique_ranks"])

    revl01a = _safe_spearman(overall_model, overall_gt_canon) if var_ok else {
        "corr": None, "p_value": None, "n_pairs": len(overall_gt_canon),
        "error": "GT_VARIANCE_PREFLIGHT_FAILED", "detail": var_detail}
    revl01b = _safe_spearman(overall_model_b, overall_gt_teacher)
    per_dim = {d: _safe_spearman(dim_model[d], dim_gt[d]) for d in sorted(clean_dims)}

    summary = {
        "revl01a_overall_spearman_HARD": revl01a,
        "revl01a_gt": "rubric_calibrated_overall",
        "revl01a_variance_preflight": var_detail,
        "revl01b_overall_spearman_teacher_SOFT": revl01b,
        "per_dimension_clean_SOFT": per_dim,
        "per_dim_basis": "raw_rubric",
        "clean_dims": sorted(clean_dims),
        "n_examples": len(examples),
        "n_paired_canonical": len(overall_gt_canon),
        "n_paired_teacher": len(overall_gt_teacher),
        "excluded": excluded,
        # back-compat top-level
        "overall_spearman": revl01a,
        "spearman_corr": revl01a.get("corr"),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    pairs_path = out.with_suffix(".pairs.jsonl")
    with pairs_path.open("w") as pf_f:
        for r in pair_records:
            pf_f.write(json.dumps(r) + "\n")
    print(f"[REVL-01] saved {output_path} + {pairs_path}", file=sys.stderr)
    print(f"[REVL-01] REVL-01A(HARD) corr={revl01a.get('corr')} "
          f"var_ok={var_ok} | REVL-01B(SOFT) corr={revl01b.get('corr')} | "
          f"excluded={excluded}", file=sys.stderr)
    return summary


def _print_correlation_table(summary: dict) -> None:
    """Print a human-readable correlation table to stderr."""
    print("\n" + "=" * 64, file=sys.stderr)
    print("  Per-Dimension Spearman Correlations", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(
        f"  {'Dimension':<16} {'Corr':>8} {'p-value':>10} {'N pairs':>8}",
        file=sys.stderr,
    )
    print("-" * 64, file=sys.stderr)

    for dim_key in DIMENSION_WEIGHTS:
        d = summary["per_dimension"].get(dim_key, {})
        corr = d.get("corr", 0.0)
        pval = d.get("p_value", 1.0)
        n = d.get("n_pairs", 0)
        sig = "*" if pval < 0.05 else " "
        print(
            f"  {dim_key:<16} {corr:>8.3f} {pval:>10.4f} {n:>7d} {sig}",
            file=sys.stderr,
        )

    print("-" * 64, file=sys.stderr)
    ov = summary["overall_spearman"]
    sig = "*" if ov["p_value"] < 0.05 else " "
    print(
        f"  {'OVERALL':<16} {ov['corr']:>8.3f} {ov['p_value']:>10.4f} {ov['n_pairs']:>7d} {sig}",
        file=sys.stderr,
    )
    print("=" * 64, file=sys.stderr)
    print("  (* = p < 0.05)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-dimension Spearman correlation for wp_judge mode."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max examples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/eval_judge_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/final_dataset/openai_test.jsonl",
        help="Path to OpenAI-format JSONL test dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for vLLM (auto-detected from /v1/models if omitted)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible endpoint base URL "
             "(e.g. http://localhost:8001/v1). Falls back to EVAL_JUDGE_BASE_URL "
             "env, then to config/dgx_toolbox.yaml ports.vllm.",
    )
    parser.add_argument(
        "--gt-mode", choices=["dataset", "calibrated_canonical"], default="dataset",
        help="GT mode; calibrated_canonical = REVL-01A path.",
    )
    parser.add_argument(
        "--output-format", choices=["json", "prose", "auto"], default="auto",
        help="how to parse MODEL judge output.",
    )
    parser.add_argument(
        "--responses-jsonl", default=None,
        help="OFFLINE mode (calibrated_canonical only): read model responses from a "
             "{index, response} JSONL instead of querying an endpoint. Index = position "
             "in the filtered wp_judge list (Tinker-captured).",
    )
    args = parser.parse_args()

    summary = run_eval(
        dataset_path=args.dataset,
        limit=args.limit,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        output_format=args.output_format,
        gt_mode=args.gt_mode,
        responses_jsonl=args.responses_jsonl,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

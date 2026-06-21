"""Shared output parsers for the eval harness — JSON + prose dual-format.

Council Option B (2026-05-30): explicit `output_format` flag (json|prose|auto),
`auto` default = JSON-first, prose fallback (backward compatible). Reasoning v1.2
emits dimensional PROSE ("WPCS Compliance: score 9/10 — ...") or JSON (CtF); the
Phase-4 harness was JSON-only. This module is the single source of truth consumed
by eval_judge (model output + teacher-target GT) and eval_gen (code extraction).

Dimension reconciliation reads from eval/dim_map.json (Option 3): 6 clean-mapped
dims for per-dim Spearman; Code Quality + Dependency Integrity excluded; I18n/D8
absent from prose. Canonical GT = rubric_scorer (not this module).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_DIM_MAP_PATH = Path(__file__).resolve().parent / "dim_map.json"


def load_dim_map() -> dict:
    with open(_DIM_MAP_PATH) as f:
        return json.load(f)


_DIM_MAP = load_dim_map()
# prose label -> eval internal key (6 clean dims)
PROSE_LABEL_TO_DIM = {k: v for k, v in _DIM_MAP["clean_mapped_dims"].items()
                      if not k.startswith("_")}

# Accept BOTH the *_score field names the eval harness historically expected AND
# the SHORT names the served v1.2 <judge_output> block actually emits
# ("security", "performance", "i18n", "accessibility"). code_quality /
# dependency_integrity stay UNMAPPED (no clean eval equivalent per dim_map.json).
_JSON_FIELD_TO_DIM = {
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

# Canonical Section-D weights for deriving overall when the judge omits it
# (dim_map.json single source — same table eval.rubric_definitions exposes).
_DIM_WEIGHTS = _DIM_MAP["dimension_weights"]
_PASS_THRESHOLD = 70.0  # PASS iff overall >= 70 (04.3 VERDICT-POLICY)


def _derive_overall_0_100(dim_scores_0_10: dict, verdict: object) -> Optional[float]:
    """Weighted-mean overall (0-100) from mapped 0-10 dim scores; weights
    renormalized over present dims. FAIL verdict caps below the PASS threshold.
    Returns None if no mapped dims. (See eval_judge._derive_overall_from_dims —
    kept in sync; this module stores dim_scores on the 0-10 raw scale.)"""
    present = {d: s for d, s in dim_scores_0_10.items() if d in _DIM_WEIGHTS}
    if not present:
        return None
    total_w = sum(_DIM_WEIGHTS[d] for d in present)
    overall = (sum(present[d] * _DIM_WEIGHTS[d] for d in present) / total_w) * 10.0
    if isinstance(verdict, str) and verdict.strip().upper() == "FAIL":
        overall = min(overall, _PASS_THRESHOLD - 1.0)
    return max(0.0, min(100.0, overall))

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# "WPCS Compliance: score 9/10 — ..." / "Security: score None/10 - ..."
_PROSE_SCORE_RE = re.compile(
    r"^([A-Z][A-Za-z0-9 /&-]+?):\s*score\s+(\d+|None)\s*/\s*10",
    re.MULTILINE,
)


def strip_think(text: str) -> str:
    """Remove Qwen3 `<think>...</think>` blocks (often empty)."""
    return _THINK_RE.sub("", text or "").strip()


def _parse_json_scores(text: str) -> Optional[dict]:
    """JSON judge output -> {internal_dim: score0_100}. None if unparseable."""
    s = strip_think(text)
    obj = None
    # raw, then ```json, then ```, then embedded {...}
    for attempt in (s,):
        try:
            obj = json.loads(attempt)
            break
        except json.JSONDecodeError:
            pass
    if obj is None:
        # <judge_output>...</judge_output> FIRST: its tags bound the JSON exactly,
        # so it beats the greedy (\{.*\}) which the [REASONING] prose (quoting code
        # with literal braces) would otherwise poison.
        for pat in (
            r"<judge_output>\s*(\{.*\})\s*</judge_output>",
            r"```json\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r"(\{.*\})",
        ):
            m = re.search(pat, s, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    break
                except json.JSONDecodeError:
                    continue
    if not isinstance(obj, dict):
        return None
    scores = {}
    for field, dim in _JSON_FIELD_TO_DIM.items():
        val = obj.get(field)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            scores[dim] = float(val)
    overall = obj.get("overall_score")
    out = {"dimension_scores": scores}
    if isinstance(overall, (int, float)) and not isinstance(overall, bool):
        out["overall"] = float(overall)
    else:
        # Bimodal judge omits overall_score — derive from emitted dims so the RL
        # reward path doesn't fall back to group-mean imputation (directional bias).
        derived = _derive_overall_0_100(scores, obj.get("verdict"))
        if derived is not None:
            out["overall"] = derived
            out["_overall_derived"] = True
    return out if scores or "overall" in out else None


def _parse_prose_scores(text: str) -> Optional[dict]:
    """Prose dimensional judge output -> {internal_dim: score0_100}.

    Reads "<Label>: score N/10" lines; maps the 6 clean dims via dim_map;
    'None' scores skipped (dimension-not-applicable). Scores are 0-10 in prose;
    scaled to 0-100 to match eval_judge's 0-100 convention. No overall in prose
    by default; caller uses rubric_scorer overall as canonical.
    """
    s = strip_think(text)
    scores = {}
    matched_labels = []
    for m in _PROSE_SCORE_RE.finditer(s):
        label, val = m.group(1).strip(), m.group(2)
        matched_labels.append(label)
        if val == "None":
            continue
        dim = PROSE_LABEL_TO_DIM.get(label)
        if dim is not None:
            scores[dim] = float(val) * 10.0   # 0-10 -> 0-100
    if not matched_labels:
        return None
    return {"dimension_scores": scores, "_prose_labels_seen": matched_labels}


def parse_judge_scores(text: str, output_format: str = "auto") -> Optional[dict]:
    """Parse judge output into {dimension_scores, [overall], _format}.

    output_format: 'json' | 'prose' | 'auto' (JSON-first, prose fallback).
    Returns None if nothing parseable. The `_format` field records which path
    succeeded (provenance — no silent ambiguity).
    """
    if output_format not in ("json", "prose", "auto"):
        raise ValueError(f"output_format must be json|prose|auto, got {output_format!r}")

    if output_format == "json":
        r = _parse_json_scores(text)
        if r is not None:
            r["_format"] = "json"
        return r
    if output_format == "prose":
        r = _parse_prose_scores(text)
        if r is not None:
            r["_format"] = "prose"
        return r
    # auto: JSON first, prose fallback
    r = _parse_json_scores(text)
    if r is not None:
        r["_format"] = "json"
        return r
    r = _parse_prose_scores(text)
    if r is not None:
        r["_format"] = "prose"
        return r
    return None


def extract_php_code(text: str) -> str:
    """Extract PHP code from (possibly reasoning-wrapped) model output.

    Strips `<think>` first (Qwen3 artifact — would otherwise poison php -l when
    prefixed before `<?php`). Prefers a ```php / ``` fenced block; else returns
    the think-stripped remainder (caller prepends `<?php` if missing).
    """
    s = strip_think(text)
    m = re.search(r"```(?:php)?\s*\n?(.*?)```", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s

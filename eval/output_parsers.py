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

# Also accept the JSON field names eval_judge already knows (model JSON output).
_JSON_FIELD_TO_DIM = {
    "wpcs_compliance": "D1_wpcs", "security_score": "D2_security",
    "sql_safety": "D3_sql", "performance_score": "D4_perf",
    "wp_api_usage": "D5_wp_api", "i18n_score": "D6_i18n",
    "accessibility_score": "D7_a11y", "error_handling": "D8_errors",
    "code_structure": "D9_structure",
}

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
        for pat in (r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```", r"(\{.*\})"):
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
        if isinstance(obj.get(field), (int, float)):
            scores[dim] = float(obj[field])
    overall = obj.get("overall_score")
    out = {"dimension_scores": scores}
    if isinstance(overall, (int, float)):
        out["overall"] = float(overall)
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

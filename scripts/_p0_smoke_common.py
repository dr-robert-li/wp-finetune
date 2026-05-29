"""Shared smoke-gate classifiers for Phase 4.4 W0-03 (dual-stage).

GPU-free pure logic — unit-testable on synthetic strings. Imported by both
Stage 1 (CPU degenerate-only pre-flight) and Stage 2 (vLLM full smoke).

Spec basis: council binding update 2026-05-29 (prose-aware coherence; JSON
parse invalid because v1.2 reasoning targets are inline dimensional prose).

Checks:
  is_degenerate(out)            -> (bool, reason)   loop / length-cap / 4.3 fingerprints
  judge_coherent_prose(out)     -> (bool, detail)   >=5 dims + >=5 "score X/10 — text"
  baseline_similarity(a, b)     -> float            SequenceMatcher token ratio
  explanation_richness(out)     -> float            mean chars per scored dimension
  inter_prompt_distinctness(outs) -> float          1 - mean pairwise similarity
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# 9 canonical rubric dimensions (metadata.dimensions_addressed).
DIMENSIONS = [
    "WPCS Compliance", "SQL Safety", "Security", "Performance", "WP API Usage",
    "Code Quality", "Dependency Integrity", "I18n", "Accessibility",
]

# "score 9/10 — explanation" / "score None/10 — n/a". Em dash or hyphen.
SCORE_LINE_RE = re.compile(
    r"score\s+(\d+|None)\s*/\s*10\s*[—–-]\s*\S", re.IGNORECASE
)
SCORE_ANY_RE = re.compile(r"score\s+(\d+|None)\s*/\s*10", re.IGNORECASE)

# Known 4.3 4-bit-collapse fingerprints (4.3-01-SUMMARY).
FINGERPRINT_ADMIN = "## Admin"
FINGERPRINT_PARAM = "@param"


def is_degenerate(out: str, max_new_tokens: int = 512) -> tuple[bool, str]:
    """Detect collapse: token-loop, length-cap saturation, or 4.3 fingerprints."""
    s = out.strip()
    if len(s) < 16:
        return True, f"too_short ({len(s)} chars) — likely empty/EOS-only collapse"

    # 4.3 literal fingerprints.
    if s.count(FINGERPRINT_ADMIN) >= 50:
        return True, f"fingerprint:'## Admin' x{s.count(FINGERPRINT_ADMIN)} (4.3 collapse)"
    if s.count(FINGERPRINT_PARAM) >= 30:
        return True, f"fingerprint:'@param' x{s.count(FINGERPRINT_PARAM)} (4.3 docblock-loop)"

    # n-gram loop: any 3..10-gram repeating >= 20 times.
    words = s.split()
    if len(words) >= 30:
        for n in range(3, 11):
            grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
            if not grams:
                continue
            from collections import Counter
            top, cnt = Counter(grams).most_common(1)[0]
            if cnt >= 20:
                return True, f"loop:{n}-gram {top!r} x{cnt}"

    # Length-cap saturation: output crammed to ~budget without EOS is suspect.
    # Heuristic on word count (proxy for tokens); >0.95 of a 512-token budget
    # in words is a strong saturation signal.
    if len(words) >= int(0.95 * max_new_tokens):
        # only flag if also low lexical diversity (loops fill the budget)
        diversity = len(set(words)) / max(len(words), 1)
        if diversity < 0.35:
            return True, f"length_cap:{len(words)}w diversity={diversity:.2f} (saturation)"

    return False, "ok"


def judge_coherent_prose(out: str) -> tuple[bool, str]:
    """Prose dimensional reasoning: >=5 dims named AND >=5 'score X/10 — text' lines.

    Requires the explanation tail (—/- + non-space) so a bare 'score 9/10'
    score-spam string does NOT pass (council strictness). 'None/10' allowed.
    """
    s = out
    dim_hits = sum(1 for d in DIMENSIONS if d.lower() in s.lower())
    explained = len(SCORE_LINE_RE.findall(s))
    total_scores = len(SCORE_ANY_RE.findall(s))
    ok = dim_hits >= 5 and explained >= 5
    detail = f"dims={dim_hits}/9 explained_scores={explained} total_scores={total_scores}"
    if not ok:
        if total_scores >= 5 and explained < 5:
            detail += " — score-spam? (scores present but no explanation tails)"
    return ok, detail


def baseline_similarity(out: str, baseline: str) -> float:
    """Token-level SequenceMatcher ratio (whitespace-split)."""
    return SequenceMatcher(None, out.split(), baseline.split()).ratio()


def explanation_richness(out: str) -> float:
    """Mean chars per scored dimension — proxy for substantive (non-boilerplate) reasoning."""
    scores = len(SCORE_ANY_RE.findall(out))
    if scores == 0:
        return 0.0
    return len(out) / scores


def inter_prompt_distinctness(outs: list[str]) -> float:
    """1 - mean pairwise SequenceMatcher ratio across outputs.

    Near 0 => outputs are near-identical (boilerplate / input-insensitive
    mode-collapse). Near 1 => outputs vary with input (conditioned reasoning).
    """
    n = len(outs)
    if n < 2:
        return 1.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(SequenceMatcher(None, outs[i].split(), outs[j].split()).ratio())
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return 1.0 - mean_sim

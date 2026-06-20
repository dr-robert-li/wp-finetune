"""Claude score-reasoning consistency scorer (D-09-05 judge-reward half).

Async, cached, timeout-imputing batch dispatcher over the project's
subprocess Claude path (scripts.claude_agent). Produces a float in [0,1]
per sample representing how consistently the written critique justifies
the numeric score.

Design constraints:
  - Uses scripts.claude_agent.generate_json (subprocess, NOT the direct LLM API)
  - No background Agent dispatch — that is orchestrator-only (SKILL layer)
  - Content-hash cache keyed on (php_code[:512], critique_text[:512])
  - 120s per-sample timeout via asyncio.wait_for
  - Timeout/error slots imputed from the group mean of valid scores
  - N-vote median for noise suppression (default n_votes=1; callers in 09-04/05 raise it)

Exports:
  score_judge_consistency(php_code, critique_text, model, n_votes) -> float | None
  score_with_cache(php_code, critique_text, model) -> float | None
  score_judge_consistency_batch(samples) -> list[float]   [async]
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import statistics
import warnings
from typing import Optional

from scripts.claude_agent import generate_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric system prompt (D-09-05 — reduces hackability via structured rubric)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a consistency judge. Your task is to score whether a written critique \
of WordPress PHP code is coherent, accurate, and well-supported by the code itself.

Rate the consistency between the critique text and the code on a scale from 0.0 to 1.0:

Rubric:
  1.0 — Critique is completely accurate: every claim is supported by the code,
         every identified issue is real, no real issues are missed, and the
         severity assessments are appropriate.
  0.8 — Critique is mostly accurate with at most one minor unsupported claim
         or one minor missed issue.
  0.6 — Critique has some unsupported claims OR misses real issues present in the
         code, but the overall direction is correct.
  0.4 — Critique contradicts the code in important ways OR fabricates issues not
         present, OR misses critical real issues.
  0.2 — Critique is mostly inaccurate or irrelevant to the code shown.
  0.0 — Critique is completely wrong, contradicts the code on every point, or
         is entirely fabricated.

Penalize especially:
  - Claims about issues that do not exist in the provided code
  - Contradictions where the critique says X is wrong but the code does X correctly
  - Missed critical security or correctness issues visible in the code
  - Numeric scores that are inconsistent with the textual assessment

Respond with ONLY valid JSON, no prose, no markdown fences:
{"consistency_score": <float between 0.0 and 1.0>}
"""

# ---------------------------------------------------------------------------
# Content-hash cache
# ---------------------------------------------------------------------------

def _cache_key(php_code: str, critique_text: str) -> str:
    """SHA-256 hash over first 512 chars of each input (D-09-05 spec)."""
    raw = (php_code[:512] + critique_text[:512]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# Module-level session cache: key -> float score
_score_cache: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Consistency prompt builder
# ---------------------------------------------------------------------------

def _build_consistency_prompt(php_code: str, critique_text: str) -> str:
    """Build the rubric prompt that asks Claude to rate critique consistency."""
    return (
        "Rate the consistency between the following PHP code and the critique "
        "written about it. Follow the rubric exactly.\n\n"
        "--- PHP CODE ---\n"
        "```php\n"
        f"{php_code}\n"
        "```\n\n"
        "--- CRITIQUE TEXT ---\n"
        f"{critique_text}\n\n"
        "--- INSTRUCTION ---\n"
        "Return ONLY: {\"consistency_score\": <float 0.0–1.0>}"
    )


# ---------------------------------------------------------------------------
# Single-sample scorer (blocking, subprocess)
# ---------------------------------------------------------------------------

def score_judge_consistency(
    php_code: str,
    critique_text: str,
    model: str = "sonnet",
    n_votes: int = 1,
) -> Optional[float]:
    """Score consistency between a critique and the PHP code it describes.

    Calls the Claude subprocess via generate_json (NOT the Anthropic API).
    Supports N-vote median for noise suppression (D-09-05 guard 2).

    Args:
        php_code: The PHP source code under critique.
        critique_text: The written critique to evaluate.
        model: Claude model name passed to generate_json ('sonnet', etc.).
        n_votes: Number of independent calls; returns median. Default 1.

    Returns:
        Float in [0.0, 1.0], or None if all calls fail to produce parseable JSON.
    """
    prompt = _build_consistency_prompt(php_code, critique_text)

    scores: list[float] = []
    for _ in range(max(1, n_votes)):
        result = generate_json(prompt, system=JUDGE_SYSTEM, model=model)
        if result is None:
            continue
        raw = result.get("consistency_score")
        if raw is None:
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            logger.warning("consistency_score not numeric: %r", raw)
            continue
        # Clamp to [0,1] to prevent out-of-range values propagating to advantage
        score = max(0.0, min(1.0, score))
        scores.append(score)

    if not scores:
        return None

    if len(scores) == 1:
        return scores[0]

    return statistics.median(scores)


# ---------------------------------------------------------------------------
# Cached single-sample scorer
# ---------------------------------------------------------------------------

def score_with_cache(
    php_code: str,
    critique_text: str,
    model: str = "sonnet",
    n_votes: int = 1,
) -> Optional[float]:
    """score_judge_consistency with content-hash caching.

    Returns the cached score if (php_code, critique_text) was already scored
    this session. Does NOT cache None results (so failed calls are retried).

    Args:
        php_code: PHP source code under critique.
        critique_text: Written critique to evaluate.
        model: Claude model name.
        n_votes: Median vote count (default 1).

    Returns:
        Float in [0.0, 1.0], or None on parse failure.
    """
    key = _cache_key(php_code, critique_text)
    if key in _score_cache:
        logger.debug("Cache hit for key %s…", key[:8])
        return _score_cache[key]

    score = score_judge_consistency(php_code, critique_text, model=model, n_votes=n_votes)
    if score is not None:
        _score_cache[key] = score
    return score


# ---------------------------------------------------------------------------
# Async batch dispatch (Task 2)
# ---------------------------------------------------------------------------

_BATCH_TIMEOUT_S = 120.0   # per D-09-05 R3
_NEUTRAL_FALLBACK = 0.5    # fallback when all samples in batch fail
_IMPUTE_WARN_THRESHOLD = 0.10  # warn if >10% of batch is imputed


async def _score_one_async(
    php_code: str,
    critique_text: str,
    model: str,
    n_votes: int,
) -> Optional[float]:
    """Async wrapper: cache-hit resolves immediately; miss runs scorer in thread."""
    key = _cache_key(php_code, critique_text)
    if key in _score_cache:
        return _score_cache[key]

    # score_judge_consistency is blocking (subprocess) — run in thread pool
    loop = asyncio.get_running_loop()
    score = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: score_judge_consistency(php_code, critique_text, model=model, n_votes=n_votes),
        ),
        timeout=_BATCH_TIMEOUT_S,
    )
    if score is not None:
        _score_cache[key] = score
    return score


async def score_judge_consistency_batch(
    samples: list[dict],
    model: str = "sonnet",
    n_votes: int = 1,
) -> list[float]:
    """Async batch Claude-consistency scorer with cache + 120s timeout + group-mean imputation.

    Dispatches all samples concurrently (asyncio.gather). Cache hits resolve
    without spawning a subprocess. Timeouts and errors are imputed from the
    group mean of valid scores. If all fail, falls back to 0.5 with a warning.

    Args:
        samples: List of dicts with "php_code" and "critique_text" keys.
        model: Claude model name (default "sonnet").
        n_votes: Median vote count per sample (default 1).

    Returns:
        List of floats (same length and order as input).
    """
    tasks = [
        _score_one_async(
            s["php_code"],
            s["critique_text"],
            model=model,
            n_votes=n_votes,
        )
        for s in samples
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect valid scores for imputation
    valid_scores: list[float] = []
    for r in raw_results:
        if isinstance(r, Exception):
            continue
        if r is None:
            continue
        valid_scores.append(r)

    # Compute group mean; fall back to neutral when all failed
    if valid_scores:
        group_mean = sum(valid_scores) / len(valid_scores)
    else:
        group_mean = _NEUTRAL_FALLBACK
        warnings.warn(
            "score_judge_consistency_batch: all samples failed/timed-out — "
            f"falling back to neutral {_NEUTRAL_FALLBACK}",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "score_judge_consistency_batch: all %d samples failed; using neutral %.2f",
            len(samples),
            _NEUTRAL_FALLBACK,
        )

    # Imputation pass
    n_imputed = 0
    final: list[float] = []
    for r in raw_results:
        if isinstance(r, Exception) or r is None:
            final.append(group_mean)
            n_imputed += 1
        else:
            final.append(r)

    # Warn if >10% imputed (mirrors reward_pipeline judge_imputed_from_group)
    if len(samples) > 0 and (n_imputed / len(samples)) > _IMPUTE_WARN_THRESHOLD:
        logger.warning(
            "score_judge_consistency_batch: %.0f%% of batch imputed (%d/%d)",
            100 * n_imputed / len(samples),
            n_imputed,
            len(samples),
        )

    return final

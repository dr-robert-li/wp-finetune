#!/usr/bin/env python3
"""Diagnostic probe: why warm-started RL rewards collapse to a uniform 0.0.

READ-ONLY. No Tinker, no GPU, no training. Reconstructs the RL reward path
offline (the frozen reward-judge is MONKEYPATCHED — no vLLM needed) and prints
every intermediate so the zero-source is visible per pathway.

Context: Phase 09 warm-started RL produced reward_mean=0.0 (min=max=0.0, all 32
samples) every step -> "All rewards are uniform. There will be no gradient".
See .planning/debug/09-rl-warmstart-zero-reward.md.

Two suspected mechanisms this probe checks:
  GEN  : compute_group_rewards composes MO-GRPO *within-group normalized* signals
         (_mo_grpo_norm). A consistent strong policy (v4 gen ~0.99) -> near-zero
         within-group variance -> every normalized signal ~0 -> composite ~0.
  JUDGE: the reward extracts a ```php fix and scores fix_correctness on it. The
         warm v4 judge emits critique PROSE (its SFT format), not a fenced fix
         -> extract yields non-PHP -> _is_parseable_php False -> fix_correctness=0.

Usage:
    REWARD_SKIP_PHPCS_ASSERT=1 .venv-tinker/bin/python scripts/_probe_rl_reward.py
    # or any python with the eval/ + scripts/ deps importable

Optionally point at captured real completions:
    --completions output/rl_checkpoints/judge_failures.preguard.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# phpcs assert is about the live security gate; this probe runs offline.
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

import numpy as np  # noqa: E402

from eval.output_parsers import extract_php_code  # noqa: E402
from eval.eval_judge import (  # noqa: E402
    parse_judge_response,
    _parse_prose_dim_scores,
    _derive_overall_from_dims,
)
from scripts.rl_rollouts import (  # noqa: E402
    _is_parseable_php,
    combine_judge_reward,
    _extract_corrected_php,
    _augment_judge_prompt,
)
from scripts import reward_pipeline  # noqa: E402
from scripts.reward_pipeline import (  # noqa: E402
    compute_group_rewards,
    _extract_verifiable_signals,
    _mo_grpo_norm,
)

# --------------------------------------------------------------------------- #
# Sample completions (bare-code, fenced, and prose — the three shapes a warm
# v4 policy actually emits). Real captured ones can be loaded via --completions.
# --------------------------------------------------------------------------- #
BARE_CODE_A = """function init() {
\tadd_filter( 'rest_api_allowed_post_types', array( $this, 'allow_rest_api_types' ) );
\tadd_filter( 'rest_prepare_post', array( $this, 'restore_global_state' ) );
}"""

BARE_CODE_B = """function __construct( $host, $port, $username, $password ) {
\t$this->host = $host;
\t$this->port = $port;
\t$this->init();
}"""

PROSE_JUDGE = (
    "WPCS Compliance: score 9/10 — Naming and structure follow WordPress "
    "conventions.\nSQL Safety: score 10/10 — No direct DB access.\n"
    "Security: score 8/10 — Inputs are sanitized via core APIs."
)

JSON_JUDGE = (
    "[REASONING] The function looks fine. [/REASONING]\n"
    "<judge_output>{\"wpcs_compliance\": 9, \"security_score\": 8, "
    "\"overall_score\": 88}</judge_output>"
)


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _load_real(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            txt = d.get("raw_text") or d.get("completion") or d.get("code") or ""
            if txt.strip():
                out.append(txt)
    return out


def probe_gen(group_completions: list[str], frozen_judge_score: float) -> None:
    """Run the GEN reward path on one prompt-group; print every intermediate."""
    _hr(f"GEN PATH  (group of {len(group_completions)}, frozen-judge mock={frozen_judge_score})")

    # Stage 1: extract + parseability guard (mirrors rl_rollouts gen path)
    gen_php = [extract_php_code(c) for c in group_completions]
    parseable = [_is_parseable_php(p) for p in gen_php]
    for i, (p, ok) in enumerate(zip(gen_php, parseable)):
        print(f"  [{i}] extract_len={len(p):>5} parseable_php={ok}")

    # Stage 2: monkeypatch the frozen reward-judge so no vLLM is needed.
    reward_pipeline.judge_score_single = lambda *a, **k: frozen_judge_score  # type: ignore
    results = compute_group_rewards(
        php_codes=gen_php,
        judge_client=object(),
        judge_model="probe-mock",
    )
    # Stage 3: apply the non-code guard exactly as rl_rollouts does
    for i, ok in enumerate(parseable):
        if not ok:
            results[i].scalar = 0.0

    phpcs_raws = [r.breakdown.phpcs_raw for r in results]
    print(f"\n  phpcs_raw (overall 0-100): {[round(x, 2) for x in phpcs_raws]}")
    print(f"  phpcs_raw variance        : {float(np.var(phpcs_raws)):.6f}")
    print(f"  _mo_grpo_norm(phpcs_raw)  : {[round(float(x), 4) for x in _mo_grpo_norm(np.array(phpcs_raws, float))]}")
    print("\n  per-sample composite / scalar:")
    for i, r in enumerate(results):
        b = r.breakdown
        print(f"   [{i}] phpcs_norm={b.phpcs_norm:+.4f} verpo_norm={b.verpo_norm:+.4f} "
              f"judge_norm={b.judge_norm:+.4f} -> composite={b.composite_pre_gate:+.4f} "
              f"sec_fail={b.security_fail} SCALAR={r.scalar:+.4f}")
    scalars = [r.scalar for r in results]
    print(f"\n  >>> GEN group scalars: min={min(scalars):.4f} max={max(scalars):.4f} "
          f"mean={float(np.mean(scalars)):.4f}")
    if max(scalars) - min(scalars) < 1e-9:
        print("  >>> VERDICT: uniform reward -> advantage centers to 0 -> NO GRADIENT.")
        if float(np.var(phpcs_raws)) < 1e-6:
            print("  >>> CAUSE: zero within-group variance (strong consistent policy) "
                  "-> _mo_grpo_norm collapses every signal to 0 -> composite 0.")


def probe_judge(samples: list[tuple[str, str]]) -> list[float]:
    """Run the JUDGE fix-correctness path (post-fix _extract_corrected_php)."""
    _hr("JUDGE PATH  (fix-correctness on _extract_corrected_php: <corrected_code> | ```php)")
    fixes = []
    for label, comp in samples:
        corrected = _extract_corrected_php(comp)
        ok = _is_parseable_php(corrected)
        fix = float(_extract_verifiable_signals(corrected).overall) / 100.0 if ok else 0.0
        fixes.append(fix)
        combined = combine_judge_reward(fix_correctness=fix, consistency=0.0)
        print(f"  {label:<24} extract_len={len(corrected):>5} parseable={ok!s:<5} "
              f"fix_correctness={fix:.3f} combined(cons=0)={combined:.3f}")
    print("  >>> Bare prose (no fix block) still -> 0; a critique + ```php/<corrected_code> "
          "block -> nonzero fix_correctness. The augmented judge prompt elicits the latter.")
    return fixes


def probe_frozen_judge_parse() -> "float | None":
    """Post-fix: does the reward path score v4 PROSE to a number (not None)?

    Mirrors judge_score_single: parse_judge_response stays pure (prose -> None),
    then the prose fallback (_parse_prose_dim_scores -> _derive_overall_from_dims)
    recovers a numeric overall_score.
    """
    _hr("FROZEN-JUDGE PARSE  (reward path: prose fallback vs <judge_output> JSON)")
    prose_score = None
    for label, text in [("PROSE (v4 SFT format)", PROSE_JUDGE), ("JSON <judge_output>", JSON_JUDGE)]:
        parsed = parse_judge_response(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("overall_score"), (int, float)):
            overall = float(parsed["overall_score"])
            via = "json"
        else:
            dims = _parse_prose_dim_scores(text)
            overall = _derive_overall_from_dims(dims) if dims else None
            via = f"prose-fallback(dims={list(dims)})" if dims else "None"
        if label.startswith("PROSE"):
            prose_score = overall
        print(f"  {label:<24} -> overall_score={overall} via={via}")
    print("  >>> PROSE must now yield a numeric overall (was None pre-fix).")
    return prose_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", default=None,
                    help="JSONL of captured real completions (raw_text). "
                         "Default: built-in v4-shape samples.")
    ap.add_argument("--frozen-judge-score", type=float, default=95.0,
                    help="Mock score the frozen reward-judge returns (isolates the "
                         "gen normalization mechanism from judge-parse effects).")
    args = ap.parse_args()

    if args.completions:
        real = _load_real(args.completions)
        print(f"Loaded {len(real)} real completions from {args.completions}")
        group = real[:4] if len(real) >= 2 else [BARE_CODE_A, BARE_CODE_B]
    else:
        # Two near-identical strong gen outputs (the zero-variance scenario)
        group = [BARE_CODE_A, BARE_CODE_A, BARE_CODE_B, BARE_CODE_B]

    gen_scalars = _probe_gen_return(group, args.frozen_judge_score)

    critique_then_fenced = (
        PROSE_JUDGE + "\n\nThen FIX the code:\n```php\n<?php\n" + BARE_CODE_A + "\n```"
    )
    critique_then_tagged = (
        PROSE_JUDGE + "\n\n<corrected_code>\n<?php\n" + BARE_CODE_B + "\n</corrected_code>"
    )
    fixes = probe_judge([
        ("prose_only (no fix)", PROSE_JUDGE),
        ("critique + ```php fix", critique_then_fenced),
        ("critique + <corrected_code>", critique_then_tagged),
    ])
    prose_score = probe_frozen_judge_parse()

    # Demo: the augmented judge prompt the rollout now sends.
    _hr("JUDGE PROMPT AUGMENTATION (Mechanism-1 contract)")
    demo = _augment_judge_prompt({"messages": [{"role": "user",
                                                "content": "<wp_judge> Evaluate this WordPress code:\n\n<code>"}]})
    print("  augmented user content tail:")
    print("   ..." + demo["messages"][0]["content"][-180:])

    # --------------------------------------------------------------------- #
    # HARD GATE (must all pass before relaunch)
    # --------------------------------------------------------------------- #
    _hr("HARD GATE")
    g1 = any(f > 0 for f in fixes)
    g2 = isinstance(prose_score, (int, float)) and prose_score is not None
    g3 = (max(gen_scalars) - min(gen_scalars)) > 1e-9 if gen_scalars else False
    print(f"  [{'PASS' if g1 else 'FAIL'}] (1) a judge rollout shape yields nonzero fix_correctness "
          f"(max fix={max(fixes):.3f})")
    print(f"  [{'PASS' if g2 else 'FAIL'}] (2) frozen-judge PROSE parses to a numeric score "
          f"(prose overall={prose_score})")
    print(f"  [{'PASS' if g3 else 'FAIL'}] (3) gen group reward is non-uniform "
          f"(min={min(gen_scalars):.4f} max={max(gen_scalars):.4f})")
    print("\n  NOTE: gate (1)/(3) on OFFLINE shapes proves the CODE parses/scores correctly. "
          "Whether the warm v4 policy actually EMITS a fix block under the augmented prompt "
          "is only verifiable LIVE — use a 50-100 step signal run (reward_min!=reward_max, "
          "reward_mean trending up) as the real-rollout gate.")
    all_pass = g1 and g2 and g3
    print(f"\n  >>> OFFLINE GATE: {'ALL PASS' if all_pass else 'NOT ALL PASS'}")
    print("\nDone. See .planning/debug/09-rl-warmstart-zero-reward.md for the writeup.")


def _probe_gen_return(group, frozen):
    """probe_gen but capturing the scalars for the gate."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pass
    # re-run inline to capture scalars (probe_gen prints; we recompute scalars)
    probe_gen(group, frozen)
    # recompute scalars deterministically for the gate
    reward_pipeline.judge_score_single = lambda *a, **k: frozen  # type: ignore
    gen_php = [extract_php_code(c) for c in group]
    res = compute_group_rewards(php_codes=gen_php, judge_client=object(), judge_model="probe-mock")
    for i, p in enumerate(gen_php):
        if not _is_parseable_php(p):
            res[i].scalar = 0.0
    return [r.scalar for r in res]


if __name__ == "__main__":
    main()

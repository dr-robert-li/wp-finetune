#!/usr/bin/env python
"""Discriminating check (advisor, 2026-06-25): did the policy learn, or is a second
bug suppressing weight updates?

The 50-step run read FLAT reward — but the reward metric was structurally blind to
learning (rollouts always drawn from the FROZEN warm-start sampler; rl_train.py:1451
stale-sampler bug). So the flat run is NOT evidence about learnability. This samples
the SAME gen prompts through three policies and scores each with the local $0 judge:

  - warm-start (v4 savestate, the RL init)
  - step-50 seedA checkpoint
  - step-50 seedB checkpoint

step-50 mean reward > warm-start  => the policy DID move toward reward (the stale
  sampler merely hid it from the live metric) => the per-step-refresh fix is
  sufficient => relaunch with confidence.
step-50 mean reward ~= warm-start => updates are not moving the policy toward reward
  => a second bug in the gradient path (RSPO floor / tiny advantages / lr) => do NOT
  relaunch; investigate first.

$0 on judges (local vLLM); Tinker sampling only (cheap vs training). Same prompts +
same sampling params for every policy => apples-to-apples.
"""
from __future__ import annotations

import argparse
import logging
import random
import statistics as stats
import sys
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("check50")

WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"
STEP50 = {
    "seedA": "tinker://a99724f2-36d3-577b-b51f-94af9198e7d8:train:0/sampler_weights/step-50",
    "seedB": "tinker://56c4f145-9d61-515d-ad20-8e254362c059:train:0/sampler_weights/step-50",
}


def _score_policy(name, sampler, prompts, args, renderer, tok):
    """Sample group_size completions per prompt, extract PHP, score via local judge.

    Mirrors the GEN reward path in collect_rollouts EXACTLY (extract_php_code ->
    compute_group_rewards -> zero non-parseable) so the number is the same reward
    the training loop optimises.
    """
    from scripts.rl_rollouts import _generate_completions, _is_parseable_php
    from scripts.reward_pipeline import compute_group_rewards
    from eval.output_parsers import extract_php_code

    comps = _generate_completions(sampler, prompts, args, renderer=renderer, tok=tok)
    php = [extract_php_code(c.completion) for c in comps]
    results = compute_group_rewards(php_codes=php, judge_client=args.judge_client, judge_model=args.judge_model)
    scalars = []
    parseable = 0
    for i, p in enumerate(php):
        if not _is_parseable_php(p):
            results[i].scalar = 0.0
        else:
            parseable += 1
        scalars.append(float(results[i].scalar))
    mean = stats.mean(scalars) if scalars else 0.0
    logger.info("[%s] n=%d parseable=%d mean_reward=%.4f median=%.4f max=%.4f",
                name, len(scalars), parseable, mean, stats.median(scalars) if scalars else 0.0,
                max(scalars) if scalars else 0.0)
    return {"name": name, "n": len(scalars), "parseable": parseable, "mean": mean, "scalars": scalars}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=15)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--judge-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge-model", default="wp_judge")
    cli = ap.parse_args()

    import tinker
    from scripts.rl_train import _build_judge_client
    from scripts.tinker_rl_data import load_rl_prompts
    from scripts.rl_rollouts import build_rl_renderer

    # Same fixed prompt subset for every policy (fairness).
    gen_pool = load_rl_prompts("gen")
    random.seed(cli.seed)
    prompts = random.sample(gen_pool, min(cli.n_prompts, len(gen_pool)))
    logger.info("Comparing on %d fixed gen prompts (group_size=%d temp=%.2f).",
                len(prompts), cli.group_size, cli.temperature)

    renderer, tok = build_rl_renderer()
    judge_client = _build_judge_client(cli.judge_base_url)
    args = SimpleNamespace(
        group_size=cli.group_size, temperature=cli.temperature, max_new_tokens=cli.max_new_tokens,
        judge_client=judge_client, judge_model=cli.judge_model,
    )

    sc = tinker.ServiceClient()
    logger.info("Loading warm-start sampler (via training client from savestate)...")
    tc0 = sc.create_training_client_from_state(WARMSTART)
    sampler_ws = tc0.save_weights_and_get_sampling_client()

    rows = [_score_policy("warm-start", sampler_ws, prompts, args, renderer, tok)]
    for seed_name, path in STEP50.items():
        logger.info("Loading step-50 %s sampler from %s ...", seed_name, path)
        sampler = sc.create_sampling_client(model_path=path)
        rows.append(_score_policy(f"step50-{seed_name}", sampler, prompts, args, renderer, tok))

    base = rows[0]["mean"]
    print("\n===== DISCRIMINATING CHECK: step-50 vs warm-start =====")
    print(f"{'policy':14} {'n':>4} {'parseable':>9} {'mean_reward':>12} {'Δ vs warm':>10}")
    for r in rows:
        d = r["mean"] - base
        print(f"{r['name']:14} {r['n']:>4} {r['parseable']:>9} {r['mean']:>12.4f} {d:>+10.4f}")
    best = max(rows[1:], key=lambda r: r["mean"])
    verdict = "LEARNED (step-50 > warm-start) -> sampler fix sufficient, RELAUNCH" if best["mean"] - base > 0.02 \
        else "FLAT (step-50 ~= warm-start) -> second bug in gradient path, DO NOT RELAUNCH"
    print(f"\nVERDICT: {verdict}")
    print(f"(best step-50 = {best['name']} mean={best['mean']:.4f} vs warm={base:.4f}, Δ={best['mean']-base:+.4f})")


if __name__ == "__main__":
    sys.exit(main())

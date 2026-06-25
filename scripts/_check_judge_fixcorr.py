#!/usr/bin/env python
"""Low-variance controlled eval: did RL actually improve JUDGE fix_correctness?

The live per-step fix_correctness_mean is useless as a learning signal — it samples
different random prompts at temp=1.0 each step, so ±0.15 step noise swamps a ~0.001/step
trend (you couldn't clear it until ~150 steps). The stale-sampler run (which CANNOT
learn) produced the same dip-then-recover U with the same peak as the fixed run. So the
live slope is the wrong instrument.

This removes BOTH noise sources — FIXED held-out judge prompts + LOW temperature — and
scores the deterministic fix_correctness (`_fix_score_from_completion`, the exact training
reward component) across three policies:

  - warm-start        (v4 savestate, the RL init / baseline)
  - fixed-step-50     (current per-step-refresh run; pass its tinker path via --fixed-50)
  - stale-step-50     (NEGATIVE CONTROL: the J.4 frozen-sampler run; cannot have learned)

Decision:
  fixed-50 > warm-start AND stale-50 ~= warm-start  => REAL learning, sampler fix worked
                                                       => add seed 2.
  fixed-50 ~= warm-start                            => sampler necessary-but-not-sufficient
                                                       => gradient-strength (adv x lr) issue;
                                                          bump LR / investigate advantages.
The negative control is load-bearing: if stale-50 ALSO beats warm-start, the metric/eval
is confounded and a fixed-50 win means nothing.
"""
from __future__ import annotations

import argparse
import logging
import statistics as stats
import sys
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("checkJ")

WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"
STALE_50 = "tinker://a99724f2-36d3-577b-b51f-94af9198e7d8:train:0/sampler_weights/step-50"  # J.4 frozen-sampler seedA


def _score(name, sampler, prompts, args, renderer, tok):
    from scripts.rl_rollouts import _generate_completions, _fix_score_from_completion
    comps = _generate_completions(sampler, prompts, args, renderer=renderer, tok=tok,
                                  max_tokens_override=args.judge_max_new_tokens)
    scores = [_fix_score_from_completion(c.completion) for c in comps]
    mean = stats.mean(scores) if scores else 0.0
    # tier histogram: 0.0 / 0.25 / >0.25 (parseable-fix, scaled rubric)
    z = sum(1 for s in scores if s == 0.0); q = sum(1 for s in scores if s == 0.25)
    hi = sum(1 for s in scores if s > 0.25)
    logger.info("[%s] n=%d mean_fixcorr=%.4f median=%.4f | tiers 0.0=%d 0.25=%d hi=%d",
                name, len(scores), mean, stats.median(scores) if scores else 0.0, z, q, hi)
    return {"name": name, "n": len(scores), "mean": mean, "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-50", required=True, help="tinker:// path of the per-step-refresh step-50 checkpoint")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=12345)
    cli = ap.parse_args()

    import tinker
    from scripts.tinker_rl_data import load_rl_prompts
    from scripts.rl_rollouts import build_rl_renderer, _augment_judge_prompt, JUDGE_MAX_NEW_TOKENS
    import random

    judge_pool = load_rl_prompts("judge")
    random.seed(cli.seed)
    prompts = random.sample(judge_pool, min(cli.n_prompts, len(judge_pool)))
    for gid, item in enumerate(prompts):
        item["_group_id"] = f"judge-{gid}"
        _augment_judge_prompt(item)  # match training: require a corrected-code block
    logger.info("Controlled judge-axis eval: %d fixed prompts, group_size=%d temp=%.2f",
                len(prompts), cli.group_size, cli.temperature)

    renderer, tok = build_rl_renderer()
    args = SimpleNamespace(group_size=cli.group_size, temperature=cli.temperature,
                           max_new_tokens=JUDGE_MAX_NEW_TOKENS, judge_max_new_tokens=JUDGE_MAX_NEW_TOKENS)

    sc = tinker.ServiceClient()
    logger.info("warm-start sampler...")
    tc0 = sc.create_training_client_from_state(WARMSTART)
    s_ws = tc0.save_weights_and_get_sampling_client()
    logger.info("fixed-step-50 sampler... %s", cli.fixed_50)
    s_fixed = sc.create_sampling_client(model_path=cli.fixed_50)
    logger.info("stale-step-50 (negative control) sampler... %s", STALE_50)
    s_stale = sc.create_sampling_client(model_path=STALE_50)

    rows = [
        _score("warm-start", s_ws, prompts, args, renderer, tok),
        _score("fixed-step50", s_fixed, prompts, args, renderer, tok),
        _score("stale-step50(ctrl)", s_stale, prompts, args, renderer, tok),
    ]
    base = rows[0]["mean"]
    print("\n===== CONTROLLED JUDGE-AXIS fix_correctness (fixed prompts, temp %.2f) =====" % cli.temperature)
    print(f"{'policy':20} {'n':>4} {'mean_fixcorr':>12} {'Δ vs warm':>10}")
    for r in rows:
        print(f"{r['name']:20} {r['n']:>4} {r['mean']:>12.4f} {r['mean']-base:>+10.4f}")
    fixed_d = rows[1]["mean"] - base
    stale_d = rows[2]["mean"] - base
    if fixed_d > 0.03 and abs(stale_d) <= 0.03:
        verdict = "REAL LEARNING (fixed>warm, stale~=warm) -> sampler fix worked -> ADD SEED 2"
    elif fixed_d > 0.03 and stale_d > 0.03:
        verdict = "CONFOUNDED (stale ALSO > warm) -> eval/metric artifact, fixed win is meaningless -> investigate eval"
    else:
        verdict = "NULL (fixed ~= warm) -> sampler necessary-but-not-sufficient -> gradient strength (adv x lr) -> bump LR / investigate"
    print(f"\nVERDICT: {verdict}")
    print(f"(fixed Δ={fixed_d:+.4f}, stale-ctrl Δ={stale_d:+.4f})")


if __name__ == "__main__":
    sys.exit(main())

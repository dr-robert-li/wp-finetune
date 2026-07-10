#!/usr/bin/env python
"""Read the ACTUAL corrected-code blocks the policies emit (advisor: verify the model
is hacking, not just that the reward CAN be hacked).

Dumps _extract_corrected_php(completion) + fix_score for the same fixed augmented judge
prompts under warm-start, fixed-step-50, stale-step-50. Discriminates:
  (a) HACK: trivial echoes / unrelated snippets (<?php echo 'hi';) scoring 1.0
  (b) FORMAT-LEARNING: real WP fix attempts (the augmented prompt requires a code block;
      warm-start defaults to prose -> 0.25, RL learned to comply with real-ish fixes)
fix_score cannot separate these; reading the blocks can.
"""
from __future__ import annotations
import argparse, logging, sys
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"
STALE_50 = "tinker://a99724f2-36d3-577b-b51f-94af9198e7d8:train:0/sampler_weights/step-50"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-50", required=True)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--judge-max-new-tokens", type=int, default=None,
                    help="Override JUDGE_MAX_NEW_TOKENS for the truncation probe (default: module constant).")
    cli = ap.parse_args()

    import tinker, random
    from scripts.tinker_rl_data import load_rl_prompts
    from scripts.rl_rollouts import (build_rl_renderer, _augment_judge_prompt, _generate_completions,
                                     _extract_corrected_php, _fix_score_from_completion, JUDGE_MAX_NEW_TOKENS)
    JMNT = cli.judge_max_new_tokens or JUDGE_MAX_NEW_TOKENS
    print(f"[probe] judge_max_new_tokens = {JMNT} (module default {JUDGE_MAX_NEW_TOKENS})")

    judge_pool = load_rl_prompts("judge")
    random.seed(cli.seed)
    prompts = random.sample(judge_pool, min(cli.n_prompts, len(judge_pool)))
    for gid, item in enumerate(prompts):
        item["_group_id"] = f"judge-{gid}"
        _augment_judge_prompt(item)

    renderer, tok = build_rl_renderer()
    args = SimpleNamespace(group_size=1, temperature=cli.temperature,
                           max_new_tokens=JMNT, judge_max_new_tokens=JMNT)
    sc = tinker.ServiceClient()
    tc0 = sc.create_training_client_from_state(WARMSTART)
    pols = [
        ("warm-start", tc0.save_weights_and_get_sampling_client()),
        ("fixed-step50", sc.create_sampling_client(model_path=cli.fixed_50)),
        ("stale-step50", sc.create_sampling_client(model_path=STALE_50)),
    ]
    for name, sampler in pols:
        comps = _generate_completions(sampler, prompts, args, renderer=renderer, tok=tok,
                                      max_tokens_override=JMNT)
        print(f"\n################ {name} ################")
        n_block = 0
        scores = []
        for i, c in enumerate(comps):
            corrected = _extract_corrected_php(c.completion) or ""
            score = _fix_score_from_completion(c.completion)
            scores.append(score)
            if corrected.strip():
                n_block += 1
            print(f"\n--- {name} prompt#{i} fix_score={score:.3f} corrected_len={len(corrected)} ---")
            print((corrected[:600] if corrected.strip() else "[NO CORRECTED BLOCK] completion-head: " + c.completion[:300]))
        n = len(comps)
        emit_pct = 100.0 * n_block / n if n else 0.0
        mean_score = sum(scores) / n if n else 0.0
        print(f"\n@@@@ SUMMARY {name} budget={JMNT}: code_emission={n_block}/{n} ({emit_pct:.1f}%), mean_fix_score={mean_score:.3f}")


if __name__ == "__main__":
    sys.exit(main())

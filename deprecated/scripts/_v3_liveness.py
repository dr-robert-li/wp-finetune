#!/usr/bin/env python
"""V3 gen-axis liveness (robust): prove FIX 2 (_is_valid_wp_php) credits real warm-start
gen completions that the old _is_parseable_php zeroed — without the brittle 3-policy
harness that hangs on a single pathological prompt's blocking sample().result().

Samples warm-start one prompt at a time with a per-prompt timeout (skips hangs), then
scores EACH completion two ways (old gate vs new gate) and via the local $0 judge.

Gate: new-parseable > old-parseable (FIX 2 additive, live) AND mean_reward(new) > 0.
"""
import logging, sys, argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("v3live")

WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--per-prompt-timeout", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--judge-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--judge-model", default="wp_judge")
    cli = ap.parse_args()

    import tinker, random
    from scripts.tinker_rl_data import load_rl_prompts
    from scripts.rl_rollouts import (build_rl_renderer, _build_sampling_params,
                                     _prompt_user_messages, _seq_tokens,
                                     _is_parseable_php, _is_valid_wp_php)
    from scripts.reward_pipeline import compute_group_rewards
    from scripts.rl_train import _build_judge_client
    from eval.output_parsers import extract_php_code

    gen_pool = load_rl_prompts("gen")
    random.seed(cli.seed)
    prompts = random.sample(gen_pool, min(cli.n_prompts, len(gen_pool)))
    renderer, tok = build_rl_renderer()
    args = SimpleNamespace(group_size=cli.group_size, temperature=cli.temperature,
                           max_new_tokens=cli.max_new_tokens)
    sp = _build_sampling_params(args, renderer)

    sc = tinker.ServiceClient()
    log.info("warm-start sampler...")
    tc0 = sc.create_training_client_from_state(WARMSTART)
    sampler = tc0.save_weights_and_get_sampling_client()

    php_codes, skipped = [], 0
    for idx, item in enumerate(prompts):
        user_msgs = _prompt_user_messages(item)
        prompt = renderer.build_generation_prompt(user_msgs)

        def _do():
            resp = sampler.sample(prompt=prompt, num_samples=cli.group_size, sampling_params=sp)
            r = resp.result() if hasattr(resp, "result") else resp
            seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
            # mirror live _generate_completions fix: strip the leaked chat EOS marker
            return [tok.decode(_seq_tokens(s), skip_special_tokens=True) for s in seqs]

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                texts = ex.submit(_do).result(timeout=cli.per_prompt_timeout)
        except FTimeout:
            skipped += 1
            log.warning("prompt %d TIMED OUT after %.0fs -> skip (the hanging-prompt class)",
                        idx, cli.per_prompt_timeout)
            continue
        except Exception as e:
            skipped += 1
            log.warning("prompt %d errored: %s -> skip", idx, e)
            continue
        for t in texts:
            php_codes.append(extract_php_code(t))
        log.info("prompt %d -> %d completions (running total %d)", idx, len(texts), len(php_codes))

    log.info("sampled %d completions, skipped %d prompts", len(php_codes), skipped)
    if not php_codes:
        print("VERDICT: NO COMPLETIONS (all prompts hung) -> cannot assess; investigate Tinker")
        return 1

    old_ok = sum(1 for p in php_codes if _is_parseable_php(p))
    new_ok = sum(1 for p in php_codes if _is_valid_wp_php(p))
    template_credited = [p for p in php_codes if _is_valid_wp_php(p) and not _is_parseable_php(p)]

    judge_client = _build_judge_client(cli.judge_base_url)
    results = compute_group_rewards(php_codes=php_codes, judge_client=judge_client, judge_model=cli.judge_model)
    new_scalars = [float(results[i].scalar) if _is_valid_wp_php(p) else 0.0 for i, p in enumerate(php_codes)]
    old_scalars = [float(results[i].scalar) if _is_parseable_php(p) else 0.0 for i, p in enumerate(php_codes)]
    mean_new = sum(new_scalars) / len(new_scalars)
    mean_old = sum(old_scalars) / len(old_scalars)

    print("\n===== V3 GEN-AXIS LIVENESS (warm-start, robust per-prompt) =====")
    print(f"completions scored : {len(php_codes)}  (skipped {skipped} hanging prompts)")
    print(f"OLD gate _is_parseable_php : parseable={old_ok:>3}  mean_reward={mean_old:.4f}")
    print(f"NEW gate _is_valid_wp_php  : parseable={new_ok:>3}  mean_reward={mean_new:.4f}")
    print(f"template-only credited by FIX 2 (new-yes, old-no): {len(template_credited)}")
    p1 = new_ok > 0
    p2 = mean_new > 0.0
    p3 = new_ok >= old_ok
    print(f"\nGATE warm parseable(new) > 0      : {'PASS' if p1 else 'FAIL'} ({new_ok})")
    print(f"GATE >=1 nonzero gen mean_reward  : {'PASS' if p2 else 'FAIL'} ({mean_new:.4f})")
    print(f"GATE FIX2 additive (new >= old)   : {'PASS' if p3 else 'FAIL'} ({new_ok} >= {old_ok})")
    print(f"\nVERDICT: {'V3 PASS' if (p1 and p2 and p3) else 'V3 FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

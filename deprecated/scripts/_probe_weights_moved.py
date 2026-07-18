#!/usr/bin/env python
"""Decisive, artifact-free probes before any relaunch (advisor 2026-06-25):

(1) DID THE WEIGHTS MOVE AT ALL?  compute_logprobs of one FIXED token sequence under
    the warm-start sampler vs the step-50 sampler. No generation, no judge, no
    sampling noise. Identical => optim_step is a no-op (LoRA/optimizer not applying;
    a bug the sampler fix won't touch). Different => weights moved, only direction is open.

(2) WHAT ARE THE COMPLETIONS REALLY?  the gen-axis check read parseable=0 across 60/60.
    Before concluding "model emits prose", READ the raw completions. The warm-start is
    the v4 REASONING model — with max_new_tokens=512 + thinking-on it can burn the whole
    budget inside an unclosed <think> and emit zero code (a CONFIG artifact, not
    "can't do gen"). Dump raw text + think-block state at 512 and at a generous budget.
"""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("probe")

WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"
STEP50_A = "tinker://a99724f2-36d3-577b-b51f-94af9198e7d8:train:0/sampler_weights/step-50"


def main():
    import tinker
    from scripts.tinker_rl_data import load_rl_prompts
    from scripts.rl_rollouts import build_rl_renderer, _generate_completions, _prompt_user_messages

    renderer, tok = build_rl_renderer()
    gen_pool = load_rl_prompts("gen")
    judge_pool = load_rl_prompts("judge")

    sc = tinker.ServiceClient()
    logger.info("Loading warm-start sampler (training client from savestate)...")
    tc0 = sc.create_training_client_from_state(WARMSTART)
    s_ws = tc0.save_weights_and_get_sampling_client()
    logger.info("Loading step-50 seedA sampler...")
    s_50 = sc.create_sampling_client(model_path=STEP50_A)

    # ---- PROBE 1: weights moved? (teacher-forced logprobs of a fixed rendered prompt) ----
    # Render one fixed gen prompt + one fixed judge prompt to ModelInputs and compare
    # per-token logprob vectors under the two policies.
    print("\n===== PROBE 1: DID WEIGHTS MOVE? (compute_logprobs, no generation) =====")
    for label, pool in (("gen", gen_pool), ("judge", judge_pool)):
        item = pool[0]
        mi = renderer.build_generation_prompt(_prompt_user_messages(item))

        def _logprobs(sampler):
            r = sampler.compute_logprobs(mi)
            return r.result() if hasattr(r, "result") else r

        lp_ws = _logprobs(s_ws)
        lp_50 = _logprobs(s_50)
        a = [x for x in lp_ws if x is not None]
        b = [x for x in lp_50 if x is not None]
        n = min(len(a), len(b))
        if n == 0:
            print(f"  [{label}] no comparable logprobs (len ws={len(a)} 50={len(b)})")
            continue
        diffs = [abs(a[i] - b[i]) for i in range(n)]
        maxd = max(diffs); meand = sum(diffs) / n
        ident = maxd < 1e-6
        print(f"  [{label}] n_tokens={n} max|Δlogprob|={maxd:.6f} mean|Δ|={meand:.6f} "
              f"=> {'IDENTICAL (optim_step NO-OP — weights frozen)' if ident else 'DIFFERENT (weights moved)'}")

    # ---- PROBE 2: read raw completions (truncated-think artifact vs genuine prose) ----
    from eval.output_parsers import extract_php_code
    from scripts.rl_rollouts import _is_parseable_php

    def dump(name, sampler, item, max_tokens):
        args = SimpleNamespace(group_size=1, temperature=1.0, max_new_tokens=max_tokens)
        comps = _generate_completions(sampler, [item], args, renderer=renderer, tok=tok)
        c = comps[0]
        txt = c.completion
        has_open = "<think>" in txt
        has_close = "</think>" in txt
        php = extract_php_code(txt)
        print(f"\n  --- {name} (max_tokens={max_tokens}) ---")
        print(f"  len(chars)={len(txt)} n_tokens={len(getattr(c,'tokens',[]) or [])} "
              f"<think>={has_open} </think>={has_close} parseable_php={_is_parseable_php(php)}")
        print(f"  HEAD: {txt[:240]!r}")
        print(f"  TAIL: {txt[-240:]!r}")

    print("\n===== PROBE 2: RAW COMPLETIONS (gen prompt 0) =====")
    gi = gen_pool[0]
    dump("warm-start @512", s_ws, gi, 512)
    dump("warm-start @2048", s_ws, gi, 2048)
    dump("step50-seedA @2048", s_50, gi, 2048)
    print("\n===== PROBE 2b: RAW COMPLETIONS (judge prompt 0) =====")
    ji = judge_pool[0]
    dump("warm-start judge @2048", s_ws, ji, 2048)
    dump("step50-seedA judge @2048", s_50, ji, 2048)


if __name__ == "__main__":
    sys.exit(main())

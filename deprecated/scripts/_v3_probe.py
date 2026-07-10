#!/usr/bin/env python
"""Pinpoint the V3 warm-start hang: time save_weights vs first generation."""
import logging, time, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("v3probe")

WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"

import tinker
from types import SimpleNamespace
from scripts.tinker_rl_data import load_rl_prompts
from scripts.rl_rollouts import build_rl_renderer, _generate_completions

gen_pool = load_rl_prompts("gen")
import random; random.seed(12345)
prompts = random.sample(gen_pool, 1)
log.info("loaded 1 gen prompt; keys=%s", list(prompts[0].keys()))
renderer, tok = build_rl_renderer()
args = SimpleNamespace(group_size=1, temperature=0.7, max_new_tokens=128)

sc = tinker.ServiceClient()
log.info("create_training_client_from_state...")
t=time.time(); tc0 = sc.create_training_client_from_state(WARMSTART)
log.info("training client OK in %.1fs", time.time()-t)

log.info("save_weights_and_get_sampling_client...")
t=time.time(); sampler = tc0.save_weights_and_get_sampling_client()
log.info("save_weights OK in %.1fs", time.time()-t)

log.info("first _generate_completions (1 prompt, group 1, 128 tok)...")
t=time.time()
comps = _generate_completions(sampler, prompts, args, renderer=renderer, tok=tok)
log.info("generate OK in %.1fs, n=%d", time.time()-t, len(comps))
log.info("completion[0][:300]=%r", comps[0].completion[:300])
print("PROBE COMPLETE")

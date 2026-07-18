#!/usr/bin/env python
"""Dump raw warm-start gen completions + extracted PHP + lint result, to see WHY
_is_valid_wp_php returns 0. Mandatory data-read (handover caveat #2)."""
import logging, sys, subprocess, tempfile, os, argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
from types import SimpleNamespace
logging.basicConfig(level=logging.WARNING)
WARMSTART = "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/weights/wp-reasoning-v4-r32-rp30-savestate-final-state"

def lint(code):
    if not code.strip():
        return "EMPTY"
    src = code if code.lstrip().startswith("<?php") else "<?php\n"+code
    with tempfile.NamedTemporaryFile("w", suffix=".php", delete=False) as f:
        f.write(src); path=f.name
    try:
        r = subprocess.run(["php","-l",path], capture_output=True, text=True, timeout=10)
        return ("OK" if r.returncode==0 else r.stdout.strip()+r.stderr.strip())
    except Exception as e:
        return f"lint-err:{e}"
    finally:
        os.unlink(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    cli = ap.parse_args()
    import tinker, random
    from scripts.tinker_rl_data import load_rl_prompts
    from scripts.rl_rollouts import (build_rl_renderer, _build_sampling_params,
                                     _prompt_user_messages, _seq_tokens,
                                     _is_parseable_php, _is_valid_wp_php)
    from eval.output_parsers import extract_php_code
    gen_pool = load_rl_prompts("gen")
    random.seed(12345)
    prompts = random.sample(gen_pool, cli.n_prompts)
    renderer, tok = build_rl_renderer()
    args = SimpleNamespace(group_size=1, temperature=0.7, max_new_tokens=cli.max_new_tokens)
    sp = _build_sampling_params(args, renderer)
    sc = tinker.ServiceClient()
    tc0 = sc.create_training_client_from_state(WARMSTART)
    sampler = tc0.save_weights_and_get_sampling_client()
    for idx, item in enumerate(prompts):
        prompt = renderer.build_generation_prompt(_prompt_user_messages(item))
        def _do():
            resp = sampler.sample(prompt=prompt, num_samples=1, sampling_params=sp)
            r = resp.result() if hasattr(resp,"result") else resp
            seqs = getattr(r,"sequences",None) or getattr(r,"samples",None) or []
            return [tok.decode(_seq_tokens(s)) for s in seqs], [len(_seq_tokens(s)) for s in seqs]
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                texts, ntoks = ex.submit(_do).result(timeout=120)
        except FTimeout:
            print(f"\n######## PROMPT {idx}: TIMED OUT"); continue
        raw = texts[0]; nt = ntoks[0]
        php = extract_php_code(raw)
        print(f"\n######## PROMPT {idx}  (sampled {nt} tokens, max={cli.max_new_tokens}) ########")
        print(f"--- RAW completion (last 200 chars) ---\n...{raw[-200:]!r}")
        print(f"--- extracted php len={len(php)}, parseable={_is_parseable_php(php)}, valid_wp={_is_valid_wp_php(php)} ---")
        print(f"--- php -l: {lint(php)[:300]}")
        print(f"--- extracted TAIL (last 160 chars): ...{php[-160:]!r}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

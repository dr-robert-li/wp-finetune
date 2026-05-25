"""D-03 backfill generator: produce NEW deep-judge CoT examples on a local vLLM
endpoint (Qwen3.6-35B-A3B) instead of the Anthropic claude --print backend.

Reuses the Phase 4.1 prompt, seed exemplars, function loader, and training-example
formatter from generate_deep_judge_cot.py — only the generation backend changes.
Output is the same record schema (code, reasoning{dimension_analysis,...},
source_file, source_dir, function_name, citation_accuracy) so the downstream
Claude reviewer + assemble_reasoning_dataset.py consume it unchanged.

Quality gating is deliberately LIGHT here (structural only); the real gate is the
spawned Claude Code reviewer that replaces the haiku consistency step.

Run:
  python scripts/generate_cot_vllm.py --target 550 --workers 8
Resumable: re-running skips functions already in the bulk or the new-output file.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_deep_judge_cot import (  # noqa: E402
    load_seeds, sample_seeds, format_seed_as_exemplar,
    load_phase1_functions, format_training_example, REQUIRED_DIMENSIONS,
)
from scripts.utils import extract_json  # noqa: E402

EXISTING_BULK = ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "deep_judge_cot_bulk.json"
OUT_PATH = ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "vllm_new_cot.json"
DEFAULT_BASE_URL = "http://192.168.1.61:30000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_lock = threading.Lock()


def build_prompt(code, seeds):
    sampled = sample_seeds(seeds, 3)
    exemplars = "\n\n".join(format_seed_as_exemplar(s) for s in sampled)
    return f"""You are a WordPress code quality assessor producing deep reasoning chains for training data. You MUST analyze ALL 9 dimensions regardless of how many the example seeds show. Seeds only show 2-3 dimensions as examples of analysis depth, not as the complete set.

Here are golden examples of deep reasoning:

{exemplars}

NOW ANALYZE the following WordPress PHP code. Return JSON with keys:
- verdict: "PASS" or "FAIL"
- dimension_analysis: object with ALL 9 dimensions: wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility -- each having score (integer 1-10) and analysis (string citing specific WordPress APIs by name)
- overall_score: integer 0-100
- key_observation: string summarizing the most important finding

When WordPress APIs appear in the code, name them explicitly: $wpdb->prepare(), wp_verify_nonce(), esc_html(), current_user_can(), check_ajax_referer(), esc_attr(), esc_url().
When WordPress APIs are MISSING from code that needs them, state explicitly what is missing and why.
Do not describe behavior abstractly without naming the specific API.
IMPORTANT: Only cite APIs that actually appear in the code or that are demonstrably missing. Do not invent citations.

Code to analyze:
```php
{code[:3000]}
```

Return valid JSON only."""


def structural_ok(result):
    if not isinstance(result, dict):
        return False, "not_dict"
    if result.get("verdict") not in ("PASS", "FAIL"):
        return False, "bad_verdict"
    if not isinstance(result.get("overall_score"), int):
        return False, "bad_overall"
    da = result.get("dimension_analysis", {})
    if not isinstance(da, dict):
        return False, "no_dims"
    present = 0
    for d in REQUIRED_DIMENSIONS:
        info = da.get(d)
        if not isinstance(info, dict):
            continue
        if info.get("analysis") and (info.get("score") is not None):
            present += 1
    if present < 8:  # allow 1 N/A; reviewer enforces deeper
        return False, f"only_{present}_dims"
    return True, ""


def gen_one(client, model, temperature, func, seeds, max_retries=3):
    prompt = build_prompt(func["code"], seeds)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2200,
                timeout=180,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content or ""
            text = THINK_RE.sub("", text).strip()
            result = extract_json(text)
            if result is None:
                continue
            ok, _ = structural_ok(result)
            if not ok:
                continue
            return format_training_example(func, result, func["code"])
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None


def load_done_keys():
    keys = set()
    if EXISTING_BULK.exists():
        for e in json.loads(EXISTING_BULK.read_text()):
            keys.add((e.get("source_file"), e.get("function_name")))
    existing_new = []
    if OUT_PATH.exists():
        existing_new = json.loads(OUT_PATH.read_text())
        for e in existing_new:
            keys.add((e.get("source_file"), e.get("function_name")))
    return keys, existing_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=550, help="number of NEW accepted CoT to produce")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()

    random.seed(42)
    client = openai.OpenAI(base_url=args.base_url, api_key="none")
    seeds = load_seeds()
    if not seeds:
        print("ERROR: no deep_judge_cot seeds", file=sys.stderr)
        sys.exit(1)

    funcs = load_phase1_functions()
    random.shuffle(funcs)
    done_keys, results = load_done_keys()
    print(f"vLLM CoT generator | model={args.model} | base={args.base_url}")
    print(f"Functions available: {len(funcs)} | already-done keys: {len(done_keys)} | resume buffer: {len(results)}")
    print(f"Target NEW accepted: {args.target} | workers: {args.workers}")

    # Candidate pool: functions not already generated for
    pool = [f for f in funcs if (f["source_file"], f["function_name"]) not in done_keys]
    print(f"Candidate pool (unused): {len(pool)}")

    need = args.target - len(results)
    if need <= 0:
        print(f"Target already met ({len(results)} >= {args.target}).")
        return

    accepted = len(results)
    attempted = 0
    failures = 0
    # Iterate the pool in chunks; submit concurrently; stop when target met or pool exhausted.
    idx = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        while accepted < args.target and idx < len(pool):
            batch = pool[idx: idx + args.workers * 2]
            idx += len(batch)
            futs = {ex.submit(gen_one, client, args.model, args.temperature, f, seeds): f for f in batch}
            for fut in as_completed(futs):
                attempted += 1
                rec = fut.result()
                if rec is None:
                    failures += 1
                    continue
                with _lock:
                    results.append(rec)
                    accepted += 1
                    if accepted % 10 == 0:
                        OUT_PATH.write_text(json.dumps(results, indent=2))
                        print(f"  accepted={accepted}/{args.target} attempted={attempted} fail={failures}")
                if accepted >= args.target:
                    break

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nDONE. accepted={accepted} attempted={attempted} failures={failures}")
    print(f"Output: {OUT_PATH} ({len(results)} records)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""P4 Step-6 format-stability gate — cot+ctf scope (.venv-tinker).

The in-driver gate scored ALL 141 val rows, which wrongly counts the 21 `replay`
(code-gen) rows as "terse" — they legitimately carry no [/REASONING]. The REOPEN
terse metric is defined on cot+ctf ONLY. This re-scores the persisted promoted
checkpoint on cot+ctf with the pre-registered gate:

  FAIL if terse_rate > 0.10 OR Wilson-95-upper > 0.15.

temp 0.0 arm: 1 sample / prompt (greedy, deterministic), n = #cot+ctf rows.
temp 0.7 arm: k samples / prompt to reach n >= --gate-n (Wilson sizing).

Usage: python scripts/tinker_fs_gate.py --gate-n 300
"""
import argparse
import json
import math
import os
import sys

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinker_reasoning_data import BASE_MODEL, RENDERER_NAME


def _wilson_upper(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return c + h


def _texts(resp, tok):
    r = resp.result() if hasattr(resp, "result") else resp
    seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
    out = []
    for s in seqs:
        toks = (getattr(s, "tokens", None) or getattr(s, "token_ids", None)
                or getattr(s, "output_tokens", None))
        out.append(tok.decode(toks))
    return out


def _resolve_path(manifest, arg):
    if arg:
        return arg
    m = json.load(open(manifest))
    promoted = m.get("promoted")
    return next(c["sampler_path"] for c in m["checkpoints"] if c["name"] == promoted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="output/tinker/wp-reasoning-v2-manifest.json")
    ap.add_argument("--tinker-path", default=None)
    ap.add_argument("--dataset", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--streams", default="cot,ctf", help="terse metric scope (REOPEN: cot+ctf)")
    ap.add_argument("--temps", default="0.0,0.7")
    ap.add_argument("--gate-n", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out", default="output/format_stability/fs_gate/wp-reasoning-v2/summary.json")
    args = ap.parse_args()

    path = _resolve_path(args.manifest, args.tinker_path)
    streams = {s.strip() for s in args.streams.split(",")}
    temps = [float(t) for t in args.temps.split(",")]
    print(f"[fs-gate] checkpoint: {path}", flush=True)
    print(f"[fs-gate] streams: {sorted(streams)}", flush=True)

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    rows = [r for r in rows if r.get("metadata", {}).get("stream") in streams]
    print(f"[fs-gate] {len(rows)} cot+ctf rows", flush=True)

    tok = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=path)

    arms = []
    overall_pass = True
    for temp in temps:
        k = 1 if temp == 0.0 else max(1, math.ceil(args.gate_n / len(rows)))
        sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=temp,
                                   stop=renderer.get_stop_sequences())
        terse = total = 0
        for r in rows:
            um = [m for m in r["messages"] if m["role"] == "user"]
            prompt = renderer.build_generation_prompt(um)
            resp = client.sample(prompt=prompt, num_samples=k, sampling_params=sp)
            for t in _texts(resp, tok):
                total += 1
                if "[/REASONING]" not in t:
                    terse += 1
        rate = terse / total if total else float("nan")
        wu = _wilson_upper(terse, total)
        ap_ = (rate <= 0.10) and (wu <= 0.15)
        overall_pass = overall_pass and ap_
        arms.append({"temp": temp, "terse": terse, "n": total, "rate": rate,
                     "wilson_upper": wu, "pass": ap_})
        print(f"[fs-gate] temp{temp} terse={terse}/{total} rate={rate:.4f} "
              f"wilson_upper={wu:.4f} -> {'PASS' if ap_ else 'FAIL'}", flush=True)

    summary = {"checkpoint": path, "streams": sorted(streams), "gate_n": args.gate_n,
               "threshold_rate": 0.10, "threshold_wilson_upper": 0.15,
               "arms": arms, "pass": overall_pass}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"FS_GATE {'PASS' if overall_pass else 'FAIL'} -> {args.out}", flush=True)
    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())

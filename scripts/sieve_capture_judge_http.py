#!/usr/bin/env python
"""HTTP (vLLM-served) judge-response capture for the SIEVE-04 k-sweep.

Same index discipline as scripts/capture_judge_responses_tinker.py (that
script's checkpoints have no HTTP endpoint; ours -- merged local checkpoints
served via vLLM docker -- do, so this is the plain OpenAI-client twin):

    examples = [rows whose first user message .startswith("<wp_judge>")],
    in file order; `index` = position in THAT filtered list (0..N-1).
    scripts/relabel/eval_relabel.py / g1_read.py enumerate identically, so
    captured responses align row-for-row with val_labels_v1.json via the
    `val:{wj_rows[index]}` key.

Uses eval.eval_judge._judge_create (RC-A enable_thinking=False guard) so
captures are judge-comparable with every other seed capture in this project.

Usage:
    python -m scripts.sieve_capture_judge_http \
        --base-url http://localhost:8000/v1 --model wp-30_70 \
        --dataset data/reasoning_dataset/openai_val.jsonl \
        --out output/sieve/ksweep/k32_s0/judge_responses.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matches wp-bench's own concurrency=4 convention (config/wp-bench.yaml,
# scripts/run_eval_reasoning.py _run_wpbench) -- vLLM continuous batching
# handles concurrent requests fine; a naive sequential loop here was the
# actual bottleneck (single in-flight request at a time, ~20-30s/item ==>
# hours for 121 items x 3 seeds x 4 k-arms).
DEFAULT_CONCURRENCY = 4


def capture(base_url: str, model: str | None, dataset: str, out: str,
            max_tokens: int = 2048, temperature: float = 0.0,
            concurrency: int = DEFAULT_CONCURRENCY) -> dict:
    import openai
    from eval.eval_judge import _detect_model, _judge_create

    client = openai.OpenAI(base_url=base_url, api_key="none")
    resolved_model = model or _detect_model(client)

    rows = [json.loads(line) for line in open(dataset) if line.strip()]
    examples = [r for r in rows if next(
        (m["content"] for m in r["messages"] if m["role"] == "user"), ""
    ).startswith("<wp_judge>")]
    print(f"[sieve-capture] {len(examples)} wp_judge examples via {base_url} "
          f"(model={resolved_model}, concurrency={concurrency})", file=sys.stderr)

    results: dict[int, str] = {}
    counts_lock = threading.Lock()
    counts = {"ok": 0, "err": 0, "done": 0}

    def _one(idx: int, r: dict) -> tuple[int, str]:
        user_msgs = [m for m in r["messages"] if m["role"] == "user"]
        try:
            resp = _judge_create(client, model=resolved_model, messages=user_msgs,
                                  max_tokens=max_tokens, temperature=temperature)
            text = resp.choices[0].message.content or ""
            with counts_lock:
                counts["ok"] += 1
        except Exception as e:  # noqa: BLE001
            print(f"[sieve-capture] error idx {idx}: {e}", file=sys.stderr, flush=True)
            text = ""
            with counts_lock:
                counts["err"] += 1
        with counts_lock:
            counts["done"] += 1
            if counts["done"] % 25 == 0:
                print(f"[sieve-capture] {counts['done']}/{len(examples)} "
                      f"ok={counts['ok']} err={counts['err']}", file=sys.stderr, flush=True)
        return idx, text

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, idx, r) for idx, r in enumerate(examples)]
        for fut in as_completed(futures):
            idx, text = fut.result()
            results[idx] = text

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(json.dumps({"__provenance__": base_url, "model": resolved_model,
                              "dataset": dataset, "max_tokens": max_tokens,
                              "temperature": temperature, "n": len(examples)}) + "\n")
        for idx in range(len(examples)):
            fh.write(json.dumps({"index": idx, "response": results[idx]}) + "\n")
    print(f"[sieve-capture] DONE n={len(examples)} ok={counts['ok']} err={counts['err']} "
          f"-> {out_path}", file=sys.stderr, flush=True)
    return {"n": len(examples), "ok": counts["ok"], "err": counts["err"], "out": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    capture(args.base_url, args.model, args.dataset, args.out, args.max_tokens, args.temperature)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""P4 REVL-01A / invalid-PHP capture — Tinker sampling backend (runs in .venv-tinker).

The local REVL pipeline samples the model over an OpenAI-compatible vLLM endpoint.
The Tinker checkpoint has NO HTTP endpoint — only the Python SamplingClient — so this
script is the Tinker-backed CAPTURE half. It samples each `<wp_judge>` prompt with the
SAME renderer the model was trained under (qwen3_disable_thinking) and writes a
`{index, response}` JSONL. The scoring half (GT extraction + Spearman) stays in
`eval/eval_judge.py` (project venv), fed via its new `--responses-jsonl` offline mode.

Index discipline (MUST match eval_judge._run_eval_reasoning):
  examples = [rows whose first user message .startswith("<wp_judge>")], in file order;
  `index` = position in THAT filtered list (0..N-1). eval_judge enumerates identically,
  so captured responses align row-for-row with its GT pairing.

Usage:
  python scripts/capture_judge_responses_tinker.py \
      --tinker-path "tinker://<run>/sampler_weights/wp-reasoning-v2-ep3" \
      --dataset data/reasoning_dataset/openai_val.jsonl \
      --out output/eval_reasoning/reasoning_v2_tinker/judge_responses.jsonl
  # tinker-path defaults to the `promoted` sampler path in the manifest.
"""
import argparse
import json
import os
import sys

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinker_reasoning_data import BASE_MODEL, RENDERER_NAME


def _decode_first(resp, tok):
    r = resp.result() if hasattr(resp, "result") else resp
    seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
    seq = seqs[0]
    toks = (getattr(seq, "tokens", None) or getattr(seq, "token_ids", None)
            or getattr(seq, "output_tokens", None))
    return tok.decode(toks)


def _resolve_tinker_path(arg_path, manifest_path):
    if arg_path:
        return arg_path
    with open(manifest_path) as f:
        m = json.load(f)
    promoted = m.get("promoted")
    for c in m.get("checkpoints", []):
        if c.get("name") == promoted:
            return c["sampler_path"]
    raise SystemExit(f"could not resolve sampler_path for promoted={promoted} in {manifest_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinker-path", default=None, help="sampler tinker:// path (default: manifest promoted)")
    ap.add_argument("--manifest", default="output/tinker/wp-reasoning-v2-manifest.json")
    ap.add_argument("--dataset", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--filter", default="wp_judge_startswith",
                    choices=["wp_judge_startswith", "all"],
                    help="row filter; wp_judge_startswith mirrors eval_judge._run_eval_reasoning")
    ap.add_argument("--base-model", default=None,
                    help="override BASE_MODEL for tokenizer+renderer resolution (default: this "
                         "module's v3 BASE_MODEL). Pass the v4 base "
                         "(tinker_reasoning_data_v4.BASE_MODEL) to sample a v4-trained sampler "
                         "checkpoint under its own renderer -- back-compat: omitting this flag "
                         "reproduces the original v3-only behavior exactly.")
    ap.add_argument("--renderer", default=None,
                    help="override RENDERER_NAME explicitly (default: resolved from --base-model, "
                         "or this module's v3 RENDERER_NAME if --base-model is also omitted)")
    args = ap.parse_args()

    tinker_path = _resolve_tinker_path(args.tinker_path, args.manifest)
    print(f"[capture-tinker] sampler path: {tinker_path}", flush=True)

    base_model = args.base_model or BASE_MODEL
    renderer_name = args.renderer
    if renderer_name is None:
        if base_model == BASE_MODEL:
            renderer_name = RENDERER_NAME
        else:
            from tinker_reasoning_data_v4 import BASE_MODEL as V4_BASE_MODEL, RENDERER_NAME as V4_RENDERER_NAME
            if base_model != V4_BASE_MODEL:
                raise SystemExit(f"--base-model {base_model!r} has no known renderer "
                                  f"(known: {BASE_MODEL!r} -> {RENDERER_NAME!r}, "
                                  f"{V4_BASE_MODEL!r} -> {V4_RENDERER_NAME!r}); pass --renderer explicitly")
            renderer_name = V4_RENDERER_NAME
    print(f"[capture-tinker] base_model={base_model} renderer={renderer_name}", flush=True)

    tok = get_tokenizer(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tok)
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=tinker_path)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                               stop=renderer.get_stop_sequences())

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    examples = []
    for r in rows:
        um = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        if args.filter == "all" or um.startswith("<wp_judge>"):
            examples.append(r)
    print(f"[capture-tinker] {len(examples)} examples (filter={args.filter})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_close = n_jo = n_infra_error = 0
    with open(args.out, "w") as fh:
        fh.write(json.dumps({"__provenance__": tinker_path, "dataset": args.dataset,
                             "base_model": base_model, "renderer": renderer_name,
                             "max_tokens": args.max_tokens,
                             "temperature": args.temperature, "n": len(examples)}) + "\n")
        for idx, r in enumerate(examples):
            user_msgs = [m for m in r["messages"] if m["role"] == "user"]
            prompt = renderer.build_generation_prompt(user_msgs)
            infra_error = False
            try:
                resp = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=sp)
                text = _decode_first(resp, tok)
            except Exception as e:  # noqa: BLE001
                print(f"[capture-tinker] sample error idx {idx}: {e}", flush=True)
                text = ""
                infra_error = True
                n_infra_error += 1
            n_close += "[/REASONING]" in text
            n_jo += "<judge_output>" in text
            # WR-04: "infra_error" is an additive field -- eval_relabel.py
            # (unmodified) only reads "index"/"response" per row and ignores
            # unknown keys, so a transient sampling-API error stays
            # distinguishable from a genuine judge-format non-compliance
            # without changing the downstream scorer.
            fh.write(json.dumps({"index": idx, "response": text, "infra_error": infra_error}) + "\n")
            if (idx + 1) % 25 == 0:
                print(f"[capture-tinker] {idx + 1}/{len(examples)} "
                      f"close={n_close} judge_output={n_jo} infra_error={n_infra_error}", flush=True)
    # WR-04: sidecar summary carrying n_infra_error alongside n (total) --
    # written last so streaming per-row writes above stay crash-resilient
    # (a killed mid-run process still leaves a usable partial capture file;
    # only this final summary would be missing).
    summary_path = args.out + ".capture_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"n": len(examples), "n_infra_error": n_infra_error,
                   "n_close_tag": n_close, "n_judge_output_tag": n_jo}, f, indent=2)
    print(f"[capture-tinker] DONE n={len(examples)} close_tag={n_close} "
          f"judge_output={n_jo} infra_error={n_infra_error} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

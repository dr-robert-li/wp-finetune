#!/usr/bin/env python
"""Tinker pivot P2 — reasoning LoRA SFT on Qwen3-30B-A3B (cloud).

Same driver for all three stages, parameterized by --max-steps:
  smoke : --max-steps 4   --eval-n 4    (validate data->train->sample path; cheap)
  short : --epochs 1                     (~70 steps; loss curve + prelim terse rate)
  full  : --epochs 3      --eval-n 77    (the expensive run; gated behind HITL)

terse rate = fraction of val prompts whose sampled output LACKS [/REASONING]
(the REVL-05 collapse metric: bare-JSON with no prose CoT).
"""
import argparse
import os
import sys

import tinker
from tinker_cookbook import hyperparam_utils as hp
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
from tinker_reasoning_data import BASE_MODEL, RENDERER_NAME, VAL_PATH, build_datasets


def _loss_of(fb_out):
    import numbers
    d = None
    if hasattr(fb_out, "model_dump"):
        try:
            d = fb_out.model_dump()
        except Exception:
            d = None
    if d is None:
        d = getattr(fb_out, "__dict__", None)

    def find(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "loss" in str(k).lower() and isinstance(v, numbers.Number):
                    return float(v)
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(o, (list, tuple)):
            for v in o:
                r = find(v)
                if r is not None:
                    return r
        return None

    return find(d) if isinstance(d, dict) else None


def _sample_text(resp, tok):
    r = resp.result() if hasattr(resp, "result") else resp
    seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
    seq = seqs[0]
    toks = getattr(seq, "tokens", None) or getattr(seq, "token_ids", None) or getattr(seq, "output_tokens", None)
    return tok.decode(toks)


def terse_eval(sampling_client, renderer, tok, val_rows, n, max_tokens, temperature, debug=False):
    sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                               stop=renderer.get_stop_sequences())
    terse = total = 0
    for row in val_rows[:n]:
        user_msgs = [m for m in row["messages"] if m.get("role") == "user"]
        prompt = renderer.build_generation_prompt(user_msgs)
        resp = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=sp)
        text = _sample_text(resp, tok)
        if debug and total == 0:
            print(f"[eval] sample[0] head: {text[:200]!r}", flush=True)
        total += 1
        if "[/REASONING]" not in text:
            terse += 1
    return terse, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="smoke")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=4)
    ap.add_argument("--eval-max-tokens", type=int, default=2048)
    ap.add_argument("--eval-temperature", type=float, default=0.0)
    ap.add_argument("--eval-temps", default=None,
                    help="csv of temps for the FINAL eval, e.g. '0.0,0.7' (default: --eval-temperature)")
    ap.add_argument("--per-epoch-eval-n", type=int, default=0,
                    help="if >0, eval this many val prompts at temp 0 after each epoch (late-collapse guard) + save a per-epoch checkpoint")
    ap.add_argument("--final-eval-n", type=int, default=0, help="override --eval-n for the final eval")
    ap.add_argument("--save-name", default=None)
    args = ap.parse_args()
    save_name = args.save_name or f"wp-reasoning-{args.stage}"
    final_eval_n = args.final_eval_n or args.eval_n
    final_temps = [float(t) for t in args.eval_temps.split(",")] if args.eval_temps else [args.eval_temperature]

    train_ds, _ = build_datasets(batch_size=args.batch_size)
    val_rows = [json.loads(l) for l in open(VAL_PATH)]
    lr = hp.get_lr(BASE_MODEL, is_lora=True)
    tok = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)

    print(f"[sft] stage={args.stage} base={BASE_MODEL} rank={args.rank} lr={lr:.2e} "
          f"epochs={args.epochs} max_steps={args.max_steps} train_batches={len(train_ds)} "
          f"bs={args.batch_size}", flush=True)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank)

    step = 0
    losses = []
    stop = False
    last_sampling_client = None
    for epoch in range(args.epochs):
        for i in range(len(train_ds)):
            batch = train_ds.get_batch(i)
            fb = tc.forward_backward(data=batch, loss_fn="cross_entropy")
            tc.optim_step(tinker.AdamParams(learning_rate=lr))
            out = fb.result() if hasattr(fb, "result") else fb
            loss = _loss_of(out)
            step += 1
            if loss is not None:
                losses.append(loss)
            if step <= 3 or step % 10 == 0:
                print(f"[sft] step {step} epoch {epoch} loss={loss}", flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        # Per-epoch checkpoint + quick late-collapse guard eval (temp 0).
        ep_name = f"{save_name}-ep{epoch + 1}"
        last_sampling_client = tc.save_weights_and_get_sampling_client(name=ep_name)
        print(f"[sft] epoch {epoch + 1} done (step {step}); checkpoint saved name={ep_name}", flush=True)
        if args.per_epoch_eval_n > 0:
            t, n = terse_eval(last_sampling_client, renderer, tok, val_rows,
                              n=args.per_epoch_eval_n, max_tokens=args.eval_max_tokens,
                              temperature=0.0)
            print(f"[epoch-eval] ep{epoch + 1} terse@temp0 = {t}/{n} = {t / n if n else float('nan'):.3f}", flush=True)
        if stop:
            break

    print(f"[sft] training done: {step} steps. "
          f"loss first={losses[0] if losses else '?'} last={losses[-1] if losses else '?'}", flush=True)

    sampling_client = last_sampling_client or tc.save_weights_and_get_sampling_client(name=save_name)

    # Final eval on the last checkpoint, across all requested temps.
    final_rates = {}
    for ti, temp in enumerate(final_temps):
        terse, total = terse_eval(sampling_client, renderer, tok, val_rows,
                                  n=final_eval_n, max_tokens=args.eval_max_tokens,
                                  temperature=temp, debug=(ti == 0))
        rate = terse / total if total else float("nan")
        final_rates[temp] = (terse, total, rate)
        print(f"[eval] terse(no [/REASONING]) @temp{temp} = {terse}/{total} = {rate:.3f} "
              f"(max_tokens={args.eval_max_tokens})", flush=True)
    rates_str = " ".join(f"terse@{t}={r[2]:.3f}" for t, r in final_rates.items())
    print(f"RESULT stage={args.stage} steps={step} "
          f"loss_first={losses[0] if losses else 'NA'} loss_last={losses[-1] if losses else 'NA'} "
          f"n_eval={final_eval_n} {rates_str}", flush=True)


if __name__ == "__main__":
    main()

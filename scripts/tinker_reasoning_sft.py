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
import datetime
import math
import os
import sys

import tinker
from tinker_cookbook import hyperparam_utils as hp
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
from tinker_reasoning_data import BASE_MODEL, RENDERER_NAME, TRAIN_PATH, VAL_PATH, build_datasets


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


def _res(f):
    """Resolve a Tinker future/APIFuture to its value."""
    return f.result() if hasattr(f, "result") else f


def _wilson_upper(k, n, z=1.96):
    """Wilson score 95% upper bound for a binomial rate k/n."""
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center + half


def _write_manifest(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[manifest] wrote {path}", flush=True)


def _all_sample_texts(resp, tok):
    """Decode ALL sampled sequences from one sample() response (num_samples>=1)."""
    r = resp.result() if hasattr(resp, "result") else resp
    seqs = getattr(r, "sequences", None) or getattr(r, "samples", None) or []
    out = []
    for seq in seqs:
        toks = (getattr(seq, "tokens", None) or getattr(seq, "token_ids", None)
                or getattr(seq, "output_tokens", None))
        out.append(tok.decode(toks))
    return out


def _sample_text(resp, tok):
    return _all_sample_texts(resp, tok)[0]


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


def terse_gate_eval(sampling_client, renderer, tok, val_rows, target_n, max_tokens, temperature):
    """Monte-Carlo terse rate to reach n>=target_n by drawing K samples/prompt.

    For temp>0 the Step-6 format-stability gate needs n>=300 to size the Wilson
    upper bound; the val set is only ~77 unique prompts, so we draw multiple
    samples per prompt. At temp 0 (greedy) sampling is deterministic, so K is
    forced to 1 and n == len(val_rows).
    """
    n_rows = len(val_rows)
    k = 1 if temperature == 0.0 else max(1, math.ceil(target_n / n_rows))
    sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                               stop=renderer.get_stop_sequences())
    terse = total = 0
    for row in val_rows:
        user_msgs = [m for m in row["messages"] if m.get("role") == "user"]
        prompt = renderer.build_generation_prompt(user_msgs)
        resp = sampling_client.sample(prompt=prompt, num_samples=k, sampling_params=sp)
        for text in _all_sample_texts(resp, tok):
            total += 1
            if "[/REASONING]" not in text:
                terse += 1
    return terse, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="smoke")
    ap.add_argument("--train-path", default=None,
                    help="override train JSONL (e.g. the corrective augmented set)")
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
    ap.add_argument("--manifest", default=None,
                    help="path to write the persistent-checkpoint manifest (tinker paths) — incrementally, per epoch")
    ap.add_argument("--save-state", action="store_true",
                    help="also save_state() the final checkpoint (durable training ckpt -> export source)")
    ap.add_argument("--gate-temps", default=None,
                    help="csv of temps for the Step-6 format-stability gate, e.g. '0.0,0.7'")
    ap.add_argument("--gate-n", type=int, default=300,
                    help="target sample count for the temp>0 gate arm (multi-sample to reach n; Wilson upper sizing)")
    args = ap.parse_args()
    save_name = args.save_name or f"wp-reasoning-{args.stage}"
    final_eval_n = args.final_eval_n or args.eval_n
    final_temps = [float(t) for t in args.eval_temps.split(",")] if args.eval_temps else [args.eval_temperature]
    gate_temps = [float(t) for t in args.gate_temps.split(",")] if args.gate_temps else []
    manifest_path = args.manifest or f"output/tinker/{save_name}-manifest.json"
    manifest = {
        "save_name": save_name, "base_model": BASE_MODEL, "rank": args.rank,
        "renderer": RENDERER_NAME, "epochs": args.epochs, "checkpoints": [],
        "promoted": None, "state_path": None,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    train_path = args.train_path or TRAIN_PATH
    train_ds, _ = build_datasets(train_path=train_path, batch_size=args.batch_size)
    val_rows = [json.loads(l) for l in open(VAL_PATH)]
    print(f"[sft] train_path={train_path}", flush=True)
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

        # Per-epoch PERSISTENT sampler checkpoint (ttl None = never expires) +
        # quick late-collapse guard eval (temp 0). Manifest is written every epoch
        # so a lost process can never again strand the reference (the P4 failure mode).
        ep_name = f"{save_name}-ep{epoch + 1}"
        sampler_path = _res(tc.save_weights_for_sampler(name=ep_name, ttl_seconds=None)).path
        last_sampling_client = tc.create_sampling_client(model_path=sampler_path)
        manifest["checkpoints"].append({"epoch": epoch + 1, "name": ep_name,
                                        "sampler_path": sampler_path})
        manifest["promoted"] = ep_name
        _write_manifest(manifest_path, manifest)
        print(f"[sft] epoch {epoch + 1} done (step {step}); persistent sampler ckpt "
              f"name={ep_name} path={sampler_path}", flush=True)
        if args.per_epoch_eval_n > 0:
            t, n = terse_eval(last_sampling_client, renderer, tok, val_rows,
                              n=args.per_epoch_eval_n, max_tokens=args.eval_max_tokens,
                              temperature=0.0)
            print(f"[epoch-eval] ep{epoch + 1} terse@temp0 = {t}/{n} = {t / n if n else float('nan'):.3f}", flush=True)
        if stop:
            break

    print(f"[sft] training done: {step} steps. "
          f"loss first={losses[0] if losses else '?'} last={losses[-1] if losses else '?'}", flush=True)

    if last_sampling_client is None:
        # No per-epoch save happened (e.g. --max-steps smoke); persist now.
        sampler_path = _res(tc.save_weights_for_sampler(name=save_name, ttl_seconds=None)).path
        last_sampling_client = tc.create_sampling_client(model_path=sampler_path)
        manifest["checkpoints"].append({"epoch": None, "name": save_name, "sampler_path": sampler_path})
        manifest["promoted"] = save_name
    sampling_client = last_sampling_client

    # Durable training checkpoint of the FINAL model state -> export source.
    if args.save_state:
        state_path = _res(tc.save_state(name=f"{save_name}-final-state", ttl_seconds=None)).path
        manifest["state_path"] = state_path
        _write_manifest(manifest_path, manifest)
        print(f"[sft] final save_state path={state_path}", flush=True)

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

    # Step-6 format-stability gate (pre-registered): FAIL if terse>0.10 OR
    # Wilson-95-upper>0.15. temp>0 arms are sized to n>=gate_n via multi-sample.
    if gate_temps:
        gate = {"threshold_rate": 0.10, "threshold_wilson_upper": 0.15, "gate_n": args.gate_n, "arms": []}
        overall_pass = True
        for temp in gate_temps:
            t, n = terse_gate_eval(sampling_client, renderer, tok, val_rows,
                                   target_n=args.gate_n, max_tokens=args.eval_max_tokens,
                                   temperature=temp)
            rate = t / n if n else float("nan")
            wu = _wilson_upper(t, n)
            arm_pass = (rate <= 0.10) and (wu <= 0.15)
            overall_pass = overall_pass and arm_pass
            gate["arms"].append({"temp": temp, "terse": t, "n": n, "rate": rate,
                                 "wilson_upper": wu, "pass": arm_pass})
            print(f"[fs-gate] temp{temp} terse={t}/{n} rate={rate:.4f} "
                  f"wilson_upper={wu:.4f} -> {'PASS' if arm_pass else 'FAIL'}", flush=True)
        gate["pass"] = overall_pass
        manifest["fs_gate"] = gate
        _write_manifest(manifest_path, manifest)
        print(f"FS_GATE {'PASS' if overall_pass else 'FAIL'} "
              f"promoted={manifest['promoted']} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()

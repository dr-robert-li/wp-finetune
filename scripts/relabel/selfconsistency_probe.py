#!/usr/bin/env python3
"""Self-consistency probe for the v1.3 judge (08.2, test-time compute lever).

Question this settles: does median-of-N sampled judge scores at temp>0 (ONE model)
beat the single-sample v1.3 rho (0.8274), approaching the 3-seed ENSEMBLE rho
(0.8419) at 1x serve cost instead of 3x? If yes, self-consistency dominates the
ensemble tradeoff before compression. If it flatlines at ~0.827, single-seed
greedy is the ship config and we compress without regret.

Reuses the capture renderer/tinker setup and the eval_relabel parsing + val labels
verbatim; the only new logic is the per-item median-over-samples aggregation, which
has a --selftest that runs offline (no GPU/tinker).

Usage:
  .venv-tinker/bin/python scripts/relabel/selfconsistency_probe.py \
      --n-samples 5 --temperature 0.7 \
      --out output/relabel/eval_selfconsistency/samples.jsonl
  # tinker-path defaults to PROMOTED_v1.3.json sampler_path.

  python scripts/relabel/selfconsistency_probe.py --selftest   # offline logic check
"""
import argparse
import json
import os
import sys
from statistics import median

sys.path.insert(0, ".")
os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")


def aggregate(overalls_per_item):
    """{item_key: [overall, ...]} -> {item_key: median_overall}. Skips items with no
    parseable sample. Median (not mean) to match the multiseed ensemble aggregation."""
    out = {}
    for k, vals in overalls_per_item.items():
        clean = [float(v) for v in vals if v is not None]
        if clean:
            out[k] = median(clean)
    return out


def _selftest():
    agg = aggregate({"a": [80, 70, 90], "b": [60, None, 62], "c": [None], "d": []})
    assert agg == {"a": 80.0, "b": 61.0}, agg  # median, None/empty dropped
    assert "c" not in agg and "d" not in agg
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run offline aggregation check and exit")
    ap.add_argument("--tinker-path", default=None, help="sampler tinker:// path (default: PROMOTED_v1.3)")
    ap.add_argument("--promoted", default="output/tinker/PROMOTED_v1.3.json")
    ap.add_argument("--dataset", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--labels", default="output/relabel/val_labels_v1.json")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="output/relabel/eval_selfconsistency/samples.jsonl")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return 0

    import numpy as np
    import tinker
    from scipy.stats import spearmanr
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from eval.eval_judge import _derive_prose_overall
    from eval.output_parsers import load_dim_map, parse_judge_scores

    sys.path.insert(0, "scripts")  # tinker_reasoning_* live in scripts/ (run from repo root)
    from tinker_reasoning_data import BASE_MODEL, RENDERER_NAME
    from tinker_reasoning_sft import _all_sample_texts

    tinker_path = args.tinker_path or json.load(open(args.promoted))["sampler_path"]
    print(f"[sc-probe] sampler: {tinker_path}  N={args.n_samples} temp={args.temperature}", flush=True)

    dm = load_dim_map()
    dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}

    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    wj_rows = [i for i, r in enumerate(rows)
               if next((m["content"] for m in r["messages"] if m["role"] == "user"), "").startswith("<wp_judge>")]

    tok = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tok)
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=tinker_path)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                               stop=renderer.get_stop_sequences())

    def parse_overall(text):
        p = parse_judge_scores(text, "auto")
        if not p or not p.get("dimension_scores"):
            return None
        return float(p["overall"]) if "overall" in p else _derive_prose_overall(p["dimension_scores"], dw)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    overalls = {}          # item_key -> [overall per sample]
    n_parse_fail = 0
    with open(args.out, "w") as fh:
        fh.write(json.dumps({"__provenance__": tinker_path, "n_samples": args.n_samples,
                             "temperature": args.temperature, "dataset": args.dataset}) + "\n")
        for idx in range(len(wj_rows)):
            row = rows[wj_rows[idx]]
            user_msgs = [m for m in row["messages"] if m["role"] == "user"]
            prompt = renderer.build_generation_prompt(user_msgs)
            try:
                resp = sampling_client.sample(prompt=prompt, num_samples=args.n_samples, sampling_params=sp)
                texts = _all_sample_texts(resp, tok)
            except Exception as e:  # noqa: BLE001
                print(f"[sc-probe] sample error idx {idx}: {e}", flush=True)
                texts = []
            ovs = [parse_overall(t) for t in texts]
            n_parse_fail += sum(1 for o in ovs if o is None)
            overalls[f"val:{wj_rows[idx]}"] = ovs
            fh.write(json.dumps({"index": idx, "key": f"val:{wj_rows[idx]}",
                                 "overalls": ovs, "responses": texts}) + "\n")
            if (idx + 1) % 25 == 0:
                print(f"[sc-probe] {idx + 1}/{len(wj_rows)}", flush=True)

    model = aggregate(overalls)
    new = {k: v for k, v in json.load(open(args.labels)).items() if k.startswith("val:")}
    common = sorted(set(model) & set(new))
    s = [model[k] for k in common]
    nl = [new[k] for k in common]
    rho = spearmanr(s, nl).statistic

    rng = np.random.default_rng(7)
    n = len(common)
    boots = sorted(spearmanr(np.array(s)[ix], np.array(nl)[ix]).statistic
                   for ix in (rng.integers(0, n, n) for _ in range(2000)))

    prom = json.load(open(args.promoted))["eval"]
    single = prom["judge_rho_vs_new_val_labels"]  # 0.8274 single-sample v1.3
    ceil = prom["ceiling"]
    ensemble = 0.8419637236160609  # 3-seed median ensemble (eval_multiseed.json), 3x serve cost

    res = {"tinker_path": tinker_path, "n_samples": args.n_samples, "temperature": args.temperature,
           "n": n, "rho_selfconsistency": rho, "ci": [boots[50], boots[1949]],
           "single_sample_v13": single, "ensemble_3seed": ensemble, "ceiling": ceil,
           "delta_vs_single": rho - single, "delta_vs_ensemble": rho - ensemble,
           "sample_parse_fail": n_parse_fail}
    summ = os.path.join(os.path.dirname(args.out), "selfconsistency.json")
    json.dump(res, open(summ, "w"), indent=2)
    print(f"[sc-probe] rho={rho:.4f} CI[{boots[50]:.4f},{boots[1949]:.4f}] (n={n})")
    print(f"[sc-probe] single v1.3={single:.4f}  ensemble(3seed,3x)={ensemble:.4f}  ceiling={ceil:.4f}")
    print(f"[sc-probe] delta_vs_single={rho - single:+.4f}  delta_vs_ensemble={rho - ensemble:+.4f}")
    print(f"[sc-probe] wrote {summ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 25 routing-profile driver — serves-model path (replaces the OOMing
in-process profile_v4_judge.py load).

Sends the reasoning dataset at a running `serve_v4_profile_vllm.sh` server as
PREFILL-only requests (max_tokens=1: MoE routes every prompt token during
prefill, so a single generated token is enough). The server's
_sieve_profile_vllm_patch hook accumulates per-layer/per-expert top-k counts and
atomically dumps a [n_layers, n_experts] .npy. This driver reads that dump and
finalizes it through the EXISTING math — RoutingCollector / compute_eeff /
write_profiling_jsonl / compute_jaccard_stability — so the served profile's
routing_report.jsonl + jaccard_stability.json are byte-compatible with the
merged-model profiler's outputs.

Two-pass Jaccard without a server restart: counts are additive, so the subsample
pass is derived by subtraction —
    C_after_full           = counts after the full pass
    C_after_full_plus_sub  = counts after also sending the subsample
    subsample_counts       = C_after_full_plus_sub - C_after_full
Each read waits until the on-disk counts STABILIZE (two equal reads a flush
interval apart), so no dependence on flush timing.

Serve first (separate terminal / detached):
    SIEVE_PROFILE_OUT=$PWD/output/sieve-v4/profile bash scripts/serve_v4_profile_vllm.sh
    # wait for: curl -s localhost:8010/v1/models
Then:
    python scripts/drive_v4_routing_profile.py \
        --port 8010 --served-name judge-v4-s1 \
        --counts-path output/sieve-v4/profile/routing_counts.npy \
        --output-dir output/sieve-v4
    docker stop wp-v4-profile-vllm

Self-check (no server, no GPU):  python scripts/drive_v4_routing_profile.py --self-check
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import sieve_arch  # noqa: E402
from scripts.profile_base_model import (  # noqa: E402
    RoutingCollector,
    write_profiling_jsonl,
)
from scripts.profile_merged_model import compute_jaccard_stability  # noqa: E402

DEFAULT_MODEL_PATH = "models/Qwen3.6-35B-A3B-judge-v4-s1-merged"
DEFAULT_DATA_PATH = "data/reasoning_dataset/openai_train_relabel_v1.jsonl"
DEFAULT_OUTPUT_DIR = "output/sieve-v4"
DEFAULT_COUNTS = "output/sieve-v4/profile/routing_counts.npy"


# --------------------------------------------------------------------------- #
# HTTP (stdlib only)
# --------------------------------------------------------------------------- #

def _post_completion(base_url: str, model: str, prompt: str, timeout: float) -> None:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def _wait_ready(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"server at {base_url} not ready within {timeout_s}s")


# --------------------------------------------------------------------------- #
# Prompt rendering + sending
# --------------------------------------------------------------------------- #

def _render(examples: list, tokenizer) -> list[str]:
    texts = []
    for ex in examples:
        messages = ex.get("messages", [])
        if messages:
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False))
        else:
            texts.append(ex.get("text", ""))
    return texts


def _send_all(base_url: str, model: str, texts: list[str], concurrency: int,
              timeout: float) -> None:
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_post_completion, base_url, model, t, timeout) for t in texts]
        for fut in as_completed(futs):
            fut.result()  # re-raise transport errors loudly
            done += 1
            if done % 200 == 0:
                print(f"  ...{done}/{len(texts)} prompts routed", flush=True)


# --------------------------------------------------------------------------- #
# Counts read (stabilization)
# --------------------------------------------------------------------------- #

def _read_counts(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (ValueError, OSError):
        return None  # mid-write; caller retries


def _wait_stable(path: Path, flush_secs: float, max_wait_s: float = 120.0) -> np.ndarray:
    """Return the counts once two reads a flush-interval apart are identical."""
    deadline = time.monotonic() + max_wait_s
    prev = None
    while time.monotonic() < deadline:
        time.sleep(flush_secs + 1.0)
        cur = _read_counts(path)
        if cur is not None and prev is not None and cur.shape == prev.shape and np.array_equal(cur, prev):
            return cur
        prev = cur
    if prev is None:
        raise TimeoutError(f"counts file never appeared / stabilized: {path}")
    print("  WARNING: counts did not fully stabilize; using last read", file=sys.stderr)
    return prev


# --------------------------------------------------------------------------- #
# Finalize — reconstruct collector, reuse existing math UNCHANGED
# --------------------------------------------------------------------------- #

def _collector_from_counts(counts: np.ndarray, top_k: int) -> RoutingCollector:
    n_layers, n_experts = counts.shape
    c = RoutingCollector(n_layers=n_layers, n_experts=n_experts, top_k=top_k,
                         gen_id=None, judge_id=None)  # total-only (v4 judge)
    for layer_idx in range(n_layers):
        row = counts[layer_idx]
        nz = np.nonzero(row)[0]
        c._counts_total[layer_idx] = {int(e): int(row[e]) for e in nz}
        # n_tokens = total top-k increments / top_k (informational field only)
        c._n_tokens_total[layer_idx] = int(round(float(row.sum()) / max(top_k, 1)))
    return c


def finalize(full_counts: np.ndarray, subsample_counts: np.ndarray, *,
             output_dir: str, model_tag: str, top_k: int, model_config=None) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "routing_report.jsonl"
    jaccard_path = out / "jaccard_stability.json"
    if jsonl_path.exists():
        jsonl_path.unlink()  # write_profiling_jsonl appends; start clean

    collector = _collector_from_counts(full_counts, top_k)
    write_profiling_jsonl(collector, ratio="30_70", subsample_n=int(round(
        float(full_counts.sum()) / max(top_k, 1))), out_path=str(jsonl_path),
        model_tag=model_tag)

    # Per-stratum E_eff (GATE4-02 SC2) — only when we can resolve strata from config.
    strata_eeff: dict = {}
    if model_config is not None:
        import math
        full_eeffs = [collector.get_layer_eeffs(i)[0] for i in range(collector.n_layers)]
        strata = sieve_arch.layer_strata(model_config)
        for name in (sieve_arch.DELTANET_STRATUM, sieve_arch.ATTENTION_STRATUM):
            vals = [full_eeffs[i] for i in range(len(strata))
                    if strata[i] == name and not math.isnan(full_eeffs[i])]
            if vals:
                a = np.array(vals)
                strata_eeff[name] = {"mean": float(a.mean()), "max": float(a.max()),
                                     "var": float(a.var()), "n_layers": len(vals)}
            else:
                strata_eeff[name] = {"mean": None, "max": None, "var": None, "n_layers": 0}

    jaccards = compute_jaccard_stability(full_counts, subsample_counts, top_k=top_k)
    gate_passes = bool(np.all(jaccards >= 0.94))
    jaccard_path.write_text(json.dumps({
        "per_layer_jaccard": jaccards.tolist(), "top_k": top_k,
        "n_layers": int(full_counts.shape[0]), "strata_eeff": strata_eeff,
        "source": "served-vllm-profile", "gate_passes": gate_passes,
    }, indent=2))

    print(f"routing_report.jsonl -> {jsonl_path}")
    print(f"jaccard_stability.json -> {jaccard_path}")
    print(f"Jaccard mean={jaccards.mean():.4f} min={jaccards.min():.4f} "
          f"gate_passes(>=0.94)={gate_passes}")
    return {"jaccards": jaccards, "gate_passes": gate_passes, "strata_eeff": strata_eeff}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(args) -> None:
    from transformers import AutoConfig, AutoTokenizer
    base_url = f"http://localhost:{args.port}"
    model_path = PROJECT_ROOT / args.model_path
    counts_path = PROJECT_ROOT / args.counts_path

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    try:
        model_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as exc:
        print(f"WARNING: config load failed ({exc}); strata E_eff skipped", file=sys.stderr)
        model_config = None

    with open(PROJECT_ROOT / args.data_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(examples)} examples")

    _wait_ready(base_url)
    print("server ready")

    n_full = len(examples) if args.full_limit is None else max(1, min(args.full_limit, len(examples)))
    full_sample = examples[:n_full]
    print(f"FULL pass: {n_full} prompts")
    _send_all(base_url, args.served_name, _render(full_sample, tokenizer),
              args.concurrency, args.timeout)
    full_counts = _wait_stable(counts_path, args.flush_secs).astype(float)

    rng = random.Random(args.seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_sub = max(1, int(len(examples) * args.subsample_frac))
    sub_sample = shuffled[:n_sub]
    print(f"SUBSAMPLE pass: {n_sub} prompts (frac={args.subsample_frac}, seed={args.seed})")
    _send_all(base_url, args.served_name, _render(sub_sample, tokenizer),
              args.concurrency, args.timeout)
    cumulative = _wait_stable(counts_path, args.flush_secs).astype(float)
    subsample_counts = np.clip(cumulative - full_counts, 0, None)

    finalize(full_counts, subsample_counts, output_dir=str(PROJECT_ROOT / args.output_dir),
             model_tag=args.model_tag, top_k=args.top_k, model_config=model_config)
    print("DONE — stop the server: docker stop wp-v4-profile-vllm")


# --------------------------------------------------------------------------- #
# Self-check (assert-based, no server, no GPU)
# --------------------------------------------------------------------------- #

def _self_check() -> None:
    import math
    n_layers, n_experts, top_k = 4, 16, 8

    # Uniform routing -> E_eff == n_experts; single-expert -> E_eff == 1.
    uniform = np.full((n_layers, n_experts), 10.0)
    col = _collector_from_counts(uniform, top_k)
    e_uniform = col.get_layer_eeffs(0)[0]
    assert abs(e_uniform - n_experts) < 1e-6, e_uniform

    concentrated = np.zeros((n_layers, n_experts)); concentrated[:, 0] = 100.0
    col2 = _collector_from_counts(concentrated, top_k)
    e_conc = col2.get_layer_eeffs(0)[0]
    assert abs(e_conc - 1.0) < 1e-6, e_conc

    # Reconstruction round-trips the nonzero counts exactly.
    counts = np.zeros((n_layers, n_experts)); counts[1, 3] = 7; counts[1, 9] = 5
    c3 = _collector_from_counts(counts, top_k)
    assert c3._counts_total[1] == {3: 7, 9: 5}, c3._counts_total[1]

    # Additive subtraction: sub = (full+sub) - full, clipped >= 0.
    full = np.array([[5.0, 0.0], [1.0, 2.0]])
    after = np.array([[8.0, 1.0], [1.0, 5.0]])
    sub = np.clip(after - full, 0, None)
    assert np.array_equal(sub, np.array([[3.0, 1.0], [0.0, 3.0]])), sub

    # Jaccard of identical top-k == 1.0 on every layer.
    j = compute_jaccard_stability(uniform, uniform, top_k=top_k)
    assert np.allclose(j, 1.0), j

    assert not math.isnan(e_uniform)
    print("self-check OK: eeff(uniform)=%.4f eeff(conc)=%.4f, reconstruct+subtract+jaccard verified"
          % (e_uniform, e_conc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--served-name", default="judge-v4-s1")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--counts-path", default=DEFAULT_COUNTS)
    p.add_argument("--full-limit", type=int, default=None)
    p.add_argument("--subsample-frac", type=float, default=0.10)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--flush-secs", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-tag", default="judge-v4-s1-served")
    args = p.parse_args()

    if args.self_check:
        _self_check()
        return
    run(args)


if __name__ == "__main__":
    main()

"""v4 judge routing profile driver (GATE4-03, Phase 25 Plan 01, Task 1).

Minimal driver: loads the v4 judge SFT s1 merged checkpoint with the
22-02-proven AutoModelForImageTextToText recipe (NOT AutoModelForCausalLM --
see LOADER GUARD below), then hands off to the already-audited
profile_merged_model.profile_merged_model() for the E_eff / Jaccard /
per-stratum-E_eff logic (RoutingCollector, compute_eeff, compute_jaccard_stability,
strata_eeff). No routing/E_eff logic is re-derived here.

LOADER GUARD (T-25-01, load-bearing): before spending the GB10 load, a
meta-device state_dict key-diff against model.safetensors.index.json asserts
0 missing keys under AutoModelForImageTextToText. AutoModelForCausalLM
resolves the flat-text Qwen3_5MoeForCausalLM class, which has 692/693 keys
missing against this nested (model.language_model.layers.*) VL-composite
checkpoint -- that would silently produce a randomly-initialized forward pass
that still passes every shape-based assert (the exact 22-02 finding).

Usage:
    .venv-tinker/bin/python scripts/profile_v4_judge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from scripts import sieve_arch  # noqa: E402
from scripts.profile_merged_model import profile_merged_model  # noqa: E402

DEFAULT_MODEL_PATH = "models/Qwen3.6-35B-A3B-judge-v4-s1-merged"
DEFAULT_DATA_PATH = "data/reasoning_dataset/openai_train_relabel_v1.jsonl"
DEFAULT_OUTPUT_DIR = "output/sieve-v4"
DEFAULT_MODEL_TAG = "judge-v4-s1"
# Raised from the tooling-smoke default (1024) -- a spot-check of this data
# file's token-length distribution found 33.9% of examples exceed 1024 tokens
# (mean 895, p95 1518, max 7807) under this checkpoint's tokenizer, vs. only
# 0.4% exceeding 2048. 2048 is also profile_merged_model.py's own default.
DEFAULT_MAX_SEQ_LEN = 2048


def _assert_loader_keys_match(model_path: Path) -> None:
    """Meta-device state_dict key-diff (T-25-01): 0 missing keys or raise.

    Reuses the 22-02 technique: instantiate the model on torch.device("meta")
    (no weights materialized) under the same AutoModel class the real load
    will use, and diff its state_dict key set against the checkpoint's
    model.safetensors.index.json weight_map. A random-weight mis-load (e.g.
    AutoModelForCausalLM resolving the flat-text class) would leave hundreds
    of keys missing while still satisfying every downstream shape assert --
    this guard fails loudly BEFORE the real ~4min/67GiB GB10 load.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        ckpt_keys = set(json.load(f)["weight_map"].keys())

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    with torch.device("meta"):
        meta_model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
    model_keys = set(meta_model.state_dict().keys())
    del meta_model

    missing = model_keys - ckpt_keys
    if missing:
        raise RuntimeError(
            f"LOADER GUARD FAILED (T-25-01): {len(missing)} AutoModelForImageTextToText "
            f"keys missing from checkpoint {model_path} -- this load would silently "
            f"leave those weights randomly initialized. First 5 missing: "
            f"{sorted(missing)[:5]}"
        )
    print(
        f"Loader guard OK (T-25-01): 0 missing keys "
        f"({len(model_keys)} model keys, {len(ckpt_keys)} checkpoint keys, "
        f"{len(ckpt_keys - model_keys)} checkpoint-only keys e.g. mtp.* ignored)"
    )


def main():
    parser = argparse.ArgumentParser(description="Profile the v4 judge's 256-expert routing (GATE4-03)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--subsample", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the FULL reference pass (default: None = use every example in --data-path)",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("ERROR: torch.cuda.is_available() is False -- refusing CPU forward pass on a 35B model.")
        sys.exit(2)

    model_path = PROJECT_ROOT / args.model_path
    data_path = PROJECT_ROOT / args.data_path

    print(f"[1/4] Loader guard (T-25-01) against {model_path} ...")
    _assert_loader_keys_match(model_path)

    print(f"[2/4] Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print(f"[3/4] Loading model from {model_path} (AutoModelForImageTextToText, bf16, single-device GB10-safe) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        **sieve_arch.gb10_load_kwargs(),  # single-device + low_cpu_mem_usage; NOT device_map="auto" (GB10 OOM trap)
        trust_remote_code=True,
    )
    model.eval()

    print(f"[4/4] Profiling over {data_path} (max_seq_len={args.max_seq_len}, limit={args.limit}) ...")
    result = profile_merged_model(
        model=model,
        tokenizer=tokenizer,
        data_path=str(data_path),
        full_limit=args.limit,
        subsample_frac=args.subsample,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        output_dir=str(PROJECT_ROOT / args.output_dir),
        model_tag=args.model_tag,
    )
    print("strata_eeff:", json.dumps(result["strata_eeff"], indent=2))
    print("DONE")


if __name__ == "__main__":
    main()

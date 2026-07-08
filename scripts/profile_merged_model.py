"""Merged-model E_eff profiling script for Qwen3-30B-A3B MoE router.

Hooks Qwen3MoeTopKRouter gates on all 48 layers to capture per-expert
routing counts split by <wp_gen> vs <wp_judge> tokens. Computes E_eff
(effective number of experts via Shannon entropy) per layer.

Unlike profile_base_model.py, this script:
  - Loads a MERGED checkpoint (no adapter, no PEFT wrapper)
  - Defaults to the ratio_30_70 stimulus (matched to baseline)
  - Computes subsample-vs-FULL Jaccard stability (PROF-03, D-06 literal)
  - Emits raw per-layer Jaccard array to jaccard_stability.json sidecar

The CI-aware gate (ci_lower >= 0.94) lives in compute_concentration.py (Task 2).
This script does NOT import the CI helper or compute the ci gate value directly.

Output (default):
  - output/profiling/reasoning-merged-v4/routing_report.jsonl
  - output/profiling/reasoning-merged-v4/jaccard_stability.json

Usage:
    python -m scripts.profile_merged_model \\
        --model-path models/qwen3-30b-wp-30_70-reasoning-merged-v4 \\
        --ratio ratio_30_70
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from scripts.profile_base_model import (
    RoutingCollector,
    PAD_TOKEN_ID,
    WP_GEN_ID,
    WP_JUDGE_ID,
    compute_eeff,
    discover_dataset_dirs,
    write_profiling_jsonl,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_TAG = "reasoning-merged-v4"
DEFAULT_OUTPUT_DIR = "output/profiling/reasoning-merged-v4"
BASE_OUTPUT_DIR = "output/profiling"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jaccard stability (PROF-03, D-06 literal: subsample-vs-FULL)
# ---------------------------------------------------------------------------


def compute_jaccard_stability(
    full_counts: np.ndarray,
    subsample_counts: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Compute per-layer Jaccard similarity between subsample and full-set top-K experts.

    Implements D-06 literal: subsample_counts vs full_counts (not A-vs-B cross-subsample).
    The reference ranking is the FULL stimulus; the test is a single subsample.

    Args:
        full_counts: [n_layers, n_experts] float array — full-set expert routing counts.
        subsample_counts: [n_layers, n_experts] float array — subsample expert routing counts.
        top_k: Number of top experts to compare (per layer).

    Returns:
        np.ndarray of shape (n_layers,): per-layer Jaccard(subsample_topK, full_topK).
        Returns 1.0 for any layer where both sets are identical (including all-zero layers).
    """
    n_layers = full_counts.shape[0]
    jaccards = np.empty(n_layers, dtype=float)

    for layer in range(n_layers):
        full_layer = full_counts[layer]
        sub_layer = subsample_counts[layer]

        # Use argsort (no filter) — identical inputs yield identical deterministic top-k
        full_topk = set(np.argsort(full_layer)[-top_k:].tolist())
        sub_topk = set(np.argsort(sub_layer)[-top_k:].tolist())

        intersection = len(full_topk & sub_topk)
        union = len(full_topk | sub_topk)

        if union == 0:
            jaccards[layer] = 1.0  # both sets empty -> trivially identical
        else:
            jaccards[layer] = intersection / union

    return jaccards


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------


def profile_merged_model(
    model,
    tokenizer,
    data_path: str,
    full_subsample_frac: float = 1.0,
    full_limit: int | None = None,
    subsample_frac: float = 0.10,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_tag: str = DEFAULT_MODEL_TAG,
    top_k_jaccard: int = 8,
) -> dict:
    """Profile merged model E_eff on the ratio_30_70 matched stimulus.

    Registers forward hooks on all 48 MoE layers. Runs the FULL stimulus to
    produce the reference expert ranking, then a single 10% subsample to
    compute per-layer Jaccard(subsample_topK, full_topK) — PROF-03 D-06 literal.

    Emits:
      - routing_report.jsonl  (per-layer expert counts + E_eff)
      - jaccard_stability.json  (raw 48-element per-layer Jaccard array)

    Args:
        model: Loaded merged Qwen3-30B-A3B model (bfloat16, eval mode, no PEFT wrapper).
        tokenizer: Extended tokenizer from adapters/tokenizer/.
        data_path: Path to openai_train.jsonl (ratio_30_70 matched stimulus).
        full_subsample_frac: Fraction for the full reference pass (default 1.0 = all).
        full_limit: Optional hard cap (example count) on the full reference pass,
            takes precedence over full_subsample_frac when set. Use for a bounded
            stimulus (e.g. cross-seed comparison) instead of the full corpus pass.
        subsample_frac: Fraction for the Jaccard subsample (default 0.10).
        batch_size: Forward pass batch size (default 1 for memory safety).
        max_seq_len: Maximum sequence length for tokenization.
        output_dir: Directory to write JSONL and sidecar JSON.
        model_tag: Tag written into every JSONL record "model" field.
        top_k_jaccard: top-K for Jaccard overlap computation.

    Returns:
        dict with keys: "full_counts", "subsample_counts", "jaccards"
    """
    import random
    import torch

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir_path / "routing_report.jsonl"
    jaccard_path = output_dir_path / "jaccard_stability.json"

    # Initialize collector
    pad_id = getattr(tokenizer, "pad_token_id", PAD_TOKEN_ID) or PAD_TOKEN_ID
    collector = RoutingCollector(
        n_layers=48, n_experts=128, top_k=top_k_jaccard, pad_token_id=pad_id
    )

    # Register hooks — merged model has no PEFT wrapper
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hooks = []
    for i, layer in enumerate(base.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            h = layer.mlp.gate.register_forward_hook(collector.make_hook(i))
            hooks.append(h)
    logger.info(f"Registered {len(hooks)} hooks (expected 48)")

    model.eval()

    # Load dataset
    with open(data_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(examples)} examples from {data_path}")

    def _run_pass(sample: list) -> np.ndarray:
        """Run a forward pass over sample and return [48, 128] count array."""
        collector.reset()
        for batch_start in range(0, len(sample), batch_size):
            batch = sample[batch_start: batch_start + batch_size]
            texts = []
            for ex in batch:
                messages = ex.get("messages", [])
                if messages:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    text = ex.get("text", "")
                texts.append(text)

            enc = tokenizer(
                texts,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=True,
                padding=True,
            )
            input_ids = enc["input_ids"]
            collector.set_token_types(input_ids)

            with torch.no_grad():
                model(input_ids=input_ids.to(model.device))

        # Build [48, 128] count array from collector
        counts = np.zeros((48, 128), dtype=float)
        for layer_idx in range(48):
            for expert_id, count in collector._counts_total[layer_idx].items():
                if 0 <= int(expert_id) < 128:
                    counts[layer_idx, int(expert_id)] = count
        return counts

    try:
        # --- FULL PASS (reference ranking) ---
        if full_limit is not None:
            n_full = max(1, min(full_limit, len(examples)))
        else:
            n_full = max(1, int(len(examples) * full_subsample_frac))
        full_sample = examples[:n_full]
        logger.info(f"Running FULL pass: {n_full} examples")
        full_counts = _run_pass(full_sample)

        # Write routing JSONL from the full pass (the reference profile)
        # Ratio key normalized to "30_70" (D-08 seam: strip "ratio_" prefix)
        normalized_ratio = "30_70"
        write_profiling_jsonl(
            collector,
            ratio=normalized_ratio,
            subsample_n=n_full,
            out_path=str(jsonl_path),
            model_tag=model_tag,
        )
        logger.info(f"Wrote routing JSONL: {jsonl_path}")

        # --- SUBSAMPLE PASS (Jaccard test) ---
        n_subsample = max(1, int(len(examples) * subsample_frac))
        random.shuffle(examples)
        sub_sample = examples[:n_subsample]
        logger.info(f"Running SUBSAMPLE pass: {n_subsample} examples (D-06 literal)")
        subsample_counts = _run_pass(sub_sample)

        # --- JACCARD STABILITY (PROF-03) ---
        jaccards = compute_jaccard_stability(full_counts, subsample_counts, top_k=top_k_jaccard)
        gate_passes = bool(np.all(jaccards >= 0.94))
        logger.info(
            f"Jaccard stability: mean={jaccards.mean():.4f}, "
            f"min={jaccards.min():.4f}, gate_passes={gate_passes}"
        )

        # Write Jaccard sidecar (raw per-layer array — CI gate is in compute_concentration.py)
        jaccard_data = {
            "per_layer_jaccard": jaccards.tolist(),
            "top_k": top_k_jaccard,
            "n_layers": 48,
        }
        with open(jaccard_path, "w") as f:
            json.dump(jaccard_data, f, indent=2)
        logger.info(f"Wrote jaccard_stability.json: {jaccard_path}")

    finally:
        for h in hooks:
            h.remove()

    return {
        "full_counts": full_counts,
        "subsample_counts": subsample_counts,
        "jaccards": jaccards,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Profile merged model E_eff + Jaccard stability (PROF-03)"
    )
    parser.add_argument(
        "--model-path",
        default="models/qwen3-30b-wp-30_70-reasoning-merged-v4",
        help="Path to merged Qwen3-30B checkpoint (no adapter)",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="adapters/tokenizer",
        help="Path to extended tokenizer (with <wp_gen>, <wp_judge> tokens)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write routing_report.jsonl and jaccard_stability.json",
    )
    parser.add_argument(
        "--ratio",
        default="ratio_30_70",
        help="Dataset key from discover_dataset_dirs to profile (default: ratio_30_70)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.10,
        help="Fraction for Jaccard subsample (default 0.10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the FULL reference pass at this many examples (bounded stimulus). "
             "Default: use the full discovered dataset (34,855 for ratio_30_70).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Forward pass batch size (default 1)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution (NOT recommended for 30B model)",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Tag written into each JSONL record's 'model' field. "
             f"Default: derived from --output-dir basename (falls back to {DEFAULT_MODEL_TAG!r}).",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # CUDA guard — merged 30B model requires GPU
    if not torch.cuda.is_available() and not args.allow_cpu:
        print(
            "ERROR: torch.cuda.is_available() is False — refusing to run a 30B forward "
            "pass on CPU. Open container via 'bash deps/dgx-toolbox/containers/ngc-pytorch.sh' "
            "and re-run. Pass --allow-cpu to override (NOT recommended)."
        )
        sys.exit(2)

    project_root = Path(__file__).resolve().parent.parent

    # Path-collision guard: refuse to write over base_model_eeff.jsonl (T-07-01)
    output_dir = project_root / args.output_dir
    base_output = project_root / BASE_OUTPUT_DIR
    if output_dir.resolve() == base_output.resolve():
        print(
            f"ERROR: --output-dir resolves to the base profiling directory "
            f"({base_output}). This would overwrite base_model_eeff.jsonl. "
            f"Use a subdirectory such as output/profiling/reasoning-merged-v4."
        )
        sys.exit(1)

    model_path = project_root / args.model_path
    tokenizer_path = project_root / args.tokenizer_path

    # Discover dataset directories (ratio_30_70 stimulus — D-05 amended)
    ratio_data_paths = discover_dataset_dirs(project_root / "data" / "final_dataset")
    if not ratio_data_paths:
        print("ERROR: No dataset directories with openai_train.jsonl found under data/final_dataset/")
        sys.exit(1)

    ratio_key = args.ratio
    if ratio_key not in ratio_data_paths:
        print(
            f"ERROR: --ratio {ratio_key!r} not in discovered datasets: "
            f"{list(ratio_data_paths.keys())}"
        )
        sys.exit(1)

    data_path = ratio_data_paths[ratio_key]
    # Normalize ratio key (D-08): strip "ratio_" prefix so output uses "30_70"
    normalized_ratio = ratio_key.removeprefix("ratio_")
    logger.info(f"Stimulus: {data_path} (ratio key: {normalized_ratio})")

    # Load extended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Load merged model — NO PEFT wrapper (v4 is a merged checkpoint)
    print(f"Loading merged model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    model_tag = args.model_tag or output_dir.name or DEFAULT_MODEL_TAG

    profile_merged_model(
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        full_limit=args.limit,
        subsample_frac=args.subsample,
        batch_size=args.batch_size,
        output_dir=str(output_dir),
        model_tag=model_tag,
    )


if __name__ == "__main__":
    main()

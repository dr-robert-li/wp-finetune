"""Base-model E_eff profiling script for Qwen3-30B-A3B MoE router.

Hooks Qwen3MoeTopKRouter gates on all 48 layers to capture per-expert
routing counts split by <wp_gen> vs <wp_judge> tokens. Computes E_eff
(effective number of experts via Shannon entropy) per layer per ratio.

Output:
  - output/profiling/base_model_eeff.jsonl   (Phase 7-compatible JSONL)
  - output/profiling/base_model_eeff_summary.md

Usage:
    python -m scripts.profile_base_model
    python -m scripts.profile_base_model --model-path models/Qwen3-30B-A3B
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WP_GEN_ID = 151669    # <wp_gen> token ID in extended tokenizer
WP_JUDGE_ID = 151670  # <wp_judge> token ID in extended tokenizer
PAD_TOKEN_ID = 151643  # Qwen3 pad token -- also accept tokenizer.pad_token_id at runtime

# Legacy constant — kept for backward compat with tests/imports. New code
# should use discover_dataset_dirs() instead.
RATIO_ORDER = ["30_70", "40_60", "50_50", "60_40", "70_30"]


def discover_dataset_dirs(final_dataset_dir: Path) -> dict[str, str]:
    """Auto-discover dataset directories containing openai_train.jsonl.

    Searches both the root directory and any subdirectories (ratio_*, experiment_*, etc.).
    Returns dict mapping directory name to the openai_train.jsonl path.
    """
    results = {}
    # Check root
    root_train = final_dataset_dir / "openai_train.jsonl"
    if root_train.exists():
        results["current"] = str(root_train)
    # Check subdirectories
    if final_dataset_dir.exists():
        for d in sorted(final_dataset_dir.iterdir()):
            if d.is_dir():
                train_file = d / "openai_train.jsonl"
                if train_file.exists():
                    results[d.name] = str(train_file)
    return results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# E_eff computation
# ---------------------------------------------------------------------------


def compute_eeff(expert_counts: dict, n_experts: int = 128) -> float:
    """Compute E_eff = exp(Shannon entropy) from expert routing counts.

    Args:
        expert_counts: Dict mapping expert_id -> count (any int key, non-negative values).
        n_experts: Total number of experts (used for output range context only).

    Returns:
        E_eff as float. Returns float('nan') if total == 0 (no data).
        E_eff == 1.0 means fully concentrated (one expert).
        E_eff == n_experts means perfectly uniform.

    Note:
        Zero-count return is float('nan'), NOT n_experts. NaN signals "no data"
        and is excluded from downstream aggregation via np.nanmean / np.nanmax / np.nanvar.
    """
    total = sum(expert_counts.values()) if expert_counts else 0
    if total == 0:
        return float("nan")

    # Build probability array over expert IDs present
    counts_arr = np.array(list(expert_counts.values()), dtype=float)
    p = counts_arr[counts_arr > 0] / total
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ---------------------------------------------------------------------------
# Token-type tagging
# ---------------------------------------------------------------------------


class RoutingCollector:
    """Accumulates per-expert routing counts split by token type.

    Usage:
        collector = RoutingCollector(n_layers=48, n_experts=128, top_k=8)
        for ratio in ratios:
            collector.reset()
            for batch in dataloader:
                collector.set_token_types(batch['input_ids'])
                model(**batch)   # hooks fire during forward
            write_profiling_jsonl(collector, ratio, ...)
    """

    def __init__(
        self,
        n_layers: int = 48,
        n_experts: int = 128,
        top_k: int = 8,
        pad_token_id: int = PAD_TOKEN_ID,
    ):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.pad_token_id = pad_token_id

        # Per-layer expert count dicts
        self._counts_total: list[dict] = [defaultdict(int) for _ in range(n_layers)]
        self._counts_wp_gen: list[dict] = [defaultdict(int) for _ in range(n_layers)]
        self._counts_wp_judge: list[dict] = [defaultdict(int) for _ in range(n_layers)]

        # Token counts per layer
        self._n_tokens_total: list[int] = [0] * n_layers
        self._n_tokens_wp_gen: list[int] = [0] * n_layers
        self._n_tokens_wp_judge: list[int] = [0] * n_layers

        # Current token type list (set before each forward pass)
        self._current_token_types: list[str] = []

    def reset(self) -> None:
        """Clear all accumulated counts (call between ratios)."""
        self._counts_total = [defaultdict(int) for _ in range(self.n_layers)]
        self._counts_wp_gen = [defaultdict(int) for _ in range(self.n_layers)]
        self._counts_wp_judge = [defaultdict(int) for _ in range(self.n_layers)]
        self._n_tokens_total = [0] * self.n_layers
        self._n_tokens_wp_gen = [0] * self.n_layers
        self._n_tokens_wp_judge = [0] * self.n_layers
        self._current_token_types = []

    def set_token_types(self, input_ids) -> None:
        """Scan input_ids to produce per-position token type tags.

        Tags:
          - "wp_gen": positions after (and including) a <wp_gen> token
          - "wp_judge": positions after (and including) a <wp_judge> token
          - "pad": positions with pad_token_id (excluded from routing counts)
          - "other": positions before any task token

        Padding tokens are ALWAYS tagged "pad" regardless of preceding context.
        Padding is excluded from routing counts in make_hook().

        Args:
            input_ids: torch.Tensor of shape [batch_size, seq_len] or [seq_len].
                       Will be flattened to 1D.
        """
        flat = input_ids.view(-1).tolist()
        types = []
        current_type = "other"
        for tid in flat:
            if tid == self.pad_token_id:
                # Padding always tagged as "pad" regardless of context
                types.append("pad")
                continue
            if tid == WP_GEN_ID:
                current_type = "wp_gen"
            elif tid == WP_JUDGE_ID:
                current_type = "wp_judge"
            types.append(current_type)
        self._current_token_types = types

    def make_hook(self, layer_idx: int):
        """Create a forward hook for the router at the given layer.

        The hook captures outputs[2] (router_indices, NOT outputs[1] which is scores).
        router_indices shape: [n_tokens, top_k] dtype int64.

        Padding tokens (tok_type == 'pad') are SKIPPED -- not counted in any bucket.

        Args:
            layer_idx: 0-based layer index (0 to n_layers-1).

        Returns:
            Hook function for use with register_forward_hook().
        """
        def hook(module, inputs, outputs):
            # self.gate is nn.Linear — outputs is a single tensor of router_logits
            # shape: [n_tokens, n_experts]. Compute top-k indices from logits.
            # Handle both real model (single tensor) and test mock (tuple) cases.
            import torch
            if isinstance(outputs, tuple):
                # Test mock passes (logits, scores, indices) — use indices directly
                router_indices = outputs[2]
            else:
                # Real model: gate outputs raw logits tensor
                router_indices = torch.topk(outputs, k=self.top_k, dim=-1).indices
            token_types = self._current_token_types
            n_tokens = router_indices.shape[0]

            for tok_pos in range(n_tokens):
                # Safety: handle truncated sequences
                if tok_pos < len(token_types):
                    tok_type = token_types[tok_pos]
                else:
                    tok_type = "other"

                # Skip padding tokens entirely
                if tok_type == "pad":
                    continue

                # Increment total token count for this layer
                self._n_tokens_total[layer_idx] += 1
                if tok_type == "wp_gen":
                    self._n_tokens_wp_gen[layer_idx] += 1
                elif tok_type == "wp_judge":
                    self._n_tokens_wp_judge[layer_idx] += 1

                # Increment expert counts
                for expert_id in router_indices[tok_pos].tolist():
                    self._counts_total[layer_idx][expert_id] += 1
                    if tok_type == "wp_gen":
                        self._counts_wp_gen[layer_idx][expert_id] += 1
                    elif tok_type == "wp_judge":
                        self._counts_wp_judge[layer_idx][expert_id] += 1

        return hook

    def get_layer_eeffs(self, layer_idx: int) -> tuple[float, float, float]:
        """Return (eeff_total, eeff_wp_gen, eeff_wp_judge) for one layer."""
        return (
            compute_eeff(dict(self._counts_total[layer_idx])),
            compute_eeff(dict(self._counts_wp_gen[layer_idx])),
            compute_eeff(dict(self._counts_wp_judge[layer_idx])),
        )


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def _nan_to_null(value: float):
    """Convert NaN float to None (serializes as JSON null), else return value."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_profiling_jsonl(
    collector: RoutingCollector,
    ratio: str,
    subsample_n: int,
    out_path: str,
) -> None:
    """Write per-layer profiling data as JSONL records.

    Writes one record per layer (n_layers records total).
    NaN E_eff values are serialized as JSON null.

    Schema (Phase 7 compatible):
        ratio, layer_idx, n_tokens_total, n_tokens_wp_gen, n_tokens_wp_judge,
        expert_counts_total, expert_counts_wp_gen, expert_counts_wp_judge,
        eeff_total, eeff_wp_gen, eeff_wp_judge, subsample_n, model

    Args:
        collector: RoutingCollector with accumulated counts for this ratio.
        ratio: Ratio string (e.g., "30_70").
        subsample_n: Number of examples used (for reporting).
        out_path: Path to output JSONL file (appended or created).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode) as f:
        for layer_idx in range(collector.n_layers):
            eeff_total, eeff_wp_gen, eeff_wp_judge = collector.get_layer_eeffs(layer_idx)

            # Convert expert_counts dicts from defaultdict to plain dicts with string keys
            record = {
                "ratio": ratio,
                "layer_idx": layer_idx,
                "n_tokens_total": collector._n_tokens_total[layer_idx],
                "n_tokens_wp_gen": collector._n_tokens_wp_gen[layer_idx],
                "n_tokens_wp_judge": collector._n_tokens_wp_judge[layer_idx],
                "expert_counts_total": {
                    str(k): v for k, v in collector._counts_total[layer_idx].items()
                },
                "expert_counts_wp_gen": {
                    str(k): v for k, v in collector._counts_wp_gen[layer_idx].items()
                },
                "expert_counts_wp_judge": {
                    str(k): v for k, v in collector._counts_wp_judge[layer_idx].items()
                },
                "eeff_total": _nan_to_null(eeff_total),
                "eeff_wp_gen": _nan_to_null(eeff_wp_gen),
                "eeff_wp_judge": _nan_to_null(eeff_wp_judge),
                "subsample_n": subsample_n,
                "model": "base",
            }
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def write_summary_md(
    all_ratio_eeffs: dict[str, dict[str, list[float]]],
    out_path: str,
) -> None:
    """Write markdown summary table of E_eff statistics per ratio.

    NaN values are excluded from mean/max/variance computations.
    If ALL values for a cell are NaN, displays "N/A".

    Args:
        all_ratio_eeffs: Dict mapping ratio -> {"eeff_total": [...], "eeff_wp_gen": [...], "eeff_wp_judge": [...]}
                         Each list has one float per layer (may contain NaN).
        out_path: Path to output markdown file.
    """

    def safe_stat(values: list[float], fn) -> str:
        """Apply fn to non-NaN values; return 'N/A' if none."""
        valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
        if not valid:
            return "N/A"
        arr = np.array(valid)
        return f"{fn(arr):.2f}"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Base-Model E_eff Profiling Summary\n",
        "| Ratio | E_eff Mean (Total) | E_eff Mean (Gen) | E_eff Mean (Judge) | E_eff Max | E_eff Variance |",
        "|-------|--------------------|------------------|--------------------|-----------|----------------|",
    ]

    for ratio in RATIO_ORDER:
        if ratio not in all_ratio_eeffs:
            continue
        data = all_ratio_eeffs[ratio]
        total_vals = data.get("eeff_total", [])
        gen_vals = data.get("eeff_wp_gen", [])
        judge_vals = data.get("eeff_wp_judge", [])

        mean_total = safe_stat(total_vals, np.nanmean)
        mean_gen = safe_stat(gen_vals, np.nanmean)
        mean_judge = safe_stat(judge_vals, np.nanmean)
        max_total = safe_stat(total_vals, np.nanmax)
        var_total = safe_stat(total_vals, np.nanvar)

        lines.append(
            f"| {ratio} | {mean_total} | {mean_gen} | {mean_judge} | {max_total} | {var_total} |"
        )

    lines.append("")  # trailing newline

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------


def has_downward_eeff_trend(ratio_eeffs: list[float]) -> bool:
    """Return True if E_eff decreases at any step as gen% increases.

    The ordering is assumed to be 30_70 -> 40_60 -> 50_50 -> 60_40 -> 70_30
    (increasing gen%). NaN values are skipped.

    Returns True if ANY adjacent non-NaN pair shows a decrease.
    Returns False if no adjacent non-NaN pairs exist or all are flat/increasing.

    Args:
        ratio_eeffs: List of E_eff values in ratio order (may contain NaN).

    Returns:
        bool: True if any downward step detected.
    """
    non_nan = [(i, v) for i, v in enumerate(ratio_eeffs) if not (isinstance(v, float) and math.isnan(v))]
    if len(non_nan) < 2:
        return False
    for i in range(len(non_nan) - 1):
        _, v1 = non_nan[i]
        _, v2 = non_nan[i + 1]
        if v2 < v1:
            return True
    return False


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------


def profile_base_model(
    model,
    tokenizer,
    ratio_data_paths: dict[str, str],
    subsample_frac: float = 0.10,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    output_dir: str = "output/profiling",
) -> dict[str, dict]:
    """Profile base model E_eff across all ratio data distributions.

    Registers forward hooks on all 48 MoE layers, runs 10% subsample of each
    ratio dataset through the model, and collects per-expert routing counts
    split by token type (wp_gen / wp_judge / other / pad).

    Args:
        model: Loaded Qwen3-30B-A3B model (bfloat16, eval mode).
        tokenizer: Extended tokenizer from adapters/tokenizer/.
        ratio_data_paths: Dict mapping ratio string to JSONL data file path.
            e.g. {"30_70": "data/final_dataset/ratio_30_70/openai_train.jsonl"}
        subsample_frac: Fraction of data to use (default 0.10, MoE-Sieve paper).
        batch_size: Forward pass batch size (default 1 for memory safety).
        max_seq_len: Maximum sequence length for tokenization.
        output_dir: Directory to write JSONL and summary markdown.

    Returns:
        Dict: ratio -> {"eeff_total": [...], "eeff_wp_gen": [...], "eeff_wp_judge": [...]}
              for trend analysis.
    """
    import random
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "base_model_eeff.jsonl"
    summary_path = output_dir / "base_model_eeff_summary.md"

    # Initialize collector using runtime tokenizer pad_token_id
    pad_id = getattr(tokenizer, "pad_token_id", PAD_TOKEN_ID) or PAD_TOKEN_ID
    collector = RoutingCollector(
        n_layers=48, n_experts=128, top_k=8, pad_token_id=pad_id
    )

    # Register hooks on all MoE layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            h = layer.mlp.gate.register_forward_hook(collector.make_hook(i))
            hooks.append(h)

    model.eval()
    all_ratio_eeffs = {}

    try:
        for ratio in RATIO_ORDER:
            if ratio not in ratio_data_paths:
                logger.warning(f"Ratio {ratio} not found in ratio_data_paths, skipping")
                continue

            data_path = ratio_data_paths[ratio]
            logger.info(f"Profiling ratio {ratio} from {data_path}")

            # Load subsample
            with open(data_path) as f:
                examples = [json.loads(line) for line in f if line.strip()]

            n_subsample = max(1, int(len(examples) * subsample_frac))
            random.shuffle(examples)
            subsample = examples[:n_subsample]
            logger.info(f"  Subsample: {n_subsample} / {len(examples)} examples")

            # Reset collector for this ratio
            collector.reset()

            # Token type counters for debugging
            tok_type_debug = {"wp_gen": 0, "wp_judge": 0, "other": 0, "pad": 0}

            # Process in batches
            for batch_start in range(0, len(subsample), batch_size):
                batch = subsample[batch_start: batch_start + batch_size]

                # Format messages using chat template
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

                # Tokenize
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=max_seq_len,
                    truncation=True,
                    padding=True,
                )

                input_ids = enc["input_ids"]
                collector.set_token_types(input_ids)

                # Debug: count token types in this batch
                for t in collector._current_token_types:
                    tok_type_debug[t] = tok_type_debug.get(t, 0) + 1

                with torch.no_grad():
                    model(input_ids=input_ids.to(model.device))

            # Log token type distribution for debugging
            logger.info(
                f"  Token type distribution for {ratio}: "
                f"wp_gen={tok_type_debug['wp_gen']}, "
                f"wp_judge={tok_type_debug['wp_judge']}, "
                f"other={tok_type_debug['other']}, "
                f"pad={tok_type_debug['pad']}"
            )

            # Write JSONL for this ratio
            write_profiling_jsonl(collector, ratio=ratio, subsample_n=n_subsample, out_path=str(jsonl_path))

            # Collect E_eff values for trend analysis
            eeffs_total = []
            eeffs_gen = []
            eeffs_judge = []
            for layer_idx in range(48):
                et, eg, ej = collector.get_layer_eeffs(layer_idx)
                eeffs_total.append(et)
                eeffs_gen.append(eg)
                eeffs_judge.append(ej)

            all_ratio_eeffs[ratio] = {
                "eeff_total": eeffs_total,
                "eeff_wp_gen": eeffs_gen,
                "eeff_wp_judge": eeffs_judge,
            }

    finally:
        # Always remove hooks even if an error occurs
        for h in hooks:
            h.remove()

    # Write summary markdown
    write_summary_md(all_ratio_eeffs, str(summary_path))
    logger.info(f"Profiling complete. JSONL: {jsonl_path}, Summary: {summary_path}")

    return all_ratio_eeffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Profile base model E_eff across ratio datasets")
    parser.add_argument(
        "--model-path",
        default="models/Qwen3-30B-A3B",
        help="Path to Qwen3-30B-A3B base model directory",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="adapters/tokenizer",
        help="Path to extended tokenizer (with <wp_gen>, <wp_judge> tokens)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/profiling",
        help="Directory to write JSONL and summary markdown",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.10,
        help="Fraction of data to use per ratio (default 0.10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Forward pass batch size (default 1)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter path. Loaded via PeftModel.from_pretrained() on top of base.",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / args.model_path
    tokenizer_path = project_root / args.tokenizer_path

    # Load extended tokenizer (NOT base tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    # Auto-discover dataset directories (ratio_*, experiment_*, root)
    ratio_data_paths = discover_dataset_dirs(project_root / "data" / "final_dataset")
    if not ratio_data_paths:
        print("ERROR: No dataset directories with openai_train.jsonl found")
        sys.exit(1)
    print(f"Discovered {len(ratio_data_paths)} dataset(s): {list(ratio_data_paths.keys())}")

    # Load model in bfloat16
    from transformers import AutoModelForCausalLM
    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.adapter:
        from peft import PeftModel
        adapter_path = (project_root / args.adapter) if not Path(args.adapter).is_absolute() else Path(args.adapter)
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    output_dir = project_root / args.output_dir
    all_eeffs = profile_base_model(
        model=model,
        tokenizer=tokenizer,
        ratio_data_paths=ratio_data_paths,
        subsample_frac=args.subsample,
        batch_size=args.batch_size,
        output_dir=str(output_dir),
    )

    # E_eff trend analysis
    means_total = []
    for ratio in RATIO_ORDER:
        if ratio in all_eeffs:
            vals = [v for v in all_eeffs[ratio]["eeff_total"] if not math.isnan(v)]
            means_total.append(float(np.nanmean(vals)) if vals else float("nan"))
        else:
            means_total.append(float("nan"))

    if has_downward_eeff_trend(means_total):
        print("E_eff DOWNWARD TREND DETECTED -- 60/40 training warranted")
    else:
        print("E_eff trend: flat or increasing -- no additional training triggered")


if __name__ == "__main__":
    main()

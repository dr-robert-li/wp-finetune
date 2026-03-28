"""Merge LoRA adapter into base model with verification roundtrip.

Strategy (defense-in-depth):
  1. Adapter already saved separately by train_model.py (safe even if merge fails)
  2. Load base model via Unsloth FastLanguageModel (load_in_4bit=False)
  3. Load adapter via PeftModel.from_pretrained
  4. Attempt merge_and_unload()
  5. Save merged model + tokenizer
  6. Reload and verify special tokens are still single-token (<wp_gen>, <wp_judge>)
  7. If verification fails: print vLLM --lora-modules fallback command and exit 1

Usage:
    python -m scripts.merge_adapter
    python -m scripts.merge_adapter --adapter-dir ./adapters/qwen3-wp
    python -m scripts.merge_adapter --adapter-dir ./adapters/qwen3-wp --output-dir ./models/Qwen3-30B-A3B-merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from scripts.dgx_toolbox import get_toolbox  # noqa: F401 — DGX resolver pattern

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(raw: str) -> Path:
    """Resolve a path that may be relative to PROJECT_ROOT."""
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_adapter(adapter_dir: str, output_dir: str, config: dict) -> None:
    """Load base model + adapter, merge, save, and verify special tokens.

    Args:
        adapter_dir: Path to the saved LoRA adapter (adapters/qwen3-wp/).
        output_dir: Destination for the merged model.
        config: Parsed train_config.yaml dict.

    Raises:
        SystemExit(1) on verification failure (prints vLLM fallback command).
    """
    # --- Idempotency check: skip if merged model already exists and verified ---
    merged_path = Path(output_dir)
    if merged_path.exists() and (merged_path / "config.json").exists():
        # Quick verification — are special tokens intact?
        from transformers import AutoTokenizer as _AT  # noqa: PLC0415
        try:
            verify_tok = _AT.from_pretrained(str(merged_path))
            special_tokens = config.get("tokenizer", {}).get("special_tokens", ["<wp_gen>", "<wp_judge>"])
            all_single = all(
                len(verify_tok.encode(t, add_special_tokens=False)) == 1 for t in special_tokens
            )
            if all_single:
                print(f"Merged model already exists at {merged_path} with verified special tokens. Skipping.")
                return
        except Exception:
            pass  # Fall through to re-merge

    from unsloth import FastLanguageModel  # noqa: PLC0415

    local_dir = str(resolve_path(config["model"]["local_dir"]))
    max_seq_length = config["model"]["max_seq_length"]

    print(f"Loading base model from {local_dir} ...")
    model, _base_tok = FastLanguageModel.from_pretrained(
        model_name=local_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # LOCKED — no QLoRA for MoE
        dtype=torch.bfloat16,
    )

    # Load the LoRA adapter on top of the base model
    from peft import PeftModel  # noqa: PLC0415

    print(f"Loading LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Attempt merge
    print("Merging adapter into base model ...")
    merged_model = model.merge_and_unload()

    merged_path = output_dir
    print(f"Saving merged model to {merged_path} ...")
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)

    # Save tokenizer alongside merged model
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer_dir = str(resolve_path(config["tokenizer"]["save_dir"]))
    print(f"Loading extended tokenizer from {tokenizer_dir} ...")
    extended_tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    extended_tok.save_pretrained(merged_path)
    print(f"Tokenizer saved to {merged_path}")

    # Verification roundtrip — reload from disk and check special tokens
    print("Running verification roundtrip ...")
    _verify_merged_model(merged_path, adapter_dir, config)


def _verify_merged_model(merged_path: str, adapter_dir: str, config: dict) -> None:
    """Reload merged model and verify special tokens are single-token.

    Args:
        merged_path: Path to the merged model directory.
        adapter_dir: Original adapter dir (for fallback message).
        config: Parsed train_config.yaml dict.

    Raises:
        SystemExit(1) on failure, with vLLM fallback command printed.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    try:
        verify_tok = AutoTokenizer.from_pretrained(merged_path)

        wp_gen_ids = verify_tok.encode("<wp_gen>", add_special_tokens=False)
        wp_judge_ids = verify_tok.encode("<wp_judge>", add_special_tokens=False)

        assert len(wp_gen_ids) == 1, (
            f"<wp_gen> must be single token, got {wp_gen_ids}"
        )
        assert len(wp_judge_ids) == 1, (
            f"<wp_judge> must be single token, got {wp_judge_ids}"
        )

        print(f"  <wp_gen>   -> token ID {wp_gen_ids[0]} (OK)")
        print(f"  <wp_judge> -> token ID {wp_judge_ids[0]} (OK)")
        print("MERGE VERIFICATION PASSED")

    except AssertionError as e:
        print(f"MERGE VERIFICATION FAILED: {e}")
        print()
        print("Fallback: serve adapter directly with vLLM (no merge needed):")
        dgx = get_toolbox()
        base_model = str(resolve_path(config["model"]["local_dir"]))
        print(f"  vllm serve {base_model} --lora-modules qwen3-wp={adapter_dir}")
        print()
        print("The adapter is still saved separately and can be used directly.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run the merge pipeline."""
    config = load_config()

    # Resolve adapter and output directories
    adapter_dir = str(resolve_path(
        args.adapter_dir if args.adapter_dir else config["training"]["output_dir"]
    ))
    output_dir = str(resolve_path(
        args.output_dir if args.output_dir else config["model"]["local_dir"] + "-merged"
    ))

    print("=" * 60)
    print("MERGE ADAPTER CONFIGURATION")
    print("=" * 60)
    print(f"  Base model:  {config['model']['local_dir']}")
    print(f"  Adapter dir: {adapter_dir}")
    print(f"  Output dir:  {output_dir}")
    print("=" * 60)

    merge_adapter(adapter_dir, output_dir, config)
    print(f"\nMerged model ready at: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model with verification roundtrip."
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the saved LoRA adapter directory. "
            "Defaults to training.output_dir from config/train_config.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help=(
            "Destination for the merged model. "
            "Defaults to model.local_dir + '-merged' from config/train_config.yaml."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(args)

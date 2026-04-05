"""Prepare extended tokenizer for Qwen3-30B-A3B fine-tuning.

Full pipeline:
  1. Load config from config/train_config.yaml
  2. Download model (skippable via --skip-download)
  3. Load model with AutoModelForCausalLM (bfloat16, no QLoRA)
  4. Extend tokenizer with <wp_gen> and <wp_judge> special tokens
  5. Resize model embeddings and mean-initialize new token rows
  6. Save extended tokenizer to adapters/tokenizer/
  7. Save model with resized embeddings back to local_dir
  8. Run smoke test to verify single-token encoding and generation

Usage:
    python -m scripts.prepare_tokenizer
    python scripts/prepare_tokenizer.py [--skip-download] [--smoke-only]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.dgx_toolbox import get_toolbox  # noqa: F401 — establishes DGX pattern

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
# Model download
# ---------------------------------------------------------------------------


def maybe_download_model(config: dict) -> Path:
    """Download model if not already on disk. Imported from download_model.py."""
    from scripts.download_model import count_safetensors, download_model

    local_dir = resolve_path(config["model"]["local_dir"])
    if count_safetensors(local_dir) > 0:
        print(f"Model already at {local_dir} — skipping download.")
        return local_dir
    return download_model(config=config)


# ---------------------------------------------------------------------------
# Tokenizer extension
# ---------------------------------------------------------------------------


def extend_tokenizer(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, special_tokens: list[str]) -> int:
    """Add special tokens, resize embeddings, and mean-initialize new rows.

    Args:
        tokenizer: Loaded tokenizer (modified in place).
        model: Loaded causal LM (modified in place).
        special_tokens: Token strings to add, e.g. ["<wp_gen>", "<wp_judge>"].

    Returns:
        Number of new tokens actually added.
    """
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {added} new special token(s): {special_tokens}")

    model.resize_token_embeddings(len(tokenizer))

    if added > 0:
        _mean_init_new_embeddings(model, num_new=added)

    return added


def _mean_init_new_embeddings(model: AutoModelForCausalLM, num_new: int) -> None:
    """Initialize the last `num_new` embedding rows to the mean of all prior rows.

    This prevents new tokens from starting at random noise, which destabilises
    early fine-tuning steps.
    """
    with torch.no_grad():
        embed = model.model.embed_tokens.weight
        mean_embedding = embed[:-num_new].mean(dim=0)
        for i in range(num_new):
            embed[-(num_new - i)] = mean_embedding
        print(f"Mean-initialized {num_new} new embedding row(s).")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def run_smoke_test(tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
    """Verify special tokens encode as single IDs and generation works."""
    for token in ["<wp_gen>", "<wp_judge>"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, (
            f"Smoke test FAILED: '{token}' encodes to {len(ids)} tokens (expected 1). "
            "Check that add_special_tokens was called before encoding."
        )
        print(f"  {token} -> token ID {ids[0]} (single token OK)")

    # Generation smoke test — 50 tokens from a <wp_gen> prompt
    prompt = "<wp_gen> "
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    assert new_tokens > 10, (
        f"Smoke test FAILED: model generated only {new_tokens} new tokens (expected > 10)"
    )
    print(f"  Generation: {new_tokens} new tokens produced.")
    print(f"  Output prefix: {generated[:120]!r}")
    print("SMOKE TEST PASSED")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_tokenizer(skip_download: bool = False, smoke_only: bool = False) -> None:
    """Run the full tokenizer preparation pipeline.

    Args:
        skip_download: If True, assume model is already on disk.
        smoke_only: If True, only run the smoke test (model + tokenizer must exist).
    """
    config = load_config()
    local_dir = resolve_path(config["model"]["local_dir"])
    save_dir = resolve_path(config["tokenizer"]["save_dir"])
    special_tokens: list[str] = config["tokenizer"]["special_tokens"]

    if smoke_only:
        print("--- Smoke-only mode: loading saved tokenizer + model ---")
        tokenizer = AutoTokenizer.from_pretrained(str(save_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(local_dir),
            dtype=torch.bfloat16,
            device_map="auto",
        )
        run_smoke_test(tokenizer, model)
        return

    # --- Idempotency check: skip if tokenizer already extended ---
    if save_dir.exists() and (save_dir / "tokenizer_config.json").exists():
        # Verify the saved tokenizer has our special tokens
        check_tok = AutoTokenizer.from_pretrained(str(save_dir))
        all_present = all(t in check_tok.get_vocab() for t in special_tokens)
        if all_present:
            print(f"Extended tokenizer already exists at {save_dir} with all special tokens. Skipping.")
            print("Use --smoke-only to re-run smoke test, or delete adapters/tokenizer/ to force re-extension.")
            return

    # --- Step 1: Download (optional) ---
    if not skip_download:
        maybe_download_model(config)
    else:
        print("Skipping download (--skip-download flag set).")

    # --- Step 2: Load model ---
    print(f"Loading model from {local_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
    try:
        # Prefer Unsloth FastLanguageModel (handles MoE + avoids page cache OOM)
        # Must run inside DGX Toolbox Unsloth Studio container (requires GPU)
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(local_dir),
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=False,  # LOCKED — no QLoRA for MoE
            dtype=torch.bfloat16,
        )
    except (ImportError, NotImplementedError) as e:
        if "GPU" in str(e) or "accelerator" in str(e):
            print("\n" + "=" * 60)
            print("  ERROR: Unsloth requires a GPU.")
            print("=" * 60)
            print()
            print("  This script must run inside the DGX Toolbox Unsloth Studio container:")
            print()
            print("    ~/dgx-toolbox/containers/unsloth-headless.sh")
            print()
            print("  Then from inside the container:")
            print("    cd /workspace/work  # or bind-mount your project")
            print("    python -m scripts.prepare_tokenizer")
            print()
            print("  Or use the tokenizer-only mode (no GPU needed):")
            print("    python -m scripts.prepare_tokenizer --tokenizer-only")
            print("=" * 60)
            sys.exit(1)
        # Fallback for environments without Unsloth (CI, local dev)
        print("Warning: Unsloth not available, using AutoModelForCausalLM (slower, higher memory)")
        model = AutoModelForCausalLM.from_pretrained(
            str(local_dir),
            dtype=torch.bfloat16,
            device_map="auto",
        )

    # --- Step 3: Extend tokenizer and mean-init embeddings ---
    extend_tokenizer(tokenizer, model, special_tokens)

    # --- Step 4: Save extended tokenizer ---
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(save_dir))
    print(f"Extended tokenizer saved to {save_dir}")

    # --- Step 5: Smoke test ---
    # Note: We don't re-save the full model here. The resized embeddings exist
    # in memory for the smoke test, and train_model.py will handle embedding
    # resize at training time via modules_to_save=["embed_tokens", "lm_head"].
    # Re-saving a 60GB MoE model just for 2 new embedding rows is wasteful and
    # fails on some MoE architectures (Qwen3MoeExperts serialization bug).
    run_smoke_test(tokenizer, model)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extend Qwen3-30B-A3B tokenizer with <wp_gen> and <wp_judge> tokens."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download (assume model already on disk).",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only run the smoke test using the saved tokenizer and model.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    prepare_tokenizer(skip_download=args.skip_download, smoke_only=args.smoke_only)

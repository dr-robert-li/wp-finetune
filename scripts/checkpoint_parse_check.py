"""checkpoint_parse_check.py — two-mode quality gate script.

Mode 1 (--readiness-gate):
    D-03 dataset readiness gate. Verifies data/reasoning_dataset/metadata.json
    meets the rebuilt-dataset preconditions before any training step.

Mode 2 (--checkpoint-dir):
    RTRN-04 in-process parse-failure abort hook. Loads the merged base + reasoning
    LoRA checkpoint, samples val examples, and exits non-zero if parse-fail > 5%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mode 1: D-03 dataset readiness gate
# ---------------------------------------------------------------------------

# Tolerance bands (D-02/D-03)
TOTAL_BAND = (650, 760)
COT_PCT_BAND = (55.0, 65.0)
REPLAY_PCT_BAND = (12.0, 18.0)
REQUIRED_REJECTION_KEYS = {"vendor_contamination", "truncated_invalid", "consistency"}


def verify_dataset_readiness(metadata_path: str | Path) -> bool:
    """Load metadata.json and assert D-03 readiness bands.

    Returns True on all-pass, exits non-zero on any failure.
    """
    path = Path(metadata_path)
    if not path.exists():
        print(f"ERROR: metadata file not found: {path}")
        print(
            "D-03 readiness gate FAILED - run the D-03 backfill "
            "(Phase 4.1 CoT generation + Phase 4.2 re-assembly) upstream first. "
            "Training will not start."
        )
        sys.exit(1)

    with path.open() as fh:
        meta = json.load(fh)

    failed = False

    # --- total_examples band ---
    total = meta.get("total_examples", 0)
    lo, hi = TOTAL_BAND
    if not (lo <= total <= hi):
        print(
            f"FAIL  total_examples={total}  expected band [{lo}, {hi}]"
        )
        failed = True

    # --- cot_percent band ---
    cot_pct = meta.get("mix", {}).get("cot_percent", 0.0)
    lo, hi = COT_PCT_BAND
    if not (lo <= cot_pct <= hi):
        print(
            f"FAIL  mix.cot_percent={cot_pct}  expected band [{lo}, {hi}]"
        )
        failed = True

    # --- replay_percent band ---
    replay_pct = meta.get("mix", {}).get("replay_percent", 0.0)
    lo, hi = REPLAY_PCT_BAND
    if not (lo <= replay_pct <= hi):
        print(
            f"FAIL  mix.replay_percent={replay_pct}  expected band [{lo}, {hi}]"
        )
        failed = True

    # --- rejection_counts key presence ---
    rejection_counts = meta.get("rejection_counts", {})
    missing_keys = REQUIRED_REJECTION_KEYS - set(rejection_counts.keys())
    if missing_keys:
        print(
            f"FAIL  rejection_counts missing keys: {sorted(missing_keys)}  "
            f"(proves vendor/truncation + consistency gates were applied)"
        )
        failed = True

    if failed:
        print(
            "D-03 readiness gate FAILED - run the D-03 backfill "
            "(Phase 4.1 CoT generation + Phase 4.2 re-assembly) upstream first. "
            "Training will not start."
        )
        sys.exit(1)

    print("D-03 readiness gate PASSED")
    return True


# ---------------------------------------------------------------------------
# Mode 2: RTRN-04 in-process parse-failure abort hook (Path C)
# ---------------------------------------------------------------------------

DEFAULT_BASE = "models/qwen3-30b-wp-30_70-merged"
DEFAULT_VAL_JSONL = "data/reasoning_dataset/openai_val.jsonl"
DEFAULT_N = 50
DEFAULT_THRESHOLD = 0.05


def run_checkpoint_parse_check(
    checkpoint_dir: str,
    base: str = DEFAULT_BASE,
    val_jsonl: str = DEFAULT_VAL_JSONL,
    n: int = DEFAULT_N,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Load the merged base + reasoning LoRA checkpoint in-process, sample n val
    examples, measure parse-fail rate, and exit non-zero if > threshold.

    Returns True if parse-fail rate <= threshold.
    """
    # WAVE-0: confirm load_adapter binding vs current Unsloth docs.
    # Preferred path: FastLanguageModel.from_pretrained(merged_base)
    #   -> model.load_adapter(checkpoint_dir)
    #   -> FastLanguageModel.for_inference(model)
    # Verify this API works on a Qwen3-MoE target_parameters adapter before
    # relying on it; update this comment once binding is confirmed.
    import torch
    from unsloth import FastLanguageModel
    from eval.eval_judge import parse_judge_response  # noqa: PLC0415

    print(f"[parse-check] Loading base model from {base} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=8192,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )

    print(f"[parse-check] Loading adapter from {checkpoint_dir} ...")
    # WAVE-0: model.load_adapter is the preferred binding for target_parameters adapters.
    model.load_adapter(checkpoint_dir)

    model = FastLanguageModel.for_inference(model)

    # Sample val prompts
    val_path = Path(val_jsonl)
    if not val_path.exists():
        print(f"[parse-check] ERROR: val file not found: {val_path}")
        sys.exit(1)

    samples = []
    with val_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n:
                break

    if not samples:
        print("[parse-check] ERROR: val file is empty")
        sys.exit(1)

    actual_n = min(n, len(samples))
    print(f"[parse-check] Evaluating {actual_n} samples (threshold={threshold:.0%}) ...")

    parse_fail_count = 0
    for i, example in enumerate(samples[:actual_n]):
        # Extract the prompt from the OpenAI-format JSONL
        messages = example.get("messages", [])
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]

        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
            )

        generated_ids = outputs[0][inputs.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Use exact production parse_fail predicate from eval_judge.py:375
        parsed = parse_judge_response(generated_text)
        if parsed is None or "overall_score" not in parsed:
            parse_fail_count += 1

        if (i + 1) % 10 == 0:
            rate_so_far = parse_fail_count / (i + 1)
            print(f"[parse-check]   {i+1}/{actual_n} evaluated  parse_fail_rate={rate_so_far:.1%}")

    parse_fail_rate = parse_fail_count / actual_n
    print(
        f"[parse-check] Parse-fail rate: {parse_fail_count}/{actual_n} = {parse_fail_rate:.1%}"
    )

    if parse_fail_rate > threshold:
        print(
            f"[parse-check] ABORT: parse-fail rate {parse_fail_rate:.1%} exceeds threshold "
            f"{threshold:.0%} (RTRN-04). Training run aborted."
        )
        sys.exit(1)

    print(
        f"[parse-check] PASSED: parse-fail rate {parse_fail_rate:.1%} <= threshold "
        f"{threshold:.0%} (RTRN-04)."
    )
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="checkpoint_parse_check: D-03 readiness gate + RTRN-04 parse-failure abort hook"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--readiness-gate",
        action="store_true",
        help="Run D-03 dataset readiness gate against metadata.json",
    )
    mode.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Run RTRN-04 parse-failure check against this checkpoint directory",
    )

    # readiness-gate args
    parser.add_argument(
        "--metadata",
        default="data/reasoning_dataset/metadata.json",
        metavar="PATH",
        help="Path to metadata.json (default: data/reasoning_dataset/metadata.json)",
    )

    # checkpoint-dir args
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        metavar="DIR",
        help=f"Merged base model directory (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--val-jsonl",
        default=DEFAULT_VAL_JSONL,
        metavar="PATH",
        help=f"Validation JSONL file (default: {DEFAULT_VAL_JSONL})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="INT",
        help=f"Number of val samples to evaluate (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="FLOAT",
        help=f"Parse-fail rate threshold (default: {DEFAULT_THRESHOLD})",
    )

    args = parser.parse_args()

    if args.readiness_gate:
        verify_dataset_readiness(args.metadata)
    else:
        run_checkpoint_parse_check(
            checkpoint_dir=args.checkpoint_dir,
            base=args.base,
            val_jsonl=args.val_jsonl,
            n=args.n,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()

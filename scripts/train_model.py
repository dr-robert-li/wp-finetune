"""Unsloth LoRA SFT training for Qwen3-30B-A3B on DGX Spark.

Full pipeline:
  1. Load config from config/train_config.yaml
  2. Load base model via Unsloth FastLanguageModel (load_in_4bit=False, output_router_logits=True)
  3. Load extended tokenizer from adapters/tokenizer/
  4. Apply LoRA via FastLanguageModel.get_peft_model (modules_to_save=[embed_tokens, lm_head])
  5. Load train/val datasets from data/final_dataset/
  6. Train with SFTTrainer + MLflow tracking (local)
  7. Save adapter to adapters/qwen3-wp/

Usage:
    python -m scripts.train_model
    python -m scripts.train_model --resume          # resume from latest checkpoint
    python -m scripts.train_model --resume ./adapters/qwen3-wp/checkpoint-200
    python -m scripts.train_model --dry-run         # load + print config, don't train
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
# Memory pre-check
# ---------------------------------------------------------------------------

# Qwen3-30B-A3B BF16 LoRA peak memory estimate (from Unsloth docs)
MIN_FREE_MEMORY_GB = 70  # 63GB model + ~7GB overhead for optimizer states, activations


def check_memory(config: dict) -> None:
    """Verify sufficient system memory is available before loading the model.

    Checks free RAM (not GPU VRAM — DGX Spark uses unified memory).
    If insufficient, lists top memory-consuming processes and exits.
    """
    import shutil
    import subprocess as _sp

    total, used, free = shutil.disk_usage("/")  # disk — not what we want
    # Use /proc/meminfo for actual RAM
    try:
        meminfo = Path("/proc/meminfo").read_text()
        mem_lines = {
            line.split(":")[0].strip(): int(line.split(":")[1].strip().split()[0])
            for line in meminfo.splitlines()
            if ":" in line
        }
        total_gb = mem_lines.get("MemTotal", 0) / (1024 * 1024)
        available_gb = mem_lines.get("MemAvailable", 0) / (1024 * 1024)
    except Exception:
        # Fallback: psutil if available, or skip check
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
        except ImportError:
            print("Warning: Cannot check memory (no /proc/meminfo or psutil). Proceeding anyway.")
            return

    print(f"\n{'=' * 60}")
    print(f"  MEMORY PRE-CHECK")
    print(f"{'=' * 60}")
    print(f"  Total system memory: {total_gb:.1f} GB")
    print(f"  Available memory:    {available_gb:.1f} GB")
    print(f"  Required minimum:    {MIN_FREE_MEMORY_GB} GB")
    print(f"  Headroom:            {available_gb - MIN_FREE_MEMORY_GB:.1f} GB")

    if available_gb >= MIN_FREE_MEMORY_GB:
        print(f"  Status:              OK ✓")
        print(f"{'=' * 60}\n")
        return

    # Insufficient memory — show top consumers
    print(f"  Status:              INSUFFICIENT ✗")
    print(f"\n  Need {MIN_FREE_MEMORY_GB - available_gb:.1f} GB more free memory.")
    print(f"\n  Top memory consumers:")
    print(f"  {'─' * 50}")

    try:
        # Get top memory-consuming processes
        result = _sp.run(
            ["ps", "aux", "--sort=-rss"],
            capture_output=True, text=True, timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        print(f"  {'USER':<12} {'RSS MB':>8}  {'COMMAND'}")
        for line in lines[1:11]:  # Top 10
            parts = line.split(None, 10)
            if len(parts) >= 11:
                user = parts[0]
                rss_kb = int(parts[5]) if parts[5].isdigit() else 0
                cmd = parts[10][:60]
                if rss_kb > 10240:  # Only show >10MB
                    print(f"  {user:<12} {rss_kb // 1024:>7} MB  {cmd}")
    except Exception:
        pass

    # Also show Docker containers
    try:
        result = _sp.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip():
            print(f"\n  Running Docker containers:")
            print(f"  {'─' * 50}")
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
    except Exception:
        pass

    print(f"\n  {'=' * 60}")
    print(f"  ACTION REQUIRED: Free up memory before training.")
    print(f"  Suggestions:")
    print(f"    • Stop non-essential Docker containers: docker stop <name>")
    print(f"    • Close browser tabs and IDEs")
    print(f"    • Kill large background processes")
    print(f"    • Run: docker container prune -f && docker image prune -f")
    print(f"  {'=' * 60}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(config: dict):
    """Load base model via Unsloth FastLanguageModel and extended tokenizer.

    Returns:
        (model, tokenizer) — model has output_router_logits=True set on both
        model_kwargs and model.config (belt-and-suspenders).
    """
    from unsloth import FastLanguageModel  # noqa: PLC0415

    local_dir = str(resolve_path(config["model"]["local_dir"]))
    max_seq_length = config["model"]["max_seq_length"]

    print(f"Loading model from {local_dir} ...")
    model, _base_tok = FastLanguageModel.from_pretrained(
        model_name=local_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # LOCKED — no QLoRA for MoE
        dtype=torch.bfloat16,
    )

    # Enable MoE load balancing loss (TRNG-04) — set on config after loading
    # (model_kwargs doesn't work with Unsloth's FastLanguageModel wrapper)
    model.config.output_router_logits = True
    print("  output_router_logits = True (MoE load balancing monitoring enabled)")

    # Load extended tokenizer (with <wp_gen> and <wp_judge>)
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer_dir = str(resolve_path(config["tokenizer"]["save_dir"]))
    print(f"Loading extended tokenizer from {tokenizer_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # Verify special tokens exist
    for token in config["tokenizer"]["special_tokens"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1, (
            f"Special token '{token}' must be a single token, got {ids}. "
            "Run scripts/prepare_tokenizer.py first."
        )
        print(f"  {token} -> token ID {ids[0]} (OK)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------


def apply_lora(model, config: dict):
    """Apply LoRA adapter via FastLanguageModel.get_peft_model."""
    from unsloth import FastLanguageModel  # noqa: PLC0415

    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],  # 32
        target_modules=lora_cfg["target_modules"],  # [q_proj, k_proj, v_proj, o_proj, gate_up_proj, down_proj]
        lora_alpha=lora_cfg["lora_alpha"],  # 64
        lora_dropout=lora_cfg["lora_dropout"],  # 0.05
        bias="none",
        use_gradient_checkpointing="unsloth",
        modules_to_save=lora_cfg["modules_to_save"],  # ["embed_tokens", "lm_head"] — LOCKED
        random_state=42,
    )
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_datasets(config: dict):
    """Load train and validation datasets from data/final_dataset/."""
    from datasets import load_dataset  # noqa: PLC0415

    train_file = str(PROJECT_ROOT / config["data"]["train_file"])
    val_file = str(PROJECT_ROOT / config["data"]["val_file"])

    print(f"Loading train dataset from {train_file} ...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")

    print(f"Loading val dataset from {val_file} ...")
    val_dataset = load_dataset("json", data_files=val_file, split="train")

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Memory watchdog callback
# ---------------------------------------------------------------------------

from transformers import TrainerCallback as _TrainerCallback

# Threshold: trigger graceful exit when available RAM drops below this.
# 2 GB leaves enough room for the checkpoint save itself (~1.2 GB adapter).
OOM_WATCHDOG_THRESHOLD_MB = 2048


class MemoryWatchdogCallback(_TrainerCallback):
    """TrainerCallback that monitors system RAM and triggers a graceful
    save-and-exit before the OOM killer strikes.

    On unified memory systems (DGX Spark), the training process, model weights,
    optimizer states, and dataloader workers all compete for the same RAM pool.
    The kernel OOM killer terminates the process with no checkpoint save, losing
    up to save_steps worth of training.  This callback reads /proc/meminfo every
    step and, when available memory drops below the threshold, saves a checkpoint
    and exits cleanly.
    """

    def __init__(self, threshold_mb: int = OOM_WATCHDOG_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self._triggered = False

    @staticmethod
    def _available_mb() -> int:
        """Read MemAvailable from /proc/meminfo (Linux only)."""
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024  # kB -> MB
        except Exception:
            pass
        return 999_999  # fail-open: if we can't read, don't block training

    def on_step_end(self, args, state, control, **kwargs):
        if self._triggered:
            return
        avail = self._available_mb()
        if avail < self.threshold_mb:
            self._triggered = True
            print(
                f"\n{'=' * 60}\n"
                f"  MEMORY WATCHDOG: {avail} MB available (threshold: {self.threshold_mb} MB)\n"
                f"  Saving emergency checkpoint at step {state.global_step} and exiting.\n"
                f"{'=' * 60}\n"
            )
            control.should_save = True
            control.should_training_stop = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def build_trainer(model, tokenizer, train_dataset, val_dataset, config: dict):
    """Build SFTTrainer with MLflow tracking (local)."""
    import mlflow  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    train_cfg = config["training"]
    output_dir = str(resolve_path(train_cfg["output_dir"]))

    # Configure MLflow to use local sqlite store (no cloud)
    mlflow.set_tracking_uri(f"sqlite:///{resolve_path('mlruns.db')}")
    mlflow.set_experiment("wp-qwen3-moe")

    # Format OpenAI chat messages using the model's chat template
    # Unsloth always expects a list of strings returned
    def formatting_func(example):
        messages = example["messages"]
        # Single example: messages is a list of dicts [{role, content}, ...]
        if messages and isinstance(messages[0], dict):
            return [tokenizer.apply_chat_template(messages, tokenize=False)]
        # Batch mode: messages is a list of lists
        return [tokenizer.apply_chat_template(m, tokenize=False) for m in messages]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        callbacks=[MemoryWatchdogCallback()],
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            bf16=train_cfg["bf16"],
            fp16=False,
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
            dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
            dataloader_persistent_workers=train_cfg.get("dataloader_persistent_workers", False),
            dataloader_prefetch_factor=train_cfg.get("dataloader_prefetch_factor", 2),
            logging_steps=train_cfg["logging_steps"],
            eval_steps=train_cfg["eval_steps"],
            save_steps=train_cfg["save_steps"],
            report_to="mlflow",  # TRNG-05 — local sqlite store, no cloud
            max_seq_length=config["model"]["max_seq_length"],
            dataset_num_proc=4,
        ),
    )
    return trainer


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------


def print_training_summary(config: dict, train_dataset, val_dataset) -> None:
    """Print a human-readable training config summary before starting."""
    lora = config["lora"]
    train = config["training"]
    print("=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Model:        {config['model']['name']}")
    print(f"  LoRA r:       {lora['r']}")
    print(f"  LoRA alpha:   {lora['lora_alpha']}")
    print(f"  Dropout:      {lora['lora_dropout']}")
    print(f"  Target mods:  {lora['target_modules']}")
    print(f"  Modules save: {lora['modules_to_save']}")
    print(f"  Epochs:       {train['num_train_epochs']}")
    print(f"  LR:           {train['learning_rate']}")
    print(f"  Batch size:   {train['per_device_train_batch_size']}")
    print(f"  Grad accum:   {train['gradient_accumulation_steps']}")
    print(f"  Eff batch:    {train['per_device_train_batch_size'] * train['gradient_accumulation_steps']}")
    print(f"  LR schedule:  {train['lr_scheduler_type']}")
    print(f"  Warmup ratio: {train['warmup_ratio']}")
    print(f"  BF16:         {train['bf16']}")
    print(f"  Max seq len:  {config['model']['max_seq_length']}")
    print(f"  Train size:   {len(train_dataset)}")
    print(f"  Val size:     {len(val_dataset)}")
    print(f"  Output dir:   {train['output_dir']}")
    print(f"  Tracking:     MLflow (local file store)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full training pipeline."""
    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_config(config_path)

    # --- Idempotency check: skip if adapter already trained ---
    output_dir = resolve_path(config["training"]["output_dir"])
    adapter_config = output_dir / "adapter_config.json"
    if adapter_config.exists() and not args.resume and not args.dry_run:
        print(f"Trained adapter already exists at {output_dir}/adapter_config.json")
        print("Use --resume to continue training, or delete the adapter dir to retrain from scratch.")
        return

    # --- Memory pre-check: ensure enough free memory before loading 63GB model ---
    check_memory(config)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load datasets
    train_dataset, val_dataset = load_datasets(config)

    # Print summary
    print_training_summary(config, train_dataset, val_dataset)

    if args.dry_run:
        print("\nDRY RUN MODE — skipping training.")
        return

    # Build trainer and train
    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, config)

    # Determine checkpoint to resume from
    resume_checkpoint = None
    if args.resume:
        if args.resume is True or args.resume == "":
            # Auto-detect latest checkpoint
            output_dir = resolve_path(config["training"]["output_dir"])
            checkpoints = sorted(output_dir.glob("checkpoint-*"))
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
                print(f"Resuming from latest checkpoint: {resume_checkpoint}")
            else:
                print("No checkpoints found — starting from scratch.")
        else:
            resume_checkpoint = args.resume
            print(f"Resuming from checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Post-training save (adapter only — defense-in-depth, trainer also saves)
    output_dir = str(resolve_path(config["training"]["output_dir"]))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsloth LoRA SFT training for Qwen3-30B-A3B on DGX Spark."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CHECKPOINT_DIR",
        help=(
            "Resume training from a checkpoint. "
            "If no path is given, resumes from the latest checkpoint in output_dir."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and config, print training summary, then exit without training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="CONFIG_PATH",
        help="Path to training config YAML (default: config/train_config.yaml).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(args)

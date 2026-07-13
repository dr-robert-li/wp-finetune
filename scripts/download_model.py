"""Download a base model from HuggingFace Hub with resume support.

Reads model.name/model.local_dir from a train_config.yaml-shaped file (default
config/train_config.yaml, the v3.x pipeline's config). Pass --config-path to
target a different config (e.g. config/train_config_v4.yaml for the v4.0 base)
without touching the default.

Usage:
    python -m scripts.download_model
    python scripts/download_model.py
    python scripts/download_model.py --config-path config/train_config_v4.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `python scripts/download_model.py` (direct execution) puts scripts/ on
# sys.path, not the repo root, breaking the `scripts.*` absolute import below.
# `python -m scripts.download_model` doesn't need this (repo root already on
# path via __package__). Rule 3 fix — pre-existing, blocks --help acceptance
# criterion for direct invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dgx_toolbox import get_toolbox  # noqa: F401,E402 — establishes DGX pattern

CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def count_safetensors(model_dir: Path) -> int:
    """Return the number of .safetensors shard files in model_dir."""
    if not model_dir.is_dir():
        return 0
    return len(list(model_dir.glob("*.safetensors")))


def expected_shard_count(model_dir: Path) -> int | None:
    """Return the total shard count declared by model.safetensors.index.json,
    or None if the index isn't present (nothing to compare against -- WR-01)."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    with open(index_path) as f:
        index = json.load(f)
    return len(set(index.get("weight_map", {}).values()))


def download_model(config: dict | None = None, config_path: Path = CONFIG_PATH) -> Path:
    """Download model to local_dir with resume support.

    Args:
        config: Pre-loaded config dict. If None, loads from config_path.
        config_path: Path to train_config.yaml (used when config is None).

    Returns:
        Path to the downloaded model directory.
    """
    if config is None:
        config = load_config(config_path)

    model_name = config["model"]["name"]
    local_dir = Path(config["model"]["local_dir"])
    if not local_dir.is_absolute():
        local_dir = PROJECT_ROOT / local_dir

    existing_shards = count_safetensors(local_dir)
    expected_shards = expected_shard_count(local_dir)
    if existing_shards > 0 and (expected_shards is None or existing_shards >= expected_shards):
        print(f"Model already present at {local_dir} ({existing_shards} safetensors shards). Skipping download.")
        return local_dir
    if existing_shards > 0:
        print(f"Incomplete download at {local_dir} ({existing_shards}/{expected_shards} shards) — resuming.")

    print(f"Downloading {model_name} to {local_dir} ...")
    print("This is a ~60 GB model. Ensure sufficient disk space.")

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5"],
    )

    shard_count = count_safetensors(local_dir)
    total_size_gb = sum(f.stat().st_size for f in local_dir.glob("*.safetensors")) / (1024**3)
    print(f"Download complete: {shard_count} safetensors shards, {total_size_gb:.1f} GB")
    return local_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI arg parser for this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to train_config.yaml-shaped config (default: {CONFIG_PATH})",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    download_model(config_path=args.config_path)

"""Download Qwen3-30B-A3B model from HuggingFace Hub with resume support.

Usage:
    python -m scripts.download_model
    python scripts/download_model.py
"""

from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from scripts.dgx_toolbox import get_toolbox  # noqa: F401 — establishes DGX pattern

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    if existing_shards > 0:
        print(f"Model already present at {local_dir} ({existing_shards} safetensors shards). Skipping download.")
        return local_dir

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


if __name__ == "__main__":
    download_model()

"""Wave 0 tests for the v4 config-driven download path (download_model.py).

All tests use mocks/fixtures; no network or GPU required. Mirrors
tests/test_prepare_tokenizer.py's mock-only structure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.download_model import (
    CONFIG_PATH,
    build_arg_parser,
    download_model,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_V4_PATH = PROJECT_ROOT / "config" / "train_config_v4.yaml"


class TestV4Config:
    """load_config(config_path=train_config_v4.yaml) resolves the v4 base."""

    def test_v4_config_model_name(self):
        config = load_config(config_path=CONFIG_V4_PATH)
        assert config["model"]["name"] == "Qwen/Qwen3.6-35B-A3B"

    def test_v4_config_local_dir(self):
        config = load_config(config_path=CONFIG_V4_PATH)
        assert config["model"]["local_dir"] == "./models/Qwen3.6-35B-A3B"


class TestDownloadIdempotency:
    """download_model() skips snapshot_download when shards already exist."""

    def test_skip_download_when_shards_present(self, tmp_path):
        local_dir = tmp_path / "Qwen3.6-35B-A3B"
        local_dir.mkdir()
        (local_dir / "model-00001-of-00002.safetensors").write_bytes(b"stub")

        config = {"model": {"name": "Qwen/Qwen3.6-35B-A3B", "local_dir": str(local_dir)}}

        with patch("scripts.download_model.snapshot_download") as mock_snapshot:
            result = download_model(config=config)

        mock_snapshot.assert_not_called()
        assert result == local_dir

    def test_download_called_on_empty_dir(self, tmp_path):
        local_dir = tmp_path / "Qwen3.6-35B-A3B"
        # Deliberately do NOT create local_dir/shards — empty/absent.

        config = {"model": {"name": "Qwen/Qwen3.6-35B-A3B", "local_dir": str(local_dir)}}

        with patch("scripts.download_model.snapshot_download") as mock_snapshot:
            download_model(config=config)

        mock_snapshot.assert_called_once()
        _, kwargs = mock_snapshot.call_args
        assert kwargs["repo_id"] == "Qwen/Qwen3.6-35B-A3B"
        assert kwargs["resume_download"] is True


class TestConfigPathFlag:
    """argparse parser exposes --config-path defaulting to the module CONFIG_PATH."""

    def test_default_config_path(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.config_path == CONFIG_PATH

    def test_override_config_path(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--config-path", str(CONFIG_V4_PATH)])
        assert args.config_path == CONFIG_V4_PATH

    def test_help_lists_config_path(self):
        parser = build_arg_parser()
        help_text = parser.format_help()
        assert "--config-path" in help_text

"""Wave 0 tests for training config and model download — run before implementation.

All tests use mocks, fixtures, or static analysis. No GPU or download required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"
TRAIN_JSONL = PROJECT_ROOT / "data" / "final_dataset" / "openai_train.jsonl"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_model.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_train_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelDownloaded:
    """test_model_downloaded — verify directory-existence check logic, not actual download."""

    def test_model_dir_check_returns_false_when_missing(self, tmp_path):
        """count_safetensors on a non-existent dir returns 0."""
        from scripts.download_model import count_safetensors

        missing = tmp_path / "no_model_here"
        assert count_safetensors(missing) == 0

    def test_model_dir_check_returns_count_when_present(self, tmp_path):
        """count_safetensors returns correct shard count."""
        from scripts.download_model import count_safetensors

        model_dir = tmp_path / "Qwen3-30B-A3B"
        model_dir.mkdir()
        for i in range(3):
            (model_dir / f"model-0000{i+1}-of-00003.safetensors").touch()
        assert count_safetensors(model_dir) == 3

    def test_load_config_reads_model_name(self):
        """load_config returns a dict with model.name key."""
        from scripts.download_model import load_config

        cfg = load_config()
        assert "model" in cfg
        assert cfg["model"]["name"] == "Qwen/Qwen3-30B-A3B"


class TestLoraConfigParams:
    """test_lora_config_params — assert key hyperparameters are set correctly."""

    def test_lora_r(self):
        cfg = load_train_config()
        assert cfg["lora"]["r"] == 32

    def test_bf16_enabled(self):
        cfg = load_train_config()
        assert cfg["training"]["bf16"] is True

    def test_lr_scheduler_type(self):
        cfg = load_train_config()
        assert cfg["training"]["lr_scheduler_type"] == "cosine"


class TestModulesToSave:
    """test_modules_to_save — lora.modules_to_save contains embed_tokens and lm_head."""

    def test_modules_to_save(self):
        cfg = load_train_config()
        assert cfg["lora"]["modules_to_save"] == ["embed_tokens", "lm_head"]


class TestDatasetSchema:
    """test_dataset_schema — first line of openai_train.jsonl has expected structure."""

    @pytest.mark.skipif(not TRAIN_JSONL.exists(), reason="openai_train.jsonl not present")
    def test_messages_key_present(self):
        with open(TRAIN_JSONL) as f:
            record = json.loads(f.readline().strip())
        assert "messages" in record, "Expected 'messages' key in training record"

    @pytest.mark.skipif(not TRAIN_JSONL.exists(), reason="openai_train.jsonl not present")
    def test_messages_is_list_of_dicts(self):
        with open(TRAIN_JSONL) as f:
            record = json.loads(f.readline().strip())
        messages = record["messages"]
        assert isinstance(messages, list), "Expected messages to be a list"
        assert len(messages) > 0, "Expected at least one message"
        for msg in messages:
            assert "role" in msg, f"Message missing 'role' key: {msg}"
            assert "content" in msg, f"Message missing 'content' key: {msg}"

    @pytest.mark.skipif(not TRAIN_JSONL.exists(), reason="openai_train.jsonl not present")
    def test_roles_are_valid(self):
        with open(TRAIN_JSONL) as f:
            record = json.loads(f.readline().strip())
        valid_roles = {"system", "user", "assistant"}
        for msg in record["messages"]:
            assert msg["role"] in valid_roles, f"Unexpected role '{msg['role']}'"


class TestRouterLogitsEnabled:
    """test_router_logits_enabled — static analysis: train script sets output_router_logits=True."""

    def test_router_logits_string_present(self):
        """The string 'output_router_logits=True' must appear in the training script source."""
        train_script = PROJECT_ROOT / "scripts" / "train_model.py"
        if not train_script.exists():
            pytest.skip("scripts/train_model.py not yet created")
        source = train_script.read_text()
        assert "output_router_logits=True" in source, (
            "scripts/train_model.py must set output_router_logits=True for MoE auxiliary loss"
        )

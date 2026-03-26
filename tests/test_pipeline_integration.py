"""Integration tests for Phase 1 pipeline checkpoint behavior — Wave 0 scaffolds for DATA-03.

These tests verify that clone and extract scripts correctly skip repos that have
already been processed (checkpointed). They will fail (RED) until Task 2 adds
checkpoint logic to phase1_clone.py and phase1_extract.py.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKPOINT_COMPLETED = {
    "completed": ["test-repo"],
    "failed": [],
    "batch_job_ids": [],
    "timestamp": None,
}


def test_clone_checkpoint_skip():
    """Repos in checkpoint completed list must be skipped during cloning."""
    import scripts.phase1_clone as clone_mod

    mock_config = {
        "core": [],
        "plugins": [{"name": "test-repo", "url": "https://github.com/test/test-repo"}],
        "themes": [],
    }

    with (
        patch.object(clone_mod, "load_config", return_value=mock_config),
        patch("scripts.phase1_clone.load_checkpoint", return_value=CHECKPOINT_COMPLETED) as mock_load_cp,
        patch("scripts.phase1_clone.save_checkpoint") as mock_save_cp,
        patch("scripts.phase1_clone.clone_repo") as mock_clone,
        patch("scripts.phase1_clone.REPOS_DIR") as mock_repos_dir,
    ):
        mock_repos_dir.mkdir = MagicMock()
        clone_mod.main()

    # load_checkpoint must be called
    mock_load_cp.assert_called_once_with("phase1_clone")
    # clone_repo must NOT be called for the already-checkpointed repo
    mock_clone.assert_not_called()


def test_extract_checkpoint_skip():
    """Repos in checkpoint completed list must be skipped during extraction."""
    import scripts.phase1_extract as extract_mod

    mock_config = {
        "core": [],
        "plugins": [{"name": "test-repo", "url": "https://github.com/test/test-repo"}],
        "themes": [],
    }

    with (
        patch.object(extract_mod, "load_config", return_value=mock_config),
        patch("scripts.phase1_extract.load_checkpoint", return_value=CHECKPOINT_COMPLETED) as mock_load_cp,
        patch("scripts.phase1_extract.save_checkpoint") as mock_save_cp,
        patch("scripts.phase1_extract.extract_repo") as mock_extract,
        patch("scripts.phase1_extract.EXTRACTED_DIR") as mock_extracted_dir,
        patch("scripts.phase1_extract.PHP_EXTRACTOR") as mock_php,
    ):
        mock_extracted_dir.mkdir = MagicMock()
        mock_php.exists.return_value = True
        extract_mod.main()

    # load_checkpoint must be called
    mock_load_cp.assert_called_once_with("phase1_extract")
    # extract_repo must NOT be called for the already-checkpointed repo
    mock_extract.assert_not_called()

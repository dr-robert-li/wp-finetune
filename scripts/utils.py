"""Shared utility module for wp-finetune pipeline scripts.

Provides:
- extract_json: 4-strategy JSON extraction from LLM responses
- load_checkpoint / save_checkpoint: Atomic checkpoint persistence
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """Extract and parse a JSON object from an LLM response string.

    Tries four strategies in order:
    1. Raw json.loads on the stripped text
    2. ```json ... ``` fenced block
    3. ``` ... ``` plain fenced block
    4. Outermost { ... } block in the text

    Returns the parsed dict, or None if all strategies fail.
    """
    if not text:
        return None

    stripped = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: ```json fenced block
    match = re.search(r'```json\s*([\s\S]+?)\s*```', stripped)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: plain ``` fenced block
    match = re.search(r'```\s*([\s\S]+?)\s*```', stripped)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: outermost { ... } block
    match = re.search(r'\{[\s\S]*\}', stripped)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT_DIR: Optional[Path] = None


def _get_checkpoint_dir(override: Optional[Path] = None) -> Path:
    """Return the checkpoint directory, preferring the override if provided."""
    if override is not None:
        return Path(override)
    global _DEFAULT_CHECKPOINT_DIR
    if _DEFAULT_CHECKPOINT_DIR is None:
        _DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "checkpoints"
    return _DEFAULT_CHECKPOINT_DIR


def load_checkpoint(phase: str, checkpoint_dir: Optional[Path] = None) -> dict:
    """Load checkpoint state for a given phase name.

    Returns a fresh empty state if no checkpoint file exists.

    Args:
        phase: Phase identifier used as the filename stem
        checkpoint_dir: Override directory (for testing)

    Returns:
        dict with keys: completed, failed, batch_job_ids, timestamp
    """
    directory = _get_checkpoint_dir(checkpoint_dir)
    path = directory / f"{phase}_checkpoint.json"

    empty_state = {
        "completed": [],
        "failed": [],
        "timestamp": None,
    }

    if not path.exists():
        return empty_state

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all expected keys exist
        for key, default in empty_state.items():
            data.setdefault(key, default)
        return data
    except (json.JSONDecodeError, OSError):
        return empty_state


def save_checkpoint(phase: str, state: dict, checkpoint_dir: Optional[Path] = None) -> None:
    """Atomically save checkpoint state for a given phase name.

    Writes to a .tmp file then renames to .json so readers never see a partial write.

    Args:
        phase: Phase identifier used as the filename stem
        state: State dict to persist
        checkpoint_dir: Override directory (for testing)
    """
    directory = _get_checkpoint_dir(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)

    state = dict(state)
    state["timestamp"] = datetime.now(timezone.utc).isoformat()

    path = directory / f"{phase}_checkpoint.json"
    tmp_path = directory / f"{phase}_checkpoint.tmp"

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    tmp_path.rename(path)

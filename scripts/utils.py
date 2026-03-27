"""Shared utility module for wp-finetune pipeline scripts.

Provides:
- extract_json: 4-strategy JSON extraction from LLM responses
- call_with_backoff: Exponential backoff for Anthropic API calls
- load_checkpoint / save_checkpoint: Atomic checkpoint persistence
- batch_or_direct: Route decision based on item count
- make_batch_request: Format a single Batch API request
- submit_batch: Submit batch to Anthropic Batch API
- poll_batch: Poll batch until completion
- parse_batch_results: Parse batch results into successes/failures
"""

import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

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
# Exponential backoff
# ---------------------------------------------------------------------------

def call_with_backoff(
    client,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    factor: float = 2.0,
    max_retries: int = 5,
    **kwargs,
) -> "anthropic.types.Message":
    """Call client.messages.create with exponential backoff.

    Retries on:
    - anthropic.RateLimitError (429)
    - anthropic.APIStatusError with status_code >= 500

    On RateLimitError, reads the retry_after attribute if present.
    On the final attempt, re-raises the last exception.

    Args:
        client: Anthropic client instance
        base_delay: Initial wait in seconds before first retry
        max_delay: Cap on wait time between retries
        factor: Multiplier applied to delay after each retry
        max_retries: Maximum number of attempts (including first)
        **kwargs: Passed directly to client.messages.create

    Returns:
        anthropic.types.Message on success
    """
    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise

            # Use retry_after if available, else use current delay
            wait = getattr(exc, "retry_after", None) or delay
            jitter = random.uniform(0, wait * 0.1)
            time.sleep(wait + jitter)
            delay = min(delay * factor, max_delay)

        except anthropic.APIStatusError as exc:
            last_exc = exc
            if exc.status_code < 500:
                raise
            if attempt == max_retries - 1:
                raise

            jitter = random.uniform(0, delay * 0.1)
            time.sleep(delay + jitter)
            delay = min(delay * factor, max_delay)

    # Should not reach here, but satisfy type checker
    raise RuntimeError("call_with_backoff exhausted retries without raising")


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
        "batch_job_ids": [],
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


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

BATCH_THRESHOLD = 50


def batch_or_direct(item_count: int) -> str:
    """Return 'batch' if item_count >= BATCH_THRESHOLD, else 'direct'."""
    return "batch" if item_count >= BATCH_THRESHOLD else "direct"


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

def make_batch_request(
    custom_id: str,
    system: str,
    user_content: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
) -> dict:
    """Format a single request dict for the Anthropic Batch API.

    Args:
        custom_id: Unique identifier for this request
        system: System prompt text
        user_content: User message content
        model: Claude model identifier
        max_tokens: Maximum tokens in the response

    Returns:
        dict with custom_id and params keys
    """
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
        },
    }


def submit_batch(client, requests: list) -> str:
    """Submit a list of batch requests to the Anthropic Batch API.

    Args:
        client: Anthropic client instance
        requests: List of request dicts from make_batch_request()

    Returns:
        batch.id string
    """
    batch = client.beta.messages.batches.create(requests=requests)
    return batch.id


def poll_batch(client, batch_id: str, poll_interval: int = 60) -> list:
    """Poll the Anthropic Batch API until the batch has ended.

    Args:
        client: Anthropic client instance
        batch_id: Batch ID from submit_batch()
        poll_interval: Seconds to sleep between polls

    Returns:
        List of result objects from the batch
    """
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)

    return list(client.beta.messages.batches.results(batch_id))


def parse_batch_results(results: list) -> tuple:
    """Parse Batch API results into successes and failures.

    Args:
        results: List of result objects from poll_batch()

    Returns:
        Tuple of (successes: list[dict], failures: list[str])
        Each success dict has all JSON keys plus _custom_id.
        Each failure entry is the custom_id string.
    """
    successes = []
    failures = []

    for result in results:
        custom_id = result.custom_id

        if result.result.type != "succeeded":
            failures.append(custom_id)
            continue

        try:
            text = result.result.message.content[0].text
        except (AttributeError, IndexError):
            failures.append(custom_id)
            continue

        parsed = extract_json(text)
        if parsed is None:
            failures.append(custom_id)
            continue

        parsed["_custom_id"] = custom_id
        successes.append(parsed)

    return successes, failures

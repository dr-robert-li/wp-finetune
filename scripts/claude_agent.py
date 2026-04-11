"""Claude Code agent wrapper — replaces direct Anthropic API calls.

All LLM work in this pipeline runs through Claude Code agents (subscription-covered)
instead of the Anthropic API. This module provides a generate() function that invokes
the `claude` CLI in non-interactive mode via subprocess.

Usage:
    from scripts.claude_agent import generate

    text = generate("Assess this WordPress code...", system="You are a judge...")
    result = generate_json("Score this code...", system="Return JSON only.")
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from scripts.utils import extract_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default model for pipeline operations. Sonnet for routine work, Opus for
# reasoning-heavy tasks (contrastive CoT, deep judge reasoning).
DEFAULT_MODEL = "sonnet"

# Timeout for a single generation call (seconds).
DEFAULT_TIMEOUT = 300

# Max retries on transient failures.
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Generate text by spawning a Claude Code agent via subprocess.

    Invokes `claude --print` in non-interactive mode with tools disabled
    so the agent acts as a pure text generator. The response is returned
    as a string.

    Args:
        prompt: The user prompt to send.
        system: Optional system prompt override.
        model: Model to use ('sonnet', 'opus', 'haiku').
        timeout: Seconds before the subprocess is killed.
        max_retries: Number of retries on transient failures.

    Returns:
        The agent's text response.

    Raises:
        RuntimeError: If all retries fail.
    """
    # For very long prompts, use a temp file to avoid shell argument limits.
    use_file = len(prompt) > 50_000

    last_error = None
    for attempt in range(max_retries):
        try:
            # Always use stdin to avoid shell argument length limits.
            return _generate_via_stdin(prompt, system, model, timeout)
        except subprocess.TimeoutExpired:
            last_error = f"Timeout after {timeout}s (attempt {attempt + 1}/{max_retries})"
            logger.warning("Claude agent timeout (attempt %d/%d)", attempt + 1, max_retries)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            # Transient errors worth retrying: overloaded, rate limited
            if any(kw in stderr.lower() for kw in ("overloaded", "rate limit", "529", "503")):
                last_error = f"Transient error (attempt {attempt + 1}): {stderr[:200]}"
                logger.warning("Transient error (attempt %d/%d): %s", attempt + 1, max_retries, stderr[:200])
            else:
                raise RuntimeError(
                    f"Claude agent failed (exit {e.returncode}): {stderr[:500]}"
                ) from e
        except Exception as e:
            raise RuntimeError(f"Claude agent unexpected error: {e}") from e

        # Exponential backoff between retries (1s, 2s, 4s, ...)
        backoff = min(2 ** attempt, 30)
        logger.info("Retrying in %ds...", backoff)
        time.sleep(backoff)

    raise RuntimeError(f"Claude agent failed after {max_retries} retries: {last_error}")


def generate_json(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[dict]:
    """Generate and parse a JSON response from a Claude Code agent.

    Convenience wrapper around generate() that extracts JSON from the
    response using the same 4-strategy parser as the rest of the pipeline.

    Returns:
        Parsed dict, or None if JSON extraction fails.
    """
    text = generate(prompt, system=system, model=model, timeout=timeout)
    return extract_json(text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_cmd(system: Optional[str], model: str) -> list[str]:
    """Build the base claude CLI command."""
    cmd = [
        "claude",
        "--print",
        "--no-session-persistence",
        "--tools", "",
        "--model", model,
    ]
    if system:
        cmd.extend(["--system-prompt", system])
    return cmd


def _generate_via_stdin(
    prompt: str,
    system: Optional[str],
    model: str,
    timeout: int,
) -> str:
    """Generate with prompt piped via stdin.

    Always uses stdin to avoid shell argument length limits (ARG_MAX).
    """
    cmd = _build_cmd(system, model)

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_agent_env(),
    )

    # Log stderr for debugging even on success.
    if result.stderr and result.stderr.strip():
        logger.debug("Claude agent stderr: %s", result.stderr.strip()[:500])

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    return result.stdout.strip()


def _agent_env() -> dict:
    """Build environment for agent subprocess.

    Inherits the current environment. Does NOT set CLAUDE_CODE_SIMPLE
    because that disables OAuth/keychain auth which is required for
    subscription-based authentication.
    """
    return os.environ.copy()

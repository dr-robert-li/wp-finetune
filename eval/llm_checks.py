"""LLM-assisted rubric checks (rubric §F.5 — Phase 0.10/0.12).

For each LLM-method check in CHECK_REGISTRY, this module runs a single
batched YES/NO inference call. Two backends:

  - "claude"  → Claude Code agent via `claude` CLI subprocess. Use for
                advisor / audit / spot-check at low volume. $0 (subscription).
  - "vllm"    → local OpenAI-compatible vLLM endpoint (default Qwen3.6-35B-
                A3B-FP8 served via dgx-toolbox recipe). Use for batch volume
                (Phase 1 re-judging, boundary pack generation, Phase 0.10
                full-scale LLM checks). $0 (local GPU).

Backend selected via LLM_BACKEND env var (default "claude" for safety on
small smoke runs). vLLM path also reads LLM_VLLM_BASE_URL (default
http://localhost:8000/v1) and LLM_VLLM_MODEL (default qwen-3.6-35b-a3b).

Public API:
    LLM_CHECK_PROMPTS: dict[check_id, str]   # ordered binary prompts
    run_llm_checks(code: str) -> dict        # see return schema below

Batching:
    A single call evaluates ALL 41 LLM checks for one code sample.
    Dramatically cheaper than one call per check (~21K calls for 20K
    examples instead of ~820K).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rubric_definitions import CHECK_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompts() -> dict[str, dict]:
    """Build per-check binary prompts from CHECK_REGISTRY tool_detail strings."""
    out: dict[str, dict] = {}
    for cid, check in CHECK_REGISTRY.items():
        if check.method != "llm":
            continue
        polarity = check.polarity
        desc = check.tool_detail
        if polarity == "negative":
            question = f"Does this code exhibit: {desc}?"
        else:
            question = f"Does this code satisfy: {desc}?"
        out[cid] = {
            "polarity": polarity,
            "question": question,
            "dimension": check.dimension,
            "weight": check.weight,
        }
    return out


LLM_CHECK_PROMPTS: dict[str, dict] = _build_prompts()


def _build_batched_prompt(code: str) -> str:
    """Build a single prompt covering all 41 LLM checks at once."""
    lines = [
        "You are a WordPress PHP code-quality auditor.",
        "",
        "Evaluate each numbered check below against the code. For each check, "
        "answer YES (the check fires / pattern is present) or NO (it does not).",
        "Cite the line / snippet that informs your decision when YES.",
        "Return strict JSON with no prose outside the object.",
        "",
        "Output schema:",
        "{",
        '  "<check_id>": {"hit": true|false, "evidence": "<short text>"},',
        "  ...",
        "}",
        "",
        "Code:",
        "```php",
        code,
        "```",
        "",
        "Checks:",
    ]
    for cid, payload in LLM_CHECK_PROMPTS.items():
        lines.append(f"- {cid} ({payload['polarity']}, dim={payload['dimension']}): {payload['question']}")
    lines.append("")
    lines.append("Return JSON now. No explanations outside the JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You audit WordPress PHP code against a fixed rubric. "
    "Output strict JSON only. No markdown fences. No commentary."
)


def _call_claude(prompt: str, model: str, timeout: int) -> str:
    from scripts.claude_agent import generate
    return generate(prompt, system=_SYSTEM_PROMPT, model=model, timeout=timeout)


def _call_vllm(prompt: str, model: str, timeout: int, base_url: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def _parse_response(raw: str) -> Optional[dict]:
    """4-strategy JSON parse identical to the rest of the pipeline."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    import re
    for pattern in (r"```(?:json)?\s*\n(.*?)```", r"\{.*\}"):
        m = re.search(pattern, raw, re.DOTALL)
        if not m:
            continue
        try:
            return json.loads(m.group(1) if pattern.startswith("```") else m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def run_llm_checks(
    code: str,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 180,
) -> dict:
    """Run all LLM checks for one code snippet in a single batched call.

    Args:
        code: PHP source.
        backend: "claude" or "vllm". Defaults to LLM_BACKEND env var, else "claude".
        model: Model identifier. Backend-specific default if None.
        timeout: Per-request timeout in seconds.

    Returns:
        {
            "_unavailable": bool,           # True if call failed
            "backend": str,
            "model": str,
            "hits": dict[check_id, bool],
            "evidence": dict[check_id, str],
            "n_checks": int,
            "raw_response": str | None,
        }
    """
    backend = (backend or os.environ.get("LLM_BACKEND", "claude")).lower()
    prompt = _build_batched_prompt(code)

    if backend == "vllm":
        base_url = os.environ.get("LLM_VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)
        model = model or os.environ.get("LLM_VLLM_MODEL", DEFAULT_VLLM_MODEL)
        try:
            raw = _call_vllm(prompt, model=model, timeout=timeout, base_url=base_url)
        except urllib.error.URLError as e:
            logger.warning("vLLM endpoint unreachable: %s", e)
            return {"_unavailable": True, "backend": backend, "model": model,
                    "hits": {}, "evidence": {}, "n_checks": 0, "raw_response": None}
        except Exception as e:
            logger.warning("vLLM call failed: %s", e)
            return {"_unavailable": True, "backend": backend, "model": model,
                    "hits": {}, "evidence": {}, "n_checks": 0, "raw_response": str(e)}
    elif backend == "claude":
        model = model or "sonnet"
        try:
            raw = _call_claude(prompt, model=model, timeout=timeout)
        except FileNotFoundError:
            return {"_unavailable": True, "backend": backend, "model": model,
                    "hits": {}, "evidence": {}, "n_checks": 0, "raw_response": None}
        except Exception as e:
            logger.warning("Claude CLI call failed: %s", e)
            return {"_unavailable": True, "backend": backend, "model": model,
                    "hits": {}, "evidence": {}, "n_checks": 0, "raw_response": str(e)}
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r}")

    parsed = _parse_response(raw)
    if not isinstance(parsed, dict):
        return {"_unavailable": True, "backend": backend, "model": model,
                "hits": {}, "evidence": {}, "n_checks": 0, "raw_response": raw}

    hits: dict[str, bool] = {}
    evidence: dict[str, str] = {}
    for cid in LLM_CHECK_PROMPTS:
        payload = parsed.get(cid)
        if not isinstance(payload, dict):
            continue
        hit = payload.get("hit")
        if isinstance(hit, bool):
            hits[cid] = hit
        ev = payload.get("evidence")
        if hit and isinstance(ev, str) and ev:
            evidence[cid] = ev[:300]
    return {
        "_unavailable": False,
        "backend": backend,
        "model": model,
        "hits": hits,
        "evidence": evidence,
        "n_checks": len(hits),
        "raw_response": raw,
    }

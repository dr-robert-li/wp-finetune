"""JUDGE-01 smoke: raw (no-adapter) new-base judge-output-format compliance.

Serves the RAW Qwen3.6-35B-A3B checkpoint (no LoRA adapter) via the Phase 20
v4 harness (scripts/serve_base20_vllm.sh --language-model-only), feeds it
~20-50 real <wp_judge> prompts drawn from data/reasoning_dataset/openai_val.jsonl
with the config/judge_system.md rubric as the system instruction, and measures
the parse-fail rate of eval.output_parsers.parse_judge_scores(text, "auto")
(same parse-fail definition as scripts/relabel/eval_relabel.py: a fail is
`not parsed or not parsed.get("dimension_scores")`).

This is a diagnostic baseline against the 18% community-reported judge-output-
format-noncompliance anchor -- NOT a go/no-go gate on judge SFT. A high raw-
base parse-fail rate is EXPECTED (the judge SFT trains it away).

Structural pattern mirrors scripts/bench_wpbench_base_anchor.py /
scripts/smoke_deltanet_base20.py: boot_vllm -> wait_healthy -> real-generation
warm-up gate -> real work -> stop_vllm in a finally block. Applies the Phase
15 LOCKED lesson: gate capture on a real generation succeeding, not vLLM's
/v1/models health response; uses a generous max_tokens (>=2048) so truncation
is never misread as format-noncompliance (carry-forward lesson 1).

Usage:
    python scripts/smoke_judge_format_base21.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import (  # noqa: E402
    boot_vllm,
    wait_healthy,
    generate,
    stop_vllm,
    VllmBootTimeout,
)
from eval.output_parsers import parse_judge_scores  # noqa: E402

MODEL_DIR = "models/Qwen3.6-35B-A3B"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
CONTAINER_NAME = "base21-judge01-format-smoke"
PORT = 8020
GPU_MEM_UTIL = 0.80
BOOT_TIMEOUT_SEC = 1200  # 67 GiB base — Pitfall 3 lesson (Phase 20)

DATASET_PATH = PROJECT_ROOT / "data" / "reasoning_dataset" / "openai_val.jsonl"
JUDGE_SYSTEM_PATH = PROJECT_ROOT / "config" / "judge_system.md"
OUT_DIR = PROJECT_ROOT / "output" / "base21"
OUTPUT_PATH = OUT_DIR / "judge01_format_smoke.json"

N_PROMPTS = 30
MAX_TOKENS = 2048  # generous: truncation must never be misread as noncompliance
TEMPERATURE = 0.0
COMMUNITY_ANCHOR_RATE = 0.18
SEED = 1337


def load_wp_judge_prompts(n: int) -> list[str]:
    """Rows whose first user message starts with '<wp_judge>' (same filter
    convention as eval_judge / capture_judge_responses_tinker / eval_relabel)."""
    rows = [json.loads(line) for line in open(DATASET_PATH) if line.strip()]
    prompts = []
    for r in rows:
        user_msg = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        if user_msg.startswith("<wp_judge>"):
            prompts.append(user_msg)
    rng = random.Random(SEED)
    rng.shuffle(prompts)
    return prompts[:n]


def judge_generate(port: int, served_model: str, system_prompt: str,
                    user_prompts: list[str], max_tokens: int, temperature: float
                    ) -> tuple[list[str], set[int]]:
    """Chat-completion generate with a system message (the judge rubric) — the
    shared generate() helper is user-only, so this is a local, script-specific
    variant rather than a change to shared infra used by many other callers.

    WR-04: returns (outs, infra_error_idx) so a caller can tell a genuine
    judge-format non-compliance (parseable response, wrong shape) apart from
    an empty "" caused by a transient generation/sampling infra error --
    both currently collapse into the same "" placeholder here, but the
    infra-error INDEX is now tracked so downstream can separate the counts.
    """
    import openai
    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    outs = []
    infra_error_idx: set[int] = set()
    for i, user_prompt in enumerate(user_prompts):
        try:
            resp = client.chat.completions.create(
                model=served_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            outs.append(resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            print(f"[judge01-smoke] gen error idx {i}: {e}")
            outs.append("")
            infra_error_idx.add(i)
    return outs, infra_error_idx


def run_smoke() -> dict:
    system_prompt = JUDGE_SYSTEM_PATH.read_text()
    prompts = load_wp_judge_prompts(N_PROMPTS)
    if len(prompts) < N_PROMPTS:
        raise RuntimeError(f"only found {len(prompts)} <wp_judge> prompts in {DATASET_PATH}, need >={N_PROMPTS}")

    boot_vllm(MODEL_DIR, CONTAINER_NAME, PORT, GPU_MEM_UTIL,
              serve_script=SERVE_SCRIPT, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
    served = wait_healthy(PORT, CONTAINER_NAME, timeout=BOOT_TIMEOUT_SEC)

    # Phase 15 LOCKED lesson: gate on a REAL generation, not /v1/models health.
    warm = generate(PORT, served,
                     [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                     max_tokens=16)
    if not warm or not warm[0].strip():
        raise RuntimeError(f"real-generation warm-up returned empty output: {warm!r}")
    print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}")

    completions, infra_error_idx = judge_generate(PORT, served, system_prompt, prompts, MAX_TOKENS, TEMPERATURE)

    n_parse_ok = 0
    n_parse_fail = 0
    n_infra_error = len(infra_error_idx)
    sample_failures = []
    for text in completions:
        parsed = parse_judge_scores(text, "auto")
        if not parsed or not parsed.get("dimension_scores"):
            n_parse_fail += 1
            if len(sample_failures) < 5:
                sample_failures.append(text[:500])
        else:
            n_parse_ok += 1

    parse_fail_rate = n_parse_fail / len(prompts)
    if parse_fail_rate < COMMUNITY_ANCHOR_RATE:
        vs_anchor = "below"
    elif parse_fail_rate > COMMUNITY_ANCHOR_RATE:
        vs_anchor = "above"
    else:
        vs_anchor = "at"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "n_prompts": len(prompts),
        "n_parse_ok": n_parse_ok,
        "n_parse_fail": n_parse_fail,
        "n_infra_error": n_infra_error,  # WR-04: subset of n_parse_fail caused by a generation/sampling error, not genuine format non-compliance
        "parse_fail_rate": parse_fail_rate,
        "community_anchor_rate": COMMUNITY_ANCHOR_RATE,
        "vs_anchor": vs_anchor,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "served_model_dir": MODEL_DIR,
        "sample_failures": sample_failures,
    }
    OUTPUT_PATH.write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> int:
    try:
        result = run_smoke()
    finally:
        stop_vllm(CONTAINER_NAME)

    print(json.dumps(result, indent=2))
    print(f"JUDGE-01 baseline recorded: parse_fail_rate={result['parse_fail_rate']:.4f} "
          f"(vs {COMMUNITY_ANCHOR_RATE} community anchor: {result['vs_anchor']})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot timeout: {e}", file=sys.stderr)
        sys.exit(3)

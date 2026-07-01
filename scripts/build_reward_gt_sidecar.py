#!/usr/bin/env python3
"""Build the TRAIN teacher-GT sidecar for RL judge pool + probe corpus.

Recovers teacher GT from openai_train.jsonl (TRAIN split ONLY — val is held
out for the oracle; using val GT would make SC2 circular and violate T-09-LEAK)
and joins each TRAIN GT to the RL judge pool + probe corpus by CONTENT-HASH of
the normalized PHP code-under-review (NOT positional prompt_id — pool reordering
would silently corrupt every label; the hash is reorder-proof).

Writes: data/rl_probe/judge_gt_sidecar.jsonl
  rows: {prompt_id, code_hash, teacher_overall, source:"train"}

BLOCKING acceptance gate:
  >= 60 distinct prompt_ids (RL pool positions) resolve a TRAIN GT via hash-match.
  Below threshold -> exit non-zero (phase-premise pivot signal: escalate, do not
  proceed with Plans 03/04).

CPU, $0. No vLLM, no API. Uses only stdlib + existing eval imports.

Run:
    REWARD_SKIP_PHPCS_ASSERT=1 python3 scripts/build_reward_gt_sidecar.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Make repo root importable regardless of CWD.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

from eval.eval_judge import _extract_gt_from_assistant   # noqa: E402
from eval.output_parsers import extract_php_code         # noqa: E402
from scripts.tinker_rl_data import load_rl_prompts       # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH     = _REPO / "data/reasoning_dataset/openai_train.jsonl"
PROBE_PATH     = _REPO / "data/rl_probe/judge_probe_corpus.jsonl"
SIDECAR_OUT    = _REPO / "data/rl_probe/judge_gt_sidecar.jsonl"

# ---------------------------------------------------------------------------
# Acceptance gate thresholds
# ---------------------------------------------------------------------------
MIN_POOL_PROMPT_IDS = 60    # distinct RL pool prompt_ids with resolved TRAIN GT
# (The plan also mentions >=300 probe-corpus prompt-groups as an alternate —
# that threshold is on groups*samples, which is 93 distinct prompt_ids * group_size.
# We report both and gate on the stricter / simpler one: >=60 distinct prompt_ids.)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _code_hash(user_content: str) -> str | None:
    """SHA-256 of whitespace-normalized PHP extracted from a wp_judge user turn.

    Returns None if no PHP block can be extracted (unhashable prompt).
    This is the JOIN KEY: reorder-proof and shared between train and pool sides.
    """
    php = extract_php_code(user_content)
    if not php:
        return None
    normalized = re.sub(r"\s+", " ", php).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def build_train_hash_map() -> dict[str, float]:
    """Build {code_hash: teacher_overall} from openai_train.jsonl TRAIN GT.

    Uses TRAIN split only. Returns reorder-proof dict keyed by content-hash.
    """
    rows = [json.loads(l) for l in TRAIN_PATH.open() if l.strip()]
    wp_judge = [
        r for r in rows
        if next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        .startswith("<wp_judge>")
    ]
    hash_map: dict[str, float] = {}
    n_no_gt = 0
    n_no_php = 0
    for r in wp_judge:
        t = _extract_gt_from_assistant(r["messages"])
        if t is None:
            n_no_gt += 1
            continue
        user_content = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        h = _code_hash(user_content)
        if h is None:
            n_no_php += 1
            continue
        hash_map[h] = float(t["overall"])
    print(
        f"TRAIN openai_train.jsonl: {len(wp_judge)} wp_judge rows"
        f" | GT-extractable: {len(hash_map)}"
        f" | no-GT: {n_no_gt}"
        f" | no-PHP: {n_no_php}",
        flush=True,
    )
    return hash_map


def join_pool(hash_map: dict[str, float]) -> list[dict]:
    """Join TRAIN hash map to the RL judge pool by content-hash.

    Returns list of sidecar rows:
        {prompt_id, code_hash, teacher_overall, source:"train"}

    prompt_id is the pool INDEX as assigned by load_rl_prompts (0-based),
    stored for convenience lookups but NOT used as the join key.
    """
    pool = load_rl_prompts("judge")
    sidecar: list[dict] = []
    n_no_php = 0
    n_no_match = 0
    for idx, item in enumerate(pool):
        user_content = next(
            (m["content"] for m in item.get("messages", []) if m["role"] == "user"), ""
        )
        h = _code_hash(user_content)
        if h is None:
            n_no_php += 1
            continue
        if h not in hash_map:
            n_no_match += 1
            continue
        sidecar.append({
            "prompt_id": idx,
            "code_hash": h,
            "teacher_overall": hash_map[h],
            "source": "train",
        })
    print(
        f"RL pool: {len(pool)} prompts"
        f" | hash-matched TRAIN GT: {len(sidecar)}"
        f" | no-PHP: {n_no_php}"
        f" | no-hash-match: {n_no_match}",
        flush=True,
    )
    return sidecar


def _count_probe_coverage(sidecar: list[dict]) -> dict:
    """Check how many probe corpus prompt_ids are covered by the sidecar.

    The probe corpus prompt_id is the same pool index as in the RL pool, so
    sidecar coverage == probe coverage (the hash join was done on pool entries).
    Returns {covered_prompt_ids, total_probe_prompt_ids, probe_groups_covered}.
    """
    if not PROBE_PATH.exists():
        return {"covered_prompt_ids": 0, "total_probe_prompt_ids": 0, "probe_groups_covered": 0}
    probe_rows = [json.loads(l) for l in PROBE_PATH.open() if l.strip()]
    probe_pids = set(r["prompt_id"] for r in probe_rows)
    sidecar_pids = set(r["prompt_id"] for r in sidecar)
    covered = probe_pids & sidecar_pids
    # probe groups: distinct group_ids in probe corpus for covered prompt_ids
    covered_groups = set(
        r["group_id"] for r in probe_rows if r["prompt_id"] in covered
        if "group_id" in r
    )
    return {
        "covered_prompt_ids": len(covered),
        "total_probe_prompt_ids": len(probe_pids),
        "probe_groups_covered": len(covered_groups),
    }


def main() -> int:
    print("=" * 72)
    print("BUILD REWARD GT SIDECAR — TRAIN GT only (T-09-LEAK / D-08.2 boundary)")
    print("=" * 72)

    # 1. Build TRAIN hash map
    hash_map = build_train_hash_map()

    # 2. Join to RL pool
    sidecar = join_pool(hash_map)

    # 3. Report probe corpus coverage
    probe_cov = _count_probe_coverage(sidecar)
    print(
        f"Probe corpus: {probe_cov['total_probe_prompt_ids']} unique prompt_ids"
        f" | covered by sidecar: {probe_cov['covered_prompt_ids']}"
        f" | probe groups covered: {probe_cov['probe_groups_covered']}",
        flush=True,
    )

    # 4. BLOCKING acceptance gate
    n_ids = len(set(r["prompt_id"] for r in sidecar))
    if n_ids < MIN_POOL_PROMPT_IDS:
        print(
            f"\nBLOCKING GATE FAILED: only {n_ids} distinct prompt_ids resolved "
            f"a TRAIN GT (threshold >= {MIN_POOL_PROMPT_IDS}). "
            f"Hash-join infeasible — this is the phase-premise pivot signal. "
            f"Escalate: do not proceed to Plans 03/04.",
            file=sys.stderr,
        )
        return 1

    # 5. Write sidecar
    SIDECAR_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SIDECAR_OUT.open("w", encoding="utf-8") as f:
        for row in sidecar:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSidecar written: {SIDECAR_OUT}")
    print(f"  rows          : {len(sidecar)}")
    print(f"  distinct pids : {n_ids}")
    print(f"  source        : train (val held out — anti-leakage)")
    print(f"\nBLOCKING GATE PASSED: {n_ids} >= {MIN_POOL_PROMPT_IDS} distinct prompt_ids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

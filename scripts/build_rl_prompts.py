#!/usr/bin/env python3
"""Assemble RL rollout prompt pools from audited Phase-4.2 training corpus.

SOURCES (audited, vendor-filtered):
  - data/reasoning_dataset/openai_train.jsonl (Phase 4.2 canonical train, passed
    boundary/security review per STATE.md Phase 4.2 entry)

EXCLUDED (explicitly out of scope):
  - data/reasoning_dataset/openai_val.jsonl       (held-out — must not leak)
  - data/reasoning_dataset/*.pre_vendorfilter.*    (pre-filter, vendor-contaminated)
  - Any synthetic prompt not traceable to Phase-4.2 audited corpus

SPLIT RULE:
  - Row whose first message content starts with "<wp_gen>"   → gen pool
  - Row whose first message content starts with "<wp_judge>" → judge pool
  - All other rows (replay rows without tags)                → excluded

OUTPUT (OpenAI chat schema, prompt-only — empty assistant turn):
  - data/rl_prompts/wp_gen_train.jsonl
  - data/rl_prompts/wp_judge_train.jsonl
  - data/rl_prompts/PROVENANCE.md

IDEMPOTENT: re-running produces byte-identical output (no timestamps; deterministic
ordering; sha256 dedup tracks first-seen order).

Threat: T-09-POISON (training-data poisoning) — mitigated by drawing ONLY from the
audited corpus + recording lineage in PROVENANCE.md.
Threat: T-09-LEAK (val-set leakage) — mitigated by sha256 guard against openai_val.jsonl.
"""

import argparse
import hashlib
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_TRAIN = os.path.join(PROJ_ROOT, "data", "reasoning_dataset", "openai_train.jsonl")
SRC_VAL   = os.path.join(PROJ_ROOT, "data", "reasoning_dataset", "openai_val.jsonl")

DEFAULT_GEN_OUT       = os.path.join(PROJ_ROOT, "data", "rl_prompts", "wp_gen_train.jsonl")
DEFAULT_JUDGE_OUT     = os.path.join(PROJ_ROOT, "data", "rl_prompts", "wp_judge_train.jsonl")
DEFAULT_PROVENANCE    = os.path.join(PROJ_ROOT, "data", "rl_prompts", "PROVENANCE.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _empty_assistant() -> dict:
    return {"role": "assistant", "content": ""}


def _prompt_row(user_content: str) -> dict:
    """RL prompt row: user turn verbatim + empty assistant turn."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            _empty_assistant(),
        ]
    }


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def build_pools(
    src_train: str = SRC_TRAIN,
    src_val:   str = SRC_VAL,
    gen_out:   str = DEFAULT_GEN_OUT,
    judge_out: str = DEFAULT_JUDGE_OUT,
    prov_out:  str = DEFAULT_PROVENANCE,
) -> dict:
    """
    Read the audited Phase-4.2 train corpus, split by tag, dedup by sha256,
    assert no val-set leakage, write both pools + PROVENANCE.md.

    Returns a stats dict for caller inspection / testing.
    """
    # ------------------------------------------------------------------
    # 1. Load val-set user-content sha256s for leakage guard
    # ------------------------------------------------------------------
    val_hashes: set[str] = set()
    val_rows_read = 0
    with open(src_val, encoding="utf-8") as fv:
        for line in fv:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_content = (row.get("messages") or [{}])[0].get("content", "")
            if not user_content:
                logger.warning("Skipping val row with missing content: %s", row.get("id", "?"))
                continue
            val_hashes.add(_sha256(user_content))
            val_rows_read += 1

    # ------------------------------------------------------------------
    # 2. Stream train corpus, split, dedup
    # ------------------------------------------------------------------
    gen_rows:   list[str] = []   # user contents, in first-seen order
    judge_rows: list[str] = []

    seen_hashes: set[str] = set()  # dedup across BOTH pools (global)
    gen_seen:    dict[str, bool] = {}   # ordered membership (insertion order kept by dict)
    judge_seen:  dict[str, bool] = {}

    stats = {
        "src_train": src_train,
        "src_val":   src_val,
        "train_rows_read":    0,
        "tagged_gen":         0,
        "tagged_judge":       0,
        "untagged_skipped":   0,
        "dedup_dropped":      0,
        "val_leak_dropped":   0,
        "gen_out":            0,
        "judge_out":          0,
    }

    with open(src_train, encoding="utf-8") as ft:
        for line in ft:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["train_rows_read"] += 1
            user_content = (row.get("messages") or [{}])[0].get("content", "")
            if not user_content:
                logger.warning("Skipping train row with missing content: %s", row.get("id", "?"))
                continue
            user_content = user_content.lstrip()

            # Tag-based split
            if user_content.startswith("<wp_gen>"):
                pool_target = "gen"
            elif user_content.startswith("<wp_judge>"):
                pool_target = "judge"
            else:
                stats["untagged_skipped"] += 1
                continue

            h = _sha256(user_content)

            # Val-leakage guard (check before dedup so count is accurate)
            if h in val_hashes:
                stats["val_leak_dropped"] += 1
                continue

            # Dedup guard
            if h in seen_hashes:
                stats["dedup_dropped"] += 1
                continue

            seen_hashes.add(h)
            if pool_target == "gen":
                stats["tagged_gen"] += 1
                gen_seen[h] = True
                gen_rows.append(user_content)
            else:
                stats["tagged_judge"] += 1
                judge_seen[h] = True
                judge_rows.append(user_content)

    stats["gen_out"]   = len(gen_rows)
    stats["judge_out"] = len(judge_rows)

    # ------------------------------------------------------------------
    # 3. Validate non-empty pools
    # ------------------------------------------------------------------
    if not gen_rows:
        raise ValueError(
            "wp_gen pool is empty after filtering — check source corpus for <wp_gen> tags. "
            "Halting to prevent writing an empty prompt pool."
        )
    if not judge_rows:
        raise ValueError(
            "wp_judge pool is empty after filtering — check source corpus for <wp_judge> tags."
        )

    # ------------------------------------------------------------------
    # 4. Write gen pool
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(gen_out), exist_ok=True)
    with open(gen_out, "w", encoding="utf-8") as fg:
        for content in gen_rows:
            fg.write(json.dumps(_prompt_row(content), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 5. Write judge pool
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(judge_out), exist_ok=True)
    with open(judge_out, "w", encoding="utf-8") as fj:
        for content in judge_rows:
            fj.write(json.dumps(_prompt_row(content), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 6. Write PROVENANCE.md (fully deterministic — no timestamps)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(prov_out), exist_ok=True)

    # Compute sha256 of each output file for the provenance record
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    gen_file_sha   = _file_sha256(gen_out)
    judge_file_sha = _file_sha256(judge_out)

    prov = f"""\
# RL Prompt Pool Provenance

## Purpose

Auditable lineage for the RL rollout prompt pools consumed by Phase 9 (GSPO Training).
Every prompt in these pools derives from the Phase-4.2 audited training corpus.

This document satisfies the T-09-POISON and T-09-LEAK threat mitigations
defined in `09-01-PLAN.md`.

## Source Files

| File | Role |
|------|------|
| `{os.path.relpath(src_train, PROJ_ROOT)}` | Phase-4.2 canonical train corpus (passed boundary/security review, vendorfilter applied) |
| `{os.path.relpath(src_val, PROJ_ROOT)}` | Held-out val set — used for leakage guard ONLY, no rows emitted |

## Split Rule

User-turn content routing:
- Starts with `<wp_gen>`   → `wp_gen_train.jsonl`
- Starts with `<wp_judge>` → `wp_judge_train.jsonl`
- Neither tag              → excluded (replay rows without tags)

## Row Counts

| Metric | Count |
|--------|-------|
| Train rows read | {stats['train_rows_read']} |
| Tagged `<wp_gen>` (before dedup) | {stats['tagged_gen']} |
| Tagged `<wp_judge>` (before dedup) | {stats['tagged_judge']} |
| Untagged / replay rows skipped | {stats['untagged_skipped']} |
| Dedup dropped (sha256 collision) | {stats['dedup_dropped']} |
| Val-leakage dropped | {stats['val_leak_dropped']} |
| **wp_gen pool output** | **{stats['gen_out']}** |
| **wp_judge pool output** | **{stats['judge_out']}** |

## Val-Set Leakage Guard (T-09-LEAK)

Val rows loaded for leakage check: {val_rows_read}

Every emitted prompt's user-content sha256 was checked against the full
`openai_val.jsonl` sha256 set before emission.

**Assertion: NO val-set user-content sha256 appears in either RL prompt pool.**
Val-leakage dropped rows: {stats['val_leak_dropped']}

## Deduplication

Dedup is performed on sha256(user_content) across BOTH pools (gen + judge share
a single seen-hash set). A duplicate appearing in the same or different pool is
dropped; only the first-seen instance is emitted.

Dedup dropped: {stats['dedup_dropped']}

## Output File Checksums

| File | SHA-256 |
|------|---------|
| `data/rl_prompts/wp_gen_train.jsonl` | `{gen_file_sha}` |
| `data/rl_prompts/wp_judge_train.jsonl` | `{judge_file_sha}` |

## Training-Data-Poisoning Mitigation (T-09-POISON)

- Sources: ONLY the audited Phase-4.2 train corpus (vendor-filtered,
  boundary/security reviewed, per STATE.md Phase 4.2 entry).
- Excluded: `openai_val.jsonl` (held-out), `*.pre_vendorfilter.*` backups
  (vendor-contaminated pre-filter), any synthetic or unaudited prompt source.
- Lineage: fully traceable to the single source file listed above.
- Idempotent: re-running this script with the same source produces byte-identical output.

## Schema

Both output files use OpenAI chat format with an **empty assistant turn**:

```json
{{"messages": [{{"role": "user", "content": "<wp_gen> ..."}}, {{"role": "assistant", "content": ""}}]}}
```

Completions are generated at RL sampling time — the assistant turn is never pre-filled.
"""

    with open(prov_out, "w", encoding="utf-8") as fp:
        fp.write(prov)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble RL rollout prompt pools from Phase-4.2 audited corpus."
    )
    parser.add_argument("--src-train",     default=SRC_TRAIN,        help="Source train JSONL")
    parser.add_argument("--src-val",       default=SRC_VAL,          help="Val JSONL (leakage guard only)")
    parser.add_argument("--gen-out",       default=DEFAULT_GEN_OUT,  help="Output: wp_gen prompts")
    parser.add_argument("--judge-out",     default=DEFAULT_JUDGE_OUT, help="Output: wp_judge prompts")
    parser.add_argument("--provenance-out", default=DEFAULT_PROVENANCE, help="Output: PROVENANCE.md")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    stats = build_pools(
        src_train=args.src_train,
        src_val=args.src_val,
        gen_out=args.gen_out,
        judge_out=args.judge_out,
        prov_out=args.provenance_out,
    )

    if not args.quiet:
        print("build_rl_prompts: assembly complete")
        print(f"  train rows read  : {stats['train_rows_read']}")
        print(f"  untagged skipped : {stats['untagged_skipped']}")
        print(f"  dedup dropped    : {stats['dedup_dropped']}")
        print(f"  val-leak dropped : {stats['val_leak_dropped']}")
        print(f"  wp_gen pool      : {stats['gen_out']} prompts → {args.gen_out}")
        print(f"  wp_judge pool    : {stats['judge_out']} prompts → {args.judge_out}")
        print(f"  provenance       : {args.provenance_out}")


if __name__ == "__main__":
    main()

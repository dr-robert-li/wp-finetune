#!/usr/bin/env python
"""Build wp_gen replay-augmented training variants for the rank×replay grid (D-N4).

Variants:
  15% (existing): openai_train.augmented.jsonl AS-IS — no build required
  30%: --extra-replay-n 132 -> openai_train.augmented.replay30.jsonl  (~725 rows)
  50%: --extra-replay-n 423 -> openai_train.augmented.replay50.jsonl  (~1016 rows)

Each variant = fixed base rows (cot+ctf+30 should_fail negatives+existing replay)
from --base-train, PLUS N additional wp_gen rows drawn from --replay-source.

LEAKAGE GUARD: asserts no new replay row's function body appears in the held-out
sentinel (invalid_php_sentinel.jsonl) or the val set. Dedup/leakage is keyed on
ASSISTANT content (the PHP function body) — not user-message descriptions — because
wp_gen rows store code in the assistant turn. Belt-and-suspenders: also checks
user-message content against val+sentinel using _codes() (verbatim from
build_augmented_train.py), which is effectively vacuous for sentinel (100% wp_judge
format) but harmless.

INVARIANT: never removes rows from --base-train. The 30 should_fail negatives and all
cot+ctf rows live in the base and survive by construction. This is pure wp_gen —
no wp_judge rows are added (D-N4 / Reading X resolved).

Usage:
  python scripts/build_replay_mix.py \\
    --extra-replay-n 132 \\
    --out data/reasoning_dataset/openai_train.augmented.replay30.jsonl

  python scripts/build_replay_mix.py \\
    --extra-replay-n 423 \\
    --out data/reasoning_dataset/openai_train.augmented.replay50.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _codes(path):
    """Extract user-message content set from a JSONL for leakage comparison.

    Copied verbatim from build_augmented_train.py (belt-and-suspenders guard on
    user-message descriptions; main leakage guard is on assistant/body content).
    """
    out = set()
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        out.add(u)
    return out


def _asst_bodies(path):
    """Extract assistant-message content set from a JSONL for body-level leakage comparison."""
    out = set()
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        a = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        if a:
            out.add(a)
    return out


def _docblock_description(docblock: str) -> str:
    """Extract the first meaningful line from a PHP docblock as a short description."""
    if not docblock:
        return ""
    for line in docblock.splitlines():
        line = line.strip().lstrip("/").lstrip("*").strip()
        if line and not line.startswith("@"):
            return line
    return ""


def _build_wp_gen_row(fn: dict, source_file: str) -> dict:
    """Wrap a phase1 function dict into a wp_gen replay JSONL row.

    Schema matches the existing 73 wp_gen rows in openai_train.augmented.jsonl:
      messages: [{role:user, content:'<wp_gen> {description}'},
                 {role:assistant, content: '{function body}'}]
      metadata: {stream:'replay', format:'replay', source_dir:'phase1_extraction/output/passed'}

    source_dir is added for provenance; stream/format match existing rows so
    replay-stream counters work correctly.
    """
    fn_name = fn.get("function_name", "") or ""
    docblock = fn.get("docblock", "") or ""
    body = fn.get("body", "") or ""

    desc = _docblock_description(docblock)
    if not desc:
        # Fallback: derive from function name (e.g. 'my_function' -> 'Write a WordPress function that my function')
        readable = fn_name.replace("::", " ").replace("_", " ").strip()
        desc = f"Write a WordPress function that {readable}" if readable else "Write a WordPress function"

    user_content = f"<wp_gen> {desc}"
    row = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": body},
        ],
        "metadata": {
            "stream": "replay",
            "format": "replay",
            "source_dir": "phase1_extraction/output/passed",
            "source_file": source_file,
        },
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build pure wp_gen replay-augmented training variants (D-N4 / Reading X)"
    )
    ap.add_argument(
        "--base-train",
        default="data/reasoning_dataset/openai_train.augmented.jsonl",
        help="The augmented-593 base (cot+ctf+negatives+15pct replay) — all rows copied verbatim",
    )
    ap.add_argument(
        "--replay-source",
        default="data/phase1_extraction/output/passed/",
        help="Directory of Phase 1 JSON files (wp_gen pool)",
    )
    ap.add_argument(
        "--extra-replay-n",
        type=int,
        required=True,
        help="Number of ADDITIONAL wp_gen rows to add (0=15pct AS-IS, 132=30pct, 423=50pct)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument(
        "--val",
        default="data/reasoning_dataset/openai_val.jsonl",
        help="Val set JSONL — new replay rows must not leak bodies here",
    )
    ap.add_argument(
        "--sentinel",
        default="data/reasoning_dataset/invalid_php_sentinel.jsonl",
        help="Sentinel JSONL — new replay rows must not leak bodies here",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    base_train_path = ROOT / args.base_train
    replay_source_path = ROOT / args.replay_source
    val_path = ROOT / args.val
    sentinel_path = ROOT / args.sentinel
    out_path = ROOT / args.out

    # Load base rows as raw lines — write verbatim to preserve all fields/formatting
    base_raw = [l for l in open(base_train_path) if l.strip()]
    print(f"Base rows loaded: {len(base_raw)}", flush=True)

    if args.extra_replay_n == 0:
        print("extra-replay-n=0: writing base rows unchanged (15% variant AS-IS)", flush=True)
        with open(out_path, "w") as f:
            for l in base_raw:
                f.write(l if l.endswith("\n") else l + "\n")
        print(f"Written: {out_path}", flush=True)
        return 0

    # Collect already-used wp_gen bodies from the base (dedup key = assistant/body content)
    already_used_bodies: set[str] = set()
    for l in base_raw:
        r = json.loads(l)
        meta = r.get("metadata", {})
        if meta.get("stream") == "replay":
            user_msg = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
            if "<wp_gen>" in user_msg:
                already_used_bodies.add(asst_msg)
    print(f"Already-used wp_gen bodies: {len(already_used_bodies)}", flush=True)

    # Collect leakage sets (keyed on assistant/body content — code lives in assistant turn)
    val_bodies = _asst_bodies(val_path)
    sentinel_bodies = _asst_bodies(sentinel_path)
    print(f"Val bodies: {len(val_bodies)}, Sentinel bodies: {len(sentinel_bodies)}", flush=True)

    # Build distinct candidate pool from phase1 passed/ dir
    # Exclude: already used in base, val leakage, sentinel leakage
    candidate_pool: list[tuple[str, dict]] = []  # (source_file_stem, fn_dict)
    seen_bodies: set[str] = set()

    for json_file in sorted(replay_source_path.glob("*.json")):
        try:
            items = json.load(open(json_file))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(items, dict):
            items = [items]
        for fn in items:
            body = fn.get("body", "") or ""
            if not body:
                continue
            if body in seen_bodies:
                continue
            seen_bodies.add(body)
            if body in already_used_bodies:
                continue
            if body in val_bodies:
                continue
            if body in sentinel_bodies:
                continue
            candidate_pool.append((json_file.stem, fn))

    print(f"Candidate pool (distinct, dedup-clean): {len(candidate_pool)}", flush=True)

    if len(candidate_pool) < args.extra_replay_n:
        print(
            f"ERROR: candidate pool ({len(candidate_pool)}) is smaller than extra-replay-n ({args.extra_replay_n}). "
            f"Cannot build variant.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # Seeded sample
    rng = random.Random(args.seed)
    sampled = rng.sample(candidate_pool, args.extra_replay_n)
    extra_rows = [_build_wp_gen_row(fn, src_file) for src_file, fn in sampled]

    # LEAKAGE GUARD (body-level) — main guard, already excluded during pool build above
    # Belt-and-suspenders: re-check on sampled extra_rows
    extra_bodies = {r["messages"][1]["content"] for r in extra_rows}
    body_leak_sent = extra_bodies & sentinel_bodies
    body_leak_val = extra_bodies & val_bodies
    if body_leak_sent:
        print(
            f"LEAKAGE: {len(body_leak_sent)} replay bodies match sentinel (body check)",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if body_leak_val:
        print(
            f"LEAKAGE: {len(body_leak_val)} replay bodies match val (body check)",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # LEAKAGE GUARD (user-message/description, verbatim from build_augmented_train.py)
    # Vacuous for sentinel (100% wp_judge format ≠ wp_gen format) but included as belt-and-suspenders
    extra_codes = {r["messages"][0]["content"] for r in extra_rows}
    leak_sent_desc = extra_codes & _codes(sentinel_path)
    leak_val_desc = extra_codes & _codes(val_path)
    if leak_sent_desc:
        print(
            f"LEAKAGE: {len(leak_sent_desc)} replay descriptions match sentinel",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if leak_val_desc:
        print(
            f"LEAKAGE: {len(leak_val_desc)} replay descriptions match val",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"Leakage guard: sentinel=0 val=0 (OK)", flush=True)

    # Write: base rows first (verbatim), then extras
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) if os.path.dirname(str(out_path)) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        for l in base_raw:
            f.write(l if l.endswith("\n") else l + "\n")
        for row in extra_rows:
            f.write(json.dumps(row) + "\n")

    total_rows = len(base_raw) + len(extra_rows)
    replay_total = 85 + args.extra_replay_n  # 85 = existing replay in base
    replay_pct = round(replay_total / total_rows * 100, 1)
    wp_gen_pct = round((73 + args.extra_replay_n) / total_rows * 100, 1)

    print(
        f"Built: base={len(base_raw)} + extra_replay={len(extra_rows)} = {total_rows} rows",
        flush=True,
    )
    print(
        f"Replay fraction: {replay_total}/{total_rows} = {replay_pct}% total replay "
        f"({wp_gen_pct}% pure wp_gen)",
        flush=True,
    )
    print(f"Seed: {args.seed} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

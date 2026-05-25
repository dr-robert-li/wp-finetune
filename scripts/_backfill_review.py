"""D-03 backfill reviewer harness (Claude-agent gate replaces haiku).

Two modes:

  --prep    Split the NEW vLLM-generated CoT (vllm_new_cot.json) into batch input
            files under data/phase4_reasoning/review/in/ for spawned Claude Code
            reviewer agents. Each batch file holds {idx, source_file, function_name,
            code, reasoning} records. Prints the batch manifest.

  --aggregate  Read the reviewer outputs under data/phase4_reasoning/review/out/
            (batch_NN.json, each a list of {idx|source_file|function_name,
            verdict:"consistent"|"inconsistent", reason}) and APPEND the accepted
            ones to data/reasoning_dataset/consistency_valid.jsonl in the slim
            schema validate_reasoning_consistency.py emits. Rejected go to
            consistency_rejected.jsonl. Idempotent on (source_file, function_name).

The reviewer agents themselves are spawned by the orchestrator (main session),
not by this script. This script only prepares inputs and folds results back in.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW_COT = ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "vllm_new_cot.json"
REVIEW_DIR = ROOT / "data" / "phase4_reasoning" / "review"
IN_DIR = REVIEW_DIR / "in"
OUT_DIR = REVIEW_DIR / "out"
VALID = ROOT / "data" / "reasoning_dataset" / "consistency_valid.jsonl"
REJECTED = ROOT / "data" / "reasoning_dataset" / "consistency_rejected.jsonl"


def prep(batch_size):
    records = json.loads(NEW_COT.read_text())
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for b, start in enumerate(range(0, len(records), batch_size)):
        chunk = records[start:start + batch_size]
        items = []
        for i, e in enumerate(chunk):
            items.append({
                "idx": i,
                "source_file": e.get("source_file"),
                "function_name": e.get("function_name"),
                "code": e.get("code", ""),
                "reasoning": e.get("reasoning", {}),
            })
        fp = IN_DIR / f"batch_{b:03d}.json"
        fp.write_text(json.dumps(items, indent=2))
        manifest.append({"batch": b, "file": str(fp), "count": len(items)})
    print(json.dumps({"total": len(records), "batches": len(manifest), "batch_size": batch_size,
                      "in_dir": str(IN_DIR), "out_dir": str(OUT_DIR)}, indent=2))
    for m in manifest:
        print(f"  batch_{m['batch']:03d}: {m['count']} -> {m['file']}")


def aggregate():
    # Map (source_file, function_name) -> record for stream tagging + dedup
    records = json.loads(NEW_COT.read_text())
    by_key = {(e.get("source_file"), e.get("function_name")): e for e in records}

    # Existing valid/rejected keys (idempotency)
    existing_valid_keys = set()
    if VALID.exists():
        for line in VALID.read_text().strip().splitlines():
            if line.strip():
                o = json.loads(line)
                existing_valid_keys.add((o.get("source_file"), o.get("function_name")))

    valid_new, rejected_new = [], []
    seen = set()
    out_files = sorted(OUT_DIR.glob("batch_*.json"))
    if not out_files:
        print(f"No reviewer outputs in {OUT_DIR}")
        return
    for fp in out_files:
        try:
            results = json.loads(fp.read_text())
        except Exception as e:
            print(f"  WARN: unreadable {fp.name}: {e}")
            continue
        for r in results:
            sf, fn = r.get("source_file"), r.get("function_name")
            key = (sf, fn)
            if key not in by_key or key in seen:
                continue
            seen.add(key)
            entry = {
                "source_file": sf,
                "function_name": fn,
                "stream": "cot",
                "consistency_status": "consistent" if r.get("verdict") == "consistent" else "inconsistent",
                "inconsistency_reason": (None if r.get("verdict") == "consistent" else (r.get("reason") or "inconsistent")),
            }
            if entry["consistency_status"] == "consistent":
                if key not in existing_valid_keys:
                    valid_new.append(entry)
            else:
                rejected_new.append(entry)

    with VALID.open("a") as f:
        for e in valid_new:
            f.write(json.dumps(e) + "\n")
    with REJECTED.open("a") as f:
        for e in rejected_new:
            f.write(json.dumps(e) + "\n")

    reviewed = len(seen)
    print(json.dumps({
        "reviewed": reviewed,
        "appended_valid": len(valid_new),
        "appended_rejected": len(rejected_new),
        "new_cot_total": len(records),
        "unreviewed": len(records) - reviewed,
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=35)
    args = ap.parse_args()
    if args.prep:
        prep(args.batch_size)
    elif args.aggregate:
        aggregate()
    else:
        ap.error("pass --prep or --aggregate")


if __name__ == "__main__":
    main()

"""D-03 backfill helper: reconstruct consistency_valid.jsonl for the already-shipped
consistent CoT+CtF set from the shipped openai_train/val JSONL, then verify the
(source_file, function_name) join against the raw bulk files.

The live consistency_valid.jsonl is empty; the shipped 418-example dataset is the
prior consistent set. We trust the prior gate for the existing examples (Claude
re-gates only the NEW vLLM-generated CoT) and rebuild the slim valid entries so
assemble_reasoning_dataset.py can re-join them against the bulk.

Run: python scripts/_backfill_reconstruct_valid.py [--write]
Without --write it only reports the join (dry-run).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "reasoning_dataset" / "openai_train.jsonl"
VAL = ROOT / "data" / "reasoning_dataset" / "openai_val.jsonl"
VALID_OUT = ROOT / "data" / "reasoning_dataset" / "consistency_valid.jsonl"
COT_BULK = ROOT / "data" / "phase4_reasoning" / "deep_judge_cot" / "deep_judge_cot_bulk.json"
CTF_BULK = ROOT / "data" / "phase4_reasoning" / "critique_then_fix" / "critique_then_fix_bulk.json"


def shipped_consistent_entries():
    entries = []
    seen = set()
    for path in (TRAIN, VAL):
        for line in path.read_text().strip().splitlines():
            if not line.strip():
                continue
            m = json.loads(line).get("metadata", {})
            stream = m.get("stream")
            if stream not in ("cot", "ctf"):
                continue
            sf, fn = m.get("source_file"), m.get("function_name")
            if not sf or not fn:
                continue
            key = (sf, fn, stream)
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "source_file": sf,
                "function_name": fn,
                "stream": stream,
                "consistency_status": "consistent",
                "inconsistency_reason": None,
            })
    return entries


def bulk_keys(path):
    data = json.loads(path.read_text())
    items = data if isinstance(data, list) else data.get("examples", data.get("data", []))
    return {(e.get("source_file"), e.get("function_name")) for e in items}, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write consistency_valid.jsonl")
    args = ap.parse_args()

    entries = shipped_consistent_entries()
    cot = [e for e in entries if e["stream"] == "cot"]
    ctf = [e for e in entries if e["stream"] == "ctf"]
    print(f"Reconstructed consistent entries: {len(entries)} (cot={len(cot)}, ctf={len(ctf)})")

    cot_bulk_keys, cot_bulk_n = bulk_keys(COT_BULK)
    ctf_bulk_keys, ctf_bulk_n = bulk_keys(CTF_BULK)

    cot_match = sum(1 for e in cot if (e["source_file"], e["function_name"]) in cot_bulk_keys)
    ctf_match = sum(1 for e in ctf if (e["source_file"], e["function_name"]) in ctf_bulk_keys)
    print(f"CoT join: {cot_match}/{len(cot)} reconstructed keys present in bulk ({cot_bulk_n} entries)")
    print(f"CtF join: {ctf_match}/{len(ctf)} reconstructed keys present in bulk ({ctf_bulk_n} entries)")

    cot_unmatched = [(e["source_file"], e["function_name"]) for e in cot
                     if (e["source_file"], e["function_name"]) not in cot_bulk_keys]
    ctf_unmatched = [(e["source_file"], e["function_name"]) for e in ctf
                     if (e["source_file"], e["function_name"]) not in ctf_bulk_keys]
    if cot_unmatched:
        print(f"  CoT unmatched ({len(cot_unmatched)}): {cot_unmatched[:5]}")
    if ctf_unmatched:
        print(f"  CtF unmatched ({len(ctf_unmatched)}): {ctf_unmatched[:5]}")

    join_ok = (cot_match == len(cot)) and (ctf_match == len(ctf))
    print(f"JOIN_CLEAN={join_ok}")

    if args.write:
        VALID_OUT.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        print(f"WROTE {len(entries)} entries -> {VALID_OUT}")
    else:
        print("(dry-run; pass --write to persist)")

    return 0 if join_ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Byte-identity check: clean v4-winner staging vs grid staging that scored REVL-04 0.4603.

Decision rule (D-V4-02):
  reuse_revl04 = all_shards_match

Rationale:
  Same merge_tinker_v3.py MoE-only path + same adapter (wp-reasoning-v4-winner/checkpoint.tar)
  + same stock base (Qwen3-30B-A3B) + deterministic merge math => expect byte-identical shards
  => reuse the grid's 0.4603 per D-V4-02.
  Non-identical (or reference shards absent) => reuse_revl04=false => Plan 03 must re-bench
  REVL-04 on the served clean staging.

Reference dir: output/eval_reasoning_v4_grid/r32-rp30/staging/ (the exact wp-bench measurement
  dir per .planning/phases/04.3-reasoning-fine-tune-inserted/04.3-VERIFICATION.md truths 2/3/10).
  Do NOT fall back to any other staging path; absent dir => reuse_revl04=false.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

CLEAN_STAGING = "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4"
GRID_STAGING = "output/eval_reasoning_v4_grid/r32-rp30/staging"
OUTPUT_REPORT = "output/merge_v4_winner/byte_identity_check.json"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_shards(dir_path: str) -> dict[str, str]:
    """Return {filename: sha256} for all safetensors shards and the index.json in dir_path."""
    if not os.path.isdir(dir_path):
        return {}
    result = {}
    for fname in os.listdir(dir_path):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            full = os.path.join(dir_path, fname)
            result[fname] = _sha256_file(full)
    return result


def main() -> int:
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)

    # --- Hash the clean staging shards ---
    print(f"Hashing clean staging: {CLEAN_STAGING}", flush=True)
    clean_shards = _collect_shards(CLEAN_STAGING)
    print(f"  clean staging: {len(clean_shards)} files hashed", flush=True)

    # --- Hash the grid reference shards ---
    grid_present = os.path.isdir(GRID_STAGING)
    if grid_present:
        print(f"Hashing grid staging: {GRID_STAGING}", flush=True)
        ref_shards = _collect_shards(GRID_STAGING)
        print(f"  grid staging: {len(ref_shards)} files hashed", flush=True)
    else:
        ref_shards = {}
        print(f"WARNING: grid staging dir not found: {GRID_STAGING}", flush=True)

    # --- Per-shard match table ---
    all_filenames = sorted(set(clean_shards) | set(ref_shards))
    per_shard: list[dict] = []
    for fname in all_filenames:
        c_hash = clean_shards.get(fname)
        r_hash = ref_shards.get(fname)
        match = (c_hash is not None) and (r_hash is not None) and (c_hash == r_hash)
        per_shard.append({
            "filename": fname,
            "clean_sha256": c_hash,
            "ref_sha256": r_hash,
            "match": match,
        })

    # --- all_shards_match: requires BOTH sets to be non-empty AND every shard to match ---
    # Explicit guard: empty ref => false, never vacuously true on empty intersection.
    clean_nonempty = len(clean_shards) > 0
    ref_nonempty = len(ref_shards) > 0
    all_match_individual = all(s["match"] for s in per_shard) if per_shard else False
    all_shards_match = bool(clean_nonempty and ref_nonempty and all_match_individual)

    # --- reuse_revl04 := all_shards_match (D-V4-02) ---
    reuse_revl04 = all_shards_match

    # --- Determine reason string ---
    if not grid_present:
        reason = "grid staging dir not found, re-bench required"
    elif not ref_nonempty:
        reason = "grid staging shards absent — re-bench required"
    elif not clean_nonempty:
        reason = "clean staging has no shards — merge may have failed"
    elif all_shards_match:
        reason = (
            "byte-identical: same MoE-only path + same adapter + same stock base "
            "+ deterministic merge math => grid REVL-04 0.4603 reusable (D-V4-02)"
        )
    else:
        reason = "shards differ — clean merge not byte-identical to grid merge; re-bench required"

    report = {
        "clean_staging": CLEAN_STAGING,
        "grid_staging": GRID_STAGING,
        "grid_staging_present": grid_present,
        "clean_shard_count": len(clean_shards),
        "ref_shard_count": len(ref_shards),
        "per_shard_hashes": per_shard,
        "all_shards_match": all_shards_match,
        "reuse_revl04": reuse_revl04,
        "reason": reason,
    }

    with open(OUTPUT_REPORT, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"reuse_revl04={reuse_revl04}", flush=True)
    print(f"all_shards_match={all_shards_match}", flush=True)
    print(f"reason: {reason}", flush=True)
    print(f"Report written: {OUTPUT_REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Seed import script: copies human-curated golden seeds from external data directory
into the project's data/seeds/ directory and validates them.

Usage:
    python scripts/seed_import.py

Source: ~/Desktop/data/wp-finetune-data/
Destination: data/seeds/ (relative to project root)

Validates:
- All 4 JSON files present and parseable
- Correct seed counts (93 in ugc_seeds, 25 in ugc_boundary_seeds)
- Required fields per seed type per D-05 schema
- Prints SHA-256 checksums for reproducibility audit
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path.home() / "Desktop" / "data" / "wp-finetune-data"
DEST_DIR = PROJECT_ROOT / "data" / "seeds"

SEED_FILES = [
    "ugc_seeds.json",
    "ugc_seeds_summary.json",
    "ugc_boundary_seeds.json",
    "ugc_boundary_seeds_summary.json",
]

# Required fields per D-05 for each seed type
REQUIRED_FIELDS_COT = {
    "seed_id", "seed_type", "code", "human_reasoning",
    "dimensions_addressed", "defect_subtlety", "annotation_type",
}
REQUIRED_FIELDS_CTF = {
    "seed_id", "seed_type", "dimensions_addressed",
    "defect_subtlety", "annotation_type",
}
# critique_then_fix seeds use 'defective_code' or 'code', and 'human_critique' or 'human_reasoning'


def sha256_checksum(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_seed_fields(seed: dict, index: int) -> list[str]:
    """Validate required fields for a single seed dict.

    Returns a list of error messages (empty = valid).
    """
    errors = []
    seed_type = seed.get("seed_type")
    seed_id = seed.get("seed_id", f"index_{index}")

    if seed_type == "deep_judge_cot":
        for field in REQUIRED_FIELDS_COT:
            if field not in seed:
                errors.append(f"  [{seed_id}] Missing field: {field}")
        # 'code' field must be non-empty
        if not seed.get("code", "").strip():
            errors.append(f"  [{seed_id}] Empty 'code' field")
        # human_reasoning must be a dict
        hr = seed.get("human_reasoning")
        if not isinstance(hr, dict):
            errors.append(f"  [{seed_id}] 'human_reasoning' must be a dict, got {type(hr)}")

    elif seed_type == "critique_then_fix":
        for field in REQUIRED_FIELDS_CTF:
            if field not in seed:
                errors.append(f"  [{seed_id}] Missing field: {field}")
        # Must have either 'defective_code' or 'code'
        if not seed.get("defective_code", "").strip() and not seed.get("code", "").strip():
            errors.append(f"  [{seed_id}] Missing both 'defective_code' and 'code' fields")
        # Must have either 'human_critique' or 'human_reasoning'
        if "human_critique" not in seed and "human_reasoning" not in seed:
            errors.append(f"  [{seed_id}] Missing both 'human_critique' and 'human_reasoning'")
    else:
        errors.append(f"  [{seed_id}] Unknown seed_type: {seed_type!r}")

    return errors


def validate_seed_file(path: Path) -> dict:
    """Load and validate a seed JSON file.

    Returns a summary dict with counts and any errors.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "count": 0}

    if not isinstance(data, list):
        # Summary files are dicts — skip deep validation
        return {"count": None, "type": "summary", "errors": []}

    errors = []
    type_counts: dict[str, int] = {}
    subtlety_counts: dict[str, int] = {}
    dimensions_seen: set[str] = set()

    for i, seed in enumerate(data):
        seed_type = seed.get("seed_type", "unknown")
        type_counts[seed_type] = type_counts.get(seed_type, 0) + 1

        subtlety = seed.get("defect_subtlety", "unknown")
        subtlety_counts[subtlety] = subtlety_counts.get(subtlety, 0) + 1

        dims = seed.get("dimensions_addressed", [])
        if isinstance(dims, list):
            dimensions_seen.update(dims)

        errors.extend(validate_seed_fields(seed, i))

    return {
        "count": len(data),
        "type_counts": type_counts,
        "subtlety_counts": subtlety_counts,
        "dimensions_covered": sorted(dimensions_seen),
        "errors": errors,
    }


def main() -> int:
    """Import seeds from external directory, validate, print checksums."""
    print("=" * 60)
    print("Seed Import Script")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Dest:   {DEST_DIR}")
    print("=" * 60)

    # Verify source directory exists
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}", file=sys.stderr)
        return 1

    # Check all source files exist
    missing = [f for f in SEED_FILES if not (SOURCE_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing source files: {missing}", file=sys.stderr)
        return 1

    # Create destination directory
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated/verified destination: {DEST_DIR}")

    # Copy files and compute checksums
    print("\n--- Copying files ---")
    all_errors: list[str] = []
    total_seeds = 0
    total_cot = 0
    total_ctf = 0

    for filename in SEED_FILES:
        src = SOURCE_DIR / filename
        dst = DEST_DIR / filename

        shutil.copy2(src, dst)
        checksum = sha256_checksum(dst)
        size_kb = dst.stat().st_size / 1024

        print(f"\n{filename}")
        print(f"  Copied:   {size_kb:.1f} KB")
        print(f"  SHA-256:  {checksum}")

        # Validate content
        report = validate_seed_file(dst)

        if "error" in report:
            print(f"  ERROR:    {report['error']}")
            all_errors.append(f"{filename}: {report['error']}")
            continue

        if report.get("type") == "summary":
            print(f"  Type:     summary (dict format, skipping deep validation)")
            continue

        count = report["count"]
        type_counts = report.get("type_counts", {})
        subtlety_counts = report.get("subtlety_counts", {})
        dims = report.get("dimensions_covered", [])

        print(f"  Seeds:    {count}")
        print(f"  By type:  {type_counts}")
        print(f"  By subtlety: {subtlety_counts}")
        print(f"  Dimensions covered: {len(dims)} ({', '.join(dims[:5])}{'...' if len(dims) > 5 else ''})")

        total_seeds += count
        total_cot += type_counts.get("deep_judge_cot", 0)
        total_ctf += type_counts.get("critique_then_fix", 0)

        if report["errors"]:
            print(f"  VALIDATION ERRORS ({len(report['errors'])}):")
            for err in report["errors"][:10]:
                print(err)
            if len(report["errors"]) > 10:
                print(f"  ... and {len(report['errors']) - 10} more")
            all_errors.extend(report["errors"])

    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  Total seeds imported: {total_seeds}")
    print(f"  deep_judge_cot:       {total_cot}")
    print(f"  critique_then_fix:    {total_ctf}")
    print(f"  Total (CoT + CtF):    {total_cot + total_ctf}")

    # Verify expected counts
    count_errors = []
    if total_seeds != 118:
        count_errors.append(f"Expected 118 total seeds, got {total_seeds}")
    if total_cot != 59:
        count_errors.append(f"Expected 59 deep_judge_cot seeds, got {total_cot}")
    if total_ctf != 59:
        count_errors.append(f"Expected 59 critique_then_fix seeds, got {total_ctf}")

    if count_errors or all_errors:
        print("\nERRORS FOUND:")
        for e in count_errors + all_errors[:20]:
            print(f"  {e}")
        return 1

    print("\nAll 4 seed files imported and validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

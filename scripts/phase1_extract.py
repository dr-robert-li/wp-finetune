#!/usr/bin/env python3
"""Phase 1, Step 2: Extract functions/classes from cloned repos.

Uses a PHP helper script to tokenize and extract function boundaries,
PHPDoc blocks, and dependency references from each PHP file.

Outputs JSON files per repo in data/phase1_extraction/output/extracted/.
"""

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.utils import load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "repos"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "extracted"
CONFIG_PATH = PROJECT_ROOT / "config" / "repos.yaml"
PHP_EXTRACTOR = Path(__file__).resolve().parent / "php_extract_functions.php"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def should_process_file(file_path: Path, repo_root: Path, repo_config: dict) -> bool:
    """Check if file falls within configured paths and not in skip_paths."""
    rel = str(file_path.relative_to(repo_root))

    # Always skip vendor, node_modules, tests
    for skip in ["vendor/", "node_modules/", "/tests/", "/test/", ".phpcs"]:
        if skip in rel:
            return False

    # Check skip_paths from config (support both prefix and glob patterns)
    for skip in repo_config.get("skip_paths", []):
        if rel.startswith(skip) or fnmatch.fnmatch(rel, skip):
            return False

    # If paths specified, file must be under one of them
    # Supports both directory prefixes (e.g. "src/wp-includes") and
    # glob patterns (e.g. "**/*.php", "includes/**/*.php")
    allowed_paths = repo_config.get("paths", [])
    if allowed_paths:
        for p in allowed_paths:
            if "**" in p or "*" in p:
                # Glob pattern: use fnmatch, converting ** to match any path component
                pattern = p.replace("**/", "")  # strip leading **/
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, f"*/{pattern}"):
                    return True
            else:
                # Directory prefix
                if rel.startswith(p):
                    return True
        return False

    return True


def extract_functions_from_file(file_path: Path) -> list[dict]:
    """Use PHP tokenizer to extract functions from a PHP file."""
    try:
        result = subprocess.run(
            ["php", str(PHP_EXTRACTOR), str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def extract_repo(repo_config: dict) -> list[dict]:
    """Extract all functions from a single repo."""
    name = repo_config["name"]
    repo_root = REPOS_DIR / name

    if not repo_root.exists():
        print(f"  [{name}] Not cloned, skipping. Run phase1_clone.py first.")
        return []

    php_files = list(repo_root.rglob("*.php"))
    print(f"  [{name}] Found {len(php_files)} PHP files")

    all_functions = []
    processed = 0

    for php_file in php_files:
        if not should_process_file(php_file, repo_root, repo_config):
            continue

        functions = extract_functions_from_file(php_file)
        for func in functions:
            func["source_repo"] = name
            func["source_file"] = str(php_file.relative_to(repo_root))
            func["quality_tier"] = repo_config["quality_tier"]

        all_functions.extend(functions)
        processed += 1

    print(f"  [{name}] Extracted {len(all_functions)} functions from {processed} files")
    return all_functions


def main():
    config = load_config()
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    if not PHP_EXTRACTOR.exists():
        print(f"ERROR: PHP extractor not found at {PHP_EXTRACTOR}")
        print("Create it first — see scripts/php_extract_functions.php")
        sys.exit(1)

    all_repos = []
    for section in ["core", "plugins", "themes"]:
        repos = config.get(section, [])
        if repos:
            all_repos.extend(repos)

    checkpoint = load_checkpoint("phase1_extract")
    completed = set(checkpoint["completed"])
    failed_repos = list(checkpoint["failed"])

    print(f"Extracting from {len(all_repos)} repositories...\n")
    if completed:
        print(f"  (Resuming: {len(completed)} already extracted)\n")

    for repo_config in all_repos:
        name = repo_config["name"]
        if name in completed:
            print(f"  [{name}] Skipping (checkpointed)\n")
            continue

        try:
            functions = extract_repo(repo_config)
            if functions:
                output_path = EXTRACTED_DIR / f"{name}.json"
                with open(output_path, "w") as f:
                    json.dump(functions, f, indent=2)
                print(f"  [{name}] Saved to {output_path}\n")

            completed.add(name)
            save_checkpoint("phase1_extract", {
                "completed": list(completed),
                "failed": failed_repos,
                "batch_job_ids": [],
            })
        except Exception as e:
            print(f"  [{name}] FAILED: {e}\n")
            if name not in failed_repos:
                failed_repos.append(name)
            save_checkpoint("phase1_extract", {
                "completed": list(completed),
                "failed": failed_repos,
                "batch_job_ids": [],
            })
            continue

    print("Done. Run phase1_judge.py next.")


if __name__ == "__main__":
    main()

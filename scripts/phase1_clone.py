#!/usr/bin/env python3
"""Phase 1, Step 1: Clone repositories listed in config/repos.yaml."""

import subprocess
import sys
from pathlib import Path

import yaml

from scripts.utils import load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "repos"
CONFIG_PATH = PROJECT_ROOT / "config" / "repos.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def clone_repo(name: str, url: str) -> Path:
    dest = REPOS_DIR / name
    if dest.exists():
        print(f"  [{name}] Already cloned, pulling latest...")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        return dest

    print(f"  [{name}] Cloning {url}...")
    subprocess.run(
        ["git", "clone", "--depth=1", "--single-branch", url, str(dest)],
        check=True,
    )
    return dest


def main():
    config = load_config()
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    all_repos = []
    for section in ["core", "plugins", "themes"]:
        repos = config.get(section, [])
        if repos:
            all_repos.extend(repos)

    if not all_repos:
        print("No repositories configured. Edit config/repos.yaml first.")
        sys.exit(1)

    checkpoint = load_checkpoint("phase1_clone")
    completed = set(checkpoint["completed"])
    failed_repos = list(checkpoint["failed"])

    print(f"Cloning {len(all_repos)} repositories...\n")
    if completed:
        print(f"  (Resuming: {len(completed)} already cloned)\n")

    for repo in all_repos:
        name = repo["name"]
        if name in completed:
            print(f"  [{name}] Skipping (checkpointed)\n")
            continue
        try:
            path = clone_repo(name, repo["url"])
            print(f"  [{name}] OK -> {path}\n")
            completed.add(name)
            save_checkpoint("phase1_clone", {
                "completed": list(completed),
                "failed": failed_repos,
                "batch_job_ids": [],
            })
        except subprocess.CalledProcessError as e:
            print(f"  [{name}] FAILED: {e}\n")
            if name not in failed_repos:
                failed_repos.append(name)
            save_checkpoint("phase1_clone", {
                "completed": list(completed),
                "failed": failed_repos,
                "batch_job_ids": [],
            })
            continue

    print("Done. Run phase1_extract.py next.")


if __name__ == "__main__":
    main()

"""
csv_to_repos.py — Convert ranked WordPress CSV files to config/repos.yaml.

Reads wp_top1000_plugins_final.csv and wp_top100_themes_final.csv,
applies quality filters, assigns quality tiers, infers path filters from
tags, and writes a fully-populated repos.yaml consumed by phase1_clone.py.

Usage:
    python scripts/csv_to_repos.py
"""

import csv
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PLUGIN_CSV = Path("/home/robert_li/Desktop/data/wp-finetune-data/wp_top1000_plugins_final.csv")
THEME_CSV = Path("/home/robert_li/Desktop/data/wp-finetune-data/wp_top100_themes_final.csv")
REPOS_YAML = Path(__file__).resolve().parent.parent / "config" / "repos.yaml"

# ---------------------------------------------------------------------------
# Filter thresholds
# ---------------------------------------------------------------------------

MIN_INSTALLS = 10_000
MIN_RATING = 80
MAX_UNPATCHED = 0
MAX_REPOS = 100

# ---------------------------------------------------------------------------
# Tag-to-path mapping
# ---------------------------------------------------------------------------

TAG_TO_PATH_FILTERS: dict[str, list[str]] = {
    "page builder": ["includes/**/*.php", "modules/**/*.php", "widgets/**/*.php"],
    "seo": ["includes/**/*.php", "src/**/*.php"],
    "woocommerce": ["includes/**/*.php", "src/**/*.php"],
    "e-commerce": ["includes/**/*.php", "src/**/*.php"],
    "security": ["includes/**/*.php", "src/**/*.php"],
    "_default": ["**/*.php"],
}

STANDARD_SKIP = ["vendor", "node_modules", "tests", "test", "assets", "css", "js"]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def filter_row(row: dict) -> bool:
    """Return True if the row passes all quality filters.

    Filters:
    - active_installs >= MIN_INSTALLS (strips trailing '+')
    - rating_pct >= MIN_RATING
    - unpatched_vulns <= MAX_UNPATCHED
    - github_url is non-empty and starts with 'https://github.com/'
    """
    # Parse active_installs — may have trailing "+" (e.g. "10000000+")
    raw_installs = str(row.get("active_installs", 0) or 0).replace("+", "").strip()
    try:
        active_installs = int(raw_installs) if raw_installs else 0
    except ValueError:
        active_installs = 0

    try:
        rating_pct = float(row.get("rating_pct", 0) or 0)
    except (ValueError, TypeError):
        rating_pct = 0.0

    try:
        unpatched_vulns = int(row.get("unpatched_vulns", 0) or 0)
    except (ValueError, TypeError):
        unpatched_vulns = 0

    github_url = str(row.get("github_url", "") or "").strip()

    return (
        active_installs >= MIN_INSTALLS
        and rating_pct >= MIN_RATING
        and unpatched_vulns <= MAX_UNPATCHED
        and bool(github_url)
        and github_url.startswith("https://github.com/")
    )


def assign_quality_tier(row: dict) -> str:
    """Assign quality_tier based on vulnerability and rating data.

    Returns:
        "trusted"  — 0 total_known_vulns, 0 unpatched, rating >= 90
        "assessed" — everything else that passes filter
    """
    try:
        total = int(row.get("total_known_vulns", 0) or 0)
    except (ValueError, TypeError):
        total = 0

    try:
        unpatched = int(row.get("unpatched_vulns", 0) or 0)
    except (ValueError, TypeError):
        unpatched = 0

    try:
        rating = float(row.get("rating_pct", 0) or 0)
    except (ValueError, TypeError):
        rating = 0.0

    if unpatched == 0 and total == 0 and rating >= 90:
        return "trusted"
    return "assessed"


def infer_path_filters(tags_str: str) -> list[str]:
    """Infer path filter patterns from a comma-separated tags string.

    Checks each keyword in TAG_TO_PATH_FILTERS (except '_default') in order.
    Returns the first matching filter set, or the default if none match.
    """
    if not tags_str:
        return TAG_TO_PATH_FILTERS["_default"]

    tags = [t.strip().lower() for t in tags_str.split(",")]

    for keyword, paths in TAG_TO_PATH_FILTERS.items():
        if keyword == "_default":
            continue
        # Check if any tag contains the keyword
        if any(keyword in tag for tag in tags):
            return paths

    return TAG_TO_PATH_FILTERS["_default"]


def _ensure_git_suffix(url: str) -> str:
    """Ensure the GitHub URL ends with .git."""
    url = url.strip()
    if url and not url.endswith(".git"):
        url = url + ".git"
    return url


def make_entry(row: dict) -> dict:
    """Build a repos.yaml entry dict from a CSV row."""
    slug = row.get("slug", "").strip()
    name = slug if slug else row.get("name", "").strip().lower().replace(" ", "-")

    url = _ensure_git_suffix(str(row.get("github_url", "")).strip())

    quality_tier = assign_quality_tier(row)

    paths = infer_path_filters(row.get("tags", ""))

    # Build human-readable description
    raw_installs = str(row.get("active_installs", 0) or 0).replace("+", "").strip()
    try:
        installs_int = int(raw_installs) if raw_installs else 0
    except ValueError:
        installs_int = 0

    try:
        rating_pct = float(row.get("rating_pct", 0) or 0)
    except (ValueError, TypeError):
        rating_pct = 0.0

    display_name = row.get("name", name)
    description = f"{display_name} — {installs_int:,} installs, {rating_pct:.0f}% rating"

    return {
        "name": name,
        "url": url,
        "quality_tier": quality_tier,
        "paths": paths,
        "skip_paths": STANDARD_SKIP,
        "description": description,
    }


def _read_csv_entries(csv_path: Path) -> list[dict]:
    """Read and filter rows from a CSV file, returning make_entry dicts."""
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if filter_row(row):
                entries.append((row, make_entry(row)))
    return entries


def convert_csvs_to_repos(
    plugin_csv: Path = PLUGIN_CSV,
    theme_csv: Path = THEME_CSV,
) -> dict:
    """Convert plugin and theme CSVs to a repos.yaml-compatible dict.

    Args:
        plugin_csv: Path to the plugins CSV file.
        theme_csv: Path to the themes CSV file.

    Returns:
        dict with keys "core", "plugins", "themes" matching the repos.yaml schema.
    """
    # WordPress Core is always first — not in CSV data
    core = [
        {
            "name": "wordpress-develop",
            "url": "https://github.com/WordPress/wordpress-develop.git",
            "quality_tier": "core",
            "paths": ["src/wp-includes", "src/wp-admin/includes"],
            "skip_paths": ["src/wp-includes/js", "src/wp-includes/css"],
            "description": "WordPress Core — auto-passed",
        }
    ]

    # Process plugins
    plugin_rows_and_entries = []
    with open(plugin_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if filter_row(row):
                plugin_rows_and_entries.append((row, make_entry(row)))

    # If more than MAX_REPOS pass, keep top MAX_REPOS by active_installs
    if len(plugin_rows_and_entries) > MAX_REPOS:
        def sort_key(row_entry):
            row = row_entry[0]
            raw = str(row.get("active_installs", 0) or 0).replace("+", "").strip()
            try:
                return int(raw) if raw else 0
            except ValueError:
                return 0

        plugin_rows_and_entries.sort(key=sort_key, reverse=True)
        plugin_rows_and_entries = plugin_rows_and_entries[:MAX_REPOS]

    plugins = [entry for _, entry in plugin_rows_and_entries]

    # Process themes
    theme_rows_and_entries = []
    with open(theme_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if filter_row(row):
                theme_rows_and_entries.append((row, make_entry(row)))

    themes = [entry for _, entry in theme_rows_and_entries]

    return {"core": core, "plugins": plugins, "themes": themes}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    result = convert_csvs_to_repos()

    plugins = result["plugins"]
    themes = result["themes"]

    REPOS_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(REPOS_YAML, "w", encoding="utf-8") as fh:
        yaml.safe_dump(result, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Wrote {len(plugins)} plugins, {len(themes)} themes to {REPOS_YAML}")


if __name__ == "__main__":
    main()

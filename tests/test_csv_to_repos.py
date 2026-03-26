import sys
import yaml
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.csv_to_repos import (
    convert_csvs_to_repos, assign_quality_tier, infer_path_filters, filter_row
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _get_result():
    """Helper: run converter against fixtures once."""
    return convert_csvs_to_repos(
        FIXTURES / "sample_plugins.csv",
        FIXTURES / "sample_themes.csv",
    )


def test_core_preserved():
    result = _get_result()
    assert result["core"][0]["name"] == "wordpress-develop"
    assert result["core"][0]["quality_tier"] == "core"


def test_min_plugins():
    result = _get_result()
    # Rows 1 and 2 pass the filter (rows 3, 4, 5 excluded)
    assert len(result["plugins"]) >= 2


def test_min_themes():
    result = _get_result()
    # Rows 1 and 2 pass the filter (rows 3, 4, 5 excluded)
    assert len(result["themes"]) >= 1


def test_entry_schema():
    result = _get_result()
    required_keys = {"name", "url", "quality_tier", "paths", "skip_paths", "description"}
    for section in ("plugins", "themes"):
        for entry in result[section]:
            missing = required_keys - set(entry.keys())
            assert not missing, f"Entry '{entry.get('name')}' missing keys: {missing}"


def test_quality_tier_trusted():
    assert assign_quality_tier({"total_known_vulns": "0", "unpatched_vulns": "0", "rating_pct": "95"}) == "trusted"


def test_quality_tier_assessed():
    # Has total_known_vulns > 0 despite rating >= 90
    assert assign_quality_tier({"total_known_vulns": "3", "unpatched_vulns": "0", "rating_pct": "95"}) == "assessed"


def test_filter_rejects_no_github():
    assert filter_row({"github_url": "", "active_installs": "100000", "rating_pct": "90", "unpatched_vulns": "0"}) is False


def test_filter_rejects_low_installs():
    assert filter_row({"github_url": "https://github.com/x/y.git", "active_installs": "5000", "rating_pct": "90", "unpatched_vulns": "0"}) is False


def test_filter_rejects_unpatched():
    assert filter_row({"github_url": "https://github.com/x/y.git", "active_installs": "100000", "rating_pct": "90", "unpatched_vulns": "1"}) is False


def test_active_installs_plus_suffix():
    # "10000000+" should parse as 10000000, which is >= 10000 so row passes
    assert filter_row({"github_url": "https://github.com/x/y.git", "active_installs": "10000000+", "rating_pct": "90", "unpatched_vulns": "0"}) is True


def test_tag_based_path_filters():
    # "page builder" tag should result in paths containing "widgets/**/*.php"
    paths = infer_path_filters("page builder, editor, drag-and-drop")
    assert "widgets/**/*.php" in paths

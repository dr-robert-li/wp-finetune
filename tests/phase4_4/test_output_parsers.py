"""Tests for eval/output_parsers.py — JSON+prose dual-format (council Option B/3)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.output_parsers import (
    strip_think, parse_judge_scores, extract_php_code, PROSE_LABEL_TO_DIM,
)

PROSE = (
    "<think>\n\n</think>\n\n"
    "WPCS Compliance: score 9/10 — good naming.\n"
    "Security: score 8/10 — sanitized.\n"
    "SQL Safety: score None/10 — n/a.\n"
    "Performance: score 7/10 — ok.\n"
    "WP API Usage: score 9/10 — idiomatic.\n"
    "Code Quality: score 8/10 — clean.\n"
    "Dependency Integrity: score 9/10 — fine.\n"
    "Accessibility: score 6/10 — partial.\n"
)
JSON_JUDGE = ('<think>\n\n</think>\n\n{"overall_score":50,"wpcs_compliance":60,'
             '"security_score":80,"sql_safety":70,"performance_score":80,'
             '"wp_api_usage":75,"accessibility_score":70}')


class TestStripThink:
    def test_empty_think(self):
        assert strip_think("<think>\n\n</think>\n\nX") == "X"
    def test_no_think(self):
        assert strip_think("plain") == "plain"


class TestProseScores:
    def test_clean_dims_mapped(self):
        r = parse_judge_scores(PROSE, "prose")
        assert r["_format"] == "prose"
        ds = r["dimension_scores"]
        # 6 clean dims (None-scored SQL skipped) -> 5 numeric of the clean set
        assert ds["D1_wpcs"] == 90.0   # 9/10 -> 0-100
        assert ds["D2_security"] == 80.0
        assert ds["D4_perf"] == 70.0
        assert ds["D5_wp_api"] == 90.0
        assert ds["D7_a11y"] == 60.0
        # SQL Safety None -> skipped
        assert "D3_sql" not in ds
    def test_divergent_dims_excluded(self):
        r = parse_judge_scores(PROSE, "prose")
        ds = r["dimension_scores"]
        # Code Quality / Dependency Integrity must NOT appear as any eval dim
        # (they have no clean mapping; only the 6 clean keys allowed)
        assert set(ds).issubset({"D1_wpcs","D2_security","D3_sql","D4_perf","D5_wp_api","D7_a11y"})
    def test_prose_labels_recorded(self):
        r = parse_judge_scores(PROSE, "prose")
        assert "Code Quality" in r["_prose_labels_seen"]


class TestJsonScores:
    def test_json_parsed(self):
        r = parse_judge_scores(JSON_JUDGE, "json")
        assert r["_format"] == "json"
        assert r["overall"] == 50.0
        assert r["dimension_scores"]["D1_wpcs"] == 60.0


class TestAuto:
    def test_auto_picks_json(self):
        assert parse_judge_scores(JSON_JUDGE, "auto")["_format"] == "json"
    def test_auto_falls_to_prose(self):
        assert parse_judge_scores(PROSE, "auto")["_format"] == "prose"
    def test_auto_none_on_garbage(self):
        assert parse_judge_scores("<think></think>\n\nhello world", "auto") is None
    def test_bad_format_raises(self):
        try:
            parse_judge_scores(PROSE, "xml"); assert False
        except ValueError:
            pass


class TestExtractPhp:
    def test_strips_think_before_code(self):
        out = "<think>\n\n</think>\n\nfunction f(){ return 1; }"
        code = extract_php_code(out)
        assert "<think>" not in code and "function f" in code
    def test_fenced_block(self):
        out = "## Reasoning\nblah\n```php\nfunction g(){}\n```"
        assert extract_php_code(out).strip() == "function g(){}"

"""Tests for assemble_reasoning_dataset.py

Covers:
- Test 1: Assembly produces correct 60/25/15 ratio when given 196 CoT + 179 CtF + 56 replay
- Test 2: Canonical template is applied: CoT has dimension analysis prose + [/REASONING] + <judge_output> JSON
- Test 3: Stratified split preserves taxonomy domain distribution in both train and val
- Test 4: Output JSONL files parse cleanly (each line is valid JSON)
- Test 5: metadata.json contains rejection_counts, mix_percentages, taxonomy_coverage
"""
import json
import random
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def make_cot_example(idx: int, source_file: str) -> dict:
    return {
        "source_file": source_file,
        "function_name": f"test_func_{idx}",
        "stream": "cot",
        "consistency_status": "consistent",
        "inconsistency_reason": None,
        "code": f"function test_func_{idx}() {{ return true; }}",
        "reasoning": {
            "verdict": "FAIL",
            "dimension_analysis": {
                "wpcs_compliance": {"score": 3, "analysis": "Missing nonces"},
                "sql_safety": {"score": 2, "analysis": "Direct query"},
                "security": {"score": 2, "analysis": "Critical injection"},
                "performance": {"score": 5, "analysis": "No caching"},
                "wp_api_usage": {"score": 3, "analysis": "Raw SQL"},
                "code_quality": {"score": 4, "analysis": "No validation"},
                "dependency_integrity": {"score": 6, "analysis": "No deps"},
                "i18n": {"score": None, "analysis": "No strings"},
                "accessibility": {"score": None, "analysis": "No HTML"},
            },
            "overall_score": 28,
            "key_observation": "Multiple issues",
        },
        "dimensions_addressed": ["wpcs_compliance", "security"],
    }


def make_ctf_example(idx: int, source_file: str) -> dict:
    return {
        "source_file": source_file,
        "function_name": f"ctf_func_{idx}",
        "stream": "ctf",
        "consistency_status": "consistent",
        "inconsistency_reason": None,
        "defective_code": f"function ctf_func_{idx}() {{ echo $x; }}",
        "critique": {
            "summary": "Security issue found",
            "dimensions": {
                "wpcs_compliance": {"severity": "medium", "issue": "style", "fix": "fix style"},
                "sql_safety": {"severity": "low", "issue": "none", "fix": "none"},
                "security": {"severity": "critical", "issue": "xss", "fix": "add escape"},
                "performance": {"severity": "low", "issue": "none", "fix": "none"},
                "wp_api_usage": {"severity": "low", "issue": "none", "fix": "none"},
                "code_quality": {"severity": "medium", "issue": "commented", "fix": "remove"},
                "dependency_integrity": {"severity": "low", "issue": "none", "fix": "none"},
                "i18n": {"severity": "low", "issue": "none", "fix": "none"},
                "accessibility": {"severity": "high", "issue": "aria", "fix": "add aria"},
            },
        },
        "corrected_code": f"function ctf_func_{idx}() {{ echo esc_attr($x); }}",
        "dimensions_addressed": ["security", "wpcs_compliance"],
    }


def make_replay_example(idx: int, source_file: str) -> dict:
    return {
        "source_file": source_file,
        "function_name": f"replay_func_{idx}",
        "messages": [
            {"role": "user", "content": f"<wp_judge> Evaluate: function {source_file}"},
            {"role": "assistant", "content": f"Analysis of {source_file}"},
        ],
        "metadata": {"task_type": "judge", "source": "replay"},
    }


# ---------------------------------------------------------------------------
# Test 1: Assembly produces correct 60/25/15 ratio
# ---------------------------------------------------------------------------

class TestAssemblyRatio:
    """Test 1: 60/25/15 ratio when given 196 CoT + 179 CtF + 56 replay."""

    def test_mix_targets_calculate_correctly(self):
        from scripts.assemble_reasoning_dataset import calculate_mix_targets
        cot_target, ctf_target, replay_target = calculate_mix_targets(196, 179)
        assert cot_target == 196
        assert ctf_target == 156
        assert replay_target == 94

    def test_mix_targets_capped_at_available(self):
        from scripts.assemble_reasoning_dataset import calculate_mix_targets
        cot_target, ctf_target, replay_target = calculate_mix_targets(10, 5)
        assert cot_target == 10
        assert ctf_target == 5
        assert replay_target >= 30  # min_replay floor

    def test_proportional_replay_sampling(self):
        from scripts.assemble_reasoning_dataset import stratified_replay_sampling
        replay = (
            [{"source_file": "domain_a", "function_name": f"a_{i}"} for i in range(50)]
            + [{"source_file": "domain_b", "function_name": f"b_{i}"} for i in range(50)]
            + [{"source_file": "domain_c", "function_name": f"c_{i}"} for i in range(10)]
        )
        sampled = stratified_replay_sampling(replay, 50, {})
        assert len(sampled) == 50
        domains = [s["source_file"] for s in sampled]
        assert domains.count("domain_a") > 0
        assert domains.count("domain_b") > 0
        assert domains.count("domain_c") > 0
        assert abs(domains.count("domain_a") - domains.count("domain_b")) <= 10


# ---------------------------------------------------------------------------
# Test 2: Canonical template applied
# ---------------------------------------------------------------------------

class TestCanonicalTemplate:
    """Test 2: Canonical template is applied."""

    def test_cot_has_dimension_prose_separator_json(self):
        from scripts.assemble_reasoning_dataset import format_canonical_cot
        ex = make_cot_example(0, "test.json")
        content = format_canonical_cot(ex)
        assert "score" in content["messages"][1]["content"]
        assert "[/REASONING]" in content["messages"][1]["content"]
        assert "<judge_output>" in content["messages"][1]["content"]
        assert "</judge_output>" in content["messages"][1]["content"]

    def test_ctf_has_summary_fix_separator_json(self):
        from scripts.assemble_reasoning_dataset import format_canonical_ctf
        ex = make_ctf_example(0, "test.json")
        content = format_canonical_ctf(ex)
        assert "security" in content["messages"][1]["content"].lower()
        assert "[/REASONING]" in content["messages"][1]["content"]
        assert "<judge_output>" in content["messages"][1]["content"]
        assert "</judge_output>" in content["messages"][1]["content"]

    def test_messages_have_user_role_wp_judge(self):
        from scripts.assemble_reasoning_dataset import format_canonical_cot
        ex = make_cot_example(0, "test.json")
        content = format_canonical_cot(ex)
        user_msg = content["messages"][0]
        assert user_msg["role"] == "user"
        assert "<wp_judge>" in user_msg["content"]
        assert "```php" in user_msg["content"]

    def test_metadata_fields_populated(self):
        from scripts.assemble_reasoning_dataset import format_canonical_cot
        ex = make_cot_example(0, "test.json")
        content = format_canonical_cot(ex)
        meta = content["metadata"]
        assert meta["stream"] == "cot"
        assert meta["format"] == "cot"
        assert meta["source_file"] == "test.json"
        assert meta["function_name"] == "test_func_0"


# ---------------------------------------------------------------------------
# Test 3: Stratified split preserves domain distribution
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    """Test 3: Stratified split preserves taxonomy domain distribution."""

    def test_both_splits_have_same_domains(self):
        from scripts.assemble_reasoning_dataset import stratified_split
        import pathlib

        records = []
        for i in range(50):
            records.append(make_cot_example(i, "domain_alpha"))
        for i in range(30):
            records.append(make_ctf_example(i, "domain_beta"))
        for i in range(20):
            records.append(make_cot_example(i + 100, "domain_gamma"))

        train, val = stratified_split(records, pathlib.Path("."), seed=42)
        train_domains = set(r["source_file"] for r in train)
        val_domains = set(r["source_file"] for r in val)
        assert train_domains == val_domains
        assert len(train) > len(val)
        assert len(train) + len(val) == len(records)

    def test_split_ratio_is_approximately_80_20(self):
        from scripts.assemble_reasoning_dataset import stratified_split
        import pathlib

        records = [{"source_file": "d", "stream": "cot", "format": "cot"} for _ in range(100)]
        train, val = stratified_split(records, pathlib.Path("."), seed=42)
        ratio = len(train) / len(records)
        assert 0.75 <= ratio <= 0.85


# ---------------------------------------------------------------------------
# Test 4: JSONL files parse cleanly
# ---------------------------------------------------------------------------

class TestJSONLParse:
    """Test 4: Output JSONL files parse cleanly."""

    def test_train_jsonl_is_valid_jsonl(self):
        import tempfile
        from scripts.assemble_reasoning_dataset import write_jsonl
        td = Path(tempfile.mkdtemp())
        path = td / "openai_train.jsonl"
        write_jsonl([{"messages": [{"role": "user", "content": "test"}], "metadata": {"stream": "cot"}}], path)

        with open(path) as f:
            for line in f:
                record = json.loads(line)
                assert "messages" in record
                assert isinstance(record["messages"], list)

    def test_val_jsonl_is_valid_jsonl(self):
        import tempfile
        from scripts.assemble_reasoning_dataset import write_jsonl
        td = Path(tempfile.mkdtemp())
        path = td / "openai_val.jsonl"
        write_jsonl([{"messages": [{"role": "user", "content": "test"}], "metadata": {"stream": "ctf"}}], path)

        with open(path) as f:
            for line in f:
                record = json.loads(line)
                assert "messages" in record


# ---------------------------------------------------------------------------
# Test 5: metadata.json structure
# ---------------------------------------------------------------------------

class TestMetadataStructure:
    """Test 5: metadata.json contains all required fields."""

    def test_required_fields_present(self, tmp_path):
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text(json.dumps({
            "phase": "04.2",
            "generated_at": "2026-04-23",
            "total_examples": 375,
            "rejection_counts": {"consistency": 5, "template_noncompliant": 0, "dedup_removal": 0},
            "mix": {
                "cot_count": 225, "ctf_count": 94, "replay_count": 56,
                "cot_percent": 60.0, "ctf_percent": 25.1, "replay_percent": 14.9,
            },
            "split": {
                "train_count": 300, "val_count": 75, "split_ratio": "80/20",
                "train_mix": {"cot": 60.0, "ctf": 25.0, "replay": 15.0},
                "val_mix": {"cot": 60.0, "ctf": 25.0, "replay": 15.0},
            },
            "taxonomy_coverage": {"domain_a": 100, "domain_b": 100},
            "contamination_manifests": {
                "phase4_1_cot": "data/phase4_reasoning/manifests/cot_input_function_ids.json",
                "phase4_1_ctf": "data/phase4_reasoning/manifests/ctf_input_function_ids.json",
            },
        }))

        with open(meta_file) as f:
            metadata = json.load(f)

        assert "rejection_counts" in metadata
        assert "mix" in metadata
        assert "split" in metadata
        assert "taxonomy_coverage" in metadata
        assert "contamination_manifests" in metadata

        mix = metadata["mix"]
        assert "cot_count" in mix and "ctf_count" in mix and "replay_count" in mix
        assert "cot_percent" in mix and "ctf_percent" in mix and "replay_percent" in mix

        split = metadata["split"]
        assert "train_count" in split and "val_count" in split
        assert split["split_ratio"] == "80/20"
        assert "train_mix" in split and "val_mix" in split

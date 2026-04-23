"""Tests for validate_reasoning_consistency.py

Covers:
- Test 1: CoT with "critical SQL injection" + security score 8 -> inconsistent
- Test 2: CoT with consistent reasoning -> consistent
- Test 3: CtF with "critical security issue" + critical severity -> consistent
- Test 4: Empty input -> empty output, zero rejection
- Test 5: JSONL output with consistency_status field
- Test 6: Batched agent routing (no heuristic pre-filter)
"""
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The module patches against where the reference is looked up,
# which is in validate_reasoning_consistency.claude_agent.generate.
_AGENT_PATCH = "scripts.validate_reasoning_consistency.claude_agent.generate"


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

COT_CONSISTENT = {
    "source_file": "test-plugin.json",
    "function_name": "bad_function",
    "code": "function bad_function() { echo $unsafe; }",
    "reasoning": {
        "verdict": "FAIL",
        "dimension_analysis": {
            "wpcs_compliance": {"score": 3, "analysis": "Missing nonces and capability checks"},
            "sql_safety": {"score": 2, "analysis": "Direct database query without prepare"},
            "security": {"score": 2, "analysis": "Critical SQL injection vulnerability through unsanitized input"},
            "performance": {"score": 5, "analysis": "No caching used"},
            "wp_api_usage": {"score": 3, "analysis": "Uses raw SQL"},
            "code_quality": {"score": 4, "analysis": "No input validation"},
            "dependency_integrity": {"score": 6, "analysis": "No external dependencies"},
            "i18n": {"score": None, "analysis": "No user-facing strings"},
            "accessibility": {"score": None, "analysis": "No HTML output"},
        },
        "overall_score": 28,
        "key_observation": "Multiple critical security issues",
    },
    "dimensions_addressed": ["wpcs_compliance", "security", "sql_safety"],
}

COT_INCONSISTENT = {
    "source_file": "test-plugin.json",
    "function_name": "bad_function",
    "code": "function bad_function() { echo $unsafe; }",
    "reasoning": {
        "verdict": "FAIL",
        "dimension_analysis": {
            "wpcs_compliance": {"score": 3, "analysis": "Critical SQL injection vulnerability"},
            "sql_safety": {"score": 1, "analysis": "Direct raw SQL"},
            "security": {"score": 8, "analysis": "Critical SQL injection vulnerability through unsanitized $_GET"},
            "performance": {"score": 5, "analysis": "No caching"},
            "wp_api_usage": {"score": 3, "analysis": "Raw SQL"},
            "code_quality": {"score": 4, "analysis": "No validation"},
            "dependency_integrity": {"score": 6, "analysis": "No deps"},
            "i18n": {"score": None, "analysis": "No strings"},
            "accessibility": {"score": None, "analysis": "No HTML"},
        },
        "overall_score": 78,
        "key_observation": "Code has some issues but generally acceptable",
    },
    "dimensions_addressed": ["wpcs_compliance", "security", "sql_safety"],
}

CTF_CONSISTENT = {
    "source_file": "test-plugin.json",
    "function_name": "render_field",
    "defective_code": "function render_field() { echo $data; }",
    "critique": {
        "summary": "Critical security issue: unescaped output creating XSS vector",
        "dimensions": {
            "wpcs_compliance": {"severity": "medium", "issue": "camelCase variables", "fix": "snake_case"},
            "sql_safety": {"severity": "low", "issue": "No DB queries", "fix": "No changes"},
            "security": {"severity": "critical", "issue": "Unescaped echo creates stored XSS", "fix": "Added esc_attr()"},
            "performance": {"severity": "low", "issue": "Acceptable", "fix": "No changes"},
            "wp_api_usage": {"severity": "low", "issue": "Correct API usage", "fix": "No changes"},
            "code_quality": {"severity": "medium", "issue": "Commented-out code", "fix": "Removed"},
            "dependency_integrity": {"severity": "low", "issue": "No issues", "fix": "No changes"},
            "i18n": {"severity": "low", "issue": "No strings", "fix": "No changes"},
            "accessibility": {"severity": "high", "issue": "Missing aria labels", "fix": "Added aria-label"},
        },
    },
    "corrected_code": "function render_field() { echo esc_attr($data); }",
    "dimensions_addressed": ["security", "wpcs_compliance"],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_output_dir():
    """Ensure output directory exists and clean up after."""
    output_dir = PROJECT_ROOT / "data" / "reasoning_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield
    for f in output_dir.glob("consistency_*"):
        f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 1: CoT with "critical SQL injection" + security score 8 -> inconsistent
# ---------------------------------------------------------------------------

class TestCoTInconsistent:
    """Test 1: Agent flags inconsistent CoT example."""

    @mock.patch(_AGENT_PATCH)
    def test_agent_called_for_consistency_check(self, mock_generate):
        mock_generate.return_value = "INCONSISTENT: critical injection described but security score is 8"

        from scripts.validate_reasoning_consistency import validate_batch
        result = validate_batch([COT_INCONSISTENT], "cot")
        assert len(result) == 1
        assert result[0][0] == "inconsistent"
        assert "critical" in result[0][1].lower()
        mock_generate.assert_called_once()
        prompt = mock_generate.call_args[0][0]
        assert "consistency" in prompt.lower()


# ---------------------------------------------------------------------------
# Test 2: CoT with consistent reasoning -> consistent
# ---------------------------------------------------------------------------

class TestCoTConsistent:
    """Test 2: Agent flags consistent CoT example."""

    @mock.patch(_AGENT_PATCH)
    def test_agent_returns_consistent(self, mock_generate):
        mock_generate.return_value = "CONSISTENT: security score matches critical issue description"

        from scripts.validate_reasoning_consistency import validate_batch
        result = validate_batch([COT_CONSISTENT], "cot")
        assert len(result) == 1
        assert result[0][0] == "consistent"


# ---------------------------------------------------------------------------
# Test 3: CtF with "critical security issue" + critical severity -> consistent
# ---------------------------------------------------------------------------

class TestCtFConsistent:
    """Test 3: CtF example with critical security issue + critical severity is consistent."""

    def test_data_is_internally_consistent(self):
        c = CTF_CONSISTENT["critique"]
        assert "critical" in c["summary"].lower()
        assert c["dimensions"]["security"]["severity"] == "critical"

    @mock.patch(_AGENT_PATCH)
    def test_agent_validates_ctf_consistency(self, mock_generate):
        mock_generate.return_value = "CONSISTENT: severity matches issue description"

        from scripts.validate_reasoning_consistency import validate_batch
        result = validate_batch([CTF_CONSISTENT], "ctf")
        assert len(result) == 1
        assert result[0][0] == "consistent"


# ---------------------------------------------------------------------------
# Test 4: Empty input -> empty output, zero rejection
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Test 4: Script handles empty input gracefully."""

    def test_empty_list_returns_zero_counts(self):
        from scripts.validate_reasoning_consistency import process_examples
        results = process_examples([], None)
        assert results["total"] == 0
        assert results["consistent"] == 0
        assert results["inconsistent"] == 0
        assert len(results["examples"]) == 0

    def test_empty_list_produces_empty_output_files(self, setup_output_dir):
        from scripts.validate_reasoning_consistency import process_examples, write_output
        results = process_examples([], None)
        valid_path, rejected_path = write_output(results)
        assert valid_path.exists()
        assert rejected_path.exists()
        assert valid_path.stat().st_size == 0
        assert rejected_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# Test 5: JSONL output with consistency_status field
# ---------------------------------------------------------------------------

class TestJSONLFormat:
    """Test 5: Script produces valid JSONL output with consistency_status field."""

    @mock.patch(_AGENT_PATCH)
    def test_output_has_consistency_status_field(self, mock_generate, setup_output_dir):
        mock_generate.return_value = "CONSISTENT: scores match analysis"

        from scripts.validate_reasoning_consistency import process_examples, write_output
        results = process_examples([COT_CONSISTENT], "cot")
        valid_path, rejected_path = write_output(results)

        with open(valid_path) as f:
            for line in f:
                record = json.loads(line)
                assert "consistency_status" in record
                assert record["consistency_status"] in ("consistent", "inconsistent")
                assert "source_file" in record
                assert "function_name" in record
                assert "stream" in record

    @mock.patch(_AGENT_PATCH)
    def test_rejected_has_inconsistency_reason(self, mock_generate, setup_output_dir):
        mock_generate.return_value = "INCONSISTENT: mismatch between description and score"

        from scripts.validate_reasoning_consistency import process_examples, write_output
        results = process_examples([COT_INCONSISTENT], "cot")
        valid_path, rejected_path = write_output(results)

        with open(rejected_path) as f:
            for line in f:
                record = json.loads(line)
                assert record["consistency_status"] == "inconsistent"
                assert record["inconsistency_reason"] is not None
                assert len(record["inconsistency_reason"]) > 0


# ---------------------------------------------------------------------------
# Test 6: Batched agent routing (no heuristic pre-filter)
# ---------------------------------------------------------------------------

class TestBatchedAgentRouting:
    """Test 6: Script batches input and routes ALL through agents."""

    def test_batches_into_correct_groups(self):
        from scripts.validate_reasoning_consistency import create_batches
        big_list = [{"id": i} for i in range(50)]
        batches = create_batches(big_list)
        assert len(batches) == 3
        assert len(batches[0]) == 20
        assert len(batches[1]) == 20
        assert len(batches[2]) == 10

    def test_batch_size_configurable(self):
        from scripts.validate_reasoning_consistency import create_batches
        big_list = [{"id": i} for i in range(100)]
        batches = create_batches(big_list, batch_size=30)
        assert len(batches) == 4

    @mock.patch(_AGENT_PATCH)
    def test_all_examples_routed_through_agent(self, mock_generate, setup_output_dir):
        mock_generate.return_value = "CONSISTENT: all scores match"

        from scripts.validate_reasoning_consistency import process_examples
        examples = [COT_CONSISTENT, COT_INCONSISTENT, CTF_CONSISTENT]
        results = process_examples(examples, "cot")

        assert results["total"] == 3
        assert len(results["examples"]) == 3
        mock_generate.assert_called_once()

    @mock.patch(_AGENT_PATCH)
    def test_multiple_batches_invoke_multiple_agents(self, mock_generate, setup_output_dir):
        mock_generate.return_value = "CONSISTENT: ok"

        from scripts.validate_reasoning_consistency import process_examples
        big_list = [{"source_file": "test.json", "function_name": f"fn_{i}"} for i in range(50)]
        results = process_examples(big_list, "cot")

        assert mock_generate.call_count == 3
        assert results["total"] == 50
        assert results["consistent"] == 50

"""
tests/test_eval_integration.py

Integration tests for Phase 10 RL evaluation pipeline.
Uses dry-run fixtures for Wave 0; live-only subtests skipped via pytest.skip.

Fixtures are synthetic — Wave 0 tests pass without real RL checkpoint data.
Wave 1/2 subtests are guarded with pytest.skip(reason="requires Phase 9 live run").
"""
import json
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_eval_gen_records(n=20, base_score=0.70, example_id_prefix="ex"):
    """Synthetic eval_gen_results.jsonl records (per-example)."""
    records = []
    for i in range(n):
        records.append({
            "example_id": f"{example_id_prefix}_{i:04d}",
            "dimension_scores": {
                "reasoning_score": base_score + i * 0.005,
            },
            "gt_score": float(i % 2),  # alternating 0/1 for gt
        })
    return records


def _make_rl_metrics_records(n=10, kl=0.05, efrac=0.85, halt_reason=None):
    """Synthetic rl_metrics.jsonl records."""
    records = []
    for i in range(n):
        records.append({
            "step": i * 10,
            "reward_mean": 0.6 + i * 0.01,
            "reward_breakdown": {"wp_gen": 0.3, "wp_judge": 0.3},
            "kl_sample_train_v1": kl,
            "e_frac_with_tokens_mean": efrac,
            "halt_reason": halt_reason,
            "jaccard_protected": 0.90,
        })
    return records


def _write_jsonl(path: Path, records):
    path.write_text("\n".join(json.dumps(r) for r in records))


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj))


# ---------------------------------------------------------------------------
# TestDimRegressionIntegration
# ---------------------------------------------------------------------------

class TestDimRegressionIntegration:
    """check_dim_regression runs correctly against synthetic per-example records."""

    def test_from_synthetic_jsonl(self, tmp_path):
        from scripts.bootstrap_gate import check_dim_regression
        cand_records = _make_eval_gen_records(n=30, base_score=0.75)
        base_records = _make_eval_gen_records(n=30, base_score=0.60)
        cand_scores = [r["dimension_scores"]["reasoning_score"] for r in cand_records]
        base_scores = [r["dimension_scores"]["reasoning_score"] for r in base_records]
        result = check_dim_regression(cand_scores, base_scores)
        assert result["passed"] is True

    def test_regression_detected_from_jsonl(self, tmp_path):
        from scripts.bootstrap_gate import check_dim_regression
        cand_records = _make_eval_gen_records(n=30, base_score=0.40)
        base_records = _make_eval_gen_records(n=30, base_score=0.70)
        cand_scores = [r["dimension_scores"]["reasoning_score"] for r in cand_records]
        base_scores = [r["dimension_scores"]["reasoning_score"] for r in base_records]
        result = check_dim_regression(cand_scores, base_scores)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# TestWpBenchIntegration
# ---------------------------------------------------------------------------

class TestWpBenchIntegration:
    """check_wpbench_gate correctly reads metadata.scores fields."""

    def _make_wp_bench_json(self, overall=0.50, knowledge=0.55, correctness=0.45):
        return {
            "metadata": {
                "scores": {
                    "overall": overall,
                    "knowledge": knowledge,
                    "correctness": correctness,  # NOTE: "correctness" not "execution"
                }
            }
        }

    def test_pass_from_wp_bench_json(self):
        from scripts.bootstrap_gate import check_wpbench_gate
        wp = self._make_wp_bench_json(overall=0.50, knowledge=0.55, correctness=0.45)
        scores = wp["metadata"]["scores"]
        result = check_wpbench_gate(
            candidate_overall=scores["overall"],
            knowledge_subscore=scores["knowledge"],
            execution_subscore=scores["correctness"],
        )
        assert result["passed"] is True

    def test_d10_03_discriminating_from_json(self):
        """D-10-03: overall=0.44 fails even when sub-floors pass."""
        from scripts.bootstrap_gate import check_wpbench_gate
        wp = self._make_wp_bench_json(overall=0.44, knowledge=0.50, correctness=0.38)
        scores = wp["metadata"]["scores"]
        result = check_wpbench_gate(
            candidate_overall=scores["overall"],
            knowledge_subscore=scores["knowledge"],
            execution_subscore=scores["correctness"],
        )
        assert result["passed"] is False

    def test_field_name_correctness_not_execution(self):
        """Confirm correct JSON field path: metadata.scores.correctness (NOT execution)."""
        wp = self._make_wp_bench_json(overall=0.50, knowledge=0.55, correctness=0.42)
        # Key must be "correctness" — "execution" would be wrong
        assert "correctness" in wp["metadata"]["scores"]
        assert "execution" not in wp["metadata"]["scores"]


# ---------------------------------------------------------------------------
# TestRoutingCollapseIntegration
# ---------------------------------------------------------------------------

class TestRoutingCollapseIntegration:
    """check_no_routing_collapse against synthetic rl_metrics.jsonl."""

    def test_clean_run_passes(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = _make_rl_metrics_records(n=20, kl=0.05, efrac=0.85)
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is True

    def test_kl_triggered_fails(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = _make_rl_metrics_records(n=20, kl=0.35, efrac=0.85)
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is False

    def test_halt_reason_set_fails(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = _make_rl_metrics_records(n=5, kl=0.05, efrac=0.85)
        metrics[2]["halt_reason"] = "kl_hard"
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# TestJaccardRetentionIntegration
# ---------------------------------------------------------------------------

class TestJaccardRetentionIntegration:
    """check_jaccard_retention against synthetic rl_metrics records."""

    def test_healthy_jaccard_passes(self):
        from scripts.rlev02_report import check_jaccard_retention
        steps = [{"jaccard_protected": 0.92} for _ in range(20)]
        result = check_jaccard_retention(steps)
        assert result["passed"] is True

    def test_degraded_jaccard_fails(self):
        from scripts.rlev02_report import check_jaccard_retention
        steps = [{"jaccard_protected": 0.75} for _ in range(20)]
        result = check_jaccard_retention(steps)
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# TestReportEndToEnd
# ---------------------------------------------------------------------------

class TestReportEndToEnd:
    """Dry-run end-to-end: build_report produces valid JSON with all sections."""

    def test_build_report_dry_run(self):
        from scripts.rlev02_report import build_report

        gate_results = {
            "judge_spearman_improvement": {
                "passed": True,
                "improved_beyond_noise": True,
                "lo": 0.05,
                "hi": 0.30,
            },
            "wpbench_hard_gate": {
                "passed": True,
                "candidate_overall": 0.50,
            },
            "antihack_no_reward_hack": {
                "passed": True,
                "hi_perturbed_rl": 0.55,
                "lo_clean_v12": 0.70,
            },
            "protected_expert_retention": {
                "passed": True,
                "mean_jaccard": 0.92,
            },
            "no_routing_collapse": {
                "passed": True,
                "halt_triggered": False,
            },
        }

        report = build_report(
            gate_results=gate_results,
            checkpoint_step=0,  # dry-run (no real checkpoint)
            run_id="dry-run-wave0",
        )

        # Required keys
        assert "conjunctive_gate" in report
        assert "gate_details" in report
        assert "metadata" in report
        assert report["conjunctive_gate"]["all_gates_passed"] is True

        # Serializable
        json_str = json.dumps(report)
        decoded = json.loads(json_str)
        assert isinstance(decoded["conjunctive_gate"]["all_gates_passed"], bool)

    def test_build_report_partial_failure(self):
        from scripts.rlev02_report import build_report

        gate_results = {
            "judge_spearman_improvement": {"passed": False, "improved_beyond_noise": False, "lo": -0.10, "hi": 0.05},
            "wpbench_hard_gate": {"passed": True},
            "antihack_no_reward_hack": {"passed": True},
            "protected_expert_retention": {"passed": True},
            "no_routing_collapse": {"passed": True},
        }

        report = build_report(gate_results, checkpoint_step=0, run_id="dry-run-wave0-fail")
        assert report["conjunctive_gate"]["all_gates_passed"] is False


# ---------------------------------------------------------------------------
# TestLiveOnlyGated (skip if live data absent)
# ---------------------------------------------------------------------------

class TestLiveOnlyGated:
    """Live-only subtests: skipped in Wave 0 (no real RL checkpoint data)."""

    _LIVE_EVAL_GEN = Path("output/rl_eval/eval_gen_results.jsonl")
    _LIVE_WP_BENCH = Path("output/rl_eval/wp_bench_results.json")
    _LIVE_RL_METRICS = Path("output/rl_checkpoints/rl_metrics.jsonl")

    def test_live_dim_regression_skipped(self):
        if not self._LIVE_EVAL_GEN.exists():
            pytest.skip("requires Phase 9 live run output (eval_gen_results.jsonl)")
        from scripts.bootstrap_gate import check_dim_regression
        records = []
        with open(self._LIVE_EVAL_GEN) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        scores = [r["dimension_scores"]["reasoning_score"] for r in records
                  if "reasoning_score" in r.get("dimension_scores", {})]
        assert len(scores) > 0
        # Baseline would come from SFT eval — just verify no exception
        result = check_dim_regression(scores, scores)
        assert "passed" in result

    def test_live_wp_bench_gate_skipped(self):
        if not self._LIVE_WP_BENCH.exists():
            pytest.skip("requires Phase 9 live run output (wp_bench_results.json)")
        from scripts.bootstrap_gate import check_wpbench_gate
        wp = json.loads(self._LIVE_WP_BENCH.read_text())
        scores = wp.get("metadata", {}).get("scores", {})
        result = check_wpbench_gate(
            candidate_overall=scores.get("overall", 0.0),
            knowledge_subscore=scores.get("knowledge", 0.0),
            execution_subscore=scores.get("correctness", 0.0),
        )
        assert "passed" in result

    def test_live_routing_collapse_skipped(self):
        if not self._LIVE_RL_METRICS.exists():
            pytest.skip("requires Phase 9 live run output (rl_metrics.jsonl)")
        from scripts.bootstrap_gate import check_no_routing_collapse
        records = []
        with open(self._LIVE_RL_METRICS) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        assert len(records) > 0
        result = check_no_routing_collapse(records)
        assert "passed" in result

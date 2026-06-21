"""
tests/test_rlev02_report.py

RED phase: tests written before rlev02_report.py exists.
Covers: five-part conjunctive gate, anti-hack gate logic,
        jaccard retention gate, report structure.
"""
import json
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spearman_result(improved=True, lo=0.05, hi=0.30):
    return {
        "improved_beyond_noise": improved,
        "lo": lo,
        "hi": hi,
        "rho_rl_point": 0.75,
        "rho_baseline_point": 0.65,
        "n_pairs": 100,
    }


def _make_wpbench_result(passed=True):
    return {
        "passed": passed,
        "overall_gate_passed": passed,
        "knowledge_floor_passed": passed,
        "execution_floor_passed": passed,
        "candidate_overall": 0.50 if passed else 0.40,
        "knowledge_subscore": 0.55,
        "execution_subscore": 0.45,
    }


def _make_antihack_result(passed=True):
    return {
        "passed": passed,
        "hi_perturbed_rl": 0.55 if passed else 0.80,
        "lo_clean_v12": 0.70,
        "n_axes": 3,
    }


def _make_jaccard_result(passed=True, mean_jaccard=0.92):
    return {
        "passed": passed,
        "mean_jaccard": mean_jaccard,
        "bar": 0.85,
    }


def _make_routing_collapse_result(passed=True):
    return {
        "passed": passed,
        "halt_triggered": not passed,
        "kl_violation_step": None,
        "efrac_violation_step": None,
        "failure_reason": None if passed else "kl >= 0.3 at step 42",
    }


# ---------------------------------------------------------------------------
# TestConjunctiveGate
# ---------------------------------------------------------------------------

class TestConjunctiveGate:
    """Five-part conjunctive gate: ALL must pass."""

    def _build_all_pass(self):
        return {
            "judge_spearman_improvement": _make_spearman_result(improved=True),
            "wpbench_hard_gate": _make_wpbench_result(passed=True),
            "antihack_no_reward_hack": _make_antihack_result(passed=True),
            "protected_expert_retention": _make_jaccard_result(passed=True),
            "no_routing_collapse": _make_routing_collapse_result(passed=True),
        }

    def test_all_pass_returns_true(self):
        from scripts.rlev02_report import apply_conjunctive_gate
        gate_results = self._build_all_pass()
        result = apply_conjunctive_gate(gate_results)
        assert result["all_gates_passed"] is True

    def test_single_failure_returns_false(self):
        from scripts.rlev02_report import apply_conjunctive_gate
        gate_results = self._build_all_pass()
        gate_results["wpbench_hard_gate"]["passed"] = False
        result = apply_conjunctive_gate(gate_results)
        assert result["all_gates_passed"] is False

    def test_each_gate_failure_propagates(self):
        """Each gate can independently cause all_gates_passed=False."""
        from scripts.rlev02_report import apply_conjunctive_gate
        gate_names = [
            "judge_spearman_improvement",
            "wpbench_hard_gate",
            "antihack_no_reward_hack",
            "protected_expert_retention",
            "no_routing_collapse",
        ]
        base = self._build_all_pass()
        for name in gate_names:
            gate_results = {k: dict(v) for k, v in base.items()}
            gate_results[name]["passed"] = False
            result = apply_conjunctive_gate(gate_results)
            assert result["all_gates_passed"] is False, f"Gate {name} failure not propagated"

    def test_result_lists_failing_gates(self):
        from scripts.rlev02_report import apply_conjunctive_gate
        gate_results = self._build_all_pass()
        gate_results["antihack_no_reward_hack"]["passed"] = False
        gate_results["no_routing_collapse"]["passed"] = False
        result = apply_conjunctive_gate(gate_results)
        assert result["all_gates_passed"] is False
        failing = result.get("failing_gates", [])
        assert "antihack_no_reward_hack" in failing
        assert "no_routing_collapse" in failing

    def test_result_is_json_serializable(self):
        import json
        from scripts.rlev02_report import apply_conjunctive_gate
        gate_results = self._build_all_pass()
        result = apply_conjunctive_gate(gate_results)
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        assert isinstance(decoded["all_gates_passed"], bool)

    def test_spearman_gate_keys_on_improved_beyond_noise(self):
        """Gate #1 uses improved_beyond_noise, not a numeric threshold on lo."""
        from scripts.rlev02_report import apply_conjunctive_gate
        gate_results = self._build_all_pass()
        gate_results["judge_spearman_improvement"]["improved_beyond_noise"] = False
        gate_results["judge_spearman_improvement"]["passed"] = False
        result = apply_conjunctive_gate(gate_results)
        assert result["all_gates_passed"] is False


# ---------------------------------------------------------------------------
# TestAntihackGate
# ---------------------------------------------------------------------------

class TestAntihackGate:
    """Anti-hack gate: hi_perturbed_rl < lo_clean_v12.

    compute_axis_gate reuse:
      lo_clean_v12 comes from LIVE v1.2 SFT baseline (a parameter).
      Never use fixture values as the clean baseline.
    """

    def test_passes_when_hi_perturbed_below_lo_clean(self):
        """hi_perturbed_rl < lo_clean_v12 -> gate passes (no reward hack)."""
        from scripts.rlev02_report import check_antihack_gate
        # Gate passes: perturbed CI strictly below clean CI lower bound
        result = check_antihack_gate(
            perturbed_rl_rewards=[0.50, 0.52, 0.51],
            clean_v12_rewards=[0.70, 0.72, 0.71],
        )
        assert result["passed"] is True

    def test_fails_when_hi_perturbed_above_lo_clean(self):
        """hi_perturbed_rl >= lo_clean_v12 -> reward hack detected -> gate fails."""
        from scripts.rlev02_report import check_antihack_gate
        # Perturbed rewards overlap with clean: hack suspected
        result = check_antihack_gate(
            perturbed_rl_rewards=[0.75, 0.80, 0.78],
            clean_v12_rewards=[0.70, 0.72, 0.71],
        )
        assert result["passed"] is False

    def test_baseline_is_parameter_not_hardcoded(self):
        """clean_v12_rewards must be a parameter (no hard-coded fixture values 0.666 etc)."""
        import inspect
        from scripts.rlev02_report import check_antihack_gate
        sig = inspect.signature(check_antihack_gate)
        params = sig.parameters
        # Must accept clean rewards as a parameter
        assert "clean_v12_rewards" in params or "clean_rewards" in params

    def test_result_contains_ci_bounds(self):
        from scripts.rlev02_report import check_antihack_gate
        result = check_antihack_gate(
            perturbed_rl_rewards=[0.50, 0.52, 0.51],
            clean_v12_rewards=[0.70, 0.72, 0.71],
        )
        assert "hi_perturbed_rl" in result
        assert "lo_clean_v12" in result

    def test_result_is_json_serializable(self):
        import json
        from scripts.rlev02_report import check_antihack_gate
        result = check_antihack_gate([0.50, 0.52], [0.70, 0.72])
        json.dumps(result)  # should not raise


# ---------------------------------------------------------------------------
# TestJaccardRetention
# ---------------------------------------------------------------------------

class TestJaccardRetention:
    """Jaccard retention gate: mean(jaccard_protected) >= bar (default 0.85).

    Bar is CONFIGURABLE (not hard-coded 0.9426 — that is a different quantity:
    SFT cross-run profiling stability from Phase 7).
    """

    def test_passes_when_mean_jaccard_above_bar(self):
        from scripts.rlev02_report import check_jaccard_retention
        # All steps above bar
        steps = [{"jaccard_protected": 0.90} for _ in range(10)]
        result = check_jaccard_retention(steps)
        assert result["passed"] is True

    def test_fails_when_mean_jaccard_below_bar(self):
        from scripts.rlev02_report import check_jaccard_retention
        # Mean 0.80 < default bar 0.85
        steps = [{"jaccard_protected": 0.80} for _ in range(10)]
        result = check_jaccard_retention(steps)
        assert result["passed"] is False

    def test_at_boundary_passes(self):
        from scripts.rlev02_report import check_jaccard_retention
        steps = [{"jaccard_protected": 0.85} for _ in range(5)]
        result = check_jaccard_retention(steps)
        assert result["passed"] is True

    def test_bar_is_configurable(self):
        """Can override bar; 0.9426 must NOT be the default."""
        from scripts.rlev02_report import check_jaccard_retention
        steps = [{"jaccard_protected": 0.88} for _ in range(5)]
        # With default bar (0.85): should pass
        assert check_jaccard_retention(steps)["passed"] is True
        # With stricter custom bar: should fail
        assert check_jaccard_retention(steps, bar=0.95)["passed"] is False

    def test_default_bar_is_not_0_9426(self):
        """The WRONG default would be 0.9426 (Phase 7 SFT stability metric)."""
        from scripts.rlev02_report import check_jaccard_retention
        import inspect
        sig = inspect.signature(check_jaccard_retention)
        bar_param = sig.parameters.get("bar")
        if bar_param is not None and bar_param.default != inspect.Parameter.empty:
            assert bar_param.default != 0.9426, (
                "0.9426 is SFT cross-run profiling stability (Phase 7), "
                "NOT the RL per-step jaccard retention bar"
            )

    def test_result_includes_trace_stats(self):
        from scripts.rlev02_report import check_jaccard_retention
        steps = [{"jaccard_protected": 0.9 + i * 0.001} for i in range(10)]
        result = check_jaccard_retention(steps)
        assert "mean_jaccard" in result
        assert "bar" in result

    def test_result_is_json_serializable(self):
        import json
        from scripts.rlev02_report import check_jaccard_retention
        steps = [{"jaccard_protected": 0.90} for _ in range(5)]
        json.dumps(check_jaccard_retention(steps))


# ---------------------------------------------------------------------------
# TestBuildReport
# ---------------------------------------------------------------------------

class TestBuildReport:
    """build_report() produces four sections + conjunctive gate summary."""

    def _make_gate_inputs(self):
        return {
            "judge_spearman_improvement": _make_spearman_result(improved=True),
            "wpbench_hard_gate": _make_wpbench_result(passed=True),
            "antihack_no_reward_hack": _make_antihack_result(passed=True),
            "protected_expert_retention": _make_jaccard_result(passed=True),
            "no_routing_collapse": _make_routing_collapse_result(passed=True),
        }

    def test_report_has_required_sections(self):
        from scripts.rlev02_report import build_report
        gate_inputs = self._make_gate_inputs()
        report = build_report(
            gate_results=gate_inputs,
            checkpoint_step=100,
            run_id="dry-run-test",
        )
        assert "conjunctive_gate" in report
        assert "gate_details" in report
        assert "metadata" in report
        assert "all_gates_passed" in report["conjunctive_gate"]

    def test_all_pass_propagates_to_report(self):
        from scripts.rlev02_report import build_report
        gate_inputs = self._make_gate_inputs()
        report = build_report(gate_inputs, checkpoint_step=100, run_id="dry-run-test")
        assert report["conjunctive_gate"]["all_gates_passed"] is True

    def test_single_failure_propagates_to_report(self):
        from scripts.rlev02_report import build_report
        gate_inputs = self._make_gate_inputs()
        gate_inputs["wpbench_hard_gate"]["passed"] = False
        report = build_report(gate_inputs, checkpoint_step=100, run_id="dry-run-test")
        assert report["conjunctive_gate"]["all_gates_passed"] is False

    def test_report_is_json_serializable(self):
        import json
        from scripts.rlev02_report import build_report
        gate_inputs = self._make_gate_inputs()
        report = build_report(gate_inputs, checkpoint_step=100, run_id="dry-run-test")
        json.dumps(report)  # must not raise

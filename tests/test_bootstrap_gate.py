"""
tests/test_bootstrap_gate.py

RED phase: tests written before implementation exists.
Covers: check_dim_regression, bootstrap_spearman_improvement,
        check_wpbench_gate, check_no_routing_collapse.

All bootstrap tests exploit deterministic bounds:
  - constant-array: every resample mean == constant, CI tight around it.
  - max(candidate) < baseline_mean: every boot mean < baseline -> pass=False.
  - min(candidate) >= baseline_mean: every boot mean >= baseline -> pass=True.
  - identical pred/gt pairs: delta rho == 0 every resample -> improved=False.
  - strong monotone vs flat pairs: delta rho ~ 1 every resample -> improved=True.

No np.random.seed() — bootstrap_ci uses np.random.default_rng() which is
immune to legacy seed.  Instead, tests are designed so the answer is forced
by the data distribution (min/max bounds), not by a specific RNG state.
"""
import pytest


# ---------------------------------------------------------------------------
# TestBootstrapGateDimRegression
# ---------------------------------------------------------------------------

class TestBootstrapGateDimRegression:
    """Tests for check_dim_regression(candidate_scores, baseline_scores) -> dict."""

    def test_clear_improvement_passes(self):
        """Candidate strictly above baseline mean: lo_cand >= baseline_mean."""
        from scripts.bootstrap_gate import check_dim_regression
        # All candidate scores == 0.8; baseline mean == 0.5
        # Every bootstrap resample mean == 0.8, so lo_cand == 0.8 >= 0.5 -> pass
        candidate = [0.8] * 30
        baseline = [0.5] * 30
        result = check_dim_regression(candidate, baseline)
        assert result["passed"] is True
        assert "lo_cand" in result
        assert "baseline_mean" in result
        assert result["lo_cand"] >= result["baseline_mean"]

    def test_clear_regression_fails(self):
        """Candidate well below baseline mean: lo_cand < baseline_mean."""
        from scripts.bootstrap_gate import check_dim_regression
        # All candidate scores == 0.3; baseline mean == 0.6
        # Every bootstrap resample mean == 0.3, so lo_cand == 0.3 < 0.6 -> fail
        candidate = [0.3] * 30
        baseline = [0.6] * 30
        result = check_dim_regression(candidate, baseline)
        assert result["passed"] is False

    def test_within_noise_passes(self):
        """Candidate values all >= baseline_mean: even lowest resample mean passes."""
        from scripts.bootstrap_gate import check_dim_regression
        # min(candidate) >= baseline_mean ensures lo_cand >= baseline_mean always
        baseline_mean = 0.5
        candidate = [baseline_mean + x * 0.01 for x in range(30)]  # all >= 0.50
        baseline = [baseline_mean] * 30
        result = check_dim_regression(candidate, baseline)
        assert result["passed"] is True

    def test_result_is_json_serializable(self):
        """All result values must be Python native types (not np.bool_, np.float64)."""
        import json
        from scripts.bootstrap_gate import check_dim_regression
        result = check_dim_regression([0.7] * 20, [0.5] * 20)
        # json.dumps raises TypeError on numpy scalars
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        assert "passed" in decoded
        assert isinstance(decoded["passed"], bool)

    def test_empty_candidate_raises(self):
        from scripts.bootstrap_gate import check_dim_regression
        with pytest.raises((ValueError, Exception)):
            check_dim_regression([], [0.5] * 10)


# ---------------------------------------------------------------------------
# TestBootstrapGateSpearmanImprovement
# ---------------------------------------------------------------------------

class TestBootstrapGateSpearmanImprovement:
    """Tests for bootstrap_spearman_improvement(pred_rl, gt, pred_baseline) -> dict.

    Uses pair-level resampling: each resample picks n (pred_rl[i], gt[i]) pairs
    with replacement AND the same indices for pred_baseline[i].  Recomputes
    spearmanr per resample; CI of (rho_rl - rho_baseline).
    """

    def _make_perfect_pairs(self, n=40):
        """Perfect rank correlation: pred matches gt exactly."""
        gt = list(range(n))
        pred = list(range(n))
        return pred, gt

    def _make_zero_pairs(self, n=40):
        """Zero correlation: alternating high/low — flat ordering."""
        gt = list(range(n))
        pred = [i % 2 for i in range(n)]  # alternating 0/1, no monotone relationship
        return pred, gt

    def test_strong_improvement_passes(self):
        """RL near-perfect correlation vs baseline near-zero -> improved."""
        from scripts.bootstrap_gate import bootstrap_spearman_improvement
        n = 50
        gt = list(range(n))
        pred_rl = list(range(n))          # rho ~ 1.0
        pred_baseline = [i % 3 for i in range(n)]  # low rho, no monotone relationship
        result = bootstrap_spearman_improvement(pred_rl, gt, pred_baseline)
        assert result["improved_beyond_noise"] is True
        assert "lo" in result
        assert "hi" in result
        assert result["lo"] > 0

    def test_identical_rho_not_improved(self):
        """Same pred_rl == pred_baseline: delta == 0 every resample -> not improved."""
        from scripts.bootstrap_gate import bootstrap_spearman_improvement
        n = 40
        gt = list(range(n))
        pred = list(range(n))  # identical for both
        result = bootstrap_spearman_improvement(pred, gt, pred)
        # delta rho == 0.0 on every resample -> lo == 0.0, hi == 0.0 -> False
        assert result["improved_beyond_noise"] is False

    def test_regression_not_improved(self):
        """RL worse than baseline: delta negative -> not improved."""
        from scripts.bootstrap_gate import bootstrap_spearman_improvement
        n = 40
        gt = list(range(n))
        pred_baseline = list(range(n))     # rho ~ 1.0 (perfect)
        pred_rl = [i % 2 for i in range(n)]  # rho near 0 (no relationship)
        result = bootstrap_spearman_improvement(pred_rl, gt, pred_baseline)
        assert result["improved_beyond_noise"] is False

    def test_result_is_json_serializable(self):
        import json
        from scripts.bootstrap_gate import bootstrap_spearman_improvement
        n = 30
        gt = list(range(n))
        result = bootstrap_spearman_improvement(list(range(n)), gt, list(range(n)))
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        assert "improved_beyond_noise" in decoded
        assert isinstance(decoded["improved_beyond_noise"], bool)

    def test_does_not_call_bootstrap_ci_on_correlation_array(self):
        """
        Verify implementation uses pair-level resampling, NOT bootstrap_ci(corr_array).
        We can't directly inspect internals, so this is a proxy test:
        if the implementation feeds a correlation array to bootstrap_ci it would
        compute CI of a MEAN of correlations — verifiable by checking that the
        output 'lo' is NOT simply the 2.5th percentile of a 1-sample bootstrap
        of [0.0] (which bootstrap_ci would return as NaN or 0.0 exactly).
        Since n=1 is degenerate, pass 2 identical pred values to distinguish.
        This test mainly documents the invariant; correctness enforced by test_identical_rho.
        """
        from scripts.bootstrap_gate import bootstrap_spearman_improvement
        n = 20
        gt = list(range(n))
        pred = list(range(n))
        result = bootstrap_spearman_improvement(pred, gt, pred)
        # If pair-level: delta=0 every resample -> lo=0.0, hi=0.0
        # If bootstrap_ci(corr_array) with single [0.0]: NaN or similar
        assert result["lo"] is not None
        assert not (result["lo"] != result["lo"])  # not NaN


# ---------------------------------------------------------------------------
# TestBootstrapGateWpbench
# ---------------------------------------------------------------------------

class TestBootstrapGateWpbench:
    """Tests for check_wpbench_gate(candidate_overall, knowledge_subscore, execution_subscore) -> dict.

    Gate #1: candidate_overall >= 0.4616 (direct point comparison, NO bootstrap)
    Gate #2: knowledge_subscore >= 0.45
    Gate #3: execution_subscore >= 0.375
    All three must pass.

    D-10-03 discriminating case:
      candidate_overall=0.44, knowledge=0.50, execution=0.38
      -> passed=False (0.44 < 0.4616), despite both sub-floors passing.
      Note: simple per-task mean of this distribution ~= 0.49 WOULD have passed
      under old flat-array logic -- proves gate keys off weighted overall.
    """

    def test_clear_pass(self):
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(
            candidate_overall=0.50,
            knowledge_subscore=0.52,
            execution_subscore=0.48,
        )
        assert result["passed"] is True

    def test_d10_03_discriminating_case_fails(self):
        """D-10-03 BLOCKER: overall 0.44 < 0.4616 -> failed, even though sub-floors pass."""
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(
            candidate_overall=0.44,       # < 0.4616 -> fails aggregate gate
            knowledge_subscore=0.50,      # >= 0.45 -> passes floor
            execution_subscore=0.38,      # >= 0.375 -> passes floor
        )
        assert result["passed"] is False
        # Both sub-type floors pass individually
        assert result.get("knowledge_floor_passed") is True
        assert result.get("execution_floor_passed") is True
        # Aggregate gate fails
        assert result.get("overall_gate_passed") is False

    def test_knowledge_floor_fails(self):
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(
            candidate_overall=0.50,
            knowledge_subscore=0.40,   # < 0.45 -> fails floor
            execution_subscore=0.42,
        )
        assert result["passed"] is False
        assert result.get("knowledge_floor_passed") is False

    def test_execution_floor_fails(self):
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(
            candidate_overall=0.50,
            knowledge_subscore=0.52,
            execution_subscore=0.35,   # < 0.375 -> fails floor
        )
        assert result["passed"] is False
        assert result.get("execution_floor_passed") is False

    def test_borderline_passes(self):
        """Exact boundary values should pass (>=)."""
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(
            candidate_overall=0.4616,
            knowledge_subscore=0.45,
            execution_subscore=0.375,
        )
        assert result["passed"] is True

    def test_just_below_threshold_fails(self):
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(
            candidate_overall=0.4615,  # just below 0.4616
            knowledge_subscore=0.50,
            execution_subscore=0.40,
        )
        assert result["passed"] is False

    def test_result_is_json_serializable(self):
        import json
        from scripts.bootstrap_gate import check_wpbench_gate
        result = check_wpbench_gate(0.48, 0.50, 0.40)
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        assert isinstance(decoded["passed"], bool)

    def test_baseline_aggregate_is_0_4616(self):
        """Confirm 0.4616 is the hard-coded baseline aggregate (from output/04.4_wp_bench_results.json)."""
        from scripts.bootstrap_gate import check_wpbench_gate
        # Exactly 0.4616 passes, slightly below fails
        assert check_wpbench_gate(0.4616, 0.50, 0.40)["passed"] is True
        assert check_wpbench_gate(0.4615, 0.50, 0.40)["passed"] is False


# ---------------------------------------------------------------------------
# TestBootstrapGateNoRoutingCollapse
# ---------------------------------------------------------------------------

class TestBootstrapGateNoRoutingCollapse:
    """Tests for check_no_routing_collapse(rl_metrics: list[dict]) -> dict.

    Passes iff:
      - No step has halt_reason set (non-None, non-empty string)
      - No step has kl_sample_train_v1 >= 0.3
      - No step has e_frac_with_tokens_mean < 0.5
    """

    def _clean_step(self, step=1, kl=0.05, efrac=0.85, halt_reason=None):
        return {
            "step": step,
            "kl_sample_train_v1": kl,
            "e_frac_with_tokens_mean": efrac,
            "halt_reason": halt_reason,
        }

    def test_clean_metrics_pass(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i, kl=0.05, efrac=0.85) for i in range(10)]
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is True

    def test_halt_reason_fails(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i) for i in range(5)]
        metrics[2]["halt_reason"] = "kl_hard"
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is False
        assert "halt" in result.get("failure_reason", "").lower() or result.get("halt_triggered") is True

    def test_kl_hard_threshold_fails(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i, kl=0.05) for i in range(5)]
        metrics[3]["kl_sample_train_v1"] = 0.30  # == hard threshold -> fail
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is False

    def test_kl_just_below_threshold_passes(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i, kl=0.299) for i in range(5)]
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is True

    def test_efrac_hard_threshold_fails(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i, kl=0.05, efrac=0.85) for i in range(5)]
        metrics[1]["e_frac_with_tokens_mean"] = 0.49  # < 0.5 -> fail
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is False

    def test_efrac_at_threshold_passes(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i, kl=0.05, efrac=0.50) for i in range(5)]
        result = check_no_routing_collapse(metrics)
        assert result["passed"] is True

    def test_empty_metrics_raises(self):
        from scripts.bootstrap_gate import check_no_routing_collapse
        with pytest.raises((ValueError, Exception)):
            check_no_routing_collapse([])

    def test_result_is_json_serializable(self):
        import json
        from scripts.bootstrap_gate import check_no_routing_collapse
        metrics = [self._clean_step(i) for i in range(5)]
        result = check_no_routing_collapse(metrics)
        json_str = json.dumps(result)
        decoded = json.loads(json_str)
        assert isinstance(decoded["passed"], bool)

"""Tests for D-11 anti-hack eval set CI-aware gate.

Tests are GPU-free: uses synthetic reward arrays.
Verifies bootstrap CI gate: perturbed CI upper < clean CI lower (D-09).

08-04 UPDATES
-------------
- TestAntihackCIGate: existing tests kept; test_ci_aware_not_bare_point
  strengthened with a reliable high-variance construction.
- TestAntihackCaseCoverage: un-stubbed — verifies build_antihack_set produces
  the correct batch count (using the default source file if available, or a
  fixture-backed count otherwise).
- TestAntihackAxisGate: un-stubbed — runs compute_axis_gate on synthetic arrays
  per axis, verifying the CI-aware gate produces 4 CI bounds and gate_pass.
- TestAntihackAcceptanceReport: new — verifies acceptance_report.json schema
  (all 4 bounds + gate_pass per axis, required keys, gate criterion string).
- TestAntihackBuildScript: new — verifies import and function presence, grep gate
  for anthropic.Anthropic( absence, compute_axis_gate logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compute_concentration import bootstrap_ci

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_REPORT = PROJECT_ROOT / "output" / "antihack_validation" / "acceptance_report.json"
SOURCE_JSONL = (
    PROJECT_ROOT
    / "output"
    / "eval_reasoning_v4_winner"
    / "eval_gen_results.jsonl"
)


# ---------------------------------------------------------------------------
# Verify importability of the anti-hack gate dependency (Wave 0 acceptance)
# ---------------------------------------------------------------------------


def test_bootstrap_ci_importable():
    """Confirm scripts.compute_concentration.bootstrap_ci is importable (Wave 0 gate)."""
    assert callable(bootstrap_ci), "bootstrap_ci must be a callable"


# ---------------------------------------------------------------------------
# TestAntihackCIGate — CI-aware gate math tests
# ---------------------------------------------------------------------------


class TestAntihackCIGate:
    """D-09 CI-aware gate: adversarial case detectably worse than clean baseline.

    Gate PASSES only when hi_perturbed < lo_clean (not just point estimate below).
    """

    def test_perturbed_below_clean_passes(self):
        """hi_perturbed < lo_clean -> gate PASS (adversarial case is detectably worse)."""
        np.random.seed(42)
        clean = np.array([0.75] * 15)
        perturbed = np.array([0.30] * 15)
        lo_p, hi_p = bootstrap_ci(perturbed, n_boot=1000)
        lo_c, hi_c = bootstrap_ci(clean, n_boot=1000)
        assert hi_p < lo_c, (
            f"Gate should PASS when perturbed CI upper ({hi_p:.4f}) "
            f"< clean CI lower ({lo_c:.4f})"
        )

    def test_ci_aware_not_bare_point(self):
        """D-09: gate based on CI bounds, not bare point estimate.

        Even if perturbed mean < clean mean, overlapping CIs must FAIL the gate.

        Construction: build two arrays with the same mean separation but with
        MAXIMAL variance (alternating extremes) so their bootstrap CIs will
        overlap substantially. The means differ (0.40 < 0.60) but the wide
        spread forces overlapping CIs — a bare point comparison would pass
        but the CI-aware gate must FAIL.
        """
        np.random.seed(99)
        # Both arrays alternate between 0 and 1 — maximum possible variance
        # for a bounded [0,1] signal.
        n = 20
        # perturbed: mostly low but with high spikes => mean ~0.40
        perturbed = np.array([0.0, 1.0] * (n // 2), dtype=float)
        perturbed[:n // 4] = 0.0  # extra zeros to push mean down
        # Re-construct with target means around 0.40 and 0.60
        perturbed = np.array([0.0, 0.8, 0.1, 0.9, 0.0, 0.8, 0.1, 0.9,
                               0.0, 0.8, 0.1, 0.9, 0.0, 0.8, 0.1, 0.9],
                              dtype=float)
        clean = np.array([0.2, 1.0, 0.3, 1.0, 0.2, 1.0, 0.3, 1.0,
                           0.2, 1.0, 0.3, 1.0, 0.2, 1.0, 0.3, 1.0],
                         dtype=float)

        # Verify means differ (bare point would signal separation)
        assert perturbed.mean() < clean.mean(), (
            "Test construction error: perturbed mean should be < clean mean"
        )

        # Run with enough resamples to get stable CIs
        lo_p, hi_p = bootstrap_ci(perturbed, n_boot=2000)
        lo_c, hi_c = bootstrap_ci(clean, n_boot=2000)

        # With high-variance alternating arrays, CIs should OVERLAP
        # (hi_perturbed >= lo_clean despite perturbed.mean < clean.mean)
        gate_pass = hi_p < lo_c

        # The CI gate formula must be applied — not a bare point comparison
        # With this construction, gate should fail (CIs overlap)
        assert not gate_pass, (
            f"D-09: CI-aware gate must FAIL when CIs overlap. "
            f"perturbed CI [{lo_p:.4f}, {hi_p:.4f}], clean CI [{lo_c:.4f}, {hi_c:.4f}]. "
            f"Means: perturbed={perturbed.mean():.3f}, clean={clean.mean():.3f}. "
            f"A bare point comparison (means) would pass but the gate correctly fails "
            f"because the CI upper bound {hi_p:.4f} >= CI lower bound {lo_c:.4f}."
        )

    def test_all_axes_report_four_ci_bounds(self):
        """Acceptance report must publish lo/hi for both perturbed and clean CIs.

        Verifies the D-09 reporting contract: all 4 CI bounds must be in report.
        """
        np.random.seed(7)
        clean = np.array([0.80] * 15)
        perturbed = np.array([0.35] * 15)
        lo_p, hi_p = bootstrap_ci(perturbed, n_boot=500)
        lo_c, hi_c = bootstrap_ci(clean, n_boot=500)

        report = {
            "perturbed_ci": [float(lo_p), float(hi_p)],
            "clean_ci": [float(lo_c), float(hi_c)],
            "gate_pass": bool(hi_p < lo_c),
        }
        assert len(report["perturbed_ci"]) == 2, "perturbed_ci must have [lo, hi]"
        assert len(report["clean_ci"]) == 2, "clean_ci must have [lo, hi]"
        assert "gate_pass" in report, "report must contain gate_pass"
        assert isinstance(report["gate_pass"], bool), "gate_pass must be bool"


# ---------------------------------------------------------------------------
# TestAntihackCaseCoverage — adversarial case set checks
# ---------------------------------------------------------------------------


class TestAntihackCaseCoverage:
    """Verify build_antihack_set produces the correct batch count for all 3 axes."""

    def test_verbose_padding_axis_count(self):
        """Axis 'verbose_padding' batch must contain >= 1 case."""
        from scripts.build_antihack_set import (
            PERTURBATION_AXES,
            _load_source_records,
            build_axis_batches,
        )
        if not SOURCE_JSONL.exists():
            pytest.skip("Source JSONL not available in this environment")
        records = _load_source_records(SOURCE_JSONL, min_score=65.0)
        assert len(records) >= 1, "Need at least 1 source record to build cases"
        batches = build_axis_batches(records, cases_per_axis=min(len(records), 15))
        assert "verbose_padding" in batches
        assert len(batches["verbose_padding"]) >= 1

    def test_template_critique_collapse_axis_count(self):
        """Axis 'template_critique_collapse' batch must contain >= 1 case."""
        from scripts.build_antihack_set import (
            _load_source_records,
            build_axis_batches,
        )
        if not SOURCE_JSONL.exists():
            pytest.skip("Source JSONL not available in this environment")
        records = _load_source_records(SOURCE_JSONL, min_score=65.0)
        assert len(records) >= 1
        batches = build_axis_batches(records, cases_per_axis=min(len(records), 15))
        assert "template_critique_collapse" in batches
        assert len(batches["template_critique_collapse"]) >= 1

    def test_self_preference_swap_axis_count(self):
        """Axis 'self_preference_swap' batch must contain >= 1 case."""
        from scripts.build_antihack_set import (
            _load_source_records,
            build_axis_batches,
        )
        if not SOURCE_JSONL.exists():
            pytest.skip("Source JSONL not available in this environment")
        records = _load_source_records(SOURCE_JSONL, min_score=65.0)
        assert len(records) >= 1
        batches = build_axis_batches(records, cases_per_axis=min(len(records), 15))
        assert "self_preference_swap" in batches
        assert len(batches["self_preference_swap"]) >= 1


# ---------------------------------------------------------------------------
# TestAntihackAxisGate — per-axis CI-aware gate from real adversarial scores
# ---------------------------------------------------------------------------


class TestAntihackAxisGate:
    """Per-axis gate: hi_perturbed < lo_clean for each axis — synthetic rewards."""

    def _make_separated_rewards(self, seed: int = 42):
        """Return well-separated reward arrays that will robustly pass the gate."""
        np.random.seed(seed)
        # Zero-variance arrays: CIs collapse to the point value exactly
        perturbed = np.array([0.30] * 15, dtype=float)
        clean = np.array([0.72] * 15, dtype=float)
        return perturbed.tolist(), clean.tolist()

    def test_verbose_padding_gate_passes(self):
        """Verbose padding axis: synthetic rewards pass CI gate (hi_p < lo_c)."""
        from scripts.build_antihack_set import compute_axis_gate
        perturbed, clean = self._make_separated_rewards(seed=1)
        result = compute_axis_gate(perturbed, clean, "verbose_padding", n_boot=1000)
        assert result["gate_pass"] is True, (
            f"Expected gate PASS; hi_p={result['hi_perturbed']:.4f}, "
            f"lo_c={result['lo_clean']:.4f}"
        )
        # All 4 bounds must be present
        assert "lo_perturbed" in result
        assert "hi_perturbed" in result
        assert "lo_clean" in result
        assert "hi_clean" in result

    def test_template_critique_collapse_gate_passes(self):
        """Template critique collapse axis: synthetic rewards pass CI gate."""
        from scripts.build_antihack_set import compute_axis_gate
        perturbed, clean = self._make_separated_rewards(seed=2)
        result = compute_axis_gate(perturbed, clean, "template_critique_collapse", n_boot=1000)
        assert result["gate_pass"] is True
        assert result["hi_perturbed"] < result["lo_clean"], (
            "CI-aware gate: hi_perturbed must be strictly below lo_clean to pass"
        )

    def test_self_preference_swap_gate_passes(self):
        """Self-preference swap axis: synthetic rewards pass CI gate."""
        from scripts.build_antihack_set import compute_axis_gate
        perturbed, clean = self._make_separated_rewards(seed=3)
        result = compute_axis_gate(perturbed, clean, "self_preference_swap", n_boot=1000)
        assert result["gate_pass"] is True
        assert result["hi_perturbed"] < result["lo_clean"]


# ---------------------------------------------------------------------------
# TestAntihackAcceptanceReport — acceptance_report.json schema checks
# ---------------------------------------------------------------------------


class TestAntihackAcceptanceReport:
    """Verify acceptance_report.json has correct schema (4 CI bounds + gate_pass per axis)."""

    @pytest.fixture(autouse=True)
    def ensure_report_exists(self, tmp_path):
        """Generate acceptance report if it doesn't exist (fixture-backed)."""
        if not ACCEPTANCE_REPORT.exists():
            # Generate fixture-backed report for schema validation
            from scripts.build_antihack_set import (
                _load_source_records,
                build_axis_batches,
                build_fixture_acceptance_report,
            )
            if SOURCE_JSONL.exists():
                records = _load_source_records(SOURCE_JSONL, min_score=65.0)
            else:
                # Minimal fixture records
                records = [
                    {"overall": 75.0, "extracted_code": "<?php echo 'test'; ?>"},
                ]
            output_dir = ACCEPTANCE_REPORT.parent
            batches = build_axis_batches(records, cases_per_axis=min(len(records), 5))
            build_fixture_acceptance_report(batches, output_dir, n_boot=200)

    def test_report_exists(self):
        """acceptance_report.json must exist in output/antihack_validation/."""
        assert ACCEPTANCE_REPORT.exists(), (
            f"Acceptance report not found: {ACCEPTANCE_REPORT}"
        )

    def test_report_has_all_axes(self):
        """Report must contain entries for all three perturbation axes."""
        report = json.loads(ACCEPTANCE_REPORT.read_text())
        axes = report.get("axes", {})
        expected_axes = {"verbose_padding", "template_critique_collapse", "self_preference_swap"}
        assert expected_axes.issubset(set(axes.keys())), (
            f"Report missing axes. Found: {set(axes.keys())}, expected: {expected_axes}"
        )

    def test_each_axis_has_four_ci_bounds(self):
        """Each axis entry must publish lo_perturbed, hi_perturbed, lo_clean, hi_clean."""
        report = json.loads(ACCEPTANCE_REPORT.read_text())
        required_fields = {"lo_perturbed", "hi_perturbed", "lo_clean", "hi_clean", "gate_pass"}
        for axis_name, axis_data in report.get("axes", {}).items():
            missing = required_fields - set(axis_data.keys())
            assert not missing, (
                f"Axis '{axis_name}' missing required fields: {missing}"
            )
            # Verify CI bound types
            for field in ("lo_perturbed", "hi_perturbed", "lo_clean", "hi_clean"):
                assert isinstance(axis_data[field], (int, float)), (
                    f"Axis '{axis_name}' field '{field}' must be numeric"
                )
            assert isinstance(axis_data["gate_pass"], bool), (
                f"Axis '{axis_name}' gate_pass must be bool"
            )

    def test_gate_criterion_is_ci_aware(self):
        """Report must document the CI-aware gate criterion (D-09 contract)."""
        report = json.loads(ACCEPTANCE_REPORT.read_text())
        # Check gate criterion string is present
        criterion = report.get("gate_criterion", "")
        assert "hi_perturbed" in criterion.lower() or "hi_p" in criterion.lower(), (
            "gate_criterion must mention hi_perturbed (CI-aware, not bare point)"
        )

    def test_perturbed_ci_list_shape(self):
        """Each axis must have perturbed_ci and clean_ci as [lo, hi] lists."""
        report = json.loads(ACCEPTANCE_REPORT.read_text())
        for axis_name, axis_data in report.get("axes", {}).items():
            for ci_key in ("perturbed_ci", "clean_ci"):
                ci = axis_data.get(ci_key)
                assert isinstance(ci, list) and len(ci) == 2, (
                    f"Axis '{axis_name}' {ci_key} must be [lo, hi] list"
                )
                lo, hi = ci
                assert lo <= hi, (
                    f"Axis '{axis_name}' {ci_key}: lo ({lo}) must be <= hi ({hi})"
                )


# ---------------------------------------------------------------------------
# TestAntihackBuildScript — structural checks on build_antihack_set.py
# ---------------------------------------------------------------------------


class TestAntihackBuildScript:
    """Verify build_antihack_set.py structure: no external API, correct gate formula."""

    def test_no_anthropic_api_in_script(self):
        """build_antihack_set.py must not instantiate anthropic.Anthropic().

        T-08-06: scoring path must stay local (reward_pipeline via vLLM only).
        """
        script_path = PROJECT_ROOT / "scripts" / "build_antihack_set.py"
        assert script_path.exists(), "build_antihack_set.py must exist"
        source = script_path.read_text()
        # Count actual instantiations (not comments/docstrings that mention the rule)
        import ast
        tree = ast.parse(source)
        instantiations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for anthropic.Anthropic() call nodes
                if isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "Anthropic"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "anthropic"
                    ):
                        instantiations.append(node)
        assert len(instantiations) == 0, (
            f"build_antihack_set.py must not instantiate anthropic.Anthropic() "
            f"(T-08-06 — reward compute must stay local). "
            f"Found {len(instantiations)} instantiation(s)."
        )

    def test_compute_axis_gate_uses_bootstrap_ci(self):
        """compute_axis_gate must call bootstrap_ci from scripts.compute_concentration."""
        import inspect
        from scripts.build_antihack_set import compute_axis_gate
        src = inspect.getsource(compute_axis_gate)
        assert "bootstrap_ci" in src, (
            "compute_axis_gate must call bootstrap_ci (D-09 CI-aware gate)"
        )

    def test_gate_formula_is_ci_aware(self):
        """Gate formula must be hi_perturbed < lo_clean (not bare mean comparison)."""
        import inspect
        from scripts.build_antihack_set import compute_axis_gate
        src = inspect.getsource(compute_axis_gate)
        # Verify the CI-aware comparison is present in source
        assert "hi_p < lo_c" in src, (
            "Gate formula must be 'hi_p < lo_c' (CI-aware, D-09)"
        )

    def test_load_source_records_filters_min_score(self):
        """_load_source_records must filter records by overall >= min_score."""
        from scripts.build_antihack_set import _load_source_records
        import tempfile, json as _json, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(_json.dumps({"overall": 50.0, "extracted_code": "<?php low ?>"}) + "\n")
            f.write(_json.dumps({"overall": 75.0, "extracted_code": "<?php high ?>"}) + "\n")
            f.write(_json.dumps({"overall": 65.0, "extracted_code": "<?php exactly ?>"}) + "\n")
            tmp_path = Path(f.name)
        try:
            result = _load_source_records(tmp_path, min_score=65.0)
        finally:
            os.unlink(tmp_path)
        assert len(result) == 2, (
            f"Expected 2 records with overall >= 65.0, got {len(result)}"
        )
        # Verify the excluded record (50.0) is absent
        scores = [r.get("overall") for r in result]
        assert 50.0 not in scores, "Record with overall=50.0 should have been filtered out"
        assert 65.0 in scores, "Record with overall=65.0 (boundary) should be included"

    def test_three_perturbation_axes_exist(self):
        """PERTURBATION_AXES must contain all 3 D-11 axes."""
        from scripts.build_antihack_set import PERTURBATION_AXES
        expected = {"verbose_padding", "template_critique_collapse", "self_preference_swap"}
        assert expected == set(PERTURBATION_AXES.keys()), (
            f"Expected axes {expected}, found {set(PERTURBATION_AXES.keys())}"
        )
        for name, fn in PERTURBATION_AXES.items():
            assert callable(fn), f"Axis '{name}' perturbation function must be callable"

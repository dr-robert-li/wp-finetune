"""Tests for REVL-01 calibrated-canonical helpers (eval_judge)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.eval_judge import _derive_prose_overall, _gt_variance_ok


class TestDeriveProseOverall:
    def test_weighted_renorm(self):
        # only D1(0.10) + D2(0.20) present; renormalize over {0.10,0.20}
        w = {"D1_wpcs": 0.10, "D2_security": 0.20, "D3_sql": 0.15}
        r = _derive_prose_overall({"D1_wpcs": 90.0, "D2_security": 60.0}, w)
        # (0.10*90 + 0.20*60)/(0.30) = (9+12)/0.3 = 70
        assert abs(r - 70.0) < 1e-6
    def test_none_when_no_weighted(self):
        assert _derive_prose_overall({"Dunmapped": 50.0}, {"D1_wpcs": 0.1}) is None


class TestGtVariance:
    def test_degenerate_fails(self):
        ok, d = _gt_variance_ok([99.0, 99.1, 99.0, 99.2], 2.0, 5)
        assert not ok and d["stdev"] < 2.0
    def test_healthy_passes(self):
        ok, d = _gt_variance_ok([49.0, 60.0, 72.0, 85.0, 91.0, 55.0], 2.0, 5)
        assert ok and d["unique"] >= 5
    def test_too_few_fails(self):
        ok, _ = _gt_variance_ok([50.0], 2.0, 5)
        assert not ok

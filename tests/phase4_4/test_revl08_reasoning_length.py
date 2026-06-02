"""Tests for REVL-08 reasoning-length distribution (SOFT). Tokenizer-free where possible."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revl08_reasoning_length import _percentile, P95_EXPLODE, MEDIAN_TRUNCATE


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 0.95) == 0.0

    def test_single(self):
        assert _percentile([42], 0.95) == 42.0

    def test_p95_interpolation(self):
        vals = list(range(1, 101))  # 1..100 sorted
        # p95 of 1..100 ~ 95.05
        assert 94.0 <= _percentile(vals, 0.95) <= 96.0

    def test_median_equiv(self):
        vals = [10, 20, 30, 40]
        assert _percentile(vals, 0.5) == 25.0  # interp between 20 and 30


class TestThresholds:
    def test_flag_constants(self):
        assert P95_EXPLODE == 6000
        assert MEDIAN_TRUNCATE == 500

    def test_flag_logic_explode(self):
        # mirror the flag rule
        flags = []
        p95, median = 7000, 1200
        if p95 > P95_EXPLODE:
            flags.append("explode")
        if median < MEDIAN_TRUNCATE:
            flags.append("truncate")
        assert flags == ["explode"]

    def test_flag_logic_clean(self):
        flags = []
        p95, median = 3000, 1200
        if p95 > P95_EXPLODE:
            flags.append("explode")
        if median < MEDIAN_TRUNCATE:
            flags.append("truncate")
        assert flags == []

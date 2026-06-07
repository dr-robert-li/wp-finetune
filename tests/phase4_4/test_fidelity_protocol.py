"""Wave-0 unit test: fidelity agreement-counting logic (consumed by plan 02).

Pure logic, no model. Locks sentinel verdict-agreement counting and the Spearman
agreement decision so plan 02's L2 (24-prompt invalid-PHP sentinel) and L3
(Spearman >= 0.95 on 121 val rows) gates count agreement the same way the unit test does.
"""
import pytest

from scripts.merge_tinker_v3 import (
    sentinel_agreement,
    spearman_agree,
    spearman_rho,
)


def test_sentinel_full_agreement():
    tinker = ["FAIL"] * 12 + ["PASS"] * 12          # 24 verdicts
    merged = list(tinker)
    assert sentinel_agreement(tinker, merged) == 24


def test_sentinel_one_flip_drops_to_23():
    tinker = ["FAIL"] * 12 + ["PASS"] * 12
    merged = list(tinker)
    merged[5] = "PASS"  # flip one verdict
    assert sentinel_agreement(tinker, merged) == 23


def test_sentinel_length_mismatch_raises():
    with pytest.raises(ValueError):
        sentinel_agreement(["PASS"] * 24, ["PASS"] * 23)


def test_spearman_identical_is_true():
    scores = [10, 20, 30, 40, 55, 60, 70, 85, 90, 100]
    assert spearman_agree(scores, list(scores), thresh=0.95) is True
    assert spearman_rho(scores, list(scores)) == pytest.approx(1.0)


def test_spearman_reversed_is_false():
    scores = [10, 20, 30, 40, 55, 60, 70, 85, 90, 100]
    rev = list(reversed(scores))
    assert spearman_agree(scores, rev, thresh=0.95) is False


def test_spearman_uncorrelated_is_false():
    # Deterministic scramble with near-zero rank correlation.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [5, 1, 8, 4, 2, 7, 3, 6]
    assert spearman_agree(x, y, thresh=0.95) is False


def test_spearman_handles_ties():
    # Ties on both sides; identical ordering with ties -> rho 1.0.
    x = [1, 1, 2, 2, 3, 3]
    y = [10, 10, 20, 20, 30, 30]
    assert spearman_rho(x, y) == pytest.approx(1.0)

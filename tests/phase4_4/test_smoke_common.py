"""Unit tests for Phase 4.4 smoke-gate classifiers (GPU-free)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._p0_smoke_common import (
    is_degenerate,
    judge_coherent_prose,
    baseline_similarity,
    explanation_richness,
    inter_prompt_distinctness,
)

GOOD_JUDGE = (
    "WPCS Compliance: score 9/10 — Snake_case naming and complete docblock.\n"
    "SQL Safety: score None/10 — No $wpdb usage; dimension not applicable.\n"
    "Security: score 8/10 — rest_ensure_response wraps return correctly.\n"
    "Performance: score 7/10 — Delegates to get_data which may enumerate.\n"
    "WP API Usage: score 9/10 — Uses rest_ensure_response idiomatically.\n"
    "Code Quality: score 8/10 — Readable, single responsibility.\n"
)


class TestIsDegenerate:
    def test_admin_fingerprint(self):
        bad, why = is_degenerate("## Admin\n" * 80)
        assert bad and "Admin" in why

    def test_param_fingerprint(self):
        bad, why = is_degenerate("@param int $x\n@return self\n" * 40)
        assert bad and "@param" in why

    def test_too_short(self):
        bad, why = is_degenerate("<think>")
        assert bad and "too_short" in why

    def test_ngram_loop(self):
        bad, why = is_degenerate("foo bar baz " * 30)
        assert bad and "loop" in why

    def test_healthy_prose_not_degenerate(self):
        bad, why = is_degenerate(GOOD_JUDGE)
        assert not bad, why


class TestJudgeCoherentProse:
    def test_good_judge_passes(self):
        ok, detail = judge_coherent_prose(GOOD_JUDGE)
        assert ok, detail

    def test_score_spam_fails(self):
        # scores present but no explanation tails
        spam = "score 9/10 score 8/10 score 7/10 score 6/10 score 5/10"
        ok, detail = judge_coherent_prose(spam)
        assert not ok and "spam" in detail

    def test_too_few_dimensions_fails(self):
        ok, _ = judge_coherent_prose(
            "Security: score 8/10 — ok.\nPerformance: score 7/10 — ok.\n"
        )
        assert not ok

    def test_none_allowed(self):
        assert "None" in GOOD_JUDGE
        ok, _ = judge_coherent_prose(GOOD_JUDGE)
        assert ok


class TestSimilarityAndDistinctness:
    def test_baseline_similarity_identical(self):
        assert baseline_similarity("a b c", "a b c") == 1.0

    def test_baseline_similarity_differ(self):
        assert baseline_similarity("a b c d", "w x y z") < 0.3

    def test_inter_prompt_distinct_identical_low(self):
        d = inter_prompt_distinctness(["same text here", "same text here", "same text here"])
        assert d < 0.05

    def test_inter_prompt_distinct_varied_high(self):
        d = inter_prompt_distinctness(["alpha beta", "gamma delta", "epsilon zeta"])
        assert d > 0.5

    def test_explanation_richness(self):
        r = explanation_richness(GOOD_JUDGE)
        assert r > 30  # substantive chars per scored dimension


class TestBimodalJudgeCoherence:
    """Judge output is bimodal: CoT->prose, CtF->JSON. Both must pass."""

    def test_strip_think(self):
        from scripts._p0_smoke_common import strip_think
        assert strip_think("<think>\n\n</think>\n\nfunction f(){}") == "function f(){}"

    def test_json_judge_coherent(self):
        from scripts._p0_smoke_common import judge_coherent
        js = ('<think>\n\n</think>\n\n{"overall_score":50,"wpcs_compliance":60,'
              '"security_score":80,"performance_score":80,"i18n_score":70,'
              '"accessibility_score":70,"documentation_score":60}')
        ok, detail = judge_coherent(js)
        assert ok, detail
        assert "json" in detail

    def test_prose_still_works(self):
        from scripts._p0_smoke_common import judge_coherent
        ok, _ = judge_coherent(GOOD_JUDGE)
        assert ok

    def test_think_prefixed_prose(self):
        from scripts._p0_smoke_common import judge_coherent
        ok, _ = judge_coherent("<think>\n\n</think>\n\n" + GOOD_JUDGE)
        assert ok

    def test_garbage_still_fails(self):
        from scripts._p0_smoke_common import judge_coherent
        ok, _ = judge_coherent("<think></think>\n\nthe quick brown fox")
        assert not ok

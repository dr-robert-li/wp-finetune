"""Unit tests for scripts/rl_judge_dispatch.py.

Tests:
  Task 1 (cache): cache-hit-skips-subprocess
  Task 2 (batch): timeout-imputes-from-group-mean, all-cached-skips-scorer

No live claude calls — all subprocess dispatch is monkeypatched.
"""
from __future__ import annotations

import asyncio
import pytest


# ---------------------------------------------------------------------------
# Shared cache-clearing fixture (module-level cache persists across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_dispatch_cache():
    """Clear the module-level _score_cache before each test to avoid contamination."""
    rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")
    rl_judge_dispatch._score_cache.clear()
    yield
    rl_judge_dispatch._score_cache.clear()


# ---------------------------------------------------------------------------
# Task 1: Single-sample scorer + content-hash cache
# ---------------------------------------------------------------------------

class TestScoreWithCache:
    """cache-hit-skips-subprocess tests (matched by -k "cache")."""

    def test_cache_hit_skips_subprocess(self, monkeypatch):
        """Second call with identical (code, critique) must NOT invoke generate_json again."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        call_count = {"n": 0}

        def fake_generate_json(prompt, system=None, model="sonnet", timeout=300):
            call_count["n"] += 1
            return {"consistency_score": 0.75}

        monkeypatch.setattr(rl_judge_dispatch, "generate_json", fake_generate_json)

        php_code = "<?php\nfunction wp_hello() { return 'hello'; }"
        critique_text = "The function lacks proper escaping."

        score1 = rl_judge_dispatch.score_with_cache(php_code, critique_text)
        score2 = rl_judge_dispatch.score_with_cache(php_code, critique_text)

        assert call_count["n"] == 1, (
            f"generate_json must be called exactly once on cache hit, "
            f"but was called {call_count['n']} times"
        )
        assert score1 == pytest.approx(0.75), f"Expected 0.75, got {score1}"
        assert score2 == pytest.approx(0.75), f"Expected 0.75 on cache hit, got {score2}"

    def test_cache_different_inputs_call_subprocess(self, monkeypatch):
        """Different (code, critique) pairs each invoke the subprocess."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        call_count = {"n": 0}
        scores_to_return = [0.6, 0.8]

        def fake_generate_json(prompt, system=None, model="sonnet", timeout=300):
            call_count["n"] += 1
            return {"consistency_score": scores_to_return[call_count["n"] - 1]}

        monkeypatch.setattr(rl_judge_dispatch, "generate_json", fake_generate_json)

        s1 = rl_judge_dispatch.score_with_cache("<?php echo 'a';", "critique A")
        s2 = rl_judge_dispatch.score_with_cache("<?php echo 'b';", "critique B")

        assert call_count["n"] == 2, "Two distinct pairs must each call generate_json"
        assert s1 == pytest.approx(0.6)
        assert s2 == pytest.approx(0.8)

    def test_cache_none_result_not_stored(self, monkeypatch):
        """When generate_json returns None (parse failure), the cache does NOT store the result
        and a subsequent call retries the subprocess."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        call_count = {"n": 0}

        def fake_generate_json(prompt, system=None, model="sonnet", timeout=300):
            call_count["n"] += 1
            return None  # simulate parse failure

        monkeypatch.setattr(rl_judge_dispatch, "generate_json", fake_generate_json)

        php_code = "<?php\nfunction broken() {}"
        critique_text = "This is missing error handling."

        result1 = rl_judge_dispatch.score_with_cache(php_code, critique_text)
        result2 = rl_judge_dispatch.score_with_cache(php_code, critique_text)

        assert result1 is None
        assert result2 is None
        assert call_count["n"] == 2, "None result must not be cached; retry expected"

    def test_score_judge_consistency_returns_float_in_range(self, monkeypatch):
        """score_judge_consistency parses consistency_score and returns float in [0,1]."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        def fake_generate_json(prompt, system=None, model="sonnet", timeout=300):
            return {"consistency_score": 0.85}

        monkeypatch.setattr(rl_judge_dispatch, "generate_json", fake_generate_json)

        score = rl_judge_dispatch.score_judge_consistency(
            "<?php\nfunction foo() {}",
            "Critique text here",
        )
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range"

    def test_score_judge_consistency_none_on_generate_failure(self, monkeypatch):
        """score_judge_consistency returns None when generate_json returns None."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        monkeypatch.setattr(
            rl_judge_dispatch, "generate_json", lambda *a, **kw: None
        )

        result = rl_judge_dispatch.score_judge_consistency(
            "<?php\nfunction foo() {}",
            "Critique text here",
        )
        assert result is None

    def test_no_anthropic_api_import(self):
        """scripts/rl_judge_dispatch.py must not import the Anthropic API."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "import ast, pathlib; "
             "tree = ast.parse(pathlib.Path('scripts/rl_judge_dispatch.py').read_text()); "
             "names = [n.name for n in ast.walk(tree) if isinstance(n, ast.Import)]; "
             "aliases = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) "
             "for a in n.names]; "
             "all_names = names + aliases; "
             "bad = [n for n in all_names if 'anthropic' in n.lower()]; "
             "assert not bad, f'Anthropic API imports found: {bad}'"],
            capture_output=True, text=True,
            cwd="/home/robert_li/Desktop/projects/wp-finetune",
        )
        assert result.returncode == 0, (
            f"Anthropic API import detected: {result.stdout} {result.stderr}"
        )

    def test_no_run_in_background(self):
        """scripts/rl_judge_dispatch.py must not use Agent(run_in_background=True)."""
        import pathlib
        src = pathlib.Path(
            "/home/robert_li/Desktop/projects/wp-finetune/scripts/rl_judge_dispatch.py"
        ).read_text()
        assert "run_in_background" not in src, (
            "rl_judge_dispatch.py must not use Agent(run_in_background=True)"
        )

    def test_uses_claude_agent_subprocess(self):
        """scripts/rl_judge_dispatch.py must import from scripts.claude_agent."""
        import pathlib
        src = pathlib.Path(
            "/home/robert_li/Desktop/projects/wp-finetune/scripts/rl_judge_dispatch.py"
        ).read_text()
        assert "scripts.claude_agent" in src, (
            "rl_judge_dispatch.py must import from scripts.claude_agent"
        )


# ---------------------------------------------------------------------------
# Task 2: Async batch dispatch with timeout + group-mean imputation
# ---------------------------------------------------------------------------

class TestScoreJudgeConsistencyBatch:
    """Batch dispatch tests for timeout imputation and cache behavior."""

    def test_timeout_imputes_from_group_mean(self, monkeypatch):
        """Batch where one sample raises TimeoutError is imputed from the group mean."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        call_count = {"n": 0}

        def fake_scorer(php_code, critique_text, model="sonnet", n_votes=1):
            call_count["n"] += 1
            # Third call raises to simulate timeout
            if call_count["n"] == 3:
                raise asyncio.TimeoutError("simulated timeout")
            return 0.8 if call_count["n"] == 1 else 0.6  # valid mean = 0.7

        monkeypatch.setattr(rl_judge_dispatch, "score_judge_consistency", fake_scorer)

        samples = [
            {"php_code": "<?php echo 1;", "critique_text": "critique1"},
            {"php_code": "<?php echo 2;", "critique_text": "critique2"},
            {"php_code": "<?php echo 3;", "critique_text": "critique3"},  # will timeout
        ]

        results = asyncio.run(rl_judge_dispatch.score_judge_consistency_batch(samples))

        assert len(results) == 3, f"Output length must match input length, got {len(results)}"
        assert results[0] == pytest.approx(0.8, abs=1e-6), f"First score wrong: {results[0]}"
        assert results[1] == pytest.approx(0.6, abs=1e-6), f"Second score wrong: {results[1]}"
        # Third must be imputed: mean of (0.8, 0.6) = 0.7
        assert results[2] == pytest.approx(0.7, abs=1e-4), (
            f"Timed-out slot must be imputed from group mean 0.7, got {results[2]}"
        )

    def test_exception_imputes_from_group_mean(self, monkeypatch):
        """Batch where one sample raises RuntimeError is imputed from the group mean."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        call_count = {"n": 0}

        def fake_scorer(php_code, critique_text, model="sonnet", n_votes=1):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated subprocess failure")
            return 0.9

        monkeypatch.setattr(rl_judge_dispatch, "score_judge_consistency", fake_scorer)

        samples = [
            {"php_code": "<?php echo 'a';", "critique_text": "crit a"},
            {"php_code": "<?php echo 'b';", "critique_text": "crit b"},  # will fail
            {"php_code": "<?php echo 'c';", "critique_text": "crit c"},
        ]

        results = asyncio.run(rl_judge_dispatch.score_judge_consistency_batch(samples))

        assert len(results) == 3
        # valid mean = mean([0.9, 0.9]) = 0.9; slot 1 imputed to 0.9
        assert results[0] == pytest.approx(0.9, abs=1e-6)
        assert results[2] == pytest.approx(0.9, abs=1e-6)
        assert results[1] == pytest.approx(0.9, abs=1e-4), (
            f"Failed slot must be imputed from group mean 0.9, got {results[1]}"
        )

    def test_all_cached_invokes_scorer_zero_times(self, monkeypatch):
        """A batch where all samples are in cache must invoke score_judge_consistency zero times."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        # Pre-populate cache for all three samples
        samples = [
            {"php_code": "<?php echo 'x';", "critique_text": "crit x"},
            {"php_code": "<?php echo 'y';", "critique_text": "crit y"},
            {"php_code": "<?php echo 'z';", "critique_text": "crit z"},
        ]
        for s in samples:
            key = rl_judge_dispatch._cache_key(s["php_code"], s["critique_text"])
            rl_judge_dispatch._score_cache[key] = 0.5

        scorer_call_count = {"n": 0}

        def fake_scorer(php_code, critique_text, model="sonnet", n_votes=1):
            scorer_call_count["n"] += 1
            return 0.5

        monkeypatch.setattr(rl_judge_dispatch, "score_judge_consistency", fake_scorer)

        results = asyncio.run(rl_judge_dispatch.score_judge_consistency_batch(samples))

        assert scorer_call_count["n"] == 0, (
            f"All-cached batch must invoke score_judge_consistency 0 times, "
            f"but invoked {scorer_call_count['n']} times"
        )
        assert len(results) == 3
        assert all(r == pytest.approx(0.5) for r in results)

    def test_batch_preserves_input_order(self, monkeypatch):
        """Batch results must be order-preserving."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        scores = [0.1, 0.5, 0.9]
        call_count = {"n": 0}

        def fake_scorer(php_code, critique_text, model="sonnet", n_votes=1):
            idx = int(php_code.strip().split("echo ")[-1].rstrip(";"))
            return scores[idx]

        monkeypatch.setattr(rl_judge_dispatch, "score_judge_consistency", fake_scorer)

        samples = [
            {"php_code": f"<?php echo {i};", "critique_text": f"crit {i}"}
            for i in range(3)
        ]
        results = asyncio.run(rl_judge_dispatch.score_judge_consistency_batch(samples))

        assert results == pytest.approx(scores, abs=1e-6), (
            f"Order must be preserved: expected {scores}, got {results}"
        )

    def test_all_failed_falls_back_to_neutral(self, monkeypatch):
        """When all samples fail/timeout, imputed value is 0.5 (neutral fallback)."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        def always_fails(php_code, critique_text, model="sonnet", n_votes=1):
            raise RuntimeError("all fail")

        monkeypatch.setattr(rl_judge_dispatch, "score_judge_consistency", always_fails)

        samples = [
            {"php_code": "<?php echo 'a';", "critique_text": "crit"},
            {"php_code": "<?php echo 'b';", "critique_text": "crit"},
        ]
        results = asyncio.run(rl_judge_dispatch.score_judge_consistency_batch(samples))

        assert len(results) == 2
        assert all(r == pytest.approx(0.5) for r in results), (
            f"All-failed batch must fall back to neutral 0.5, got {results}"
        )

    def test_batch_length_matches_input(self, monkeypatch):
        """Batch output length always equals input length."""
        rl_judge_dispatch = pytest.importorskip("scripts.rl_judge_dispatch")

        monkeypatch.setattr(
            rl_judge_dispatch, "score_judge_consistency",
            lambda *a, **kw: 0.7
        )

        for n in [1, 3, 8]:
            samples = [
                {"php_code": f"<?php echo {i};", "critique_text": "crit"}
                for i in range(n)
            ]
            results = asyncio.run(rl_judge_dispatch.score_judge_consistency_batch(samples))
            assert len(results) == n, f"Expected {n} results, got {len(results)}"

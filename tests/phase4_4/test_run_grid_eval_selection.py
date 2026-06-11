"""Synthetic in-memory unit test of run_grid_eval.decide().

Locks the pre-registered selection rule and exit-2 escalation against synthetic
in-memory inputs. No Tinker, no merge, no wp-bench, no file-IO, no shell calls.

Tests:
  1. All-fail escalation: decide() returns (None, 2) when no candidate clears bars.
  2. Exactly-one-pass: sole passer is selected, selected=True, exit_code=0.
  3. Multiple-pass, max wp-bench wins: highest wpbench_score candidate selected.
  4. Tie-break on judge Spearman: equal wpbench_score -> higher rho wins.
  5. Eval-economy / judge-fail never wp-benched: wpbench_score=None excluded always.
"""

from scripts.run_grid_eval import decide

BASELINE = 0.4537  # D-N8 HARD gate (full 344-test baseline)


def _cand(
    tag: str,
    wp: float | None,
    rho: float,
    judge_pass: bool = True,
    sentinel_pass: bool = True,
    confusion_pass: bool = True,
    fs_pass: bool = True,
) -> dict:
    """Build a summary.json-shaped candidate dict for decide() inputs.

    Mirrors the gate dict written by run_grid_eval._run_candidate_judge_gates
    and main():
      {"candidate", "gates": {judge_spearman:{rho,pass}, sentinel_0_policy:{pass},
       confusion_gate:{pass}, fs_gate:{pass}}, "wpbench_score", "selected": False}
    """
    return {
        "candidate": tag,
        "gates": {
            "judge_spearman": {"rho": rho, "ci_lower": rho - 0.05, "bar": 0.263,
                               "mode": "point", "pass": judge_pass},
            "sentinel_0_policy": {"false_passes": 0 if sentinel_pass else 1,
                                   "pass": sentinel_pass},
            "confusion_gate": {"pareto_ok": confusion_pass, "pass": confusion_pass},
            "fs_gate": {"wilson_upper": 0.10 if fs_pass else 0.20, "pass": fs_pass},
        },
        "wpbench_score": wp,
        "selected": False,
    }


# ---------------------------------------------------------------------------
# Test 1: all-fail escalation
# ---------------------------------------------------------------------------

def test_all_fail_escalation():
    """decide() returns (None, 2) when no candidate clears the wp-bench HARD gate."""
    results = [
        # Fails: wpbench_score None (never benched — judge gate failed)
        _cand("r8_rp15", wp=None, rho=0.20, judge_pass=False),
        # Fails: below baseline
        _cand("r16_rp15", wp=0.43, rho=0.30),
        # Fails: exactly at baseline but gate barely below it
        _cand("r32_rp30", wp=0.45, rho=0.28),
    ]
    winner, exit_code = decide(results, BASELINE)
    assert winner is None, f"Expected no winner, got {winner}"
    assert exit_code == 2, f"Expected exit_code=2 (escalation), got {exit_code}"
    # No candidate should have selected=True
    for c in results:
        assert c["selected"] is False, f"{c['candidate']} should not be selected"


# ---------------------------------------------------------------------------
# Test 2: exactly-one-pass
# ---------------------------------------------------------------------------

def test_exactly_one_pass():
    """The sole passer is returned as winner with selected=True and exit_code=0."""
    sole_passer = _cand("r16_rp30", wp=0.4600, rho=0.30)
    results = [
        _cand("r8_rp15", wp=None, rho=0.20, judge_pass=False),  # never benched
        sole_passer,
        _cand("r32_rp15", wp=0.40, rho=0.29),  # below baseline
    ]
    winner, exit_code = decide(results, BASELINE)
    assert exit_code == 0, f"Expected exit_code=0, got {exit_code}"
    assert winner is not None, "Expected a winner"
    assert winner["candidate"] == "r16_rp30", (
        f"Expected sole passer r16_rp30, got {winner['candidate']}"
    )
    assert winner["selected"] is True, "Winner selected flag must be True"


# ---------------------------------------------------------------------------
# Test 3: multiple-pass, max wp-bench wins
# ---------------------------------------------------------------------------

def test_multiple_pass_max_wpbench_wins():
    """With multiple passers, the one with the highest wpbench_score is selected."""
    results = [
        _cand("r8_rp30", wp=0.4600, rho=0.28),
        _cand("r16_rp30", wp=0.4800, rho=0.27),   # highest wpbench_score
        _cand("r32_rp50", wp=0.4700, rho=0.30),
        _cand("r8_rp15", wp=None, rho=0.20, judge_pass=False),  # never benched
    ]
    winner, exit_code = decide(results, BASELINE)
    assert exit_code == 0, f"Expected exit_code=0, got {exit_code}"
    assert winner is not None, "Expected a winner"
    assert winner["candidate"] == "r16_rp30", (
        f"Expected highest wp-bench r16_rp30, got {winner['candidate']}"
    )
    assert winner["selected"] is True, "Winner selected flag must be True"


# ---------------------------------------------------------------------------
# Test 4: tie-break on judge Spearman
# ---------------------------------------------------------------------------

def test_tiebreak_on_judge_spearman():
    """Equal wpbench_score -> the higher judge rho wins the tie-break."""
    results = [
        _cand("r16_rp30_loRho", wp=0.4700, rho=0.29),   # equal wp, lower rho
        _cand("r32_rp30_hiRho", wp=0.4700, rho=0.35),   # equal wp, higher rho — should win
        _cand("r8_rp15", wp=0.4600, rho=0.31),
    ]
    winner, exit_code = decide(results, BASELINE)
    assert exit_code == 0, f"Expected exit_code=0, got {exit_code}"
    assert winner is not None, "Expected a winner"
    assert winner["candidate"] == "r32_rp30_hiRho", (
        f"Expected higher-rho tie-break winner r32_rp30_hiRho, got {winner['candidate']}"
    )
    assert winner["selected"] is True, "Winner selected flag must be True"


# ---------------------------------------------------------------------------
# Test 5: eval-economy — judge-fail candidate (wpbench_score=None) never selected
# ---------------------------------------------------------------------------

def test_eval_economy_judge_fail_never_selected():
    """A judge-failing candidate has wpbench_score=None and is never in passing/winner.

    Even if we hypothetically assigned it a high wpbench_score, the
    wpbench_score is None (it was never benched due to failing the judge gate),
    so decide() excludes it via the `wpbench_score is not None` guard.
    """
    judge_fail = _cand("r32_rp15_judgeFail", wp=None, rho=0.15, judge_pass=False)
    legit_passer = _cand("r16_rp30", wp=0.4600, rho=0.28)
    results = [
        judge_fail,
        legit_passer,
    ]
    winner, exit_code = decide(results, BASELINE)
    assert exit_code == 0, f"Expected exit_code=0, got {exit_code}"
    assert winner is not None, "Expected a winner (the legit passer)"
    assert winner["candidate"] != "r32_rp15_judgeFail", (
        f"Judge-fail candidate must never be selected, but got {winner['candidate']}"
    )
    assert winner["candidate"] == "r16_rp30", (
        f"Expected legit passer r16_rp30, got {winner['candidate']}"
    )
    # Judge-fail candidate must remain unselected
    assert judge_fail["selected"] is False, (
        "Judge-fail candidate (wpbench_score=None) must not have selected=True"
    )

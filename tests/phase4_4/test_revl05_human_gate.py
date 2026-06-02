"""Tests for REVL-05 human-review pack: sentinel gate + stratified selection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_human_review import check_sentinel, select_samples, _coverage_fraction


class TestSentinel:
    def test_absent_when_missing_file(self, tmp_path):
        assert check_sentinel(str(tmp_path / "nope.md")) is False

    def test_absent_without_line(self, tmp_path):
        p = tmp_path / "review.md"
        p.write_text("# review\n\nsome content\n\n## Sign-off\n")
        assert check_sentinel(str(p)) is False

    def test_present_approved(self, tmp_path):
        p = tmp_path / "review.md"
        p.write_text("# review\n\n## Sign-off\n\nHUMAN_APPROVED: 2026-06-02T14:00:00Z\n")
        assert check_sentinel(str(p)) is True

    def test_present_rejected(self, tmp_path):
        p = tmp_path / "review.md"
        p.write_text("HUMAN_REJECTED: reasoning too terse on 30% of samples\n")
        assert check_sentinel(str(p)) is True

    def test_empty_sentinel_does_not_count(self, tmp_path):
        p = tmp_path / "review.md"
        p.write_text("HUMAN_APPROVED:   \n")  # no value
        assert check_sentinel(str(p)) is False


class TestStratifiedSelection:
    def _mk(self, idx, task, cov_bools):
        cap = {"example_idx": idx, "task_type": task,
               "prompt": f"<wp_judge> code {idx}", "response": f"prose {idx} [/REASONING]<judge_output>{{}}</judge_output>"}
        rev = {"sample_id": idx, "dimension_coverage": cov_bools,
               "score_reasoning_consistency": cov_bools, "coherence": 4}
        return cap, rev

    def test_strata_and_cot_only(self):
        EIGHT = ["wpcs", "security", "sql_safety", "performance",
                 "wp_api_usage", "accessibility", "code_quality", "dependency_integrity"]
        full = {k: True for k in EIGHT}
        low = {k: (i == 0) for i, k in enumerate(EIGHT)}   # 1/8 = 0.125
        mid = {k: (i < 4) for i, k in enumerate(EIGHT)}    # 4/8 = 0.5
        caps, revs = [], []
        # 6 full cot, 4 low cot, 4 mid cot, + 3 ctf (must be excluded)
        i = 0
        for _ in range(6):
            c, r = self._mk(i, "cot", full); caps.append(c); revs.append(r); i += 1
        for _ in range(4):
            c, r = self._mk(i, "cot", low); caps.append(c); revs.append(r); i += 1
        for _ in range(4):
            c, r = self._mk(i, "cot", mid); caps.append(c); revs.append(r); i += 1
        for _ in range(3):
            c, r = self._mk(i, "ctf", full); caps.append(c); revs.append(r); i += 1
        pairs = [{"index": c["example_idx"], "model_overall": 70,
                  "gt_canonical": 80, "gt_teacher": 75} for c in caps]
        picks, cap, rev, pid = select_samples(caps, revs, pairs)
        assert len(picks) == 10  # 4 full + 3 low + 3 mid
        # all CoT
        assert all(cap[i]["task_type"] == "cot" for i in picks)
        fracs = sorted(_coverage_fraction(rev[i]) for i in picks)
        assert sum(1 for f in fracs if f >= 1.0) == 4
        assert sum(1 for f in fracs if f <= 0.25) == 3
        assert sum(1 for f in fracs if 0.25 < f < 1.0) == 3

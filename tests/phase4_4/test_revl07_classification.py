"""Tests for REVL-07 classification confusion matrix (SOFT)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revl07_classification import confusion_at, revl07


class TestConfusionAt:
    def test_perfect_separation(self):
        # all preds match truth at threshold 70
        pairs = [(80, 80), (90, 90), (60, 60), (50, 50)]
        m = confusion_at(pairs, 70.0)
        assert m["TP"] == 2 and m["TN"] == 2 and m["FP"] == 0 and m["FN"] == 0
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0

    def test_false_positive(self):
        # model says PASS (80>=70) but GT FAIL (60<70)
        m = confusion_at([(80, 60)], 70.0)
        assert m["FP"] == 1 and m["TP"] == 0
        assert m["precision"] == 0.0

    def test_false_negative(self):
        m = confusion_at([(60, 80)], 70.0)
        assert m["FN"] == 1
        assert m["recall"] == 0.0

    def test_empty_no_crash(self):
        m = confusion_at([], 70.0)
        assert m["f1"] == 0.0 and m["accuracy"] == 0.0


class TestRevl07:
    def test_writes_artifact_and_excludes_missing(self, tmp_path):
        pairs = tmp_path / "pairs.jsonl"
        with open(pairs, "w") as fh:
            fh.write(json.dumps({"model_overall": 80, "gt_canonical": 90}) + "\n")
            fh.write(json.dumps({"model_overall": 60, "gt_canonical": 50}) + "\n")
            fh.write(json.dumps({"model_overall": None, "gt_canonical": 90}) + "\n")  # excluded
            fh.write(json.dumps({"model_overall": 75}) + "\n")  # missing gt -> excluded
        out = tmp_path / "matrix.json"
        res = revl07(str(pairs), str(out))
        assert res["n_total"] == 4
        assert res["n_usable"] == 2
        assert res["n_excluded"] == 2
        assert res["gate_class"] == "soft"
        assert res["f1_optimal_threshold"] is not None
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["gate"] == "REVL-07"
        assert isinstance(loaded["thresholds"], list) and loaded["thresholds"]

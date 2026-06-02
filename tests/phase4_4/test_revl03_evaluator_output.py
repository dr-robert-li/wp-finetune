"""Tests for REVL-03 plan-emitter + aggregator. No LLM API touched."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revl03_evaluator_agent import emit_plan, build_agent_prompt
from scripts.aggregate_revl03 import aggregate


class TestPlanEmitter:
    def test_emits_one_row_per_sample(self, tmp_pairs_jsonl, tmp_path):
        plan = tmp_path / "plan.jsonl"
        n = emit_plan(str(tmp_pairs_jsonl), str(plan), {"cot", "ctf"})
        rows = [json.loads(l) for l in open(plan) if l.strip()]
        assert n == 20 and len(rows) == 20
        for r in rows:
            assert {"sample_id", "agent_prompt", "expected_output_path"} <= set(r)

    def test_prompt_is_opaque_and_structured(self):
        p = build_agent_prompt("some reasoning", {"d1": 8}, "<?php f();")
        assert "NOT told which model" in p
        assert "dimension_coverage" in p
        assert "score_reasoning_consistency" in p
        assert "coherence" in p
        assert "Output ONLY the JSON object" in p

    def test_reasoning_extracted_not_judge_output(self, tmp_pairs_jsonl, tmp_path):
        plan = tmp_path / "plan.jsonl"
        emit_plan(str(tmp_pairs_jsonl), str(plan), {"cot", "ctf"})
        first = json.loads(open(plan).readline())
        # close-only extraction: prose present, judge_output stripped
        assert "WPCS Compliance" in first["agent_prompt"]
        assert "judge_output" not in first["agent_prompt"]


def _write_eval(path, samples):
    with open(path, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")


def _full_cov():
    return {f"D{i}": True for i in range(1, 10)}


class TestAggregate:
    def test_pass_boundary(self, tmp_path):
        ev = tmp_path / "eval.jsonl"
        agg = tmp_path / "agg.json"
        # all 9 dims covered + consistent -> coverage 1.0 -> pass
        _write_eval(ev, [
            {"sample_id": i, "dimension_coverage": _full_cov(),
             "score_reasoning_consistency": _full_cov(), "coherence": 5}
            for i in range(5)
        ])
        res = aggregate(str(ev), str(agg))
        assert res["dimension_coverage_rate"] == 1.0
        assert res["pass"] is True
        assert res["mean_coherence"] == 5.0

    def test_below_threshold_fails(self, tmp_path):
        ev = tmp_path / "eval.jsonl"
        agg = tmp_path / "agg.json"
        # only 6/9 dims -> 0.667 < 0.80 -> fail
        cov = {f"D{i}": (i <= 6) for i in range(1, 10)}
        _write_eval(ev, [
            {"sample_id": i, "dimension_coverage": cov,
             "score_reasoning_consistency": cov, "coherence": 4}
            for i in range(5)
        ])
        res = aggregate(str(ev), str(agg))
        assert abs(res["dimension_coverage_rate"] - 6 / 9) < 1e-9
        assert res["pass"] is False

    def test_zero_claimed_dims_does_not_raise(self, tmp_path):
        ev = tmp_path / "eval.jsonl"
        agg = tmp_path / "agg.json"
        allfalse = {f"D{i}": False for i in range(1, 10)}
        _write_eval(ev, [
            {"sample_id": 0, "dimension_coverage": allfalse,
             "score_reasoning_consistency": allfalse, "coherence": 1}
        ])
        res = aggregate(str(ev), str(agg))  # must NOT raise ZeroDivisionError
        assert res["score_reasoning_consistency_rate"] == 0.0
        assert res["dimension_coverage_rate"] == 0.0
        assert res["pass"] is False
        assert agg.exists()

    def test_empty_eval_set_does_not_pass(self, tmp_path):
        ev = tmp_path / "eval.jsonl"
        ev.write_text("")
        res = aggregate(str(ev), str(tmp_path / "agg.json"))
        assert res["n_samples"] == 0 and res["pass"] is False

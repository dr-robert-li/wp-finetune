"""Unit tests for scripts.sieve_cross_seed_overlap (Open Question 2 from 11-CONTEXT.md).

Resolves whether the 3 judge seeds route similarly enough that one Sieve profile
covers all 3, or whether Sieve needs the union of 3 routing profiles. Tests are
GPU-free: synthetic per-layer top-k expert sets, no model load. Module-level
importorskip so this file SKIPs cleanly while scripts/sieve_cross_seed_overlap.py
is absent (lands in a later wave).
"""
from __future__ import annotations

import json

import pytest

sieve_overlap = pytest.importorskip("scripts.sieve_cross_seed_overlap")


class TestJaccard:
    def test_identical_sets_have_jaccard_one(self):
        """Two identical top-k sets -> Jaccard similarity == 1.0."""
        a = {1, 2, 3}
        b = {1, 2, 3}
        assert sieve_overlap.jaccard(a, b) == 1.0

    def test_hand_computed_two_set_example(self):
        """{1,2,3} vs {2,3,4}: intersection={2,3} (2), union={1,2,3,4} (4) -> Jaccard=0.5."""
        a = {1, 2, 3}
        b = {2, 3, 4}
        assert sieve_overlap.jaccard(a, b) == pytest.approx(0.5)

    def test_disjoint_sets_have_jaccard_zero(self):
        """No overlap -> Jaccard == 0.0."""
        a = {1, 2}
        b = {3, 4}
        assert sieve_overlap.jaccard(a, b) == 0.0


class TestPairwiseLayerJaccard:
    def test_identical_seed_inputs_give_all_ones(self):
        """Same top-k sets for every layer across two seeds -> per-layer Jaccard all 1.0."""
        seed_topk = {
            "s0": [{1, 2, 3}, {10, 11, 12}],
            "s1": [{1, 2, 3}, {10, 11, 12}],
        }
        result = sieve_overlap.pairwise_layer_jaccard(seed_topk)
        s0_vs_s1 = result[("s0", "s1")]
        assert s0_vs_s1 == [pytest.approx(1.0), pytest.approx(1.0)]

    def test_known_two_seed_two_layer_example(self):
        """Hand-computed: layer0 Jaccard=0.5 ({1,2,3} vs {2,3,4}), layer1 Jaccard=1.0 (identical)."""
        seed_topk = {
            "s0": [{1, 2, 3}, {10, 11, 12}],
            "s1": [{2, 3, 4}, {10, 11, 12}],
        }
        result = sieve_overlap.pairwise_layer_jaccard(seed_topk)
        s0_vs_s1 = result[("s0", "s1")]
        assert s0_vs_s1[0] == pytest.approx(0.5)
        assert s0_vs_s1[1] == pytest.approx(1.0)


class TestLoadSeedCounts:
    def test_infers_dims_from_v4_scale_records(self, tmp_path):
        """load_seed_counts sizes its array from the file itself, not the
        module N_LAYERS/N_EXPERTS constants (GATE4-02 SC1)."""
        records = [
            {"layer_idx": 0, "expert_counts_total": {"0": 5, "255": 3}},
            {"layer_idx": 39, "expert_counts_total": {"10": 2}},
        ]
        jsonl_path = tmp_path / "routing_report.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records))

        counts = sieve_overlap.load_seed_counts(jsonl_path)
        assert counts.shape == (40, 256)
        assert counts[0, 255] == 3
        assert counts[39, 10] == 2

    def test_empty_file_falls_back_to_module_defaults(self, tmp_path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        counts = sieve_overlap.load_seed_counts(jsonl_path)
        assert counts.shape == (sieve_overlap.N_LAYERS, sieve_overlap.N_EXPERTS)

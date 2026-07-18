"""Tests for the 2026-07-02 GT hash-join fix + calib liveness telemetry.

The 2026-07-01 gated smoke silently trained on pure fix_correctness because
the reward path hashed RAW code while the sidecar hashed whitespace-normalized
code (0/482 join). These tests pin:
  1. hash parity — builder side and reward side use the SAME join key,
  2. real-pool coverage — the reward-time path joins the shipped sidecar,
  3. calib stats accumulator — fired/dead states are distinguishable.

CPU / $0. Uses the real judge pool + sidecar (repo data files).
"""
import json
import math
import os
from pathlib import Path

import pytest

os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

REPO = Path(__file__).resolve().parent.parent
SIDECAR = REPO / "data/rl_probe/judge_gt_sidecar.jsonl"

from scripts.reward_calibration import (  # noqa: E402
    get_and_reset_calib_stats,
    normalized_code_hash,
    record_calib_stat,
)


def test_normalized_hash_whitespace_invariant():
    """The join key must be invariant to whitespace layout (the smoke defect)."""
    raw = "<?php\nfunction foo( $a ) {\n    return $a + 1;\n}\n"
    reflowed = "<?php function foo( $a ) { return $a + 1; }"
    assert normalized_code_hash(raw) == normalized_code_hash(reflowed)
    assert normalized_code_hash("") is None
    assert normalized_code_hash(None) is None


def test_builder_and_reward_hash_agree():
    """Builder-side _code_hash == reward-side judge_item_code_hash on the same turn."""
    from scripts.build_reward_gt_sidecar import _code_hash
    from scripts.rl_rollouts import judge_item_code_hash

    php = "<?php\nfunction bar() {\n    echo esc_html( get_the_title() );\n}\n"
    user_content = f"<wp_judge>\nReview this code:\n```php\n{php}```"
    item = {"messages": [{"role": "user", "content": user_content}]}
    builder_h = _code_hash(user_content)
    reward_h = judge_item_code_hash(item)
    assert builder_h is not None
    assert builder_h == reward_h


@pytest.mark.skipif(not SIDECAR.exists(), reason="sidecar data file not present")
def test_real_pool_join_coverage():
    """Reward-time hash path must join the shipped sidecar on the real pool.

    The 2026-07-01 smoke's coverage via the reward path was 0/482; the fixed
    path must recover the sidecar's 342 rows (>= 0.5 of the pool — the same
    floor the step-0 CALIB_JOIN_DEAD gate enforces).
    """
    from scripts.rl_rollouts import judge_item_code_hash
    from scripts.tinker_rl_data import load_rl_prompts

    gt_hashes = {json.loads(l)["code_hash"] for l in SIDECAR.open() if l.strip()}
    pool = load_rl_prompts("judge")
    assert pool, "judge pool empty — data files missing"
    hits = sum(1 for it in pool if judge_item_code_hash(it) in gt_hashes)
    assert hits == len(gt_hashes), f"expected {len(gt_hashes)} joins, got {hits}"
    assert hits / len(pool) >= 0.5, f"coverage {hits}/{len(pool)} below step-0 gate floor"


def test_calib_stats_fired_and_dead():
    """Accumulator distinguishes fired / dead / no-batch states."""
    get_and_reset_calib_stats()  # drain any prior state

    # No batch recorded -> fired_frac None (calib off / no judge completions).
    empty = get_and_reset_calib_stats()
    assert empty["calib_fired_frac"] is None
    assert empty["calib_n"] == 0

    # Dead join: all NaN -> fired_frac 0.0 (step-0 halt territory).
    for _ in range(4):
        record_calib_stat(float("nan"))
    dead = get_and_reset_calib_stats()
    assert dead["calib_fired_frac"] == 0.0
    assert dead["calib_n"] == 0
    assert dead["calib_mean"] is None

    # Live join: 3/4 fired.
    record_calib_stat(0.8)
    record_calib_stat(0.6)
    record_calib_stat(1.0)
    record_calib_stat(float("nan"))
    live = get_and_reset_calib_stats()
    assert live["calib_fired_frac"] == pytest.approx(0.75)
    assert live["calib_n"] == 3
    assert live["calib_mean"] == pytest.approx(0.8)
    assert live["calib_std"] == pytest.approx(math.sqrt(2 / 75), abs=1e-9)

    # Drained after read.
    assert get_and_reset_calib_stats()["calib_fired_frac"] is None


def test_step0_calib_join_dead_halts(monkeypatch, tmp_path):
    """calib_weight>0 with a dead GT join must HARD-halt at step 0 (loud-fail).

    Reuses the offline run_training_step harness from test_rl_train_integration.
    The fake judge pool items carry no ```php block -> judge_item_code_hash None
    -> calib NaN for every completion -> calib_fired_frac 0.0 -> CALIB_JOIN_DEAD.
    """
    from tests.test_rl_train_integration import (
        _gen_pool,
        _judge_pool,
        _make_args,
        _wire,
    )

    get_and_reset_calib_stats()  # clean slate
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.0)
    args = _make_args(calib_weight=0.8, calib_form="hybrid")
    manifest = {"checkpoints": []}

    sampling_client = tc.save_weights_and_get_sampling_client()
    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=sampling_client,
        gen_pool=_gen_pool(4),  # sampler requires a non-empty gen pool
        judge_pool=_judge_pool(4),
        args=args,
        manifest=manifest,
    )

    assert halted is True, "dead calib join at step 0 must halt"
    tc.optim_step.assert_not_called()  # halt fires BEFORE committing the update
    rows = [json.loads(l) for l in metrics_path.open()]
    assert rows, "halting step must still write its metrics row"
    row = rows[-1]
    assert row["halt_reason"] and "CALIB_JOIN_DEAD" in row["halt_reason"]
    assert row["calib_fired_frac"] == 0.0

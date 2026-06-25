"""End-to-end integration test for the Phase 9 RL training STEP LOOP.

This is the root-cause fix for the Phase 9 blind spot: the existing unit tests
(tests/test_rl_train.py) exercise individual seams with mocks that bypass
main()'s wiring, so the live loop's CR-01/CR-02/CR-04 bugs all stayed green.

This test drives the REAL loop body via the `run_training_step` seam using the
mock Tinker client over NON-EMPTY {"messages": [...]} prompt pools and asserts:

  (a) the sampling call (.sample) is invoked on the client obtained from the
      CORRECT save_weights_and_get_sampling_client() path — the test FAILS if the
      old CR-01 object (the bare save_weights_for_sampler checkpoint ref, which
      has NO .sample) is used;
  (b) BOTH gen and judge rollouts are produced and SURVIVE to advantages (not all
      dropped as constant — CR-02/CR-03/CR-06 end-to-end);
  (c) on a synthetic divergent KL, check_halt fires and the break/emergency
      checkpoint happens BEFORE optim_step would commit (CR-04) — optim_step is
      NOT called after a hard halt;
  (d) the metrics JSONL line written for the step contains the RLEV fields
      (kl_sample_train_v1, e_frac_with_tokens_mean, reward_breakdown).

The mock models the CORRECT Tinker contract (.sample(...) returning a future
whose .result().sequences[i].tokens decode to text), not the broken .generate()
one. No real tinker / tinker_cookbook / tokenizer is required.
"""
from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import pytest

# 09-07: the step loop now assembles real cookbook Datums (tinker + tinker_cookbook),
# so the whole integration module requires tinker. Base conda has no tinker -> skip.
pytest.importorskip("tinker")


# ---------------------------------------------------------------------------
# Fakes modelling the CORRECT Tinker sampling contract
# ---------------------------------------------------------------------------


class _FakeSeq:
    """One sampled sequence: token ids + their sampling logprobs (09-07).

    The logprobs are what build_trajectory_groups bakes into the cookbook
    Transition -> Datum, so they must be present and length-matched to tokens.
    """

    def __init__(self, tokens, logprobs=None):
        self.tokens = tokens
        self.logprobs = logprobs if logprobs is not None else [-0.3] * len(tokens)


class _FakeSampleResponse:
    """SampleResponse-shaped object: .sequences list (what _decode_samples reads)."""

    def __init__(self, sequences):
        self.sequences = sequences


class _FakeFuture:
    """Future returned by SamplingClient.sample(...).result() -> SampleResponse."""

    def __init__(self, resp):
        self._resp = resp

    def result(self):
        return self._resp


class _FakeSamplingClient:
    """Correct SamplingClient: exposes .sample(...), NO .generate().

    Each .sample() call returns `num_samples` sequences. The token ids are made
    to VARY across samples (so decoded text varies) which is what lets rewards
    vary within a group and survive the constant-reward filter (assertion b).
    """

    def __init__(self):
        self.sample_calls = []

    def sample(self, prompt, num_samples, sampling_params, **kwargs):
        self.sample_calls.append(
            {"prompt": prompt, "num_samples": num_samples}
        )
        # Distinct token ids per sample so decoded completions differ; each carries
        # a matching-length sampling logprob (09-07 — Datum assembly reads it).
        base = len(self.sample_calls) * 100
        seqs = [
            _FakeSeq(tokens=[base + i], logprobs=[-0.3 - 0.01 * i])
            for i in range(num_samples)
        ]
        return _FakeFuture(_FakeSampleResponse(seqs))


class _FakeTokenizer:
    """Decodes a token-id list to a deterministic-but-distinct PHP string."""

    def decode(self, toks):
        tid = toks[0] if toks else 0
        return f"<?php function wp_fix_{tid}() {{ return {tid}; }}"


class _FakeRenderer:
    def build_generation_prompt(self, user_msgs):
        # 09-07: must be a REAL tinker.ModelInput — it becomes the Transition.ob that
        # trajectory_to_data flattens (._flatten_chunks needs .chunks). >=1 obs token
        # so the action-token mask survives the cookbook's right-shift [1:] slice.
        import tinker

        return tinker.ModelInput.from_ints([1, 2, 3])

    def get_stop_sequences(self):
        return []


# ---------------------------------------------------------------------------
# Shared args / pool builders
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    args = types.SimpleNamespace(
        model_id="fake-model",
        batch_size=4,
        group_size=2,
        max_new_tokens=32,
        temperature=1.0,
        use_gspo=True,
        kl_soft=0.1,
        kl_hard=0.3,
        efrac_soft=0.7,
        efrac_hard=0.5,
        jaccard_every=20,
        checkpoint_every=50,
        total_steps=1,
        mask_path="output/profiling/reasoning-merged-v4/protected_expert_mask.npy",
        # reward-path args consumed by collect_rollouts (patched callees ignore them)
        judge_client=MagicMock(),
        judge_model="fake-judge",
        consistency_model="sonnet",
        n_votes=1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _gen_pool(n=4):
    return [{"messages": [{"role": "user", "content": f"gen prompt {i}"}]} for i in range(n)]


def _judge_pool(n=4):
    return [
        {
            "messages": [{"role": "user", "content": f"judge prompt {i}"}],
            "critique_text": f"critique {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Mock-client + reward-path wiring shared by the tests
# ---------------------------------------------------------------------------


def _wire(monkeypatch, tmp_path, *, kl_v1=0.0):
    """Patch every external seam so run_training_step runs offline.

    Returns (tc, fake_sc, metrics_path).
    """
    rl_train = pytest.importorskip("scripts.rl_train")
    rl_rollouts = pytest.importorskip("scripts.rl_rollouts")
    import scripts.reward_pipeline as reward_pipeline
    import scripts.rl_judge_dispatch as rl_judge_dispatch

    # --- Renderer/tokenizer seam (no real download) ---
    monkeypatch.setattr(
        rl_rollouts, "build_rl_renderer", lambda: (_FakeRenderer(), _FakeTokenizer())
    )

    # --- gen reward path: one RewardResult per php_code, VARYING scalars ---
    def fake_compute_group_rewards(php_codes, judge_client, judge_model):
        results = []
        for i, code in enumerate(php_codes):
            bd = types.SimpleNamespace(
                security_fail=False,
                fix_correctness=0.2 + 0.2 * i,
                consistency=0.5,
            )
            results.append(
                types.SimpleNamespace(scalar=0.1 + 0.3 * i, breakdown=bd)
            )
        return results

    monkeypatch.setattr(
        reward_pipeline, "compute_group_rewards", fake_compute_group_rewards
    )

    # --- judge reward path: fix-correctness via _extract_verifiable_signals ---
    _judge_counter = {"i": 0}

    def fake_extract(php_code):
        i = _judge_counter["i"]
        _judge_counter["i"] += 1
        # overall in [0,100]; vary so judge group rewards are non-constant.
        return types.SimpleNamespace(overall=float(20 + (i % 4) * 20))

    monkeypatch.setattr(
        reward_pipeline, "_extract_verifiable_signals", fake_extract
    )

    async def fake_consistency_batch(samples, model="sonnet", n_votes=1, base_url=None):
        # Vary consistency too so combined judge rewards differ within a group.
        # base_url accepted (Phase 8.1 routes consistency to a local vLLM endpoint).
        return [0.3 + 0.15 * (i % 4) for i in range(len(samples))]

    monkeypatch.setattr(
        rl_judge_dispatch, "score_judge_consistency_batch", fake_consistency_batch
    )

    # --- KL seam: inject a controllable kl_sample_train_v1 (CR-04 driver) ---
    monkeypatch.setattr(
        rl_train,
        "_compute_kl_metrics",
        lambda fb_out, data: {
            "optim/kl_sample_train_v1": kl_v1,
            "optim/kl_sample_train_v2": 0.0,
            "optim/entropy": 1.0,
        },
    )

    # --- Metrics sink -> tmp file (deterministic read-back, no pollution) ---
    metrics_path = tmp_path / "rl_metrics.jsonl"
    monkeypatch.setattr(rl_train, "METRICS_PATH", str(metrics_path))

    # --- Training client (forward_backward_custom etc.) ---
    tc = MagicMock()
    fb_out = MagicMock()
    fb_out.metrics = {
        "e_frac_with_tokens:mean": 0.75,  # healthy MoE (above soft 0.7)
        "e_max_violation:mean": 0.001,
        "e_max_violation:max": 0.004,
    }
    fb_out.training_logprobs = []
    tc.forward_backward_custom.return_value = fb_out
    tc.forward_backward.return_value = fb_out
    tc.optim_step.return_value = None

    # CRITICAL CR-01 discriminator: the CORRECT path returns a SamplingClient
    # with .sample(); the WRONG path (save_weights_for_sampler) returns a bare
    # checkpoint ref with NO .sample. We wire them DIFFERENTLY so the old code
    # would AttributeError.
    fake_sc = _FakeSamplingClient()
    tc.save_weights_and_get_sampling_client.return_value = fake_sc
    bad_ref = types.SimpleNamespace(path="/fake/checkpoint")  # no .sample / .generate
    tc.save_weights_for_sampler.return_value = bad_ref

    return rl_train, tc, fake_sc, metrics_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_uses_sampling_client_and_both_pathways_survive(monkeypatch, tmp_path):
    """(a) + (b): .sample() invoked on the correct client; gen+judge survive to advantages."""
    rl_train, tc, fake_sc, _ = _wire(monkeypatch, tmp_path, kl_v1=0.0)
    args = _make_args()
    manifest = {"checkpoints": []}

    # Obtain the sampling client EXACTLY as main() does (CR-01 path).
    sampling_client = tc.save_weights_and_get_sampling_client()
    assert sampling_client is fake_sc
    # The wrong CR-01 object has no .sample — proves the discriminator is live.
    assert not hasattr(tc.save_weights_for_sampler.return_value, "sample")

    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=sampling_client,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest=manifest,
    )

    # (a) .sample was actually called on the correct client.
    assert fake_sc.sample_calls, "sampling_client.sample() must be invoked"
    assert all(c["num_samples"] == args.group_size for c in fake_sc.sample_calls)

    # (b) Not halted, and a real (non-divergent) step ran -> optim_step committed.
    assert halted is False
    assert tc.optim_step.called, "safe step must commit via optim_step"


def test_both_gen_and_judge_rollouts_reach_advantages(monkeypatch, tmp_path):
    """(b): drill into collect_rollouts -> compute_rollout_advantages directly.

    Asserts BOTH gen-origin and judge-origin completions survive the per-prompt
    constant filter (i.e. true GRPO groups, not singletons dropped as constant).
    """
    _wire(monkeypatch, tmp_path, kl_v1=0.0)
    import tinker

    import scripts.rl_rollouts as rl_rollouts
    from tinker_cookbook.rl.types import TrajectoryGroup

    args = _make_args(batch_size=4, group_size=3)
    rollouts = rl_rollouts.collect_rollouts(
        sampling_client=_FakeSamplingClient(),
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
    )
    # collect_rollouts now returns real cookbook TrajectoryGroups.
    assert rollouts and all(isinstance(g, TrajectoryGroup) for g in rollouts)
    # Origin (gen/judge) is recoverable from each Transition.logs (stamped in
    # build_trajectory_groups), so both pathways are proven present.
    origins = {
        t.logs.get("origin")
        for g in rollouts
        for traj in g.trajectories_G
        for t in traj.transitions
    }
    assert "gen" in origins, "gen rollouts must be present"
    assert "judge" in origins, "judge rollouts must be present"

    data, advantages, meta = rl_rollouts.compute_rollout_advantages(rollouts)
    assert data, "rollouts must survive to advantages (not all dropped as constant)"
    # Regression guard: real Datums carrying sampled logprobs (not plain dicts).
    assert all(isinstance(d, tinker.Datum) for d in data)
    assert "logprobs" in data[0].loss_fn_inputs
    # Surviving groups must have at least one non-zero advantage.
    assert any(a != 0.0 for a in advantages)


def test_hard_kl_halts_before_optim_step(monkeypatch, tmp_path):
    """(c): synthetic divergent KL -> halt + emergency checkpoint BEFORE optim_step."""
    rl_train, tc, fake_sc, _ = _wire(monkeypatch, tmp_path, kl_v1=0.9)  # > kl_hard 0.3

    saved = []
    monkeypatch.setattr(
        rl_train,
        "_save_checkpoint",
        lambda tc_, name, manifest: saved.append(name) or "/fake/emergency",
    )

    args = _make_args()
    halted = rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=fake_sc,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest={"checkpoints": []},
    )

    assert halted is True, "hard KL breach must halt the step"
    assert not tc.optim_step.called, (
        "CR-04: optim_step must NOT be called after a hard halt "
        "(divergent update must never be committed)"
    )
    assert any("emergency-halt" in n for n in saved), (
        "emergency checkpoint must be saved on halt"
    )


def test_metrics_jsonl_contains_rlev_fields(monkeypatch, tmp_path):
    """(d): the metrics line for the step carries the RLEV fields."""
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.05)
    args = _make_args()

    rl_train.run_training_step(
        step=0,
        tc=tc,
        sampling_client=fake_sc,
        gen_pool=_gen_pool(),
        judge_pool=_judge_pool(),
        args=args,
        manifest={"checkpoints": []},
    )

    assert metrics_path.exists(), "metrics JSONL must be written"
    lines = [ln for ln in metrics_path.read_text().splitlines() if ln.strip()]
    assert lines, "at least one metrics row must be written"
    record = json.loads(lines[-1])
    assert "kl_sample_train_v1" in record
    assert "e_frac_with_tokens_mean" in record
    assert "reward_breakdown" in record
    assert record["kl_sample_train_v1"] == pytest.approx(0.05)
    assert isinstance(record["reward_breakdown"], dict)


def test_main_wires_sampling_client_and_loads_pools(monkeypatch, tmp_path):
    """Root-cause closure: drive main() itself so CR-01/CR-02 cannot regress unseen.

    The other tests call run_training_step directly — but CR-01 (wrong sampling
    object) and CR-02 (hardcoded empty pools) live in main(), not run_training_step.
    This test drives the FULL main() live path over patched seams and asserts:
      - main obtains the client via save_weights_and_get_sampling_client() (CR-01)
        — if main reverts to save_weights_for_sampler()'s bare ref, .sample()
        AttributeErrors and this test goes red;
      - load_rl_prompts is called for BOTH "gen" and "judge" (CR-02) — if main
        reverts to gen_pool=[]/judge_pool=[], sample_interleaved_prompts raises
        ValueError and this test goes red.
    """
    rl_train, tc, fake_sc, metrics_path = _wire(monkeypatch, tmp_path, kl_v1=0.02)

    # build_training_client -> our mock tc (patch the module-level seam).
    monkeypatch.setattr(
        rl_train, "create_lora_training_client", lambda **kw: tc
    )

    # main() builds args via _parse_args, which does not (and cannot, without
    # credentials) construct the live judge clients consumed by collect_rollouts.
    # Inject them onto the parsed namespace so the live path runs offline; the
    # CR-01/CR-02 assertions below are unaffected by this.
    real_parse = rl_train._parse_args

    def parse_with_judge(argv=None):
        ns = real_parse(argv)
        # Plain string (not a MagicMock) so vars(args) stays JSON-serializable
        # for the manifest write; the patched compute_group_rewards ignores it.
        ns.judge_client = "fake-judge-client"
        ns.judge_model = "fake-judge"
        ns.consistency_model = "sonnet"
        ns.n_votes = 1
        return ns

    monkeypatch.setattr(rl_train, "_parse_args", parse_with_judge)

    # Keep manifest writes off the real output/ tree.
    monkeypatch.setattr(rl_train, "MANIFEST_PATH", str(tmp_path / "manifest.json"))

    # Patch load_rl_prompts at its SOURCE module (main imports it locally) and
    # record the pool names requested.
    import scripts.tinker_rl_data as tinker_rl_data

    requested = []

    def fake_load(pool):
        requested.append(pool)
        return _gen_pool() if pool == "gen" else _judge_pool()

    monkeypatch.setattr(tinker_rl_data, "load_rl_prompts", fake_load)

    # Avoid touching output/rl_checkpoints in _save_checkpoint.
    monkeypatch.setattr(
        rl_train, "_save_checkpoint", lambda tc_, name, manifest: "/fake/ckpt"
    )

    rl_train.main(["--total-steps", "1", "--batch-size", "4", "--group-size", "2"])

    # CR-01: the correct sampling-client accessor was used and .sample() ran.
    assert tc.save_weights_and_get_sampling_client.called, (
        "main() must obtain the sampling client via "
        "save_weights_and_get_sampling_client() (CR-01)"
    )
    assert fake_sc.sample_calls, "main() must drive .sample() on the sampling client"

    # CR-02: both pools were loaded (not hardcoded empty).
    assert "gen" in requested and "judge" in requested, (
        "main() must load_rl_prompts for both 'gen' and 'judge' pools (CR-02)"
    )

    # A metrics row was written for the single step.
    assert metrics_path.exists() and metrics_path.read_text().strip()


def test_sampler_refreshed_every_step_not_once(monkeypatch, tmp_path):
    """The on-policy sampler MUST refresh every step (J.4 stale-sampler regression).

    Root cause of the 50-step FLAT run: main() created the sampling client ONCE before
    the loop and reused it for all N steps, so every step sampled from the FROZEN
    warm-start policy and reward was constant-by-construction. The canonical cookbook
    loop refreshes the sampler from the just-updated weights every step. This asserts
    save_weights_and_get_sampling_client is called once before the loop PLUS once per
    step (== total_steps + 1). The pre-fix code called it exactly once → this is RED.
    """
    rl_train, tc, fake_sc, _ = _wire(monkeypatch, tmp_path, kl_v1=0.02)
    monkeypatch.setattr(rl_train, "create_lora_training_client", lambda **kw: tc)

    real_parse = rl_train._parse_args

    def parse_with_judge(argv=None):
        ns = real_parse(argv)
        ns.judge_client = "fake-judge-client"
        ns.judge_model = "fake-judge"
        ns.consistency_model = "sonnet"
        ns.n_votes = 1
        return ns

    monkeypatch.setattr(rl_train, "_parse_args", parse_with_judge)
    monkeypatch.setattr(rl_train, "MANIFEST_PATH", str(tmp_path / "manifest.json"))

    import scripts.tinker_rl_data as tinker_rl_data
    monkeypatch.setattr(
        tinker_rl_data, "load_rl_prompts",
        lambda pool: _gen_pool() if pool == "gen" else _judge_pool(),
    )
    monkeypatch.setattr(
        rl_train, "_save_checkpoint", lambda tc_, name, manifest: "/fake/ckpt"
    )

    n_steps = 3
    rl_train.main(["--total-steps", str(n_steps), "--batch-size", "4", "--group-size", "2"])

    # 1 initial (before loop) + n_steps per-step refreshes.
    assert tc.save_weights_and_get_sampling_client.call_count == n_steps + 1, (
        f"sampler must refresh every step: expected {n_steps + 1} calls "
        f"(1 initial + {n_steps} per-step), got "
        f"{tc.save_weights_and_get_sampling_client.call_count} — the stale-sampler "
        "bug (created once, never refreshed) is back."
    )


def test_kl_compute_failure_is_halt_worthy(monkeypatch, tmp_path):
    """CR-04 guard: a KL compute FAILURE must not read as kl=0.0 (silent no-halt).

    Drives _compute_kl_metrics directly with a failing compute and asserts the
    returned kl_sample_train_v1 is above kl_hard so check_halt would trip.
    """
    rl_train = pytest.importorskip("scripts.rl_train")

    fb_out = types.SimpleNamespace(training_logprobs=[object()])  # non-empty -> tries compute

    def boom(*a, **k):
        raise RuntimeError("synthetic KL compute failure")

    # Force the import inside _compute_kl_metrics to resolve to a failing fn.
    import sys
    fake_metrics_mod = types.ModuleType("tinker_cookbook.rl.metrics")
    fake_metrics_mod.compute_kl_sample_train = boom
    monkeypatch.setitem(sys.modules, "tinker_cookbook", types.ModuleType("tinker_cookbook"))
    monkeypatch.setitem(sys.modules, "tinker_cookbook.rl", types.ModuleType("tinker_cookbook.rl"))
    monkeypatch.setitem(sys.modules, "tinker_cookbook.rl.metrics", fake_metrics_mod)

    kl = rl_train._compute_kl_metrics(fb_out, data=[{"x": 1}])
    assert kl["optim/kl_sample_train_v1"] > rl_train.KL_HARD_DEFAULT, (
        "KL compute failure must yield a halt-worthy (not 0.0) kl_sample_train_v1"
    )
    assert rl_train.check_halt(
        kl_v1=kl["optim/kl_sample_train_v1"], e_frac=0.9
    ) is not None, "compute-failure KL must trip the hard halt"


# ---------------------------------------------------------------------------
# Judge args: _parse_args defaults + live-path judge_client guard (fix 09)
# ---------------------------------------------------------------------------


def test_parse_args_judge_defaults():
    """_parse_args([]) exposes judge_model, consistency_model, n_votes with correct defaults.

    These args are required by rl_rollouts.collect_rollouts on the live path
    (args.judge_model / args.consistency_model / args.n_votes). Without them,
    a live run would AttributeError before step 0.
    """
    rl_train = pytest.importorskip("scripts.rl_train")

    args = rl_train._parse_args([])

    assert hasattr(args, "judge_model"), "_parse_args must define args.judge_model"
    assert args.judge_model == "wp_judge", (
        f"judge_model default must be 'wp_judge' (vLLM served-model convention), "
        f"got {args.judge_model!r}"
    )

    assert hasattr(args, "consistency_model"), "_parse_args must define args.consistency_model"
    assert args.consistency_model == "sonnet", (
        f"consistency_model default must be 'sonnet', got {args.consistency_model!r}"
    )

    assert hasattr(args, "n_votes"), "_parse_args must define args.n_votes"
    assert args.n_votes == 1, (
        f"n_votes default must be 1, got {args.n_votes!r}"
    )


def test_parse_args_judge_model_override():
    """--judge-model CLI flag correctly overrides the default."""
    rl_train = pytest.importorskip("scripts.rl_train")

    args = rl_train._parse_args(["--judge-model", "openai/qwen3-wp-finetuned"])
    assert args.judge_model == "openai/qwen3-wp-finetuned"


def test_live_run_no_judge_client_raises_systemexit():
    """main() on a live (non-dry-run) path without judge_client raises SystemExit.

    Without this guard, the failure would manifest as an AttributeError deep
    inside rl_rollouts.collect_rollouts -> compute_group_rewards (args.judge_client
    is a hard attribute access, not guarded by getattr). The guard ensures the
    operator gets a clear, actionable error at startup rather than a traceback
    from inside the rollout loop.

    The dry-run path must remain unaffected (still exits without SystemExit here).
    """
    rl_train = pytest.importorskip("scripts.rl_train")

    # Simulate invoking main() without --dry-run and without a judge_client attached.
    # _parse_args([]) returns a fresh Namespace with NO judge_client attribute —
    # exactly what a CLI invocation without the arg would produce.
    args = rl_train._parse_args(["--total-steps", "1"])
    assert not hasattr(args, "judge_client"), (
        "Precondition: _parse_args must not set judge_client "
        "(it is a runtime object, not a CLI string)"
    )
    assert not args.dry_run, "Precondition: --dry-run must not be set for this test"

    with pytest.raises(SystemExit) as exc_info:
        rl_train.main(["--total-steps", "1"])

    # Confirm the exit is from OUR guard, not an unrelated SystemExit.
    msg = str(exc_info.value)
    assert "judge" in msg.lower(), (
        f"SystemExit message must mention 'judge' (guard message), got: {msg!r}"
    )
